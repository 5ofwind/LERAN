import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.compressor_train import REALCOMP_TRAIN
from models.jpeg import DiffJPEG
from models.modules.loss import ReconstructionLoss
from models.modules.quantization import Quantization

import time
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
import itertools
import numpy as np
import random

logger = logging.getLogger('base')

class LERAN_Stage1(BaseModel):
    def __init__(self, opt):
        super(LERAN_Stage1, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        
        self.netG = networks.define_G(opt).to(self.device)
        self.netG2 = networks.define_G2(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netG2 = DistributedDataParallel(self.netG2, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
            self.netG2 = DataParallel(self.netG2)
            
        # 创建输入网络的tensor
        tensor = torch.rand((1, 3, 480, 720),device="cuda")

        # 分析FLOPs
        flops = FlopCountAnalysis(self.netG, tensor)
        print("FLOPs: ", flops.total())

        # 分析parameters
        print(parameter_count_table(self.netG))
        
        # 创建输入网络的tensor
        tensor2 = torch.rand((1, 3, 240, 360),device="cuda")

        # 分析FLOPs
        flops = FlopCountAnalysis(self.netG2, tensor2)
        print("FLOPs: ", flops.total())

        # 分析parameters
        print(parameter_count_table(self.netG2)) 
        
        # print network
        self.print_network()
        self.print_network2()
        self.load()
        self.load2()

        self.Quantization = Quantization()
        
        if train_opt['test_val_comp_quality']:
            self.test_val_quality=train_opt['test_val_comp_quality']
            
        if train_opt['use_diffcomp']:
            if train_opt['train_comp_quality1'] and train_opt['train_comp_quality2']:
                
                if train_opt['train_comp_quality2']>train_opt['train_comp_quality1']:
                    self.current_train_quality=random.randint(train_opt['train_comp_quality1'],train_opt['train_comp_quality2']-1)
                else:
                    self.current_train_quality=train_opt['train_comp_quality1']
                
                self.diffcomp = DiffJPEG(differentiable=True, quality=self.current_train_quality).cuda()
                
            else:
                self.diffcomp = DiffJPEG(differentiable=True, quality=85).cuda()
        if train_opt['use_realcomp']:
            self.realcomp_train = REALCOMP_TRAIN(format=train_opt['comp_format'], quality=train_opt['test_val_comp_quality'])

        if self.is_train:
            self.netG.train()
            self.netG2.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            optim_params2 = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            for k, v in self.netG2.named_parameters():
                if v.requires_grad:
                    optim_params2.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
          
            self.optimizer = torch.optim.Adam(itertools.chain(optim_params,optim_params2), lr=train_opt['lr'],
                                                weight_decay=wd,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gmm_batch(self, dims):
        return self.net.module.gmm.sample(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_backward(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()
        
        # forward downscaling
        self.input = self.real_H
        
        LR = self.netG(x=self.input,network_comp_quality=self.current_train_quality)

        LR_ref = self.ref_L.detach()
        
        LR = LR + LR_ref #residual connection

        l_forw_fit1 = self.loss_forward(LR, LR_ref)

        # backward upscaling
        LR = self.Quantization(LR)
        LR_ = self.diffcomp(LR)
        y_ = LR_
        x_samples=self.netG2(x=y_,network_comp_quality=self.current_train_quality)
        
        x_samples_recon = x_samples[:, :3, :, :]
        
        l_back_rec = self.loss_backward(x_samples_recon, self.real_H)
        
        loss = l_back_rec + l_forw_fit1
        
        loss.backward()
        
        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
            nn.utils.clip_grad_norm_(self.netG2.parameters(), self.train_opt['gradient_clipping'])
        
        self.optimizer.step()

        # set log
        self.log_dict['l_forw_fit1'] = l_forw_fit1.item()
        self.log_dict['l_back_rec2'] = l_back_rec.item()
        
    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        self.netG2.eval()
        
        with torch.no_grad():
            self.forw_L = self.Quantization(self.netG(x=self.input,network_comp_quality=self.test_val_quality)+self.ref_L) #residual connection
            self.forw_L2 = self.realcomp_train(self.forw_L)
            self.fake_H = self.netG2(x=self.forw_L2,network_comp_quality=self.test_val_quality)
            

        self.netG.train()
        self.netG2.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            t0 = time.time()
            LR_img = self.Quantization(self.netG(x=HR_img,network_comp_quality=self.test_val_quality)+self.ref_L) #residual connection
            t1 = time.time()
            total_time0=t1-t0
            print("===> Average Downsampling Time: %.4f sec." % (total_time0)) #(t1 - t0))
            
        self.netG.train()

        return LR_img, total_time0

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2 - 1), Lshape[2], Lshape[3]]

        self.netG2.eval()
        with torch.no_grad():
            
            t0 = time.time()
            
            y_ = LR_img
            HR_img = self.netG2(x=y_,network_comp_quality=self.test_val_quality)
                        
            t1 = time.time()
            total_time1=t1-t0
            print("===> Average Upsampling Time: %.4f sec." % (total_time1)) #(t1 - t0))
            
        self.netG2.train()

        return HR_img, total_time1

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            #logger.info(s)

    def print_network2(self):
        s, n = self.get_network_description(self.netG2)
        if isinstance(self.netG2, nn.DataParallel) or isinstance(self.netG2, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG2.__class__.__name__,
                                             self.netG2.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG2.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            #logger.info(s)
            
    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load2(self):            
        load_path_G = self.opt['path2']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG2, self.opt['path2']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'netG', iter_label)
        self.save_network(self.netG2, 'netG2', iter_label)