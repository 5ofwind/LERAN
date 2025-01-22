import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockNoBN(nn.Module):
    def __init__(self, nf=64, model='MIMO-VRN'):
        super(ResidualBlockNoBN, self).__init__()
        self.conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=False)     
        self.conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=False)   
        # honestly, there's no significant difference between ReLU and leaky ReLU in terms of performance here
        # but this is how we trained the model in the first place and what we reported in the paper
        if model == 'LSTM-VRN':
            self.relu = nn.ReLU(inplace=True)
        elif model == 'MIMO-VRN':
            self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv3(x))
        out = self.conv4(out)      
        return identity + out
    
class PredictiveModule(nn.Module):
        def __init__(self, channel_in, block_num_rbm=6):
                super(PredictiveModule, self).__init__()

                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

                number1=64
                number2=number1//4
                self.conv_in1 = nn.Conv2d(3, number2, 3, 1, 1, bias=True)
                
                self.pixel_unshuffle = nn.PixelUnshuffle(2)
                
                residual_block = []
                residual_block2 = []
                
                for i in range(block_num_rbm):
                        residual_block.append(ResidualBlockNoBN(number1))
                residual_block2.append(ResidualBlockNoBN(number1))
                self.residual_block = nn.Sequential(*residual_block)
                self.residual_block2 = nn.Sequential(*residual_block2)
                self.conv_last = nn.Conv2d(number1, 3, 3, 1, 1, bias=False)

                for p in self.parameters():
                    p.requires_grad=False

                self.conv_adjust=nn.Conv2d(number1, number1, 1, 1, 0, bias=True)
                self.conv_adjust2=nn.Conv2d(number1, number1, 1, 1, 0, bias=True)

        def forward(self, x, network_comp_quality):
            
            
                if network_comp_quality>85:
                    network_comp_quality=85
                    
                num_multiply=(85-network_comp_quality)/200          
                
                x = self.lrelu(self.conv_in1(x))
                x = self.pixel_unshuffle(x)
                
                x=self.conv_adjust(x)*num_multiply+x
                
                for i in range(3):
                    x = self.residual_block2(x)
                    x = self.residual_block[i](x)
                
                for i in range(3,6):
                    x = self.residual_block2(x)
                    x = self.residual_block[i](x)
                x=self.conv_adjust2(x)*num_multiply+x
                
                x = self.conv_last(x)

                return x

class PredictiveModule2(nn.Module):
        def __init__(self, channel_in, block_num_rbm=10):
                super(PredictiveModule2, self).__init__()

                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

                number1=68
                self.conv_in1 = nn.Conv2d(3, number1, 3, 1, 1, bias=True)#64
                
                residual_block = []
                residual_block2 = []
                
                for i in range(block_num_rbm):
                        residual_block.append(ResidualBlockNoBN(number1))#*3#channel_in #192
                
                residual_block2.append(ResidualBlockNoBN(number1))#*3#channel_in #192
                
                self.residual_block = nn.Sequential(*residual_block)
                self.residual_block2 = nn.Sequential(*residual_block2)
                self.upconv1 = nn.Conv2d(number1, number1 * 4, 3, 1, 1, bias=False)
                self.pixel_shuffle = nn.PixelShuffle(2)
                self.HRconv = nn.Conv2d(number1, number1, 3, 1, 1, bias=False)
                self.conv_last = nn.Conv2d(number1, 3, 3, 1, 1, bias=False)
                
                for p in self.parameters():
                    p.requires_grad=False
                    
                self.conv_adjust=nn.Conv2d(number1, number1, 1, 1, 0, bias=True)
                self.conv_adjust2=nn.Conv2d(number1, number1, 1, 1, 0, bias=True)

        def forward(self, x, network_comp_quality):
            
                if network_comp_quality>85:
                    network_comp_quality=85
                    
                num_multiply=(85-network_comp_quality)/200
                
                x = self.lrelu(self.conv_in1(x))
               
                x=self.conv_adjust(x)*num_multiply+x
                
                for i in range(5):
                    x = self.residual_block2(x)
                    x = self.residual_block[i](x)
                
                for i in range(5,10):
                    x = self.residual_block2(x)
                    x = self.residual_block[i](x)
                x=self.conv_adjust2(x)*num_multiply+x

                x = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
                x = self.lrelu(self.HRconv(x))
                x = self.conv_last(x)
                
                return x
    
class Net1(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(Net1, self).__init__()
        self.pm = PredictiveModule(3)
        

    def forward(self, x, network_comp_quality=85):
        out=self.pm(x, network_comp_quality)
        return out
        
class Net2(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2):
        super(Net2, self).__init__()
        self.pm2 = PredictiveModule2(3)

    def forward(self, x, network_comp_quality=85):
        out=self.pm2(x, network_comp_quality)
        out[:, :3, :, :] = out[:, :3, :, :] + F.interpolate(x, scale_factor=2, mode="bicubic")
        return out