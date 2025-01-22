from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import time
import cv2

class REALCOMP_TRAIN:
    def __init__(self, format='JPEG', quality=85):
        self.format = format
        self.quality = quality

    def __call__(self, tensors, out_type=np.uint8, min_max=(0, 1)):
        results = []
        for tensor in tensors:
            tensor = tensor.squeeze().float().cpu().detach().clamp_(*min_max) # clamp
            img = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
            if out_type == np.uint8:
                img = (img.numpy() * 255.0).round()
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR

            with BytesIO() as f:
                t0_test = time.time()
                
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                cv2.imwrite('Training_Compressed_LR_Image.jpg', img, encode_param)
                
                t1_test = time.time()
                total_time1_test1=t1_test-t0_test
                t0_test = time.time()
                
                img_jpg=cv2.imread('Training_Compressed_LR_Image.jpg')
                
                img_jpg=img_jpg[:, :, [2, 1, 0]]
                
            results.append(transforms.ToTensor()(img_jpg))

            t1_test = time.time()
            total_time1_test2=t1_test-t0_test
            
        return torch.stack(results, 0).to(torch.device('cuda'))