## LERAN
D. Li, Y. Zhang and Y. Liu, "Lightweight Efficient Rate-Adaptive Network for Compression-Aware Image Rescaling," IEEE Signal Processing Letters., vol. 32, pp. 1–5, 2025.

https://ieeexplore.ieee.org/document/10843843

## Abstract
Compression-aware image rescaling approaches convert high-resolution images to compressed low-resolution ones to fit various display devices or save bandwidth/storage. Inverse upscaling is successively performed to enlarge the low-resolution images to the original sizes with rich details. However, previous compression-aware image rescaling methods lack adaptivity to diverse compression rates, or require multiple large models with huge computational cost for adjusting. To overcome these challenges, we propose a lightweight efficient rate-adaptive network (LERAN) for compression-aware image rescaling. We design a non-invertible framework based on quality factor-driven feature modulation modules and an expandable training strategy, to achieve the adaptivity to various compression rates with only one light and efficient model. Moreover, alternative recursive blocks are presented for lighter weights with very small performance drop. During training, we also introduce a sparse low-resolution residual feature loss which promotes easier convergence of the model without adding further computational burden. Extensive experimental results demonstrate that our method significantly outperforms state-of-the-art compression-aware image rescaling approaches for different compression rates on popular benchmarks, with an all-in-one lightweight model and much faster speed.

## Prerequisite
- Python 3
- PyTorch
- NVIDIA GPU + CUDA
- 
pip install -r requirements.txt

## For training the stage 1 and stage 2 model, enter the "LERAN_Stage1/codes" and "LERAN_Stage2/codes" respectively, and run

sh train.sh

## For testing, enter the "LERAN_Stage2/codes" folder, and run

sh test.sh

## The codes are based on SAIN and IRN-finetune-CRM.

## If you find that our work is useful, please cite:

@article{li2025lightweight,

  title={Lightweight Efficient Rate-Adaptive Network for Compression-Aware Image Rescaling},
  
  author={Li, Dingyi and Zhang, Yang and Liu, Yu},
  
  journal={IEEE Signal Processing Letters},
  
  volume={32},
  
  pages={1--5},
  
  year={2025}
  
}
