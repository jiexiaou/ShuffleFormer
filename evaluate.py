import os
import argparse

import torch.nn as nn
import torch
import time
from model import Uformer

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/DGUNet_plus/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/DGUNet_plus.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = Uformer(img_size=256, embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)

print("===>Testing using weights: ",args.weights)
#model_restoration.cuda()
#model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def main():
    NUM = 10
    print_network(model_restoration)
    with torch.no_grad():
        start_time = time.time()
        for ii in range(NUM):
            input_ = torch.randn(1, 3, 256, 256).cuda()
            restored = model_restoration(input_)
        end_time = time.time()
    print("Average time: %.4f" %((end_time - start_time)/NUM))


if __name__=='__main__':
    main()