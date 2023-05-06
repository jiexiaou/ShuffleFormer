import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
print(sys.path)
print(dir_name)

import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils
from dataset.dataset_denoise import *

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime

from losses import CharbonnierLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

# ######### Set Seeds ###########
#random.seed(1234)
#np.random.seed(1234)
#torch.manual_seed(1234)
#torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()


######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    print("Resume from " + path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)


    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)
######### DataLoader ###########
print('===> Loading datasets')

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=True, drop_last=False)

len_valset = val_dataset.__len__()
######### validation ###########
result_dir = opt.result_dir
os.makedirs(result_dir, exist_ok=True)
with torch.no_grad():
    model_restoration.eval()
    psnr_dataset = []
    psnr_model_init = []
    for ii, data_val in enumerate(tqdm(val_loader), 0):
        target = data_val[0]
        input_ = data_val[1]
        file_ = data_val[2]
        #input_ = input_.repeat(1,1,1,1)
        B = input_.shape[0]
        input_ = input_.cuda()
        target = target.cuda()

        restored = model_restoration(input_)
        #restored = torch.mean(restored, dim=0, keepdim=True)
        restored = torch.clamp(restored, 0, 1)
        psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
        psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).numpy()
        for b in range(B):
            save_file = os.path.join(result_dir, file_[b])
            utils.save_img(save_file, np.uint8(np.around(restored[b] * 255)))
    psnr_dataset = sum(psnr_dataset) / len_valset
    psnr_model_init = sum(psnr_model_init) / len_valset
    print('Input & GT (PSNR) -->%.4f dB' % (psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB' % (psnr_model_init))
