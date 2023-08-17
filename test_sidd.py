import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)

import argparse
import options
import torch.nn as nn
import scipy.io as sio
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)
import math
import utils

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from tqdm import tqdm
import losses


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score


def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


if __name__ == '__main__':
    ######### Model ###########
    model_restoration = utils.get_arch(opt)
    model_restoration = torch.nn.DataParallel(model_restoration)
    model_restoration.cuda()
    utils.load_checkpoint(model_restoration, opt.pretrain_weights)
    model_restoration.eval()

    result_dir = opt.result_dir
    os.makedirs(result_dir, exist_ok=True)
    filepath = os.path.join(opt.val_dir, 'ValidationNoisyBlocksSrgb.mat')
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
    Inoisy /= 255.

    gtpath = os.path.join(opt.val_dir, 'ValidationGtBlocksSrgb.mat')
    gt = sio.loadmat(gtpath)
    Igt = np.float32(np.array(gt['ValidationGtBlocksSrgb']))
    Igt /= 255.

    psnr_val_rgb = []
    psnr_val_rgb2 = []
    with torch.no_grad():
        restored = np.zeros_like(Inoisy)
        for i in tqdm(range(40)):
            for k in range(32):
                noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
                B, C, H, W = noisy_patch.shape
                restored_patch = model_restoration(noisy_patch)

                restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
                psnr_val_rgb.append(utils.myPSNR(restored_patch, torch.from_numpy(Igt[i, k, :, :, :])).item())
                restored[i, k, :, :, :] = restored_patch.numpy()

        print("Merge Aver PSNR: ", sum(psnr_val_rgb) / (40 * 32))

        sio.savemat(os.path.join(result_dir, 'Idenoised.mat'), {"Idenoised": restored, })