import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
sys.path.append(os.path.join(dir_name,'..'))
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
import h5py
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True
import numpy as np
from tqdm import tqdm

def splitimage(imgtensor, crop_size=256, overlap_size=30):
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


def mergeimage(split_data, starts, crop_size = 256, resolution=(1, 3, 128, 128)):
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

    result_dir_mat = os.path.join(opt.result_dir, 'mat')
    os.makedirs(result_dir_mat, exist_ok=True)

    model_restoration = utils.get_arch(opt)
    model_restoration = torch.nn.DataParallel(model_restoration)
    model_restoration.cuda()
    utils.load_checkpoint(model_restoration, opt.pretrain_weights)
    model_restoration.eval()
    israw = False
    eval_version="1.0"

    # Load info
    infos = h5py.File(os.path.join(opt.val_dir, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']

    # Process data
    with torch.no_grad():
        for i in tqdm(range(50)):
            Idenoised = np.zeros((20,), dtype=np.object)
            filename = '%04d.mat'%(i+1)
            filepath = os.path.join(opt.val_dir, filename)
            img = h5py.File(filepath, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)

            # bounding box
            ref = bb[0][i]
            boxes = np.array(info[ref]).T

            for k in range(20):
                idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
                noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1],idx[2]:idx[3],:]).unsqueeze(0).permute(0,3,1,2).cuda()

                restored_patch = model_restoration(noisy_patch)


                restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                Idenoised[k] = restored_patch

            # save denoised data
            sio.savemat(os.path.join(result_dir_mat, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )