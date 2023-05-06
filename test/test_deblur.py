import numpy as np
import os, sys, math
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
from dataset.dataset_motiondeblur import *
from utils.image_utils import splitimage, mergeimage
import utils
import options

args = options.Options().init(argparse.ArgumentParser(description='image debluring')).parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if __name__ == '__main__':
    utils.mkdir(args.result_dir)
    model_restoration= utils.get_arch(args)
    model_restoration = torch.nn.DataParallel(model_restoration)
    utils.load_checkpoint(model_restoration, args.pretrain_weights)
    print("===>Testing using weights: ", args.pretrain_weights)
    model_restoration.cuda()
    model_restoration.eval()
    inp_dir = args.val_dir
    test_dataset = get_test_data(inp_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                         pin_memory=True, drop_last=False, num_workers=4)
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for input_, file_ in tqdm(test_loader):
            input_ = input_.cuda()
            B, C, H, W = input_.shape
            split_data, starts = splitimage(input_, crop_size=512, overlap_size=32)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(split_data, starts, crop_size = 512, resolution=(B, C, H, W))
            restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).numpy()
            for j in range(B):
                restored_ = restored[j]
                save_file = os.path.join(result_dir, file_[j])
                utils.save_img(save_file, np.uint8(np.around(restored_*255)))
