import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
import h5py
import cv2
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def padding(img_lq, img_gt, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'groundtruth' 
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))
        ps = self.img_options['patch_size']
        H = clean.shape[0]
        W = clean.shape[1]

        if H < ps or W < ps:
            noisy, clean = padding(noisy, clean, ps)
        H = clean.shape[0]
        W = clean.shape[1]

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]


        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)

        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename


class DataLoaderTrain_h(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain_h, self).__init__()

        self.rgb_dir = rgb_dir
        self.target_path = os.path.join(self.rgb_dir, 'groundtruth.h5')
        self.input_path = os.path.join(self.rgb_dir, 'input.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()
        self.tar_size = len(self.keys)

        self.img_options=img_options

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = self.keys[index % self.tar_size]
        ps = self.img_options['patch_size']

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        clean = np.float32(target_h5f[tar_index]) / 255.0
        noisy = np.float32(input_h5f[tar_index]) / 255.0

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        H = clean.shape[1]
        W = clean.shape[2]

        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)

        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        apply_trans = transforms_aug[random.getrandbits(3)]
        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)
        target_h5f.close()
        input_h5f.close()
        return clean, noisy


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename


class DataLoaderVal_h(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderVal_h, self).__init__()

        self.rgb_dir = rgb_dir
        self.target_path = os.path.join(self.rgb_dir, 'groundtruth.h5')
        self.input_path = os.path.join(self.rgb_dir, 'input.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        self.keys = list(target_h5f.keys())
        target_h5f.close()
        input_h5f.close()
        self.tar_size = len(self.keys)

        self.img_options=img_options

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = self.keys[index % self.tar_size]
        ps = self.img_options['patch_size']

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        clean = np.float32(target_h5f[tar_index]) / 255.0
        noisy = np.float32(input_h5f[tar_index]) / 255.0

        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        H = clean.shape[1]
        W = clean.shape[2]

        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)

        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        target_h5f.close()
        input_h5f.close()
        return clean, noisy

class DataLoaderVal_deblur(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal_deblur, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_png_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'groundtruth', x) for x in tar_files if is_png_file(x)]

        self.img_options = img_options
        self.tar_size       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size'] if img_options is not None else None

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        index_ = index % self.tar_size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename
        
##################################################################################################

class DataLoaderTest1(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest1, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


##################################################################################################
class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform
        input_dir = 'input'

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2, 0, 1)

        return noisy, noisy_filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_training_data_h(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain_h(rgb_dir, img_options, None)


def get_validation_data_h(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_h(rgb_dir, img_options, None)


def get_validation_deblur_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)