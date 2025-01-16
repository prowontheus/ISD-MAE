import os
import torchvision
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import nibabel
import matplotlib.pyplot as plt
from scipy import ndimage


class MyDataset(Dataset):
    def __init__(self, images_dirs, transform=None, shuffle=True):
        if not isinstance(images_dirs, list):
            images_dirs = [images_dirs]

        self.images_dirs = images_dirs

        self.images = []
        for images_dir in images_dirs:
            images = glob.glob(os.path.join(images_dir,'**', '*.jpg'), recursive=True) + glob.glob(os.path.join(images_dir,'**', '*.png'), recursive=True)
            for f in images:
                if 'mask' not in f:
                    self.images.append(f)

        print('Training samples:', len(self.images))
        if shuffle:
            np.random.shuffle(self.images)
        if transform:
            self.tx = transform
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_path = os.path.join(self.images[idx])

        im = Image.open(im_path)

        im1 = self.tx(im.copy())
        im2 = self.tx(im.copy())

        return im1, im2


def resize2shape(array, shape):
    assert len(array.shape) == 4 and array.shape[0] == 3
    scale_factors = [
        shape[0] / array.shape[1],
        shape[1] / array.shape[2],
        shape[2] / array.shape[3]
    ]
    array_list = []
    for i in range(array.shape[0]):
        arr = ndimage.interpolation.zoom(array[i, :, :, :], zoom=scale_factors, order=1)
        min_val = np.min(array)
        max_val = np.max(array)
        arr = (arr - min_val) / (max_val - min_val)
        array_list.append(arr)
    array = np.stack(array_list, axis=0)
    return array


class MyDataset3D(Dataset):
    def __init__(self, images_dirs, transform=None, shuffle=True):
        if not isinstance(images_dirs, list):
            images_dirs = [images_dirs]

        self.images_dirs = images_dirs

        self.images = []
        for images_dir in images_dirs:
            images = glob.glob(os.path.join(images_dir,'**', '*.nii.gz'), recursive=True) + glob.glob(os.path.join(images_dir,'**', '*.nii'), recursive=True)
            for f in images:
                if 'mask' not in f:
                    self.images.append(f)

        print('Training samples:', len(self.images))
        if shuffle:
            np.random.shuffle(self.images)
        if transform:
            self.tx = transform
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_path = self.images[idx]
        ct_image = nibabel.load(im_path)
        ct_array = ct_image.get_fdata()

        ct_array = np.moveaxis(ct_array, 2, 0)

        C, H, W, D  = ct_array.shape
        if H != 256 or W != 256 or D != 32:
            ct_array = resize2shape(ct_array, (256, 256, 32))
        else:
            min_val = np.min(ct_array)
            max_val = np.max(ct_array)
            ct_array = (ct_array - min_val) / (max_val - min_val)

        ct_array = ct_array.astype(np.float32)
        im1 = torch.from_numpy(ct_array.copy())
        im2 = torch.from_numpy(ct_array.copy())
        # im1 = self.tx(ct_array.copy())
        # im2 = self.tx(ct_array.copy())  # 转化为numpy

        return im1, im2
