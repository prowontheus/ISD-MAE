'''
author: huhq
'''
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

        # ct_array = np.array(im)
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(ct_array[:, :, 0].astype(np.uint8), cmap='gray')
        # ax[0, 1].imshow(ct_array[:, :, 1].astype(np.uint8), cmap='gray')
        # ax[1, 0].imshow(ct_array[:, :, 2].astype(np.uint8), cmap='gray')
        # ax[1, 1].imshow(ct_array[:, :, :].astype(np.uint8))
        # plt.show()

        im1 = self.tx(im.copy())
        im2 = self.tx(im.copy())  # 转化为numpy
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(np.moveaxis(im1.numpy(),0,2)[:, :, 0], cmap='gray')
        # ax[0, 1].imshow(np.moveaxis(im1.numpy(),0,2)[:, :, 1], cmap='gray')
        # ax[1, 0].imshow(np.moveaxis(im1.numpy(),0,2)[:, :, 2], cmap='gray')
        # ax[1, 1].imshow(np.moveaxis(im1.numpy(),0,2)[:, :, :])
        # plt.show()
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(np.moveaxis(im2.numpy(),0,2)[:, :, 0], cmap='gray')
        # ax[0, 1].imshow(np.moveaxis(im2.numpy(),0,2)[:, :, 1], cmap='gray')
        # ax[1, 0].imshow(np.moveaxis(im2.numpy(),0,2)[:, :, 2], cmap='gray')
        # ax[1, 1].imshow(np.moveaxis(im2.numpy(),0,2)[:, :, :])
        # plt.show()

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
            # print(f'resize2shape({im_path}), shape={ct_array.shape}')
            ct_array = resize2shape(ct_array, (256, 256, 32))
        else:
            min_val = np.min(ct_array)
            max_val = np.max(ct_array)
            ct_array = (ct_array - min_val) / (max_val - min_val)

        # for i in range(ct_array.shape[3]):
        #     fig, ax = plt.subplots(2, 2)
        #     ax[0, 0].imshow(ct_array[0, :, :, i], cmap='gray')
        #     ax[0, 1].imshow(ct_array[1, :, :, i], cmap='gray')
        #     ax[1, 0].imshow(ct_array[2, :, :, i], cmap='gray')
        #     ax[1, 1].imshow(np.moveaxis(ct_array[:, :, :, i], 0, 2))
        #     plt.show()

        ct_array = ct_array.astype(np.float32)
        im1 = torch.from_numpy(ct_array.copy())
        im2 = torch.from_numpy(ct_array.copy())
        # im1 = self.tx(ct_array.copy())
        # im2 = self.tx(ct_array.copy())  # 转化为numpy

        return im1, im2


if __name__ == '__main__':
    import cv2
    import numpy as np
    train_file = r'X:\chest_reconstruction\penu\gram_pos'
    train_dataset = MyDataset(images_dir=train_file, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=2, persistent_workers=True)
    for step, (im1, im2) in enumerate(train_loader):
        if step == 0:            
            print(im1.shape)
            normal_dist = torch.distributions.Normal(0, 0.1)
           
            im = im1[0,...].numpy()
          
            cv2.imwrite('1_3.png', np.uint8(im*255))
