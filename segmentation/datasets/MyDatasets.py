import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
import nibabel
from scipy import ndimage
import matplotlib.pyplot as plt


class MyDataset2D(Dataset):
    def __init__(self, dataset_path: str, transform, is_training: bool = True, suffix: str = 'png'):
        self.datas = []
        self.masks = []
        self.is_training = is_training
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.get_images_and_labels()
        if transform:
            self.tx = transform
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        self.label_resize = torchvision.transforms.Resize((256, 256))
        self.label_toTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img = Image.open(self.datas[index])
        label = Image.open(self.masks[index])
        img = self.tx(img)
        label = self.label_resize(label)
        label = np.array(label).astype(np.float32)
        label[label > 1.0] = 1.0
        if len(label.shape) > 2:
            label = label[:, :, 0]
        label = self.label_toTensor(label)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(np.moveaxis(img.numpy(), 0, 2))
        # ax2.imshow(label.numpy()[0, :, :], cmap='gray')
        # plt.show()
        return img, label, self.datas[index]

    def get_images_and_labels(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Path '{self.dataset_path}' does not exist.")
        all_lists = glob.glob(os.path.join(self.dataset_path, f'**/*.{self.suffix}'), recursive=True)
        self.masks = [image for image in all_lists if 'mask' in image]
        for img in self.masks:
            self.datas.append(img[0:-(len(self.suffix)+6)] + f'.{self.suffix}')

        assert len(self.datas) > 0, f'No image files found in path {self.dataset_path}.'
        assert len(self.masks) > 0, f'No mask files found in path {self.dataset_path}.'
        # if self.is_training == True:
        #     self.datas = self.datas[0:2000]
        #     self.masks = self.masks[0:2000]
        # else:
        #     self.datas = self.datas[0:500]
        #     self.masks = self.masks[0:500]


class MyDataset3D(Dataset):
    def __init__(self, dataset_path: str, transform, is_training: bool = True, suffix: str = 'nii.gz'):
        self.datas = []
        self.masks = []
        self.is_training = is_training
        self.dataset_path = dataset_path
        self.suffix = suffix
        self.get_images_and_labels()
        if transform:
            self.tx = transform
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        self.label_resize = torchvision.transforms.Resize((256, 256))
        self.label_toTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        image_path = self.datas[index]
        mask_path = self.masks[index]
        ct_image = nibabel.load(image_path)
        ct_array = ct_image.get_fdata()
        mask_image = nibabel.load(mask_path)
        mask_array = mask_image.get_fdata()

        ct_array = np.moveaxis(ct_array, 2, 0)

        C, H, W, D = ct_array.shape
        if H != 256 or W != 256 or D != 32:
            # print(f'resize2shape({image_path}), shape={ct_array.shape}')
            scale_factors = [
                256.0 / ct_array.shape[1],
                256.0 / ct_array.shape[2],
                32.0 / ct_array.shape[3]
            ]
            array_list = []
            for i in range(ct_array.shape[0]):
                array = ndimage.interpolation.zoom(ct_array[i, :, :, :], zoom=scale_factors, order=1)
                min_val = np.min(array)
                max_val = np.max(array)
                array = (array - min_val) / (max_val - min_val)
                array_list.append(array)
            ct_array = np.stack(array_list, axis=0)
            mask_array = ndimage.interpolation.zoom(mask_array, zoom=scale_factors, order=1)
        else:
            min_val = np.min(ct_array)
            max_val = np.max(ct_array)
            ct_array = (ct_array - min_val) / (max_val - min_val)

        mask_array[mask_array >= 0.1] = 1
        mask_array[mask_array < 0.1] = 0

        ct_array = ct_array.astype(np.float32)
        mask_array = mask_array.astype(np.float32)

        # for i in range(20,ct_array.shape[3],1):
        #     fig, ax = plt.subplots(2, 2)
        #     ax[0, 0].imshow(ct_array[0, :, :, i], cmap='gray')
        #     ax[0, 1].imshow(ct_array[1, :, :, i], cmap='gray')
        #     ax[1, 0].imshow(ct_array[2, :, :, i], cmap='gray')
        #     ax[1, 1].imshow(np.moveaxis(ct_array[:, :, :, i], 0, 2))
        #     plt.show()
        #
        #     fig2, ax2 = plt.subplots(1, 2)
        #     ax2[0].imshow(mask_array[:,:,i], cmap='gray')
        #     ax2[1].imshow(mask_array[:,:,i], cmap='gray')
        #     plt.show()

        img = torch.from_numpy(ct_array)
        mask = torch.from_numpy(mask_array).unsqueeze(0)

        return img, mask, self.datas[index]

    def get_images_and_labels(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Path '{self.dataset_path}' does not exist.")
        all_lists = glob.glob(os.path.join(self.dataset_path, f'**/*.{self.suffix}'), recursive=True)
        self.datas = [image for image in all_lists if 'mask' not in image]
        for img in self.datas:
            self.masks.append(img[0:-7] + f'_mask.{self.suffix}')

        assert len(self.datas) > 0, f'No image files found in path {self.dataset_path}.'
        assert len(self.masks) > 0, f'No mask files found in path {self.dataset_path}.'
        if self.is_training == True:
            self.datas = self.datas[0:2000]
            self.masks = self.masks[0:2000]
        else:
            self.datas = self.datas[0:500]
            self.masks = self.masks[0:500]


class COVID19_CT_3D_Dataset(MyDataset3D):
    def __init__(self, dataset_path: str, transform, is_training: bool = True, suffix: str = 'nii'):
        super(COVID19_CT_3D_Dataset, self).__init__(dataset_path, transform, is_training, suffix)

    def get_images_and_labels(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Path '{self.dataset_path}' does not exist.")
        all_lists = glob.glob(os.path.join(self.dataset_path, f'**/*.{self.suffix}'), recursive=True)
        self.datas = [image for image in all_lists if 'mask' not in image]
        for img in self.datas:
            self.masks.append(img[0:-4] + f'_lung_mask.{self.suffix}')

        assert len(self.datas) > 0, f'No image files found in path {self.dataset_path}.'
        assert len(self.masks) > 0, f'No mask files found in path {self.dataset_path}.'

class Total_Segmentator_3D_Dataset(MyDataset3D):
    def __init__(self, dataset_path: str, transform, is_training: bool = True, suffix: str = 'nii.gz'):
        super(Total_Segmentator_3D_Dataset, self).__init__(dataset_path, transform, is_training, suffix)

    def get_images_and_labels(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Path '{self.dataset_path}' does not exist.")
        all_lists = glob.glob(os.path.join(self.dataset_path, f'**/*.{self.suffix}'), recursive=True)
        self.datas = [image for image in all_lists if 'mask' not in image]
        for img in self.datas:
            self.masks.append(img.replace('ct_', 'lung_mask_'))

        assert len(self.datas) > 0, f'No image files found in path {self.dataset_path}.'
        assert len(self.masks) > 0, f'No mask files found in path {self.dataset_path}.'