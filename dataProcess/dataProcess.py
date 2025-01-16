import shutil

import torch
import warnings
import os.path as osp
from PIL import Image
import os
import numpy as np
import nibabel
from nibabel.filebasedimages import ImageFileError
import random
import glob
import imageio
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def hu2gray(volume, WL=-500, WW=1200):
    '''
    convert HU value to gray scale[0,255] using lung-window(WL/WW=-500/1200)
    '''
    low = WL - 0.5 * WW
    volume = (volume - low) / WW * 255.0
    volume[volume > 255] = 255
    volume[volume < 0] = 0
    volume = np.uint8(volume)
    return volume

def rgb_image_construction(images):
    lung_channel = hu2gray(images,-500, 1200)
    medi_channel = hu2gray(images, 30, 300)
    sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    if len(lung_channel.shape) > 2:
        sobel_x = []
        sobel_y = []
        for index in range(lung_channel.shape[2]):
            sobel_x.append(cv2.Sobel(lung_channel[:,:,index], -1, 1, 0, ksize=3))
            sobel_y.append(cv2.Sobel(lung_channel[:,:,index], -1, 0, 1, ksize=3))
        sobel_x = np.stack(sobel_x, axis=-1)
        sobel_y = np.stack(sobel_y, axis=-1)
    else:
        sobel_x = cv2.Sobel(lung_channel, -1, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(lung_channel, -1, 0, 1, ksize=3)
    lung_edge = np.sqrt(sobel_x**2 + sobel_y**2)

    if len(medi_channel.shape) > 2:
        sobel_x = []
        sobel_y = []
        for index in range(medi_channel.shape[2]):
            sobel_x.append(cv2.Sobel(medi_channel[:,:,index], -1, 1, 0, ksize=3))
            sobel_y.append(cv2.Sobel(medi_channel[:,:,index], -1, 0, 1, ksize=3))
        sobel_x = np.stack(sobel_x, axis=-1)
        sobel_y = np.stack(sobel_y, axis=-1)
    else:
        sobel_x = cv2.Sobel(medi_channel, -1, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(medi_channel, -1, 0, 1, ksize=3)
    medi_edge = np.sqrt(sobel_x**2 + sobel_y**2)

    edge_channel = np.copy(medi_edge)
    edge_channel[lung_edge > medi_edge] = lung_edge[lung_edge > medi_edge]
    if len(edge_channel.shape) == 2:
        edge_channel = np.expand_dims(edge_channel, axis=-1)
    return np.uint8(np.stack((lung_channel, medi_channel, edge_channel), axis=2))


if __name__ == '__main__':
    user_home_dir = os.path.expanduser("~")
    #---------------------------------------------TotalSegmentator---------------------------------------------
    # dataset_path = f'{user_home_dir}/datasets/Totalsegmentator_dataset_v201'
    # dst_path = f'{user_home_dir}/datasets/TotalSegmentator'
    # for _, dirs, _ in os.walk(dataset_path):
    #     for dir in tqdm(dirs):
    #         for root, _, files in os.walk(os.path.join(dataset_path, dir)):
    #             mask_list = []
    #             for file in files:
    #                 if file == 'ct.nii.gz':
    #                     ct_image = nibabel.load(os.path.join(root, file))
    #                     ct_array = ct_image.get_fdata()
    #                     ct_array = rgb_image_construction(ct_array)
    #                     ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
    #                     nibabel.save(ct_img, os.path.join(dst_path, f'ct_{dir}.nii.gz'))
    #
    #                 if file.startswith('lung_'):
    #                     lung_mask = nibabel.load(os.path.join(root, file))
    #                     lung_mask = lung_mask.get_fdata()
    #                     mask_list.append(lung_mask)
    #             if len(mask_list) > 0:
    #                 lung_mask = np.logical_or.reduce(mask_list, axis=0).astype(np.uint8)
    #                 lung_img = nibabel.Nifti1Image(lung_mask, affine=np.eye(4))
    #                 nibabel.save(lung_img, os.path.join(dst_path, f'lung_mask_{dir}.nii.gz'))
    #---------------------------------------------TotalSegmentator---------------------------------------------

    #-------------------------------------------------COVID19--------------------------------------------------
    # dataset_path = f'{user_home_dir}/datasets/image_text_clinical/covid19'
    # dst_path = f'{user_home_dir}/datasets/image_text_clinical/covid19_3D'
    # all_set = set(glob.glob(os.path.join(dataset_path, '**', '*.nii.gz'), recursive=True))
    # file_set = set()
    # for file in all_set:
    #     if file.endswith('_mask.nii.gz'):
    #         image_file = file.replace('_mask.nii.gz', '.nii.gz')
    #         if image_file in all_set:
    #             file_set.add(image_file)
    # num_to_move = int(len(file_set) * 0.2)
    # val_set = set(random.sample(file_set, num_to_move))
    # train_set = file_set - val_set
    # index = 1
    # for file in tqdm(train_set):
    #     ct_image = nibabel.load(file)
    #     ct_array = ct_image.get_fdata()
    #     ct_array = rgb_image_construction(ct_array)
    #     ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
    #     nibabel.save(ct_img, os.path.join(dst_path, 'train', f'{index}.nii.gz'))
    #
    #     mask = file.replace('.nii.gz', '_mask.nii.gz')
    #     shutil.copy(mask, f'{dst_path}/train/{index}_mask.nii.gz')
    #     index += 1
    #
    # for file in tqdm(val_set):
    #     ct_image = nibabel.load(file)
    #     ct_array = ct_image.get_fdata()
    #     ct_array = rgb_image_construction(ct_array)
    #     ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
    #     nibabel.save(ct_img, os.path.join(dst_path, 'val', f'{index}.nii.gz'))
    #
    #     mask = file.replace('.nii.gz', '_mask.nii.gz')
    #     shutil.copy(mask, f'{dst_path}/val/{index}_mask.nii.gz')
    #     index += 1
    #-------------------------------------------------COVID19--------------------------------------------------

    #-----------------------------------------------Task06_Lung------------------------------------------------
    dataset_path = f'{user_home_dir}/datasets/Task06_Lung'
    dst_path = f'{user_home_dir}/datasets/Task06_Lung/3D'
    all_set = [file for file in os.listdir(os.path.join(dataset_path, 'imagesTr')) if file.endswith('.nii.gz')]
    all_set = set(all_set)
    num_to_move = int(len(all_set) * 0.2)
    val_set = set(random.sample(all_set, num_to_move))
    train_set = all_set - val_set
    index = 1
    for file in tqdm(train_set):
        ct_image = nibabel.load(os.path.join(dataset_path, 'imagesTr', file))
        ct_array = ct_image.get_fdata()
        ct_array = rgb_image_construction(ct_array)
        ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
        nibabel.save(ct_img, os.path.join(dst_path, 'train', f'{index}.nii.gz'))

        mask_path = os.path.join(dataset_path, 'labelsTr', file)
        shutil.copy(mask_path, f'{dst_path}/train/{index}_mask.nii.gz')
        index += 1

    for file in tqdm(val_set):
        ct_image = nibabel.load(os.path.join(dataset_path, 'imagesTr', file))
        ct_array = ct_image.get_fdata()
        ct_array = rgb_image_construction(ct_array)
        ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
        nibabel.save(ct_img, os.path.join(dst_path, 'val', f'{index}.nii.gz'))

        mask_path = os.path.join(dataset_path, 'labelsTr', file)
        shutil.copy(mask_path, f'{dst_path}/val/{index}_mask.nii.gz')
        index += 1
    #-----------------------------------------------Task06_Lung------------------------------------------------

    #--------------------------------------------COVID-19 CT scans---------------------------------------------
    # dataset_path = f'{user_home_dir}/datasets/COVID-19 CT scans'
    # dst_path = f'{user_home_dir}/datasets/COVID-19 CT scans/3D'
    # all_set = [file for file in os.listdir(os.path.join(dataset_path, 'ct_scans')) if file.endswith('.nii')]
    # all_set = set(all_set)
    # num_to_move = int(len(all_set) * 0.2)
    # val_set = set(random.sample(all_set, num_to_move))
    # train_set = all_set - val_set
    # index = 1
    # for file in tqdm(train_set):
    #     ct_image = nibabel.load(os.path.join(dataset_path, 'ct_scans', file))
    #     ct_array = ct_image.get_fdata()
    #     ct_array = rgb_image_construction(ct_array)
    #     ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
    #     nibabel.save(ct_img, os.path.join(dst_path, 'train', f'{index}.nii'))
    #
    #     if file.startswith('coronacases'):
    #         mask_file = file.replace('_org_', '_')
    #     elif file.startswith('radiopaedia_org_covid-19-pneumonia-'):
    #         mask_file = file.replace('radiopaedia_org_covid-19-pneumonia-', 'radiopaedia_')
    #         mask_file = mask_file.replace('-dcm.nii', '.nii')
    #     infection_mask_path = os.path.join(dataset_path, 'infection_mask', mask_file)
    #     lung_mask_path = os.path.join(dataset_path, 'lung_mask', mask_file)
    #     shutil.copy(infection_mask_path, f'{dst_path}/train/{index}_infection_mask.nii')
    #     shutil.copy(lung_mask_path, f'{dst_path}/train/{index}_lung_mask.nii')
    #     index += 1
    #
    # for file in tqdm(val_set):
    #     ct_image = nibabel.load(os.path.join(dataset_path, 'ct_scans', file))
    #     ct_array = ct_image.get_fdata()
    #     ct_array = rgb_image_construction(ct_array)
    #     ct_img = nibabel.Nifti1Image(ct_array, affine=np.eye(4))
    #     nibabel.save(ct_img, os.path.join(dst_path, 'val', f'{index}.nii'))
    #
    #     if file.startswith('coronacases'):
    #         mask_file = file.replace('_org_', '_')
    #     elif file.startswith('radiopaedia_org_covid-19-pneumonia-'):
    #         mask_file = file.replace('radiopaedia_org_covid-19-pneumonia-', 'radiopaedia_')
    #         mask_file = mask_file.replace('-dcm.nii', '.nii')
    #     infection_mask_path = os.path.join(dataset_path, 'infection_mask', mask_file)
    #     lung_mask_path = os.path.join(dataset_path, 'lung_mask', mask_file)
    #     shutil.copy(infection_mask_path, f'{dst_path}/val/{index}_infection_mask.nii')
    #     shutil.copy(lung_mask_path, f'{dst_path}/val/{index}_lung_mask.nii')
    #     index += 1
    #--------------------------------------------COVID-19 CT scans---------------------------------------------

