from PIL import Image
import os
import numpy as np
import nibabel
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
from dataProcess.dataProcess import hu2gray


def crop_image(image, mask):
    H, W, C, D = image.shape
    H1, W1, D1 = mask.shape
    assert H == H1 and W == W1 and D == D1

    x_min = np.min(np.where(mask)[1])
    x_max = np.max(np.where(mask)[1])
    y_min = np.min(np.where(mask)[0])
    y_max = np.max(np.where(mask)[0])
    height = y_max - y_min
    width = x_max - x_min
    middle_y = (y_min + y_max) // 2
    middle_x = (x_min + x_max) // 2

    if H <= 256:
        y_start_src = 0
        y_end_src = H
        y_start_dst = (256 - H) // 2
        y_end_dst = y_start_dst + H
    else:
        y_start_src = middle_y - 128
        y_end_src = middle_y + 128
        if y_start_src < 0:
            y_start_src = 0
            y_end_src = 256
        elif y_end_src > H:
            y_start_src = H - 256
            y_end_src = H
        y_start_dst = 0
        y_end_dst = 256

    if W <= 256:
        x_start_src = 0
        x_end_src = W
        x_start_dst = (256 - W) // 2
        x_end_dst = x_start_dst + W
    else:
        x_start_src = middle_x - 128
        x_end_src = middle_x + 128
        if x_start_src < 0:
            x_start_src = 0
            x_end_src = 256
        elif x_end_src > W:
            x_start_src = W - 256
            x_end_src = W
        x_start_dst = 0
        x_end_dst = 256

    padded_image = np.zeros((256, 256, C, 32), dtype=image.dtype)
    padded_mask = np.zeros((256, 256, 32), dtype=image.dtype)
    padded_image[y_start_dst:y_end_dst, x_start_dst:x_end_dst, :, :] = \
        image[y_start_src:y_end_src, x_start_src:x_end_src, :, :]
    padded_mask[y_start_dst:y_end_dst, x_start_dst:x_end_dst, :] = \
        mask[y_start_src:y_end_src, x_start_src:x_end_src, :]

    return padded_image, padded_mask



def crop3DTo2D(image_file, mask_file):
    ret = 0
    image_name = os.path.basename(image_file)
    mask_name = os.path.basename(mask_file)
    ct_image = nibabel.load(image_file)
    ct_array = ct_image.get_fdata()
    mask_image = nibabel.load(mask_file)
    mask_array = mask_image.get_fdata()
    if len(np.where(mask_array)[0]) == 0:
        return

    H, W, C, D = ct_array.shape
    H1, W1, D1 = mask_array.shape
    assert H == H1 and W == W1 and D == D1

    sum_of_dimensions = np.sum(mask_array, axis=(0, 1))
    max_index = np.argmax(sum_of_dimensions)
    if D > 32:

        if max_index - 16 < 0:
            start_index = 0
            end_index = 32
        elif max_index + 16 > mask_array.shape[2]:
            start_index = mask_array.shape[2] - 32
            end_index = mask_array.shape[2]
        else:
            start_index = max_index - 16
            end_index = max_index + 16
        padded_image = ct_array[:, :, :, start_index:end_index]
        padded_mask = mask_array[:, :, start_index:end_index]
    elif D < 32:
        padded_image = np.zeros((H, W, C, 32), dtype=ct_array.dtype)
        padded_mask = np.zeros((H, W, 32), dtype=mask_array.dtype)
        start_index = (32 - D) // 2
        end_index = start_index + D
        padded_image[:, :, :, start_index:end_index] = ct_array
        padded_mask[:, :, start_index:end_index] = mask_array
    else:
        padded_image = ct_array
        padded_mask = mask_array

    plt.show()
    if H != 256 or W != 256:
        padded_image, padded_mask = crop_image(padded_image, padded_mask)
    else:
        ret = 1

    padded_img = nibabel.Nifti1Image(padded_image, affine=np.eye(4))
    nibabel.save(padded_img, os.path.join(dst_path, image_name))
    padded_mak = nibabel.Nifti1Image(padded_mask, affine=np.eye(4))
    nibabel.save(padded_mak, os.path.join(dst_path, mask_name))
    return ret


if __name__ == '__main__':
    src_path = '~/datasets/TotalSegmentator/origin/val'
    dst_path = '~/jerry/datasets/TotalSegmentator/cropHWD/val'

    mask_list = glob.glob(f'{src_path}/**/*mask.nii.gz', recursive=True)
    file_list = []
    for mask in mask_list:
        file = mask.replace('_mask_', '_')
        file = file.replace('lung_', 'ct_')
        file_list.append(file)
    j = 0
    for i in tqdm(range(len(file_list))):
        image_file = file_list[i]
        mask_file = mask_list[i]
        j += crop3DTo2D(image_file, mask_file)
    print(f'j={j}/{len(file_list)}')

