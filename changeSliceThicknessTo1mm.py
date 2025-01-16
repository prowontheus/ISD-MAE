from PIL import Image
import os
import numpy as np
import nibabel
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
from dataProcess.dataProcess import hu2gray
from scipy import ndimage


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def crop_image(image, mask):
    H, W, D = image.shape
    H1, W1 = mask.shape
    assert H == H1 and W == W1

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

    padded_image = np.zeros((256, 256, D), dtype=image.dtype)
    padded_mask = np.zeros((256, 256), dtype=image.dtype)
    padded_image[y_start_dst:y_end_dst, x_start_dst:x_end_dst, :] = \
        image[y_start_src:y_end_src, x_start_src:x_end_src, :]
    padded_mask[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
        mask[y_start_src:y_end_src, x_start_src:x_end_src]


    return padded_image, padded_mask


def crop3DTo2D(ct_array, mask_array, mode, dst_path, filename, num_of_2D_images):
    if len(np.where(mask_array)[0]) == 0:
        return

    H, W, D = ct_array.shape
    H1, W1, D1 = mask_array.shape
    assert H == H1 and W == W1 and D == D1

    ct_array = hu2gray(ct_array)
    mask_array[-1,:,:] = 0
    mask_array[:,-1,:] = 0
    sum_of_dimensions = np.sum(mask_array, axis=(0, 1))
    sort_index = np.argsort(sum_of_dimensions)[::-1]
    slices_list = []
    for index in sort_index:
        if len(slices_list) >= num_of_2D_images:
            break
        if len(np.where(mask_array[:, :, index])[0]) == 0:
            break
        if index + 1 >= D or index == 0:
            continue
        if np.all([abs(index-j) >= 3 for j in slices_list]):
            slices_list.append(index)

    if H != 256 or W != 256:
        for slice in slices_list:
            img = ct_array[:, :, slice - 1:slice + 2]
            msk = mask_array[:, :, slice]
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(img)
            # ax2.imshow(msk, cmap='gray')
            # plt.title(f'before crop{slice}')
            # plt.show()

            if mode == 'crop':
                padded_image, padded_mask = crop_image(img, msk)
            elif mode == 'resize':
                scale_factors = [
                    256.0 / img.shape[0],
                    256.0 / img.shape[1],
                    3.0 / img.shape[2]
                ]
                padded_image = ndimage.interpolation.zoom(img, zoom=scale_factors, order=1)
                padded_mask = ndimage.interpolation.zoom(msk, zoom=scale_factors[:2], order=0)

            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(padded_image)
            # ax2.imshow(padded_mask, cmap='gray')
            # plt.title(f'after crop{slice}')
            # plt.show()
            image = Image.fromarray(padded_image)
            mask = Image.fromarray(padded_mask)
            image.save(f'{dst_path}/{filename}_{slice}.png')
            mask.save(f'{dst_path}/{filename}_{slice}_mask.png')
    else:
        for slice in slices_list:
            img = ct_array[:, :, slice - 1:slice + 2]
            msk = mask_array[:, :, slice]
            image = Image.fromarray(img)
            mask = Image.fromarray(msk)
            image.save(f'{dst_path}/{filename}_{slice}.png')
            mask.save(f'{dst_path}/{filename}_{slice}_mask.png')



if __name__ == '__main__':
    ori_path = '~/datasets/Totalsegmentator_dataset_v201'
    dst_path = '~/datasets/TotalSegmentator2D/train'
    image_list = glob.glob(f'{ori_path}/**/ct.nii.gz', recursive=True)
    lower_lobe_left = [file.replace('ct.nii.gz', 'segmentations/lung_lower_lobe_left') for file in image_list]
    lower_lobe_right = [file.replace('ct.nii.gz', 'segmentations/lung_lower_lobe_right') for file in image_list]
    middle_lobe_right = [file.replace('ct.nii.gz', 'segmentations/lung_middle_lobe_right') for file in image_list]
    upper_lobe_left = [file.replace('ct.nii.gz', 'segmentations/lung_upper_lobe_left') for file in image_list]
    upper_lobe_right = [file.replace('ct.nii.gz', 'segmentations/lung_upper_lobe_right') for file in image_list]
    open_fail = 0
    size_fail = 0
    space_fail = 0
    resample_fail = 0
    size_list = []
    ratios = []
    for i in tqdm(range(len(image_list))):
        try:
            itk_img = sitk.ReadImage(image_list[i])
            itk_lbl1 = sitk.ReadImage(lower_lobe_left[i])
            itk_lbl2 = sitk.ReadImage(lower_lobe_right[i])
            itk_lbl3 = sitk.ReadImage(middle_lobe_right[i])
            itk_lbl4 = sitk.ReadImage(upper_lobe_left[i])
            itk_lbl5 = sitk.ReadImage(upper_lobe_right[i])
        except RuntimeError:
            open_fail += 1
            print(f'open_fail={open_fail}')
            continue
        spacing = itk_img.GetSpacing()
        spacing_1 = itk_lbl1.GetSpacing()
        spacing_2 = itk_lbl2.GetSpacing()
        spacing_3 = itk_lbl3.GetSpacing()
        spacing_4 = itk_lbl4.GetSpacing()
        spacing_5 = itk_lbl5.GetSpacing()
        size = itk_img.GetSize()
        size_1 = itk_lbl1.GetSize()
        size_2 = itk_lbl2.GetSize()
        size_3 = itk_lbl3.GetSize()
        size_4 = itk_lbl4.GetSize()
        size_5 = itk_lbl5.GetSize()
        size_list.append(size)

        if size != size_1 or size != size_2 or size != size_3 or size != size_4 or size != size_5:
            size_fail += 1
            print(f'size={size_fail}')
            continue
        if spacing != spacing_1 or spacing != spacing_2 or spacing != spacing_3 or spacing != spacing_4 or spacing != spacing_5:
            if spacing != spacing_1:
                itk_lbl1.SetSpacing(spacing)
                spacing_1 = itk_lbl1.GetSpacing()
            if spacing != spacing_2:
                itk_lbl2.SetSpacing(spacing)
                spacing_2 = itk_lbl2.GetSpacing()
            if spacing != spacing_3:
                itk_lbl3.SetSpacing(spacing)
                spacing_3 = itk_lbl3.GetSpacing()
            if spacing != spacing_4:
                itk_lbl4.SetSpacing(spacing)
                spacing_4 = itk_lbl4.GetSpacing()
            if spacing != spacing_5:
                itk_lbl5.SetSpacing(spacing)
                spacing_5 = itk_lbl5.GetSpacing()
            if spacing != spacing_1 or spacing != spacing_2 or spacing != spacing_3 or spacing != spacing_4 or spacing != spacing_5:
                space_fail += 1
                print(f'spacing={space_fail}')
                continue

        new_spacing = [1.0, 1.0, 1.0]
        mask1 = sitk.GetArrayFromImage(itk_lbl1)
        mask2 = sitk.GetArrayFromImage(itk_lbl2)
        mask3 = sitk.GetArrayFromImage(itk_lbl3)
        mask4 = sitk.GetArrayFromImage(itk_lbl4)
        mask5 = sitk.GetArrayFromImage(itk_lbl5)
        merged_mask = np.logical_or.reduce([mask1, mask2, mask3, mask4, mask5]).astype(np.uint8)
        mask_ratio = np.sum(merged_mask) / np.size(merged_mask)
        ratios.append(mask_ratio)
        if mask_ratio == 0.0:
            continue
        itk_lbl = sitk.GetImageFromArray(merged_mask)
        itk_lbl.SetSpacing(spacing)
        resampled_sitk_img = resample_img(itk_img, out_spacing=new_spacing, is_label=False)
        resampled_sitk_lbl = resample_img(itk_lbl, out_spacing=new_spacing, is_label=True)
        if resampled_sitk_img.GetSpacing() != (1.0, 1.0, 1.0) or resampled_sitk_lbl.GetSpacing() != (
                1.0, 1.0, 1.0) or resampled_sitk_img.GetSize() != resampled_sitk_lbl.GetSize():
            resample_fail += 1
            print(f'resample_fail={resample_fail}')
            continue

        sitk_array = sitk.GetArrayFromImage(resampled_sitk_img)
        sitk_mask = sitk.GetArrayFromImage(resampled_sitk_lbl)
        sitk_array = np.moveaxis(sitk_array, 0, 2)
        sitk_mask = np.moveaxis(sitk_mask, 0, 2)
        sitk_mask[np.where(sitk_mask > 1)] = 0
        filename = os.path.basename(image_list[i][:-10])
        if filename == 's0068':
            i = 0
        crop3DTo2D(sitk_array, sitk_mask, 'resize', dst_path, filename, 10)
    sorted_ratios = np.sort(ratios)
    print(size_list)
    print(f'open_fail={open_fail}')
    print(f'size_fail={size_fail}')
    print(f'space_fail={space_fail}')
    print(f'resample_fail={resample_fail}')
    print(f'sorted_ratios={sorted_ratios}')