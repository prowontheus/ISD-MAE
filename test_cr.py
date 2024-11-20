'''
author: changrj
'''

import torch
from datasets.dataloader import MyDataset
import torchvision.transforms as transforms
from cfgs.config import cfg
from models.unet_ae import UnetAE
import argparse
import warnings
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as cssim

def get_args():
    parser = argparse.ArgumentParser(description='Test the CRNet for chest reconstruction')
    parser.add_argument('--encoder', '-encoder', type=str, default='mit_b2', help='The backbone for feature extraction')
    parser.add_argument('--input_size', '-input_size', type=int, default=256, help='the feed size of image')
    parser.add_argument('--data_aug', '-data_aug', type=str, default='', help='the augmentation for dataset')
    parser.add_argument('--load', '-load', type=str, default=r'D:\projects\tmi_experiments\CTAE-semimask\checkpoints\mit_b2_s08_m75_intensity_contrastive\best_weights.pth', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='test_results/mit_b0_test', help='the path to save weights')
   
    return parser.parse_args()

def random_mask_images_by_patch(images, patch_size=16, mask_ratio=0.75):
        
    N, C, H, W = images.shape
    num_patches = (H // patch_size) * (W // patch_size) # Total number of patches in each image
  
    # Reshape images to (N, C, num_patches, patch_size, patch_size)
    reshaped_images = images.view(N, C, H // patch_size, patch_size, W // patch_size, patch_size)
    reshaped_images = reshaped_images.permute(0, 2, 4, 1, 3,5).contiguous() # (N, H_patches, W_patches, C, patch_size, patch_size)

    # Generate a random mask for patches
    mask = torch.rand((N, H // patch_size, W // patch_size)) < mask_ratio
    mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, C, patch_size, patch_size)

    # Apply the mask
    masked_images = reshaped_images.clone()
    masked_images[mask_expanded] = 0

    # Reshape back to original shape
    masked_images = masked_images.permute(0, 3, 1, 4, 2, 5).contiguous()
    masked_images = masked_images.view(N, C, H, W)

    # Also reshape mask back to original image shape for loss calculation
    mask_expanded = mask_expanded.permute(0, 3, 1, 4, 2, 5).contiguous()
    mask_expanded = mask_expanded.view(N, C, H, W)

    return masked_images, mask_expanded[:,0,:,:]

def random_mask_images_by_intensity(images, n_ranges=5, mask_ratio=0.70):
       
    # 将输入的images从形状(N, 3, H, W)转换为(N, H, W, 3)
    images = images.permute(0, 2, 3, 1)
    
    ims_masked = torch.zeros_like(images)
    for i in range(images.shape[0]):
        image = images[i, ...]
        
        # 将图像从RGB转换为灰度
        r,g,b = image[...,0],image[...,1],image[...,2]
        im_gray = 0.2989*r+0.5870*g+0.1140*b
            
        range_width = 1.0 / n_ranges
        mask_ranges = [[i * range_width, (i + 1) * range_width] for i in range(n_ranges)]
        im_to_mask = image.clone()
      
        to_mask_ranges = torch.randint(0, n_ranges, (int(n_ranges * mask_ratio),))
        
        for mask_range_ind in to_mask_ranges:
            mask_low = mask_ranges[mask_range_ind][0]
            mask_high = mask_ranges[mask_range_ind][1]
            
            im_to_mask[(im_gray >= mask_low) & (im_gray < mask_high)] = torch.tensor([0, 0, 0],dtype=torch.float32)
        
        ims_masked[i, ...] = im_to_mask
    
    # 将输出的ims_masked从形状(N, H, W, 3)转换回(N, 3, H, W)
    ims_masked = ims_masked.permute(0, 3, 1, 2)
    return ims_masked


if __name__ == '__main__':
    import glob
    import os.path as osp
    from PIL import Image
    import pandas as pd

    warnings.filterwarnings('ignore')
    args = get_args()

    cfg.INPUT_SHAPE = (args.input_size, args.input_size)
    transforms = transforms.Compose([transforms.Resize((cfg.INPUT_SHAPE)), transforms.ToTensor()])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    project_heads = []
    project_outdims =  [128]
    for outdim in project_outdims:
        project_heads.append(dict(
            pooling='max',
            dropout=None,
            activation=None,
            out_dims= outdim
        ))
    model = UnetAE(encoder_name=args.encoder,  classes=3, activation='sigmoid', aux_params= project_heads)
   
    model.load_state_dict(torch.load(args.load, map_location=device))
    
    model.to(device=device)
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    ssims = []
    files = []
    imagefs = glob.glob(osp.join(r'D:\data\chest_reconstruction\pneumonia', '**', '*mask.jpg'), recursive=True)
    imagefs = imagefs[0:500]
    imagefs.extend(glob.glob(osp.join(r'D:\data\chest_reconstruction\mediastinal', '**', '*mask.jpg'), recursive=True))
    imagefs = imagefs[0:1000]
    total = len(imagefs)
    done=1
    for f in imagefs:
        print(done, total)
        done+=1
        #if done == 2000:break

        f = f.replace('_mask', '')
        
        image = Image.open(f)
        image = transforms(image)
        image = image.to(device)
        image = image[None, ...]
      
        files.append(osp.basename(f))
        with torch.no_grad():         
            #spatial_masked_ims, _ = random_mask_images_by_patch(image.clone(), 16, 0.70)
            #spatial_masked_ims= random_mask_images_by_intensity(image.clone(), 8, 0.75)
            spatial_masked_ims = image.clone()
            recons, _, = model(spatial_masked_ims)
               
            recon = recons[0,...].cpu().numpy()
            input = image[0,...].cpu().numpy()
            recon = recon.transpose((1,2,0))
            input = input.transpose((1,2,0))

            r,g,b = recon[...,0],recon[...,1],recon[...,2]
            recon = np.uint8(255*(0.2989*r+0.5870*g+0.1140*b))

            r,g,b = input[...,0],input[...,1],input[...,2]
            input = np.uint8(255*(0.2989*r+0.5870*g+0.1140*b))
            ssim = cssim(recon, input, full=False)
            print(ssim)
            ssims.append(ssim)

files = np.array(files)
ssims = np.array(ssims)

data = np.hstack([files[:,None], ssims[:,None]])
df = pd.DataFrame(data, columns=['filename','ssim'])
df.to_csv('TCS-MAE_org.csv', index=False)


        




                


