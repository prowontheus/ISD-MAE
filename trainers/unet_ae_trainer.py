import os
import time
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from .base_trainer import BaseTrainer
import losses.pytorch_ssim as ssim
import pandas as pd
from datetime import datetime
import torch.nn as nn
import matplotlib.pyplot as plt

class Trainer(BaseTrainer):
    def __init__(self, model, cfg, device='cuda'):
        super(Trainer, self).__init__(model, cfg)
        self.cfg = cfg
        self.device = device
        self.model = model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def start_train(self, train_dataset, save_model_dir, writer, pretrained_file=None):
        '''
        load datasets and model
        '''
        self._print_config()
        self._prepare_path(save_model_dir)

        
        print('\n----------------------------  start model training -----------------------------')
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.cfg.DECAY_STEPS, gamma=self.cfg.DECAY_RATE)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.AMP)

        self.model.to(device=self.device)
        
        df_summary = pd.DataFrame(columns = ['time', 'step', 'intensity_recon_loss', 'spatial_recon_loss', 'consistent_loss'])
        df_summary.to_csv(os.path.join(self.save_path, "training_summary.csv"), index=False)
        
        min_loss = 10000.0
        for epoch in range(1, self.cfg.EPOCHS+1):
            print ('\n################################### epoch:'+str(epoch)+'/'+str(self.cfg.EPOCHS))
            self.model.train()
            
            t1 = time.time()

            intensity_recon_loss, spatial_recon_loss, consistent_loss = self.__train_epoch(train_dataset, 
                                                            optimizer, 
                                                            grad_scaler, 
                                                            scheduler, writer, epoch)
            t2 = time.time()
            
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']

            print ('\n intensity recon loss: %.3f; spatial recon loss: %.3f; consistent loss: %.3f; Lr: %.6f; Used time (s): %.4f' %
                    (intensity_recon_loss, spatial_recon_loss, consistent_loss, current_lr, t2-t1))
            
            current_time = "%s"%datetime.now()#获取当前时间
            step = "Step[%d]"%epoch
            str_intensity_recon_loss = "%f"%intensity_recon_loss
            str_spatial_recon_loss = '%f'%spatial_recon_loss
            str_consistent_loss = '%f'%consistent_loss

            list = [current_time, step, str_intensity_recon_loss, str_spatial_recon_loss, str_consistent_loss]
            df_summary = pd.DataFrame([list])
            df_summary.to_csv(os.path.join(self.save_path, "training_summary.csv"),mode='a',header=False,index=False)

            mean_loss = (intensity_recon_loss+spatial_recon_loss)/2.0
            print(f'mean_loss={mean_loss}')
            if min_loss > mean_loss:
                min_loss = mean_loss
                self.checkpoint_file = os.path.join(self.save_path, "best_weights.pth")
                print ('Saving weights to %s' % (self.checkpoint_file))
                self.model.eval()
                torch.save(self.model.state_dict(), self.checkpoint_file)
               
            self._delete_old_weights(self.cfg.MAX_KEEPS_CHECKPOINTS)

        print('\n---------------------------- model training completed ---------------------------')

    def __train_epoch(self, train_dataset, optimizer, grad_scaler, scheduler, writer, epoch):
        losses = {'intensity_recon_loss':[], 'spatial_recon_loss':[], 'consistent_loss':[]}
        for step, (images1, images2) in enumerate(train_dataset):
            if step == self.cfg.STEPS_PER_EPOCH:
                break
            
            org_images1 = images1.to(self.device)#(N, 3, 256, 256)
            org_images2 = images2.to(self.device)#(N, 3, 256, 256)

            with torch.cuda.amp.autocast(enabled=self.cfg.AMP, dtype=torch.float16):
                spatial_masked_ims, _ = self.random_mask_images_by_patch(org_images1.clone(), self.cfg.SPATIALA_MASK_SIZE, self.cfg.SPATIALA_MASK_RATIO)
                spatial_recon_ims, spatial_embeddings = self.model(spatial_masked_ims)

                intensity_masked_ims = self.random_mask_images_by_intensity(org_images2.clone(), self.cfg.INTENSITY_MASK_SIZE, self.cfg.INTENSITY_MASK_RATIO)
                intensity_recon_ims, intensity_embeddings = self.model(intensity_masked_ims)


            spatial_recon_loss = self.__calc_recon_loss(spatial_recon_ims, org_images1)
            intensity_recon_loss = self.__calc_recon_loss(intensity_recon_ims, org_images2)

            consisitent_loss = self.__calc_contrastive_loss(intensity_embeddings, spatial_embeddings)

            if writer != None:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], (epoch-1) * len(train_dataset) + step)
                writer.add_scalar('intensity_recon_loss', intensity_recon_loss.item(), (epoch-1) * len(train_dataset) + step)
                writer.add_scalar('spatial_recon_loss', spatial_recon_loss.item(), (epoch-1) * len(train_dataset) + step)
                writer.add_scalar('consisitent_loss', consisitent_loss.item(), (epoch-1) * len(train_dataset) + step)
                writer.add_scalar('loss', intensity_recon_loss.item()+spatial_recon_loss.item()+consisitent_loss.item(), (epoch-1) * len(train_dataset) + step)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(intensity_recon_loss+spatial_recon_loss+consisitent_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()

            if not (torch.isnan(intensity_recon_loss) or torch.isinf(intensity_recon_loss) or intensity_recon_loss < 0.0):
                losses['intensity_recon_loss'].append(intensity_recon_loss.detach().cpu().numpy())
            if not (torch.isnan(spatial_recon_loss) or torch.isinf(spatial_recon_loss) or spatial_recon_loss < 0.0):
                losses['spatial_recon_loss'].append(spatial_recon_loss.detach().cpu().numpy())
            if not (torch.isnan(consisitent_loss) or torch.isinf(consisitent_loss) or consisitent_loss < 0.0):
                losses['consistent_loss'].append(consisitent_loss.detach().cpu().numpy())
          
            self._draw_progress_bar(step+1, self.cfg.STEPS_PER_EPOCH)
        return np.mean(losses['intensity_recon_loss']), np.mean(losses['spatial_recon_loss']), \
            np.mean(losses['consistent_loss'])
       
    def __calc_recon_loss(self, recon, gt_images):
        '''
        the reconstruction loss between preds and orignial image
        recon:(batch_size, 1, 224, 224)
        gt_images:(batch_size, 1, 224, 224)
        '''
        if recon.dtype == torch.float16:
            recon = recon.to(torch.float32)
        if self.cfg.SHAPE == '2D':
            losses = 1.0-ssim.SSIM(window_size=15, size_average=False, shape=self.cfg.SHAPE)(recon, gt_images)
        elif self.cfg.SHAPE == '3D':
            losses = 1.0-ssim.SSIM(window_size=7, size_average=False, shape=self.cfg.SHAPE)(recon, gt_images)
        average_loss = torch.mean(losses)
        return average_loss
    
    def __calc_consistent_loss(self, intensity_embeddings, spatial_embeddings):
        loss = 0
        targets = torch.ones(intensity_embeddings[0].shape[0]).to(self.device)
        for i, intensity_embed in enumerate(intensity_embeddings):
            loss += F.cosine_embedding_loss(intensity_embed, spatial_embeddings[i], targets)*1000
        
        return loss
    
    def __calc_contrastive_loss(self, embeddings1, embeddings2):
        '''
        the contrastive loss for embeddings
        embedding1: (N, 4096)
        embedding2: (N, 4096)
        '''
        loss = 0
        for i, embedding1 in enumerate(embeddings1):
            embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
            embedding2 = embeddings2[i] / embeddings2[i].norm(dim=1, keepdim=True)

            similar_matrix = self.logit_scale * (embedding1 @ embedding2.t())
            classes = similar_matrix.shape[-1]
            labels = torch.arange(classes).to(self.device)
        
            l1 = F.cross_entropy(similar_matrix, labels)
            l2 = F.cross_entropy(similar_matrix.t(), labels)
            loss+= 0.5 * (l1 + l2)
        return loss
    
   
    def random_mask_images_by_patch(self, images, patch_size=16, mask_ratio=0.75):
        """
        Apply masking to a batch of images by patches.
        Parameters:
        - images (torch.Tensor): A batch of images with shape (N, 3, 256, 256).
        - patch_size (int): The size of each square patch.
        - mask_ratio (float): The ratio of patches to be masked in each image.
        Returns:
        - torch.Tensor: A batch of masked images.
        """
        if self.cfg.SHAPE == '2D':
            N, C, H, W = images.shape
            num_patches = (H // patch_size) * (W // patch_size) # Total number of patches in each image

            # Reshape images to (N, C, num_patches, patch_size, patch_size)
            reshaped_images = images.view(N, C, H // patch_size, patch_size, W // patch_size, patch_size)
            reshaped_images = reshaped_images.permute(0, 2, 4, 1, 3, 5).contiguous() # (N, H_patches, W_patches, C, patch_size, patch_size)

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
        elif self.cfg.SHAPE == '3D':
            N, C, H, W, D = images.shape
            num_patches = (H // patch_size) * (W // patch_size) * (D // patch_size)
            
            # Reshape images to (N, C, num_patches, patch_size, patch_size)
            reshaped_images = images.view(N, C, H // patch_size, patch_size, W // patch_size, patch_size, D // patch_size, patch_size)
            reshaped_images = reshaped_images.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous() # (N, D_patches, H_patches, W_patches, C, patch_size, patch_size, patch_size)
            
            # Generate a random mask for patches
            mask = torch.rand((N, H // patch_size, W // patch_size, D // patch_size)) < mask_ratio
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, C, patch_size, patch_size, patch_size)
            
            # Apply the mask
            masked_images = reshaped_images.clone()
            masked_images[mask_expanded] = 0

            # Reshape back to original shape
            masked_images = masked_images.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            masked_images = masked_images.view(N, C, H, W, D)

            # Also reshape mask back to original image shape for loss calculation
            mask_expanded = mask_expanded.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            mask_expanded = mask_expanded.view(N, C, H, W, D)

            return masked_images, mask_expanded[:,0,:,:]
    
    def random_mask_images_by_intensity(self, images, n_ranges=5, mask_ratio=0.70):
        """
        Apply masking to a batch of images by patches.
        Parameters:
        - images (torch.Tensor): A batch of images with shape (N, 3, 256, 256).
        - n_ranges (int): The number of intensity ranges, split [0,255] into sub ranges.
        - mask_ratio (float): The ratio of ranges to be masked in each image.
        Returns:
        - torch.Tensor: A batch of masked images.
        """
        # 将输入的images从形状(N, 3, H, W)转换为(N, H, W, 3)

        if self.cfg.SHAPE == '2D':
            images = images.permute(0, 2, 3, 1)

            ims_masked = torch.zeros_like(images)

            for i in range(images.shape[0]):
                image = images[i, ...]

                # 将图像从RGB转换为灰度
                r,g,b = image[...,0],image[...,1],image[...,2]#b是edge图像
                im_gray = 0.2989*r+0.5870*g+0.1140*b

                range_width = 1.0 / n_ranges
                mask_ranges = [[i * range_width, (i + 1) * range_width] for i in range(n_ranges)]
                im_to_mask = image.clone()

                to_mask_ranges = torch.randint(0, n_ranges, (int(n_ranges * mask_ratio),))

                for mask_range_ind in to_mask_ranges:
                    mask_low = mask_ranges[mask_range_ind][0]
                    mask_high = mask_ranges[mask_range_ind][1]

                    im_to_mask[(im_gray >= mask_low) & (im_gray < mask_high)] = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)
                ims_masked[i, ...] = im_to_mask

            # 将输出的ims_masked从形状(N, H, W, 3)转换回(N, 3, H, W)
            ims_masked = ims_masked.permute(0, 3, 1, 2)
            return ims_masked
        elif self.cfg.SHAPE == '3D':
            images = images.permute(0, 2, 3, 4, 1)

            ims_masked = torch.zeros_like(images)

            for i in range(images.shape[0]):
                image = images[i, ...]

                # 将图像从RGB转换为灰度
                r, g, b = image[..., 0], image[..., 1], image[..., 2]  # b是edge图像
                im_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

                range_width = 1.0 / n_ranges
                mask_ranges = [[i * range_width, (i + 1) * range_width] for i in range(n_ranges)]
                im_to_mask = image.clone()

                to_mask_ranges = torch.randint(0, n_ranges, (int(n_ranges * mask_ratio),))

                for mask_range_ind in to_mask_ranges:
                    mask_low = mask_ranges[mask_range_ind][0]
                    mask_high = mask_ranges[mask_range_ind][1]

                    im_to_mask[(im_gray >= mask_low) & (im_gray < mask_high)] = torch.tensor([0, 0, 0], dtype=torch.float32).to(self.device)
                ims_masked[i, ...] = im_to_mask

            # 将输出的ims_masked从形状(N, H, W, 3)转换回(N, 3, H, W)
            ims_masked = ims_masked.permute(0, 4, 1, 2, 3)
            return ims_masked
    
    def random_crop(self, tensor, crop_size):
        _, _, H, W = tensor.shape
        crop_h, crop_w = crop_size

        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()

        cropped_tensor = tensor[:, :, top:top+crop_h, left:left+crop_w]
        return cropped_tensor
