import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import torchvision.transforms as transforms
from models.model import Segmentator
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datasets.MyDatasets import MyDataset2D
from torch.utils.data import DataLoader
import random
import time
from utils.utils import get_metrics, plot_roc_curve, get_resume_model_path
import pandas as pd
import losses.pytorch_ssim as ssim
from losses.criterion import DiceLoss
from utils.utils import get_dice, gethausdorff_distance
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='Train the CRNet for chest reconstruction')
    parser.add_argument('--encoder', '-encoder', type=str, default='resnet50',
                        help='The backbone for feature extraction')
    parser.add_argument('--dataname', '-dataname', type=str, default='COVID19', help='DataSet Name')
    parser.add_argument('--pretrain', '-pretrain', action='store_true', help='Use pretrained model')
    parser.add_argument('--intensity_mask_ratio', '-intensity_mask_ratio', type=float, default=0,
                        help='the intensity mask ratio of image for contrastive')
    parser.add_argument('--spatial_mask_ratio', '-spatial_mask_ratio', type=float, default=0,
                        help='the spatial mask ratio of image for contrastive')
    parser.add_argument('--frozen_encoder', '-frozen_encoder', action='store_true', help='frozen encoder')
    parser.add_argument('--resume_epoch', '-resume_epoch', type=int, default=0, help='Initial training epoch')
    parser.add_argument('--batch_size', '-batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--input_size', '-input_size', type=int, default=256, help='the feed size of image')
    parser.add_argument('--epochs', '-epochs', type=int, default=120, help='Epochs for training')
    parser.add_argument('--val_per_epoch', '-val_per_epoch', type=int, default=1, help='validation epoch')
    parser.add_argument('--steps_per_epoch', '-steps_per_epoch', type=int, default=0, help='iterations for each epoch')
    parser.add_argument('--loss', '-loss', type=str, default='BCE', help='loss function')
    parser.add_argument('--tensorboard', '-tensorboard', action='store_true', help='Use tensorboard')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Base learning rate for training')

    return parser.parse_args()


def save_weights(model, args, current_epoch, last_saved_file_name):
    if args.pretrain:
        if args.frozen_encoder:
            saved_file_name = f'{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{current_epoch}_pretrain_frozen.pth'
        else:
            saved_file_name = f'{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{current_epoch}_pretrain.pth'
    else:
        if args.frozen_encoder:
            saved_file_name = f'{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{current_epoch}_frozen.pth'
        else:
            saved_file_name = f'{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{current_epoch}.pth'
    if last_saved_file_name != '' and os.path.exists(f'./weights/{last_saved_file_name}'):
        os.remove(f'./weights/{last_saved_file_name}')
    torch.save(model.state_dict(), f'./weights/{saved_file_name}')
    return saved_file_name


def train_one_epoch(model, dataloader, optimizer, writer, epoch):
    losses = []
    model.train()
    t1 = time.time()
    if writer != None:
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    for i, batch in enumerate(tqdm(dataloader, desc=f'Training epoch {epoch}')):
        images, masks, _ = batch
        if device == 'cuda':
            images = images.to(device)
            masks = masks.to(device)
        optimizer.zero_grad()
        scores, _ = model(images)
        loss = criterion(scores, masks)
        if writer != None:
            writer.add_scalar('loss', loss, (epoch-1) * len(dataloader) + i)
        loss = torch.mean(loss)
        # print(f'loss={loss}')
        loss.backward()
        optimizer.step()
        losses.append(loss)
    t2 = time.time()
    mean_loss = torch.mean(torch.stack(losses))
    print(f'epoch: {epoch}, loss: {mean_loss}, Used time (s): {t2 - t1:.3f}')
    return mean_loss


def evaluate_one_epoch(model, dataloader, writer, epoch, phase):
    model.eval()
    thresholds = [i * 0.01 for i in range(1, 100)]
    thresholds.append(0.001)
    thresholds.append(0.0001)
    thresholds.append(0.00001)
    thresholds.append(0.000001)
    thresholds.append(0.0000001)
    thresholds.append(0.00000001)
    thresholds.append(0.000000001)
    thresholds.append(0.999)
    thresholds.append(0.9999)
    thresholds.append(0.99999)
    thresholds.append(0.999999)
    thresholds.append(0.9999999)
    thresholds.append(0.99999999)
    thresholds.append(0.999999999)
    # y_trues = []
    # y_scores = []
    hds = []
    dices = []
    total_len = 0
    dice_mode = 'mean'
    images_paths = []
    mean_dice_every = []
    for i, batch in enumerate(tqdm(dataloader, desc=f'Evaluate epoch {epoch}')):
        images, labels, img_path = batch
        for img in img_path:
            images_paths.append(img)
        if device == 'cuda':
            images = images.to(device)
            labels = labels.to(device)
        scores, _ = model(images)
        y_trues = labels.detach().cpu().numpy()
        y_scores = scores.detach().cpu().numpy()
        batch_len = len(images)
        total_len += batch_len
        dice, max_dice, max_threshold = get_dice(thresholds, y_scores, y_trues, dice_mode)
        if dice_mode == 'mean':
            dices.append([num * batch_len for num in dice])
        elif dice_mode == 'every':
            dices.append([np.sum(num) for num in dice])
            mean_dice = np.mean(dice, axis=0)
            for d in mean_dice:
                mean_dice_every.append(d)
        hd = gethausdorff_distance(max_threshold, y_scores, y_trues)
        hds.append(hd)
        # pred = scores > 0.41
        # pred = pred.detach().cpu().numpy().astype(np.uint8)
        # mask = labels.detach().cpu().numpy()
        # for index in range(mask.shape[0]):
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     ax1.imshow(mask[index, 0, :, :], cmap='gray')
        #     ax1.set_title('label')
        #     ax2.imshow(pred[index, 0, :, :], cmap='gray')
        #     ax2.set_title('pred')
        #     # plt.savefig(f'./metrics/2D/{args.loss}/{args.dataname}/{args.encoder}/with pretrained/MaskRatio-{args.intensity_mask_ratio}-{args.spatial_mask_ratio}/batch_{i}_{index}.png', format='png')
        #     plt.show()
    if dice_mode == 'every':
        sorted_list = sorted(zip(mean_dice_every, images_paths))
        sorted_mean_dice, sorted_image_path = zip(*sorted_list)
        for j in range(len(mean_dice_every) // 2):
            abs_path = os.path.dirname(sorted_image_path[j])
            basename = os.path.basename(sorted_image_path[j])
            shutil.move(sorted_image_path[j], f'{abs_path}/low_dice/{basename}')
            mask_name = basename.replace('.png', '_mask.png')
            shutil.move(f'{abs_path}/{mask_name}', f'{abs_path}/low_dice/{mask_name}')
    hd = np.mean(hds)
    max_dice = 0.0
    max_threshold = 0.0
    dsces = []
    for i in range(len(thresholds)):
        dsc = sum([d[i] for d in dices]) / total_len
        dsces.append(dsc)
        if dsc > max_dice:
            max_dice = dsc
            max_threshold = thresholds[i]
    if writer != None:
        writer.add_scalar('dice', max_dice, epoch-1)
        writer.add_scalar('hd', hd, epoch-1)
    df = pd.DataFrame(
        {'threashold': thresholds, 'dice':dsces})
    df.to_csv(f'./metrics/{phase}_epoch{epoch:02d}_{max_dice:.3f}_{max_threshold:.3f}_{hd:.3f}.csv', index=True)
    return max_dice, max_threshold, hd


if __name__ == '__main__':

    args = get_args()
    assert args.resume_epoch >= 0

    random_seed = 3407

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.backends.cudnn.deterministic = True

    device_count = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_heads = []
    project_outdims = [128]
    for outdim in project_outdims:
        project_heads.append(dict(
            pooling='max',
            dropout=None,
            activation=None,
            out_dims=outdim
        ))
    model = Segmentator( encoder_name=args.encoder,
                         in_channels=3,
                         hidden_channels=32,
                         output_channels=128,
                         n_classes=1,
                         activation = 'sigmoid',
                         aux_params=project_heads)
    if args.resume_epoch > 0:
        # resume_model = torch.load(get_resume_model_path(args.resume_epoch), map_location='cpu')
        if args.pretrain:
            if args.frozen_encoder:
                model_path = f'./weights/2D/{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{args.resume_epoch}_pretrain_frozen.pth'
            else:
                model_path = f'./weights/2D/{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{args.resume_epoch}_pretrain.pth'
        else:
            if args.frozen_encoder:
                model_path = f'./weights/2D/{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{args.resume_epoch}_frozen.pth'
            else:
                model_path = f'./weights/2D/{args.loss}_{args.dataname}_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}_{args.encoder}_epoch_{args.resume_epoch}.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'model_path({model_path}) not found')
        resume_model = torch.load(model_path, map_location='cpu')
        model.load_weights(resume_model)
    else:
        if args.pretrain:
            modelpath = f'./pretrained weight/2D/{args.dataname}/{args.encoder}/mask_ratio_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}/best_weights.pth'
            checkpoint = torch.load(modelpath, map_location=device)
            model.load_encoder(checkpoint)
    if args.frozen_encoder:
        model.frozen_encoder()
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(device_count)))
    model.to(device)

    transforms = transforms.Compose([
        # transforms.RandomAffine(degrees=(-50, 50), translate=(0.08, 0.08), scale=(0.8, 1.2), fill=0),
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor()
    ])

    user_home_dir = os.path.expanduser("~")
    if args.dataname == 'GRAM':
        datapath = f'{user_home_dir}/datasets/image_text_clinical/gram_2D'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'COVID19_CT':
        datapath = f'{user_home_dir}/datasets/COVID-19 CT scans/2D'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'COVID19_LESION':
        datapath = f'{user_home_dir}/datasets/COVID-19 CT scan lesion segmentation'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'COVID19_2D':
        datapath = f'{user_home_dir}/datasets/image_text_clinical/covid19/2D'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'Task06_Lung':
        datapath = f'{user_home_dir}/datasets/Task06_Lung/Task06_Lung2D'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'TotalSegmentator':
        datapath = f'{user_home_dir}/datasets/TotalSegmentator2D'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'Lung_nodule_seg':
        datapath = f'{user_home_dir}/datasets/Lung nodule segmentation (Decathelon Data)'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'Lung_CT_nodule':
        datapath = f'{user_home_dir}/datasets/Lung CT nodule'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'Chest_CT':
        datapath = f'{user_home_dir}/datasets/Chest CT Segmentation'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'Lungs_CT_Scan':
        datapath = f'{user_home_dir}/datasets/Lungs CT-Scan'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    elif args.dataname == 'LIDC_IDRI':
        datapath = f'{user_home_dir}/datasets/LIDC-IDRI-slices'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    else:
        raise NotImplementedError(f'Dataset {args.dataname} not implemented.')
    dataloader_train = DataLoader(datasets_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(datasets_val, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    if args.loss == 'BCE':
        criterion = nn.BCELoss()
    elif args.loss == 'SSIM':
        criterion = ssim.SSIM(window_size=15, size_average=False)
    elif args.loss == 'DICE':
        criterion = DiceLoss()

    if args.tensorboard:
        log_dir = f'./tensorboard/2D/{args.loss}/{args.dataname}/{args.encoder}/mask_ratio_{args.intensity_mask_ratio}_{args.spatial_mask_ratio}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard_writer = SummaryWriter(log_dir)
    else:
        tensorboard_writer = None

    epoch_losses = []
    best_dice = 0.0
    threshold = 0.0
    best_epoch = args.resume_epoch
    last_saved_file_name = ''
    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        print(f'time={datetime.now()},lr={scheduler.get_last_lr()}')
        mean_loss = train_one_epoch(model, dataloader_train, optimizer, tensorboard_writer, epoch)
        epoch_losses.append(mean_loss)
        scheduler.step()
        if epoch % args.val_per_epoch == 0:
            # train_max_dice, train_max_threshold = evaluate_one_epoch(model, dataloader_train, tensorboard_writer, epoch, 'train')
            # print(f'train_max_dice={train_max_dice}, train_max_threshold={train_max_threshold}')
            val_max_dice, val_max_threshold, val_hd = evaluate_one_epoch(model, dataloader_val, tensorboard_writer, epoch, 'val')
            print(f'time={datetime.now()},val_max_dice={val_max_dice}, val_max_threshold={val_max_threshold}, val_hd={val_hd}')
            if val_max_dice > best_dice:
                best_dice = val_max_dice
                threshold = val_max_threshold
                best_epoch = epoch
                last_saved_file_name = save_weights(model, args, epoch, last_saved_file_name)
        print(f'time={datetime.now()},best_dice={best_dice}, threshold={threshold}(epoch {best_epoch})')
        print('-----------------------------------------------------------------------------------------')
    tensorboard_writer.close()
