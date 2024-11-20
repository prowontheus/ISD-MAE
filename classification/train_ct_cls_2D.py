import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import torchvision.transforms as transforms
from model import Classifier
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datasets.MyDatasets import MyDataset2D
from torch.utils.data import DataLoader
import random
import time
from utils.utils import get_metrics, plot_roc_curve, get_resume_model_path
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Train the CRNet for chest reconstruction')
    parser.add_argument('--encoder', '-encoder', type=str, default='resnet50',
                        help='The backbone for feature extraction')
    parser.add_argument('--dataname', '-dataname', type=str, default='COVID19', help='DataSet Name')
    parser.add_argument('--pretrain', '-pretrain', action='store_true', help='Use pretrained model')
    parser.add_argument('--spatial_mask_ratio', '-spatial_mask_ratio', type=float, default=0,
                        help='the mask ratio of image for contrastive')
    parser.add_argument('--frozen_encoder', '-frozen_encoder', action='store_true', help='frozen encoder')
    parser.add_argument('--resume_epoch', '-resume_epoch', type=int, default=0, help='Initial training epoch')
    parser.add_argument('--batch_size', '-batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--input_size', '-input_size', type=int, default=256, help='the feed size of image')
    parser.add_argument('--epochs', '-epochs', type=int, default=120, help='Epochs for training')
    parser.add_argument('--val_per_epoch', '-val_per_epoch', type=int, default=1, help='validation epoch')
    parser.add_argument('--steps_per_epoch', '-steps_per_epoch', type=int, default=0, help='iterations for each epoch')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Base learning rate for training')

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, epoch):
    losses = []
    model.train()
    t1 = time.time()
    for i, batch in enumerate(tqdm(dataloader, desc=f'Training epoch {epoch}')):
        images, labels = batch
        if device == 'cuda':
            images = images.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    t2 = time.time()
    mean_loss = torch.mean(torch.stack(losses))
    print(f'epoch: {epoch}, loss: {mean_loss}, Used time (s): {t2 - t1:.3f}')
    return mean_loss


def evaluate_one_epoch(model, dataloader, epoch, desc):
    model.eval()
    thresholds = [i * 0.01 for i in range(1, 100)]
    y_true = []
    y_scores = []
    for i, batch in enumerate(tqdm(dataloader, desc=f'Evaluate epoch {epoch}')):
        images, labels = batch
        if device == 'cuda':
            images = images.to(device)
            labels = labels.to(device)
        logits = model(images)
        scores = torch.sigmoid(logits)
        y_true.append(labels.detach().cpu().numpy())
        y_scores.append(scores.detach().cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)
    f1s, accuracies, precisions, recalls, tns, fps, fns, tps, tprs, fprs = get_metrics(thresholds, y_true, y_scores)
    roc_auc = plot_roc_curve(epoch, y_true, y_scores, desc, is_save=True)
    df = pd.DataFrame(
        {'threashold': thresholds, 'f1_score': f1s, 'accuracy': accuracies, 'precision': precisions,
         'recall': recalls, 'tn': tns, 'fp': fps, 'fn': fns, 'tp': tps, 'tpr': tprs, 'fpr': fprs})
    df.to_csv(f'./metrics/epoch{epoch}_metrics.csv', index=True)
    return roc_auc


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
    model = Classifier(args.encoder,
                                     in_channels=3,
                                     hidden_channels=32,
                                     output_channels=128,
                                     n_classes=1,
                                     aux_params=project_heads)
    if args.resume_epoch > 0:
        resume_model = torch.load(get_resume_model_path(args.resume_epoch), map_location='cpu')
        model.load_encoder_and_classifier(resume_model)
    else:
        if args.pretrain:
            modelpath = f'./pretrained weight/2D/{args.dataname}/{args.encoder}/mask_ratio_{args.spatial_mask_ratio}/best_weights.pth'
            checkpoint = torch.load(modelpath, map_location=device)
            model.load_encoder(checkpoint)
    if args.frozen_encoder:
        model.frozen_encoder()
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(device_count)))
    model.to(device)

    transforms = transforms.Compose(
        [transforms.RandomAffine(degrees=(-50, 50), translate=(0.08, 0.08), scale=(0.8, 1.2), fill=0),
         transforms.Resize((args.input_size, args.input_size)), transforms.ToTensor()])

    user_home_dir = os.path.expanduser("~")
    if args.dataname == 'COVID19':
        datapath = f'{user_home_dir}/datasets/CT Scans for COVID-19 Classification'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='jpg')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='jpg')
    elif args.dataname == 'GRAM':
        datapath = f'{user_home_dir}/datasets/image_text_clinical/gram_2D'
        datasets_train = MyDataset2D(os.path.join(datapath, 'train'), transforms, is_training=True, suffix='png')
        datasets_val = MyDataset2D(os.path.join(datapath, 'val'), transforms, is_training=False, suffix='png')
    else:
        raise NotImplementedError(f'Dataset {args.dataname} not implemented.')
    dataloader_train = DataLoader(datasets_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(datasets_val, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    epoch_losses = []

    best_roc_auc = 0.0
    best_epoch = args.resume_epoch
    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        # mean_loss = train_one_epoch(model, dataloader_train, optimizer, epoch)
        # torch.save(model.state_dict(), f'./checkpoints/epoch_{epoch}.pth')
        # epoch_losses.append(mean_loss)
        # scheduler.step()
        if epoch % args.val_per_epoch == 0:
            roc_auc_train = evaluate_one_epoch(model, dataloader_train, epoch, 'train')
            print(f'roc_auc_train={roc_auc_train}')
            roc_auc_val = evaluate_one_epoch(model, dataloader_val, epoch, 'val')
            print(f'roc_auc_val={roc_auc_val}')
            if roc_auc_val > best_roc_auc:
                best_roc_auc = roc_auc_val
                best_epoch = epoch
        print(f'best_roc_auc={best_roc_auc}(epoch {best_epoch})')
        print('-----------------------------------------------------------------------------------------')
