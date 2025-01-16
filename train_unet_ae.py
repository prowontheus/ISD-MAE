import os

from models.unet_ae import UnetAE, UnetAE3D
import torch
from datasets.dataloader import MyDataset, MyDataset3D
import torchvision.transforms as transforms
from cfgs.config import cfg
from trainers.unet_ae_trainer import Trainer
import argparse
import warnings
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='Train the CRNet for chest reconstruction')
    parser.add_argument('--encoder', '-encoder', type=str, default='resnet50',
                        help='The backbone for feature extraction')
    parser.add_argument('--input_size', '-input_size', type=int, default=256, help='the feed size of image')
    parser.add_argument('--intensity_mask_size', '-intensity_mask_size', type=int, default=10,
                        help='the masking type, 0:patch, 1:intensity')
    parser.add_argument('--intensity_mask_ratio', '-intensity_mask_ratio', type=float, default=0.7,
                        help='the mask ratio of image for contrastive')
    parser.add_argument('--spatial_mask_size', '-spatial_mask_size', type=int, default=16,
                        help='the masking type, 0:patch, 1:intensity')
    parser.add_argument('--spatial_mask_ratio', '-spatial_mask_ratio', type=float, default=0,
                        help='the mask ratio of image for contrastive')
    parser.add_argument('--shape', '-shape', type=str, default='2D', help='2D or 3D image shape')
    parser.add_argument('--load', '-load', type=str, default='weights_best.h5', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='resnet_unet', help='the path to save weights')
    parser.add_argument('--epochs', '-epochs', type=int, default=120, help='Epochs for training')
    parser.add_argument('--steps_per_epoch', '-steps_per_epoch', type=int, default=0, help='iterations for each epoch')
    parser.add_argument('--tensorboard', '-tensorboard', action='store_true', help='Use tensorboard')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Base learning rate for training')
    parser.add_argument('--dataname', '-dataname', type=str, default='TotalSegmentator', help='dataname')

    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_args()

    user_home_dir = osp.expanduser("~")
    device_count = torch.cuda.device_count()

    cfg.INPUT_SHAPE = (args.input_size, args.input_size)
    transforms = transforms.Compose(
        [transforms.RandomAffine(degrees=(-50, 50), translate=(0.08, 0.08), scale=(0.8, 1.2), fill=0),
         transforms.Resize((cfg.INPUT_SHAPE)), transforms.ToTensor()])

    # transforms = transforms.Compose([transforms.Resize((cfg.INPUT_SHAPE)), transforms.ToTensor()])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    # load trainging dataset
    if args.shape == '3D':
        train_dataset = MyDataset3D(images_dirs=cfg.TRAIN_DATA_FILE, transform=transforms)
    else:
        train_dataset = MyDataset(images_dirs=cfg.TRAIN_DATA_FILE, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=0,
                                               persistent_workers=False)

    project_heads = []
    project_outdims = [128]
    for outdim in project_outdims:
        project_heads.append(dict(
            pooling='max',
            dropout=None,
            activation=None,
            out_dims=outdim
        ))
    if args.shape == '3D':
        model = UnetAE3D(encoder_name=args.encoder, classes=3, depth=5, activation='sigmoid', aux_params=project_heads)
    else:
        model = UnetAE(encoder_name=args.encoder, classes=3, activation='sigmoid', aux_params=project_heads)
    print(model)
    if osp.exists(args.load):
        model.load_state_dict(torch.load(args.load, map_location=device))
    model = model.to(torch.float32)
    if use_cuda:
        model = torch.nn.parallel.DataParallel(model, device_ids=list(range(device_count)))


    cfg.INTENSITY_MASK_SIZE = args.intensity_mask_size
    cfg.INTENSITY_MASK_RATIO = args.intensity_mask_ratio
    cfg.SPATIALA_MASK_SIZE = args.spatial_mask_size
    cfg.SPATIALA_MASK_RATIO = args.spatial_mask_ratio
    cfg.STEPS_PER_EPOCH = len(train_dataset) // cfg.BATCH_SIZE if args.steps_per_epoch == 0 else args.steps_per_epoch
    cfg.DECAY_STEPS = cfg.STEPS_PER_EPOCH
    cfg.LR = args.lr
    cfg.SHAPE = args.shape
    cfg.EPOCHS = args.epochs
    # initialization

    if args.tensorboard == True:
        log_dir = f'./checkpoints/{args.encoder}/mask_ratio_{cfg.INTENSITY_MASK_RATIO}_{cfg.SPATIALA_MASK_RATIO}'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        tensorboard_writer = SummaryWriter(log_dir)
    else:
        tensorboard_writer = None
    trainer = Trainer(model, cfg, device)
    trainer.start_train(train_loader, args.save_dir, tensorboard_writer, pretrained_file=args.load)
    tensorboard_writer.close()
