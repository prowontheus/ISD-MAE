import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os
import glob


class MyDataset2D(Dataset):
    def __init__(self, dataset_path: str, transform, is_training: bool = True, suffix: str = 'jpg'):
        self.datas = None
        self.labels = None
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

    def get_images_and_labels(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Path '{self.dataset_path}' does not exist.")
        pos_filepath = os.path.join(os.path.join(self.dataset_path, 'pos'))
        neg_filepath = os.path.join(os.path.join(self.dataset_path, 'neg'))
        if not os.path.exists(pos_filepath):
            raise FileNotFoundError(f"Path '{pos_filepath}' does not exist.")
        if not os.path.exists(os.path.join(self.dataset_path, 'neg')):
            raise FileNotFoundError(f"Path '{self.dataset_path}/neg' does not exist.")
        image_pos_lists = glob.glob(os.path.join(pos_filepath, f'**/*.{self.suffix}'), recursive=True)
        image_neg_lists = glob.glob(os.path.join(neg_filepath, f'**/*.{self.suffix}'), recursive=True)
        pos_lists = [image for image in image_pos_lists if 'mask' not in image]
        neg_lists = [image for image in image_neg_lists if 'mask' not in image]

        if len(pos_lists) == 0:
            raise FileNotFoundError(f"No positive image files found in path '{pos_filepath}'.")
        if len(neg_lists) == 0:
            raise FileNotFoundError(f"No negative image files found in path '{neg_filepath}'.")
        self.datas = pos_lists + image_neg_lists
        self.labels = torch.cat((torch.ones(len(pos_lists)), torch.zeros(len(neg_lists))), dim=0)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        img = Image.open(self.datas[index])
        label = self.labels[index]
        img = self.tx(img)
        return img, label
