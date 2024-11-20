import random

from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from data.chest.daraloader import MyDataset
import numpy as np


batch_size = 1
num_workers = 0
pin_memory = False
valid_size = 0.15
shuffle = False

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

train_data = './mini_dataset/'

transform = transforms.Compose([
                transforms.RandomRotation(degrees=15),  # 随机旋转
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.CenterCrop(size=224),  # 中心裁剪到224*224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4203, 0.1421, 0.1931], std=[0.3361, 0.2399, 0.1962]),  # 这个mean和std是计算过的
            ])

Training_Data = MyDataset(images_dir=train_data, transform=None)  # 为 None的话，有默认的ToTensor和 Normalize

num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory, )

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory, )

# 测试
for y, z, x in train_loader:
    x = x.numpy()
    x = np.transpose(x, (1, 2, 0))  # C*H*W -> H*W*C
    plt.imshow(x)
    plt.show()
    break
