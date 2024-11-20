from easydict import EasyDict as edict
import os
cfg = edict()

# data settings
user_home_dir = os.path.expanduser("~")
cfg.TRAIN_DATA_FILE = f'{user_home_dir}/datasets/image_text_clinical/gram_2D'
# cfg.TRAIN_DATA_FILE = f'{user_home_dir}/datasets/CT Scans for COVID-19 Classification/Original CT Scans'
cfg.EVAL_DATA_FILE = ''
cfg.TEST_DATA_FILE = f'{user_home_dir}/datasets/COVID-19_Radiography_Dataset/COVID'

# model settings
cfg.BATCH_SIZE = 24
cfg.INPUT_SHAPE = [256, 256]
cfg.SPATIALA_MASK_RATIO = 0
cfg.SPATIALA_MASK_SIZE = 8
cfg.INTENSITY_MASK_RATIO = 0.7
cfg.INTENSITY_MASK_SIZE = 5
cfg.EPOCHS = 100
cfg.LR = 0.00001
cfg.DECAY_STEPS = 100000
cfg.DECAY_RATE = 0.96
cfg.STEPS_PER_EPOCH = 100
cfg.CHECKPOINTS_ROOT = 'checkpoints'
cfg.MAX_KEEPS_CHECKPOINTS = 1
cfg.AMP = True


