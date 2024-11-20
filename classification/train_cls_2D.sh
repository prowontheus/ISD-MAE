#!/bin/bash
set -x


data_name=GRAM
encoder=resnet50
mask_ratios=(0.0 0.25 0.5 0.75)

for mask_ratio in "${mask_ratios[@]}"
do
	python ../train_unet_ae.py -encoder=${encoder} -intensity_mask_size=8 -intensity_mask_ratio=0.75 -epochs=30 -shape=2D -dataname=${data_name} -save_dir=${encoder} --spatial_mask_ratio=${mask_ratio}
	mkdir -p "./pretrained weight/2D/${data_name}/${encoder}/mask_ratio_${mask_ratio}/"
	cp ./checkpoints/${encoder}/mask_ratio_${mask_ratio}/best_weights.pth ./pretrained\ weight/2D/${data_name}/${encoder}/mask_ratio_${mask_ratio}/best_weights.pth
done
for mask_ratio in "${mask_ratios[@]}"
do
  	mkdir -p "./metrics/2D/${encoder}/${data_name}/without pretrained"
	mkdir -p "./metrics/2D/${encoder}/${data_name}/with pretrained/PatchMaskRatio-${mask_ratio}/3层全连接不冻结encoder"
	mkdir -p "./metrics/2D/${encoder}/${data_name}/with pretrained/PatchMaskRatio-${mask_ratio}/3层全连接冻结encoder"

	python train_ct_cls_2D.py --encoder=${encoder} --batch_size=32 --spatial_mask_ratio=${mask_ratio} --dataname=${data_name} --epochs=50 --pretrain
	mv metrics/*epoch* metrics/2D/${encoder}/${data_name}/with\ pretrained/PatchMaskRatio-${mask_ratio}/3层全连接不冻结encoder
	python train_ct_cls_2D.py --encoder=${encoder} --batch_size=256 --spatial_mask_ratio=${mask_ratio} --dataname=${data_name} --epochs=50 --pretrain --frozen_encoder
	mv metrics/*epoch* metrics/2D/${encoder}/${data_name}/with\ pretrained/PatchMaskRatio-${mask_ratio}/3层全连接冻结encoder
done
python train_ct_cls_2D.py --encoder=${encoder} --batch_size=64 --dataname=${data_name} --epochs=50
mv metrics/*epoch* metrics/2D/${encoder}/${data_name}/without\ pretrained

data_name=COVID19

for mask_ratio in "${mask_ratios[@]}"
do
	python ../train_unet_ae.py -encoder=${encoder} -intensity_mask_size=8 -intensity_mask_ratio=0.75 -epochs=30 -shape=2D -dataname=${data_name} -save_dir=${encoder} --spatial_mask_ratio=${mask_ratio}
	mkdir -p "./pretrained weight/2D/${data_name}/${encoder}/mask_ratio_${mask_ratio}/"
	cp ./checkpoints/${encoder}/mask_ratio_${mask_ratio}/best_weights.pth ./pretrained\ weight/2D/${data_name}/${encoder}/mask_ratio_${mask_ratio}/best_weights.pth
done
for mask_ratio in "${mask_ratios[@]}"
do
  	mkdir -p "./metrics/2D/${encoder}/${data_name}/without pretrained"
	mkdir -p "./metrics/2D/${encoder}/${data_name}/with pretrained/PatchMaskRatio-${mask_ratio}/3层全连接不冻结encoder"
	mkdir -p "./metrics/2D/${encoder}/${data_name}/with pretrained/PatchMaskRatio-${mask_ratio}/3层全连接冻结encoder"

	python train_ct_cls_2D.py --encoder=${encoder} --batch_size=32 --spatial_mask_ratio=${mask_ratio} --dataname=${data_name} --epochs=50 --pretrain
	mv metrics/*epoch* metrics/2D/${encoder}/${data_name}/with\ pretrained/PatchMaskRatio-${mask_ratio}/3层全连接不冻结encoder
	python train_ct_cls_2D.py --encoder=${encoder} --batch_size=256 --spatial_mask_ratio=${mask_ratio} --dataname=${data_name} --epochs=50 --pretrain --frozen_encoder
	mv metrics/*epoch* metrics/2D/${encoder}/${data_name}/with\ pretrained/PatchMaskRatio-${mask_ratio}/3层全连接冻结encoder
done
python train_ct_cls_2D.py --encoder=${encoder} --batch_size=64 --dataname=${data_name} --epochs=50
mv metrics/*epoch* metrics/2D/${encoder}/${data_name}/without\ pretrained

