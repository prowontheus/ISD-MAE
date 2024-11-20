#!/bin/bash
set -x

losses=(DICE)
encoders=(resnet50)
# data_names=(COVID19_CT Task06_Lung)
data_names=(COVID19_3D)
intensity_mask_ratios=(0.5)
spatial_mask_ratios=(0.5 0.75)
for loss in "${losses[@]}"
do
	for encoder in "${encoders[@]}"
	do
		for data_name in "${data_names[@]}"
		do
			for intensity_mask_ratio in "${intensity_mask_ratios[@]}"
			do
				for spatial_mask_ratio in "${spatial_mask_ratios[@]}"
				do
					if [ "$data_name" == "Task06_Lung" ] || [ "$data_name" == "COVID19_CT" ]
					then
					    epoch=30
					elif [ "$data_name" == "COVID19_3D" ] || [ "$data_name" == "TotalSegmentator" ]
					then
					    epoch=10
					fi
					python ../train_unet_ae.py -encoder=${encoder} -intensity_mask_size=8 -intensity_mask_ratio=${intensity_mask_ratio} -spatial_mask_ratio=${spatial_mask_ratio} -epochs=${epoch} -shape=3D -dataname=${data_name} -save_dir=${encoder} --tensorboard
					mkdir -p "./pretrained weight/3D/${data_name}/${encoder}/mask_ratio_${intensity_mask_ratio}_${spatial_mask_ratio}/"
					cp ./checkpoints/${encoder}/mask_ratio_${intensity_mask_ratio}_${spatial_mask_ratio}/best_weights.pth ./pretrained\ weight/3D/${data_name}/${encoder}/mask_ratio_${intensity_mask_ratio}_${spatial_mask_ratio}/best_weights.pth

					mkdir -p "./metrics/3D/${loss}/${data_name}/${encoder}/MaskRatio-${intensity_mask_ratio}-${spatial_mask_ratio}"
					python train_ct_seg_3D.py --encoder=${encoder} --batch_size=1 --intensity_mask_ratio=${intensity_mask_ratio} --spatial_mask_ratio=${spatial_mask_ratio} --dataname=${data_name} --epochs=${epoch} --loss=${loss} --pretrain --tensorboard --amp
					mv metrics/*epoch* metrics/3D/${loss}/${data_name}/${encoder}/MaskRatio-${intensity_mask_ratio}-${spatial_mask_ratio}

				done
			done
		done
	done
done