#!/bin/bash
set -x

losses=(BCE)
encoders=(mit_b2)
data_names=(COVID19_2D Lung_nodule_seg Lung_CT_nodule Lungs_CT_Scan LIDC_IDRI Chest_CT)
intensity_mask_ratios=(0.5 0.75)
spatial_mask_ratios=(0.0 0.25 0.5 0.75)

for loss in "${losses[@]}"
do
	for encoder in "${encoders[@]}"
	do
		for data_name in "${data_names[@]}"
		do
			if [ "$data_name" == "Task06_Lung" ]
			then
			    epoch=100
			elif [ "$data_name" == "Lung_nodule_seg" ] || [ "$data_name" == "Lung_CT_nodule" ] || [ "$data_name" == "Chest_CT" ] || [ "$data_name" == "LIDC_IDRI" ]
			then
				epoch=40
			elif [ "$data_name" == "COVID19_LESION" ] || [ "$data_name" == "Lungs_CT_Scan" ]
			then
				epoch=30
			elif [ "$data_name" == "TotalSegmentator" ]
			then
			    epoch=20
			elif [ "$data_name" == "COVID19_2D" ]
			then
			    epoch=40
			fi
			for intensity_mask_ratio in "${intensity_mask_ratios[@]}"
			do
				for spatial_mask_ratio in "${spatial_mask_ratios[@]}"
				do
					python ../train_unet_ae.py -encoder=${encoder} -intensity_mask_size=8 -intensity_mask_ratio=${intensity_mask_ratio} -spatial_mask_ratio=${spatial_mask_ratio} -epochs=${epoch} -shape=2D -dataname=${data_name} -save_dir=${encoder} --tensorboard
					mkdir -p "./pretrained weight/2D/${data_name}/${encoder}/mask_ratio_${intensity_mask_ratio}_${spatial_mask_ratio}/"
					cp ./checkpoints/${encoder}/mask_ratio_${intensity_mask_ratio}_${spatial_mask_ratio}/best_weights.pth ./pretrained\ weight/2D/${data_name}/${encoder}/mask_ratio_${intensity_mask_ratio}_${spatial_mask_ratio}/best_weights.pth

					mkdir -p "./metrics/2D/${loss}/${data_name}/${encoder}/MaskRatio-${intensity_mask_ratio}-${spatial_mask_ratio}"
					python train_ct_seg_2D.py --encoder=${encoder} --batch_size=16 --intensity_mask_ratio=${intensity_mask_ratio} --spatial_mask_ratio=${spatial_mask_ratio} --dataname=${data_name} --epochs=${epoch} --loss=${loss} --pretrain --tensorboard
					mv metrics/*epoch* metrics/2D/${loss}/${data_name}/${encoder}/MaskRatio-${intensity_mask_ratio}-${spatial_mask_ratio}
				done
			done
		done
	done
done