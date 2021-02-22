#!/bin/bash
lone=(0 100 500)
ltwo=(0 100 500)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase train

for l in ${lone[@]}; do
  for ll in ${ltwo[@]}; do
    python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2 --n_epochs 200 --n_epochs_decay 200 \
    --lambda_L1 $l --lambda_L2_T $ll --gpu_id 0,1,2,3 --suffix {model}_{lambda_L1}_{lambda_L2_T} \
    --dataroot /input --checkpoints_dir /checkpoints > "/checkpoints/out_${l}_${ll}.txt"
  done
done

# nvidia-docker run --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints pix
