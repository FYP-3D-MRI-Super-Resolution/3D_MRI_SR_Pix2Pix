#!/bin/bash
lone=(0 100 500 1000)
ltwo=(0 100 500 1000)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase train

for l in ${lone[@]}; do
  for i in ${!ltwo[@]}; do
    python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id ${i} --fp16 \
    --n_epochs 100 --n_epochs_decay 100 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG resnet_6blocks --upsampling linear \
    --lambda_L1 $l --gamma_TMSE ${ltwo[$i]} --suffix {model}_l{lambda_L1}_g{gamma_TMSE} \
    --dataroot /input --checkpoints_dir /checkpoints > "/checkpoints/out_${l}_${ltwo[$i]}.txt" & 
  done
  wait
done

# python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_unet_100 --gp
# nvidia-docker run --rm --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints pix_train
