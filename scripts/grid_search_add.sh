#!/bin/bash
lone=(1000)
ltwo=(500)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase train


# python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_unet --gpu_id 0 --fp16 \
# --n_epochs 200 --n_epochs_decay 200 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG unet_128 --upsampling linear \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --print_model_info \
# --dataroot /input --checkpoints_dir /checkpoints > /checkpoints/unet.txt # & \
# python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_unet_nofp16 --gpu_id 3 \
# --n_epochs 200 --n_epochs_decay 200 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG unet_128 --upsampling linear \
# --lambda_L1 0 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --print_model_info \
# --dataroot /input --checkpoints_dir /checkpoints > /checkpoints/unet_nofp16.txt # & \
python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_unet --gpu_id 0 --fp16 \
--n_epochs 200 --n_epochs_decay 200 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG unet_128 --upsampling linear \
--lambda_L1 0 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --print_model_info \
--dataroot /input --checkpoints_dir /checkpoints > /checkpoints/unet_0_0.txt # & \
# python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_unet_100 --gpu_id 2 --fp16 \
# --n_epochs 100 --n_epochs_decay 100 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG unet_128 --upsampling linear \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --print_model_info \
# --dataroot /input --checkpoints_dir /checkpoints > /checkpoints/unet_100_100_100.txt # & \
# python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_resnet --gpu_id 1 --fp16 \
# --n_epochs 200 --n_epochs_decay 200 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG resnet_9blocks --upsampling linear \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --print_model_info \
# --dataroot /input --checkpoints_dir /checkpoints > /checkpoints/resnet.txt
# python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2_up_resnet_6 --gpu_id 1 --fp16 \
# --n_epochs 200 --n_epochs_decay 200 --load_size 200 --crop_size 128 --preprocess take_center_and_crop --netG resnet_6blocks --upsampling linear \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --print_model_info \
# --dataroot /input --checkpoints_dir /checkpoints > /checkpoints/resnet_6.txt

# nvidia-docker run --rm --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints pix_train
