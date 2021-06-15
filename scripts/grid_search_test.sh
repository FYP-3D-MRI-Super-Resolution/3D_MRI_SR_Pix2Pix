#!/bin/bash
lone=(0 100 500 1000)
ltwo=(0 100 500 1000)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase validation

# python3 test.py --dataset_mode nifti --model pix2pix3d --eval --name t1t2_up_unet --gpu_id 0 --fp16 \
# --load_size 256 --preprocess pad --netG unet_128 --upsampling linear --postprocess 1 \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > /results/unet.txt & \
# python3 test.py --dataset_mode nifti --model pix2pix3d --eval --name t1t2_up_unet_nofp16 --gpu_id 5 \
# --load_size 256 --preprocess pad --netG unet_128 --upsampling linear --postprocess 1 \
# --lambda_L1 0 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > /results/unet_nofp16.txt & \
# python3 test.py --dataset_mode nifti --model pix2pix3d --eval --name t1t2_up_unet --gpu_id 2 --fp16 \
# --load_size 256 --preprocess pad --netG unet_128 --upsampling linear --postprocess 1 \
# --lambda_L1 0 --gamma_TMSE 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > /results/unet_0_0.txt & \
# python3 test.py --dataset_mode nifti --model pix2pix3d --eval --name t1t2_up_unet_100 --gpu_id 3 --fp16 \
# --load_size 256 --preprocess pad --netG unet_128 --upsampling linear --postprocess 1 \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > /results/unet_100_100_100.txt & \
# python3 test.py --dataset_mode nifti --model pix2pix3d --eval --name t1t2_up_resnet --gpu_id 4 --fp16 \
# --preprocess none --netG resnet_9blocks --upsampling linear --postprocess 1 \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > /results/resnet.txt # & \
# python3 test.py --dataset_mode nifti --model pix2pix3d --eval --name t1t2_up_resnet_6 --gpu_id 5 --fp16 \
# --preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 \
# --lambda_L1 100 --gamma_TMSE 100 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
# --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > /results/resnet_6.txt

for l in ${lone[@]}; do
  for i in ${!ltwo[@]}; do
    python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2_resnet --gpu_id ${i} --fp16 \
    --preprocess none --netG resnet_6blocks --upsampling linear --postprocess 1 --excel --eval \
    --lambda_L1 $l --gamma_TMSE ${ltwo[$i]} --suffix {model}_l{lambda_L1}_g{gamma_TMSE} --phase validation \
    --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/out_${l}_${ltwo[$i]}.txt" & 
  done
  wait
done

# nvidia-docker run --rm --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints --mount type=bind,source=/raid/gbaldini/pix_output,target=/results pix_grid_val
