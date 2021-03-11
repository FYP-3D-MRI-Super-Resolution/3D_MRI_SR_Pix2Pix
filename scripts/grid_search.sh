#!/bin/bash
lone=(0 100 500 1000)
ltwo=(0 100 500 1000)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase train

for l in ${lone[@]}; do
  for ll in ${ltwo[@]}; do
    echo "${l}, ${ll}"
    python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2 \
    --lambda_L1 $l --gamma_TMSE $ll --gpu_id 0,1 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} \
    --dataroot /input --checkpoints_dir /checkpoints > "/checkpoints/out_${l}_${ll}.txt"
  done
done

# nvidia-docker run --rm --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints pix_train
