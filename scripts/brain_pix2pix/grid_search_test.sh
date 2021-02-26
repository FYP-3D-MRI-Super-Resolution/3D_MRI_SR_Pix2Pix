#!/bin/bash
lone=(0 100 500)
ltwo=(0 100 500)

# python3 ./datasets/make_nifti_dataset.py --folder /raid/gbaldini/braindata_all --save_folder /raid/gbaldini/pix_data --phase validation

for l in ${lone[@]}; do
  for ll in ${ltwo[@]}; do
    python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2 --eval --excel \
    --lambda_L1 $l --gamma_TMSE $ll --gpu_id 0 --suffix {model}_l{lambda_L1}_g{gamma_TMSE} \
    --preprocess pad --postprocess 1 --phase validation \
    --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/out_${l}_${ll}.txt"
  done
done

# nvidia-docker run --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints,target=/checkpoints --mount type=bind,source=/raid/gbaldini/pix_output,target=/results pix_grid_val
