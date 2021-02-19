lone=(0 100 500)
ltwo=(0 100 500)

# python3 ./datasets/make_nifti_dataset.py --save_folder ./raid/gbaldini/braindata --folder /raid/gbaldini/pix_data --phase validation

for l in ${lone[@]}; do
  for ll in ${ltwo[@]}; do
    python3 test.py --dataset_mode nifti --model pix2pix3d --name t1t2 --eval --excel \
    --lambda_L1 $l --lambda_L2_T $ll --gpu_id 0 --suffix {model}_{lambda_L1}_{lambda_L2_T} \
    --preprocess pad --postprocess 1 --phase validation \
    --dataroot /input --checkpoints_dir /checkpoints --results_dir /results > "/results/out_${l}_${ll}.txt"
  done
done

# nvidia-docker run --mount type=bind,source=/raid/gbaldini/pix_data,target=/input --mount type=bind,source=/raid/gbaldini/pix_checkpoints --mount type=bind,source=/raid/gbaldini/pix_results,target=/results pix
