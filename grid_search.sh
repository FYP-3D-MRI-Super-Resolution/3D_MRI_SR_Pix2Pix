b_sizes=(1 5 10)
lone=(0 100 500)
ltwo=(0 100 500)

# python3 ./datasets/make_nifti_dataset.py --save_folder ./datasets/braindata --folder ../braindata_all/ --phase train

for b in ${b_sizes[@]}; do
  for l in ${lone[@]}; do
    for ll in ${ltwo[@]}; do
      python3 train.py --dataset_mode nifti --model pix2pix3d --name t1t2 --n_epochs 200 --n_epochs_decay 200 \
      --suffix {model}_{batch_size}_{lambda_L1}_{lambda_L2_T} \
      --dataroot ./datasets/braindata/ --batch_size $b --lambda_L1 $l --lambda_L2_T $ll --gpu_id 0,1,2,3 > "./checkpoints/out_${b}_${l}_${ll}.txt"
    done
  done
done