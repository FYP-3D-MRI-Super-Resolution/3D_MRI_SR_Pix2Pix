b_sizes=(1 5 10)
lone=(0 100 500)
ltwo=(0 100 500)


python3 ./datasets/make_nifti_dataset.py --save_folder ./datasets/braindata --folder ../braindata_all/ --phase train

for b in ${b_sizes[@]}; do
  for l in ${lone[@]}; do
    for ll in ${ltwo[@]}; do
      docker run pix --dataroot ./datasets/braindata/ --batch_size $b --lambda_L1 $l --lambda_L2_T $ll --gpu_id 0
    done
  done
done

