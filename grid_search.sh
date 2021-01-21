b_sizes=(1 5 10)
lone=(0 100 500)
ltwo=(0 100 500)


for b in ${b_sizes[@]}; do
  for l in ${lone[@]}; do
    for ll in ${ltwo[@]}; do
      python3 train.py --dataroot ../braindata/ --gpu_ids=TODO --dataset_mode nifti --model pix2pix3d --name t1t2 --preprocess pad \
      --excel --postprocess 0 --batch_size $b --lambda_L1 $l --lambda_L2_T $ll \
      --suffix {model}_{netG}_{batch_size}_{lambda_L1}_{lambda_L2_T}
    done
  done
done

