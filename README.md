# Pix2PixNIfTI

Extension of [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to run with NIfTI files (2D or 3D).
Pix2Pix is a general framework for image-to-image translation.
With our extension this cGAN can learn a mapping between two MRI sequences.

## Installation
The framework is written in python3 and requires some libraries, which can be installed with
```
pip install requirements.txt
```

## Running the program
The model can be trained with `train.py` and then the desired data set can be tested with `test.py`.

### Data and Folder Organization
This software accepts all the formats previously accepted by Pix2Pix and additionally accepts 3D NIfTI files. 
The MRI scans should be skull-stripped and registered to the same template.

For the NIfTI case we require that there is a separate folder for the source and the target sequence.
The folder structure should have a `path/to/data` folder where there is a `train` folder that contains the training data
and some test/validation folders. Each of these folders has one folder for each sequencing, so `t1`, `t2`, `t1ce`, `flair` 
and one for the ground truth `truth`.
If you are only interested in the translation between `t1` and `t2`, the other folders are not needed.
The tumor ground truth `truth` is used to compute the MSE of the tumor, and will be ignored if not present.
```
path/to/data
|
└───train
│   └───t1
│   |   │   pat1.nii
│   |   │   pat2.nii
│   |   │   ...
|   |   |   patn.nii
|   |
│   └───t2
│   |   │   pat1.nii
│   |   │   pat2.nii
│   |   │   ...
|   |   |   patn.nii
|   |
│   └─── ...
└───val
│   └───t1
│   |   │   pat1.nii
│   |   │   pat2.nii
│   |   │   ...
|   |   |   patn.nii
|   |
│   └─── ...
```

Normally, the data sets containing MRI scans have a folder `data/to/fix` 
which contains folders like `train` and `test`. These folders contain one patient folder each, and each patient folder
contains `t1`, `t2`, `t1ce`, `flair` and `truth`. 
Given a file hierarchy of this type, we created a script that takes the 
images and saves them such that they can be used by Pix2Pix.
```
python ./datasets/fix_data.py --folder data/to/fix --save_folder path/to/data --phase train
```
The `--phase train` argument can be changed to any folder name.

### Train

To start the actual training procedure run
```
python train.py --dataroot path/to/data --dataset_mode nifti --model pix2pix3d --name test_name
```
Which will create a model for 3D MRI scans. To use 2D slices instead select `--model pix2pix`.
To select a slice to run the training on, use the parameter `--chosen_slice`. 
At the moment only transverse slices are supported and the default slice is the middle transverse slice for a scan of size 240x240x155.

The network is a u128 by default, so the images should have side lengths which are multiples of 128. However, if this is not the case,
random crops will be performed to ensure that the entire structure of the brain is captured.

In case you not have a GPU, please add the option `--gpu_id -1`.

### Test
Once the model has been trained, we can retrieve the results for the `test_name` experiment with
```
python test.py --dataroot path/to/data --dataset_mode nifti --model pix2pix3d --name test_name --preprocess resize --excel --postprocess 1 --phase name/to/test
```
If the source images do not have side lengths which are multiples of 128 they can be padded with `--preprocess resize`.

The flag `--excel` prints the results of the MSE computation to a csv file. 
Otherwise, the results are only printed to screen.

`name/to/test` is the name of the folder where the MRI images for testing are located. For example, if the files to test are in `/path/to/data/name/to/test`, then 
`name/to/test` should be used. The default value is `test`.

The variable `--postprocess` can be used to postprocess the image:
* **0**: scales the image to the [0, 1] range,
* **1**: standardizes the image to have zero mean and unit variance,
* **2**: scales the image the [-1, 1] range.

It is also possible to select a filter to apply to the resulting image with `--smoothing`. 
Both the original and the filtered image will be saved as output.
The options are `median` and `average`, where `median` applies the median filter while `average` applies a simple averaging filter.
The default filter is `median`.

In case you not have a GPU, please add the option `--gpu_id -1`.
