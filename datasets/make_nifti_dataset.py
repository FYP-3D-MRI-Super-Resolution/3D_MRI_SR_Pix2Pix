import argparse
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import math
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Put the images in the right folders for GAN training and testing.')
    parser.add_argument('--folder', type=str, help='folder where the NifTi files are stored')
    parser.add_argument('--save_folder', type=str, help='folder where the NifTi files will be stored')
    parser.add_argument('--phase', required=True, choices=['train', 'test', 'validation', 'test_truth'], type=str, help='train, test or validation')

    args = parser.parse_args()

    folder = Path.cwd() / args.folder / args.phase
    store_folder = Path.cwd() / args.save_folder / args.phase
    extension = ".nii.gz"
    # This function is based on my file hierarchy, where there are different big files containing one file per patient
    for brain_folder in folder.iterdir():
        files_in_dir = sorted(brain_folder.glob('*.nii*'))
        for f in files_in_dir:
            new_folder = store_folder / f.name.split(".")[0].split("_")[-1]
            new_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(f), str(new_folder))
            new_dst_file_name = new_folder / (brain_folder.name + extension)
            os.rename(str(new_folder / f.name), new_dst_file_name)
