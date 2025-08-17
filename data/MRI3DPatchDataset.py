import torch
from torch.utils.data import Dataset
import torchio as tio
import os

class MRI3DPatchDatasetPaired(Dataset):
    def __init__(self, input_dir, target_dir, patch_size):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.patch_overlap = 0

        self.subjects = self._load_subjects()
        self.patch_counts = []
        self.cumulative_counts = []
        total = 0
        for subject in self.subjects:
            sampler = tio.GridSampler(subject, patch_size=self.patch_size, patch_overlap=self.patch_overlap)
            count = len(list(iter(sampler)))
            self.patch_counts.append(count)
            total += count
            self.cumulative_counts.append(total)

    def _load_subjects(self):
        subjects = []
        input_files = sorted(os.listdir(self.input_dir))
        for input_file in input_files:
            if not input_file.endswith(".nii") and not input_file.endswith(".nii.gz"):
                continue

            target_filename = input_file.replace("lowres_", "")
            input_path = os.path.join(self.input_dir, input_file)
            target_path = os.path.join(self.target_dir, target_filename)

            if not os.path.exists(target_path):
                continue

            subject = tio.Subject(
                input=tio.ScalarImage(input_path),
                target=tio.ScalarImage(target_path)
            )
            subjects.append(subject)
        return subjects

    def __len__(self):
        return sum(self.patch_counts)

    def _find_subject_and_local_index(self, idx):
        for subj_idx, cum_count in enumerate(self.cumulative_counts):
            if idx < cum_count:
                local_index = idx if subj_idx == 0 else idx - self.cumulative_counts[subj_idx - 1]
                return subj_idx, local_index

    def __getitem__(self, idx):
        subject_idx, local_index = self._find_subject_and_local_index(idx)
        subject = self.subjects[subject_idx]
        sampler = tio.GridSampler(subject, patch_size=self.patch_size, patch_overlap=self.patch_overlap)
        for i, patch in enumerate(sampler):
            if i == local_index:
                return patch['input'].data, patch['target'].data
        raise IndexError("Index out of range")


