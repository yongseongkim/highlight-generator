import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import copy
import os


class HighlightDataset(Dataset):
    def __init__(self, len_sequence, highlight_dir, non_highlight_dir, transform=None):
        self.simg_width = 16 * 14
        self.len_sequence = len_sequence
        self.highlight_dir = highlight_dir
        self.non_highlight_dir = non_highlight_dir
        self.hightlight_paths = [
            os.path.join(highlight_dir, f) for f in os.listdir(self.highlight_dir)
        ]
        self.non_hightlight_paths = [
            os.path.join(non_highlight_dir, f) for f in os.listdir(self.non_highlight_dir)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.hightlight_paths) + len(self.non_hightlight_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= len(self.hightlight_paths):
            idx = idx - len(self.hightlight_paths)
            img = cv2.imread(self.non_hightlight_paths[idx])
            score = 0
        else:
            img = cv2.imread(self.hightlight_paths[idx])
            score = 7
        imgs = np.asarray([img[:,
                               i * self.simg_width: (i+1) * self.simg_width,
                               :] for i in range(self.len_sequence)])
        sample = imgs, score
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        imgs, score = sample[0], sample[1]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        transpoed_imgs = np.asarray([img.transpose((2, 0, 1)) for img in imgs])
        return torch.from_numpy(transpoed_imgs).float(), torch.tensor(score, dtype=torch.float)
