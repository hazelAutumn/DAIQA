import os

import pandas as pd
import torch.utils.data as data

from src.utils.dataset_utils import pil_loader

from .dataloader_mode import DataloaderMode



# target is distortion label
class KADIS_Dataset(data.Dataset):
    def __init__(self, cfg, index, transform, mode):
        self.root = cfg.data.root
        meta_info = pd.read_csv(cfg.data.meta_info_file)

        if mode is DataloaderMode.train:
            patch_num = cfg.train.patch_num
        elif mode is DataloaderMode.val:
            patch_num = cfg.val.patch_num
        elif mode is DataloaderMode.test:
            patch_num = cfg.test.patch_num
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        sample = []
        for idx in index:
            img_name = meta_info.loc[idx]["dist_name"]
            img_name = os.path.splitext(img_name)[0] + ".png"
            img_path = os.path.join("dist_imgs", img_name)
            label2 = meta_info.loc[idx]["dist_type"]
            for _ in range(patch_num):
                sample.append((img_path, label2))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target2 = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        # if there are more than one image or more than one target
        # can organize it as
        # return [img1, img2], [targe1, target2]
        return img, target2

    def __len__(self):
        length = len(self.samples)
        return length



class KADIS_Dataset_full(data.Dataset):
    def __init__(self, cfg, index, transform, mode):
        self.root = cfg.data.root
        meta_info = pd.read_csv(cfg.data.meta_info_file)

        if mode is DataloaderMode.train:
            patch_num = cfg.train.patch_num
        elif mode is DataloaderMode.val:
            patch_num = cfg.val.patch_num
        elif mode is DataloaderMode.test:
            patch_num = cfg.test.patch_num
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        sample = []
        for idx in index:
            img_name = meta_info.loc[idx]["dist_name"]
            img_name = os.path.splitext(img_name)[0] + ".png"
            img_path = os.path.join("dist_imgs_all", img_name)
            label2 = meta_info.loc[idx]["dist_type"]
            for _ in range(patch_num):
                sample.append((img_path, label2))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target2 = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        # if there are more than one image or more than one target
        # can organize it as
        # return [img1, img2], [targe1, target2]
        return img, target2

    def __len__(self):
        length = len(self.samples)
        return length
