import os

import pandas as pd
import torch.utils.data as data

from src.utils.dataset_utils import pil_loader

from .dataloader_mode import DataloaderMode


class TID2013Dataset(data.Dataset):
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
            img_path = os.path.join("distorted_images", img_name)
            label = meta_info.loc[idx]["mos"]
            for _ in range(patch_num):
                sample.append((img_path, label))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        # if there are more than one image or more than one target
        # can organize it as
        # return [img1, img2], [targe1, target2]
        return img, target

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Dataset2o(data.Dataset):
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
            img_path = os.path.join("distorted_images", img_name)
            label = meta_info.loc[idx]["mos"]
            label2 = meta_info.loc[idx]["std"]
            for _ in range(patch_num):
                sample.append((img_path, label, label2))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, target2 = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        # if there are more than one image or more than one target
        # can organize it as
        # return [img1, img2], [targe1, target2]
        return img, target, target2

    def __len__(self):
        length = len(self.samples)
        return length


###### For 1 target output, but keep the same splits as of 2 targets
class TID2013Dataset1o(data.Dataset):
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
            img_path = os.path.join("distorted_images", img_name)
            label = meta_info.loc[idx]["mos"]
            for _ in range(patch_num):
                sample.append((img_path, label))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = pil_loader(os.path.join(self.root, path))
        if self.transform is not None:
            img = self.transform(img)

        # if there are more than one image or more than one target
        # can organize it as
        # return [img1, img2], [targe1, target2]
        return img, target

    def __len__(self):
        length = len(self.samples)
        return length

#1 ouput distortion
class TID2013Dataset_1d(data.Dataset):
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
            img_path = os.path.join("distorted_images", img_name)
            label = meta_info.loc[idx]["mos"]
            label2 = meta_info.loc[idx]["std"]
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