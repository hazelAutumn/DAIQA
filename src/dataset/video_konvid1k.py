import os

import pandas as pd
import torch.utils.data as data

from src.utils.dataset_utils import pil_loader

from .dataloader_mode import DataloaderMode
import h5py
import skvideo.io
import torch
from PIL import Image


class VideoKonvid1kDataset(data.Dataset):
    def __init__(self, cfg, index, transform, mode):
        self.root = cfg.data.root
        meta_info = pd.read_csv(cfg.data.meta_info_file)

        sample = []
        for idx in index:
            video_name = meta_info.loc[idx]["video_names"]
            video_path = os.path.join("KoNViD_1k_videos", video_name)
            label = meta_info.loc[idx]["mos"]
            sample.append((video_path, label))

        self.samples = sample
        self.transform = transform
        self.max_length = 240

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        video_data = skvideo.io.vread(os.path.join(self.root, path))

        transform = self.transform

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([self.max_length, video_channel,  224, 224])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame
        if self.max_length > video_length:
            transformed_video[video_length:,:] = frame

        #sample = {'video': transformed_video,
        #          'score': target}
        return transformed_video, target

    def __len__(self):
        length = len(self.samples)
        return length



class VideoKVQDataset(data.Dataset):
    def __init__(self, cfg, index, transform, mode):
        self.root = cfg.data.root
        meta_info = pd.read_csv(cfg.data.meta_info_file)

        sample = []
        for idx in index:
            video_name = meta_info.loc[idx]["filename"]
            video_path = os.path.join("train_video", video_name)
            label = meta_info.loc[idx]["score"]
            sample.append((video_path, label))

        self.samples = sample
        self.transform = transform
        self.key_frames = 8 #Select only keyframe
        self.fps = 30

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        video_data = skvideo.io.vread(os.path.join(self.root, path))

        transform = self.transform

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([self.key_frames, video_channel,  224, 224])
        for frame_idx in range(self.key_frames):
            true_index = frame_idx*self.fps
            if true_index >= video_length:
                frame = transformed_video[frame_idx-1]
                transformed_video[frame_idx] = frame
            else:
                frame = video_data[true_index]
                frame = Image.fromarray(frame)
                frame = transform(frame)
                transformed_video[frame_idx] = frame
        return transformed_video, target

    def __len__(self):
        length = len(self.samples)
        return length

