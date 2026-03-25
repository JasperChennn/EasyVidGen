import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize
from PIL import Image
from decord import VideoReader
from contextlib import contextmanager
import gc
import numpy as np
import random


def resize_frames(frames, height, width, crop_type='center'):  # 只添加一个参数
    """
    crop_type: 'random' or 'center'
    """
    if frames.shape[3] / frames.shape[2] > width / height:
        new_height = height
        new_width = int(frames.shape[3] * height / frames.shape[2])
    else:
        new_width = width
        new_height = int(frames.shape[2] * width / frames.shape[3])
    
    frames = resize(frames, size=[new_height, new_width], interpolation=InterpolationMode.BICUBIC)
    
    if crop_type == 'random':
        # random crop
        delta_h = frames.shape[2] - height
        delta_w = frames.shape[3] - width
        top = random.randint(0, max(0, delta_h))
        left = random.randint(0, max(0, delta_w))
        frames = crop(frames.squeeze(0), top=top, left=left, height=height, width=width)

        if random.random() < 0.5:
            frames = transforms.functional.hflip(frames)
    else:
        # central crop
        delta_h = frames.shape[2] - height
        delta_w = frames.shape[3] - width
        top, left = delta_h // 2, delta_w // 2
        frames = crop(frames.squeeze(0), top=top, left=left, height=height, width=width)
    
    return frames


class VideoDataset(Dataset):
    def __init__(
        self,
        ann_path,
        data_root=None,
        video_sample_size=512,
        video_sample_n_frames=16,
        text_drop_ratio=-1,
    ):
        # self.dataset = json.load(open(ann_path))
        if isinstance(ann_path, str):
            ann_paths = [ann_path]
        elif isinstance(ann_path, (list, tuple)):
            ann_paths = list(ann_path)
        self.dataset = []
        for ann_path in ann_paths:
            self.dataset = self.dataset + json.load(open(ann_path))
        self.data_root = data_root
        self.text_drop_ratio = text_drop_ratio
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = (video_sample_size, video_sample_size) if isinstance(video_sample_size, int) else tuple(video_sample_size)
        
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def process_image(self, image_path, size):
        """process single image"""
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        image = resize_frames(image.unsqueeze(0), size[0], size[1])
        return self.transforms(image)

    def process_video(self, video_path):
        """process single video"""
        full_path = os.path.join(self.data_root, video_path) if self.data_root else video_path
        
        vr = VideoReader(full_path, num_threads=2)
        # ensure enough frames 
        total_frames = len(vr)
        if total_frames < self.video_sample_n_frames:
            raise ValueError(f"Video has fewer frames ({total_frames}) than required ({self.video_sample_n_frames})")

        start_idx = np.random.randint(0, total_frames - self.video_sample_n_frames + 1)
        indices = range(start_idx, start_idx + self.video_sample_n_frames)
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = resize_frames(frames, self.video_sample_size[0], self.video_sample_size[1])
        frames = frames / 255.

        return self.transforms(frames)

    def is_video_file(self, file_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        return any(file_path.lower().endswith(ext) for ext in video_extensions)

    def process_file(self, file_path):
        full_path = os.path.join(self.data_root, file_path) if self.data_root else file_path
        
        if self.is_video_file(file_path):
            return self.process_video(file_path)
        else:
            # 处理图像
            pixel_values = self.process_image(full_path, self.video_sample_size, self.data_aug)
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            
            return pixel_values

    def __getitem__(self, idx):
        data = self.dataset[idx]
        try:
            pixel_values = self.process_file(data['video_path'])
            text = data['caption']
        except Exception as e:
            print(f"Error processing video {data['video_path']}: {e}")
            return self.__getitem__((idx + 1) % len(self.dataset))
        
        text = "" if self.text_drop_ratio > 0 and torch.rand(1) < self.text_drop_ratio else text

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        return {
            'pixel_values': pixel_values,
            'text': text,
        }

    def __len__(self):
        return len(self.dataset)


def collate_fn(examples):
    pixel_values = [ex['pixel_values'].unsqueeze(0).permute(0, 2, 1, 3, 4) for ex in examples]
    pixel_values = torch.cat(pixel_values, dim=0).contiguous().float()
    
    return {
        'pixel_values': pixel_values,
        'captions': [ex['text'] for ex in examples],
    }


def test():
    ann_path="/mnt/nas-data-3/anyi.cjt/datasets/Open-VFX/cakeify.json"
    data_root=None
    dataset = VideoDataset(
        ann_path=ann_path,
        data_root=data_root,
        video_sample_size=[704, 1280],
        video_sample_n_frames=81,
        text_drop_ratio=-1,
    )
    print(len(dataset))
    test_time(dataset)
    import pdb; pdb.set_trace()

def test_time(dataset):
    from time import time 
    t = time()
    x = dataset[0]
    print(time() - t)

if __name__ == "__main__":
    test()
