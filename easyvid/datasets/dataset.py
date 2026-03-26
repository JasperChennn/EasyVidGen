import json
import logging
import os
import random
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize

logger = logging.getLogger(__name__)

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def resize_frames(
    frames: torch.Tensor,
    height: int,
    width: int,
    crop_type: str = "center",
) -> torch.Tensor:
    """
    Resize frames to cover the target (height, width), then crop.

    Args:
        frames: Tensor shaped (N, C, H, W) where N is the number of frames (or 1).
        height, width: Output crop size.
        crop_type: "random" = random crop and 50% chance horizontal flip;
            "center" = center crop.
    """
    _, _, h, w = frames.shape
    if w / h > width / height:
        new_height = height
        new_width = int(w * height / h)
    else:
        new_width = width
        new_height = int(h * width / w)

    frames = resize(
        frames,
        size=[new_height, new_width],
        interpolation=InterpolationMode.BICUBIC,
    )

    delta_h = frames.shape[2] - height
    delta_w = frames.shape[3] - width

    if crop_type == "random":
        top = random.randint(0, max(0, delta_h))
        left = random.randint(0, max(0, delta_w))
        frames = crop(frames, top=top, left=left, height=height, width=width)
        if random.random() < 0.5:
            frames = transforms.functional.hflip(frames)
    else:
        top, left = delta_h // 2, delta_w // 2
        frames = crop(frames, top=top, left=left, height=height, width=width)

    return frames


class VideoDataset(Dataset):
    """
    Loads samples from JSON annotations. Each row should have ``video_path`` (video or
    image, relative or absolute) and optionally ``caption``.

    Images become a single-frame tensor ``(1, C, H, W)``; videos stay ``(T, C, H, W)``.
    ``collate_fn`` stacks these into ``(B, C, T, H, W)``.
    """

    def __init__(
        self,
        ann_path: Union[str, Sequence[str]],
        data_root: Optional[str] = None,
        video_sample_size: Union[int, Sequence[int]] = 512,
        video_sample_n_frames: int = 16,
        text_drop_ratio: float = -1.0,
        crop_type: str = "center",
        max_read_retries: int = 32,
    ):
        if isinstance(ann_path, str):
            ann_paths: List[str] = [ann_path]
        else:
            ann_paths = list(ann_path)

        self.dataset: List[dict] = []
        for path in ann_paths:
            with open(path, "r", encoding="utf-8") as f:
                self.dataset.extend(json.load(f))

        self.data_root = data_root
        self.text_drop_ratio = text_drop_ratio
        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = (
            (video_sample_size, video_sample_size)
            if isinstance(video_sample_size, int)
            else tuple(video_sample_size)
        )
        self.crop_type = crop_type
        self.max_read_retries = max(1, int(max_read_retries))

        self.transforms = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

    def _resolve_path(self, path: str) -> str:
        if self.data_root and not os.path.isabs(path):
            return os.path.join(self.data_root, path)
        return path

    @staticmethod
    def is_video_file(path: str) -> bool:
        lower = path.lower()
        return any(lower.endswith(ext) for ext in _VIDEO_EXTS)

    def process_image(self, image_path: str, size: Sequence[int]) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = transforms.ToTensor()(image)
        image = resize_frames(
            image.unsqueeze(0),
            size[0],
            size[1],
            crop_type=self.crop_type,
        )
        return self.transforms(image)

    def process_video(self, full_path: str) -> torch.Tensor:
        vr = VideoReader(full_path, num_threads=2)
        total_frames = len(vr)
        if total_frames < self.video_sample_n_frames:
            raise ValueError(
                f"Video has fewer frames ({total_frames}) than required ({self.video_sample_n_frames})"
            )

        start_idx = np.random.randint(
            0, total_frames - self.video_sample_n_frames + 1
        )
        indices = range(start_idx, start_idx + self.video_sample_n_frames)
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        frames = resize_frames(
            frames,
            self.video_sample_size[0],
            self.video_sample_size[1],
            crop_type=self.crop_type,
        )
        frames = frames / 255.0
        return self.transforms(frames)

    def process_file(self, media_path: str) -> torch.Tensor:
        full_path = self._resolve_path(media_path)
        if self.is_video_file(media_path):
            return self.process_video(full_path)
        pixel_values = self.process_image(full_path, self.video_sample_size)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        return pixel_values

    def _media_path(self, data: dict) -> str:
        path = data.get("video_path") or data.get("media_path")
        if not path:
            raise KeyError("annotation must contain 'video_path' or 'media_path'")
        return path

    def __getitem__(self, idx: int):
        n = len(self.dataset)
        last_exc: Optional[BaseException] = None

        for k in range(self.max_read_retries):
            i = (idx + k) % n
            data = self.dataset[i]
            try:
                media_path = self._media_path(data)
                pixel_values = self.process_file(media_path)
                text = data.get("caption", "")
                break
            except Exception as e:
                last_exc = e
                logger.debug(
                    "Skip sample index=%s path=%r: %s",
                    i,
                    data.get("video_path") or data.get("media_path"),
                    e,
                )
        else:
            raise RuntimeError(
                f"Failed to load any sample after {self.max_read_retries} attempts "
                f"(starting from idx={idx})"
            ) from last_exc

        if self.text_drop_ratio > 0 and torch.rand(1).item() < self.text_drop_ratio:
            text = ""

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        return {
            "pixel_values": pixel_values,
            "text": text,
        }

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(examples):
    """
    Stacks ``pixel_values`` from ``__getitem__``, each ``(T, C, H, W)``, into a batch
    ``(B, C, T, H, W)``.
    """
    pixel_values = [
        ex["pixel_values"].unsqueeze(0).permute(0, 2, 1, 3, 4) for ex in examples
    ]
    pixel_values = torch.cat(pixel_values, dim=0).contiguous().float()
    return {
        "pixel_values": pixel_values,
        "captions": [ex["text"] for ex in examples],
    }


def _test():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    ann_path = os.path.join(repo_root, "examples", "data.json")
    dataset = VideoDataset(
        ann_path=ann_path,
        data_root=None,
        video_sample_size=[704, 1280],
        video_sample_n_frames=49,
        text_drop_ratio=-1.0,
    )
    print(len(dataset))
    from time import time

    t = time()
    x = dataset[0]
    print(time() - t)


if __name__ == "__main__":
    _test()
    _test()
