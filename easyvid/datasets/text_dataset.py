"""
- 仅 ``txt_path``：每行一条 caption。
- 仅 ``precomputed_embeddings_path``（**文件夹**）：内含 ``00000000.pt``, ``00000001.pt``, ...（``:08d``）无需 ``txt_path``。
"""

import os
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


def _load_lines_from_txt(paths: Sequence[str]) -> List[str]:
    lines: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)
    return lines


def _parse_sample_pt(
    path: str, map_location: Union[str, torch.device]
) -> Tuple[str, torch.Tensor]:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, torch.Tensor):
        if obj.ndim != 2:
            raise ValueError(f"{path}: Tensor must be [n, d], got {tuple(obj.shape)}")
        return "", obj
    if isinstance(obj, dict):
        if len(obj) != 1:
            raise ValueError(
                f"{path}: expected a single {{prompt: emb}} pair, got {len(obj)} keys"
            )
        (prompt, emb), = obj.items()
        if not isinstance(emb, torch.Tensor) or emb.ndim != 2:
            raise ValueError(
                f"{path}: emb must be Tensor [n, d], got {type(emb)} {getattr(emb, 'shape', None)}"
            )
        return str(prompt), emb
    raise TypeError(f"{path}: expected Tensor or dict, got {type(obj)}")


def _emb_filename(idx: int) -> str:
    """与预计算文件命名一致：``{idx:08d}.pt``。"""
    return f"{idx:08d}.pt"


def _count_indexed_pt_dir(folder: str) -> int:
    """统计 ``00000000.pt``, ``00000001.pt``, ... 连续存在的个数（从 0 起不能断档）。"""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"precomputed_embeddings_path must be a folder: {folder}")
    n = 0
    while os.path.isfile(os.path.join(folder, _emb_filename(n))):
        n += 1
    if n == 0:
        raise ValueError(f"No {{idx:08d}}.pt files (starting from 00000000.pt) in {folder}")
    return n


class TextFileDataset(Dataset):
    """
    Args:
        txt_path: 一个或多个 ``.txt``；**与** ``precomputed_embeddings_path`` **二选一**。
        text_drop_ratio: ``>0`` 时以该概率将返回的文本置为 ``""``（嵌入不变）。
        precomputed_embeddings_path: 若设置，为目录路径；``__init__`` 只统计样本数，
            各 ``.pt`` 在 ``__getitem__`` 中再加载。不需要 ``txt_path``。
    """

    def __init__(
        self,
        txt_path: Optional[Union[str, Sequence[str]]] = None,
        text_drop_ratio: float = -1.0,
        *,
        precomputed_embeddings_path: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
    ):
        self._prompts: List[str] = []
        self._emb_dir: Optional[str] = None
        self._num_emb: int = 0
        self._map_location: Union[str, torch.device] = map_location

        if precomputed_embeddings_path is not None:
            if txt_path is not None:
                raise ValueError("Use either txt_path or precomputed_embeddings_path, not both.")
            self._emb_dir = os.path.abspath(precomputed_embeddings_path)
            self._num_emb = _count_indexed_pt_dir(self._emb_dir)
        else:
            if txt_path is None:
                raise ValueError("Provide txt_path, or precomputed_embeddings_path (a folder).")
            paths = [txt_path] if isinstance(txt_path, str) else list(txt_path)
            self._prompts = _load_lines_from_txt(paths)
            if not self._prompts:
                raise ValueError(f"No text lines loaded from {paths!r}.")

        self.text_drop_ratio = text_drop_ratio

    def __len__(self) -> int:
        if self._emb_dir is not None:
            return self._num_emb
        return len(self._prompts)

    def __getitem__(self, idx: int) -> dict:
        if self._emb_dir is not None:
            path = os.path.join(self._emb_dir, _emb_filename(idx))
            text, emb = _parse_sample_pt(path, self._map_location)
        else:
            text = self._prompts[idx]
            emb = None

        if self.text_drop_ratio > 0 and torch.rand(1).item() < self.text_drop_ratio:
            text = ""

        out: dict = {"text": text}
        if emb is not None:
            out["prompt_embeds"] = emb
        return out


def collate_fn(examples):
    if "prompt_embeds" in examples[0]:
        return collate_fn_text_embeddings(examples)
    return {"captions": [ex["text"] for ex in examples]}


def collate_fn_text_embeddings(examples):
    return {
        "captions": [ex["text"] for ex in examples],
        "prompt_embeds": torch.stack([ex["prompt_embeds"] for ex in examples], dim=0),
    }
