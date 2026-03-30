"""
使用 Accelerate 多进程/多卡并行预计算 Wan I2V 文本编码（与 ``TextFileDataset`` + ``precomputed_embeddings_path`` 约定一致）。

输出目录下为 ``00000000.pt``, ``00000001.pt``, ...（与 ``easyvid/datasets/text_dataset.py`` 中 ``_emb_filename`` 一致），
每个文件为 ``[seq_len, hidden_dim]`` 的 ``float32`` Tensor（与训练时 ``encode_prompt`` 单条 padding 后形状一致）。

用法示例::

    accelerate launch scripts/precompute_text_emb/prepare_for_wan.py \\
        --pretrained_model_name_or_path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \\
        --txt_path /path/to/captions.txt \\
        --output_dir /path/to/text_emb_dir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from easyvid.datasets.text_dataset import TextFileDataset
from easyvid.pipelines.wan.pipeline_i2v import WanImageToVideoPipeline

logger = get_logger(__name__, log_level="INFO")


class IndexedCaptionDataset(Dataset):
    """包装 ``TextFileDataset``，在样本中附带全局下标，便于按 ``{idx:08d}.pt`` 落盘。"""

    def __init__(self, txt_path: str):
        self._inner = TextFileDataset(txt_path=txt_path, text_drop_ratio=-1.0)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict:
        row = self._inner[idx]
        return {"idx": idx, "text": row["text"]}


def collate_precompute(batch: list) -> dict:
    return {
        "idx": torch.tensor([b["idx"] for b in batch], dtype=torch.long),
        "captions": [b["text"] for b in batch],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Wan text embeddings with Accelerate (multi-GPU).")
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Wan diffusers 模型目录或 Hub id（与训练一致）。",
    )
    p.add_argument("--txt_path", type=str, required=True, help="caption 文本文件，每行一条（与 TextFileDataset 一致）。")
    p.add_argument("--output_dir", type=str, required=True, help="输出目录，将写入 00000000.pt, ...")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="覆盖 accelerate 配置的混合精度；不传则使用 accelerate 默认。",
    )
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)
    p.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="与 Wan ``encode_prompt`` / 训练时一致（trainer 未显式传参时 pipeline 默认 226）。",
    )
    p.add_argument("--local_files_only", action="store_true", help="仅使用本地已缓存权重，不联网下载。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Loading Wan pipeline (text encoder only, no transformer/vae).")
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=weight_dtype,
        revision=args.revision,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )
    pipe.to(accelerator.device)
    pipe.text_encoder.eval()

    dataset = IndexedCaptionDataset(args.txt_path)
    # 显式按 rank 划分下标，保证各进程处理的样本下标集合互不相交（无重复计算）。
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        drop_last=False,
    )
    sampler.set_epoch(0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_precompute,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    loader = accelerator.prepare(loader)

    total = len(dataset)
    per_rank = len(sampler)
    if accelerator.is_main_process:
        logger.info(
            f"Total captions: {total}, processes: {accelerator.num_processes}, "
            f"samples this rank (incl. sampler padding): {per_rank}, device: {accelerator.device}"
        )

    progress = tqdm(
        loader,
        desc="encode_prompt",
        disable=not accelerator.is_local_main_process,
    )

    with torch.no_grad():
        for batch in progress:
            idx_cpu = batch["idx"]
            captions = batch["captions"]
            prompt_embeds, _ = pipe.encode_prompt(
                captions,
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_sequence_length,
                device=accelerator.device,
                dtype=weight_dtype,
            )
            # [B, seq, dim] -> 每条存 [seq, dim] float32
            for j in range(prompt_embeds.shape[0]):
                gidx = int(idx_cpu[j].item())
                # 仅当本 batch 为 sampler 为凑整 batch 而重复的 padding 样本时，gidx 会在本 rank 内重复；写盘幂等。
                path = os.path.join(args.output_dir, f"{gidx:08d}.pt")
                emb = prompt_embeds[j].detach().float().cpu()
                torch.save(emb, path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info(f"Done. Embeddings written under {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
