# 数据集（`easyvid/datasets/`）

## `dataset.py` — `VideoDataset`

**标注**：UTF-8 JSON **数组**，每项需有 `video_path` 或 `media_path`；可选 `caption`（默认 `""`）。  
**媒体**：后缀为 `.mp4/.avi/.mov/.mkv` 的当视频抽帧，否则当单张图（`T=1`）。  
**返回**：`pixel_values` `(T,C,H,W)`，`text` 为 caption。`text_drop_ratio>0` 时随机把文本置空。读失败会顺延重试（最多 `max_read_retries`）。

**`collate_fn`**：`pixel_values` → `(B,C,T,H,W)`，`captions` 为字符串列表。  
**`RepeatLastBatchSampler`**：凑满整 batch，不足时用索引 `0` 重复。

```json
[{"video_path": "a.mp4", "caption": "..."}]
```

---

## `text_dataset.py` — `TextFileDataset`

**二选一**：`txt_path` **或** `precomputed_embeddings_path`（目录）。

| 模式 | 说明 |
|------|------|
| txt | 一行一条；空行、`#` 行忽略 |
| 预计算目录 | 文件名为 `00000000.pt`, `00000001.pt`, …（`:08d`，从 0 连续）|

**每个 `.pt`**：`Tensor [n,d]`，或仅含一对的 `{prompt: emb}`。返回 `text`；预计算模式另有 `prompt_embeds`。

---

换 Dataset 时需与 Trainer 里 `batch` 字段（如 `pixel_values` / `captions` / `prompt_embeds`）一致。
