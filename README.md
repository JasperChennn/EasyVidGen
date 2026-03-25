# 🎬 EasyVidGen

> *Making video generation training as easy as building blocks.*

**[中文](README_zh.md)** · **[English](README.md)** · **[Quick start](#quick-start)**

---

## 📋 Introduction

EasyVidGen is a lightweight, modular training toolkit for video generation models. It streamlines the training workflow so you can focus on ideas instead of boilerplate—train, debug, and ship faster.

## 🚀 Features

- **Modular**: Data loading, training, and inference are decoupled.
- **Ready to run**: Minimal config; start training with one command.
- **Ecosystem-friendly**: Built on Accelerate and common video-generation stacks.
- **Lean**: Fewer dependencies, focused on training efficiency.

---

## 🗺️ Roadmap (TBD)

> The project is evolving—issues and PRs are welcome!

### ✅ Done

- [x] Project skeleton and layout
- [x] Modular architecture
- [x] DDP / DeepSpeed
- [x] I2V-Lora

### 🎯 TODO

- [ ] Support  FSDP
- [ ] One-click training for mainstream video models (e.g. Wan, LTX)
- [ ] Training logs and monitoring
- [ ] Multi-GPU / distributed training
- [ ] Plugin-style custom modules

### 🔮 Long term

- [ ] Faster iteration out of the box
- [ ] Auto-tuning for video models
- [ ] More pretrained weights and templates

---

## Quick start

### Environment

1. **Python**: 3.10+ recommended.
2. **Dependencies** (from the repo root):

   ```bash
   cd EasyVidGen
   pip install -r requirements.txt
   ```

   > If `flash_attn` fails to install, follow its official build instructions, or temporarily remove that line from `requirements.txt` and install the rest.

3. **Model weights**: Example download script (edit mirror or model IDs as needed):

   ```bash
   bash scripts/download_models.sh
   ```

   Point `--model_name` at the local folder (e.g. `checkpoints/Wan2.2-TI2V-5B-Diffusers`).

### Inference

Image-to-video example — adjust `model_name`, `CUDA_VISIBLE_DEVICES`, etc. inside the shell script if needed:

```bash
bash inference/inference_i2v.sh
```

Or call Python directly (see `inference/inference_i2v.py` for all flags):

```bash
python inference/inference_i2v.py \
  --model_name checkpoints/Wan2.2-TI2V-5B-Diffusers \
  --image_path examples/demo.jpg \
  --prompt "The dog is walking happily in the road." \
  --num_frames 49 \
  --height 704 \
  --width 1280 \
  --num_inference_steps 30 \
  --output tmp/demo.mp4 \
  --fps 16 \
  --device cuda
```

### Training

*(To be documented: launch commands, Accelerate / DeepSpeed configs, data layout, etc.)*

---

## 🤝 Contributing

We welcome bug reports, feature ideas, documentation improvements, and new modules.

## 📄 License

[MIT License](LICENSE)
