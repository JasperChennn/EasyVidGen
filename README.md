# 🎬 EasyVidGen

> *Making video generation training as easy as building blocks.*

**[中文](README_zh.md)** · **[English](README.md)** · **[Quick start](#quick-start)**

---

## 🤗 checkpoints

| Checkpoint | Hugging Face |
|------------|--------------|
| **Wan2.2-TI2V-5B-4Steps-Diffusers** | [![](https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg)](https://huggingface.co/Chenjt-pku/Wan2.2-TI2V-5B-4Steps-Diffusers) |
| **Wan2.1-1.3B-Self-Forcing-Diffusers** | [![](https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg)](https://huggingface.co/Chenjt-pku/Wan2.1-1.3B-Self-Forcing-Diffusers**) |

## 📋 Introduction

EasyVidGen is a lightweight, modular training toolkit for video generation models. It streamlines the training workflow so you can focus on ideas instead of boilerplate—train, debug, and ship faster.

## 🚀 Features

- **Modular**: Data loading, training, and inference are decoupled.
- **Ready to run**: Minimal config; fast training.
- **Ecosystem-friendly**: Built on Accelerate and common video-generation stacks.
- **Lean**: Fewer dependencies, focused on training efficiency.
- **Highly extensible**: Supports a wide range of video generation downstream tasks.

## 🚀 Updates

- **2026-03-30**: self forcing->causal transformer and teacher forcing transformer.

---

## 🗺️ Roadmap (TBD)

> The project is evolving—issues and PRs are welcome!

### ✅ Done

- [x] Project skeleton and layout
- [x] Modular architecture
- [x] DDP / DeepSpeed
- [x] I2V-Lora

### 🎯 TODO

- [ ] Support FSDP
- [ ] One-click training for more mainstream video or image generation models 
   - [x] Wan
   - [ ] CogVideoX
   - [ ] HunYuanVideo
   - [ ] Flux
   - [ ] LTX
   - [ ] SANA-Video
- [ ] Training logs and monitoring
- [ ] Plugin-style custom modules

### Downstream
- [x] DMD, Self-Forcing [60%]
- [ ] Sparse Attention for long video generation
- [ ] Lora-Moe
- [ ] MemCompression
- [ ] Self-Forcing
- [ ] GRPO
- [ ] Tiny WAM(World Action Model)

### 🔮 Long term

- [ ] Faster iteration out of the box
- [ ] Auto-tuning for video models
- [ ] More pretrained weights and templates

---

## ⚡ Quick start

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

Image-to-video — see `inference/inference_i2v.py` for all flags.

**Full checkpoint** (`Wan2.2-TI2V-5B-Diffusers`, multi-step):

```bash
python inference/inference_i2v.py \
  --model_name checkpoints/Wan2.2-TI2V-5B-Diffusers \
  --image_path assets/images/demo.jpg \
  --prompt "The dog is walking happily in the road." \
  --num_frames 81 \
  --height 704 \
  --width 1280 \
  --num_inference_steps 30 \
  --output tmp/demo.mp4 \
  --fps 16 \
  --device cuda
```

**4-step distilled checkpoint** (`Wan2.2-TI2V-5B-4Steps-Diffusers`，4 steps)



```bash
python inference/inference_i2v.py \
  --model_name checkpoints/Wan2.2-TI2V-5B-4Steps-Diffusers \
  --image_path assets/images/cat.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard." \
  --num_frames 121 \
  --height 1280 \
  --width 704 \
  --num_inference_steps 4 \
  --guidance_scale 1.0 \
  --output tmp/demo_i2v.mp4 \
  --fps 16 \
  --device cuda
```

### Lora Training

1. download data
```bash
bash examples/cakeify/download.sh
python examples/cakeify/process_data.py
```

---

## 🤝 Acknowledgements

EasyVidGen builds on open tools and research from the community. We are grateful to:

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [CogVideoX](https://github.com/zai-org/CogVideo)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [CausVid](https://github.com/tianweiy/CausVid)
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
- [Wan2.2-TI2V-5B-Turbo](https://github.com/quanhaol/Wan2.2-TI2V-5B-Turbo)

We welcome bug reports, feature ideas, documentation improvements, and new modules.

## 📚 Contact
If you have any suggestions or find our work helpful, feel free to contact us

Email: cjt@stu.pku.edu.cn

If you find our work useful, <b>please consider giving a star ⭐ to this github repository and citing it</b>:


## 📄 License

[MIT License](LICENSE)
