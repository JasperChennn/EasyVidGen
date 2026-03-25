# 🎬 EasyVidGen

> **让视频生成训练像搭积木一样简单**  
> *Making video generation training as easy as building blocks.*

**[中文](README_zh.md)** · **[English](README.md)** · **[快速开始](#快速开始)**

---

## 📋 项目简介

EasyVidGen 是一款面向视频生成模型的轻量化、模块化训练工具，致力于简化视频生成模型的训练流程，让开发者无需关注复杂的工程细节，像搭积木一样快速完成模型训练、调试与部署。

## 🚀 核心特性

- 模块化设计：数据加载、模型训练、推理部署全流程解耦
- 开箱即用：极简配置，一行命令启动训练
- 兼容主流框架：基于 Accelerator 生态，适配主流视频生成模型
- 轻量化无冗余：剔除无用依赖，专注核心训练效率

---

## 🗺️ Roadmap | 开发计划（待更新）

> 项目持续迭代中，欢迎提交 Issue / PR 参与共建！

### ✅ 已完成

- [x] 项目基础框架搭建
- [x] 核心模块化架构设计

### 🎯 近期规划

- [ ] 支持Deepspeed、FSDP
- [ ] 支持主流视频生成模型（Wan、LTX等）一键训练
- [ ] 可视化训练日志与监控面板
- [ ] 多 GPU / 分布式训练支持
- [ ] 自定义模块插件体系

### 🔮 长期愿景

- [ ] 开箱即用/快速迭代
- [ ] 视频生成模型自动调优
- [ ] 丰富的预训练模型库与模板

---

## 快速开始

### 环境

1. **Python**：建议 3.10+，并安装 [PyTorch](https://pytorch.org/)。
2. **依赖**：在仓库根目录执行：

   ```bash
   cd EasyVidGen
   pip install -r requirements.txt
   ```

   > `flash_attn` 若安装失败，可按官方说明从源码编译，或根据实际环境暂时注释 `requirements.txt` 中对应行后再安装其余依赖。

3. **工作目录**：以下命令均在仓库根目录（含 `inference/`、`src/` 等）下执行；请先将该路径加入 `PYTHONPATH`，例如：

   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **模型权重**：使用脚本从 Hugging Face 拉取（可按需改镜像或模型名），例如：

   ```bash
   bash scripts/download_models.sh
   ```

   下载完成后，推理脚本中的 `--model_name` 指向本地目录（如 `checkpoints/Wan2.2-TI2V-5B-Diffusers`）。

### 推理（Inference）

以图生视频（I2V）为例，可直接使用示例脚本（请先按脚本内路径修改 `model_name`、GPU 等）：

```bash
bash inference/inference_i2v.sh
```

或等价地直接调用 Python（参数说明见 `inference/inference_i2v.py`）：

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

### 训练（Train）

（待补充：启动命令、Accelerate / DeepSpeed 配置、数据格式等将随后更新。）

---

## 🤝 贡献指南

欢迎所有形式的贡献！包括但不限于：

- 提交 Bug 反馈
- 提出功能建议
- 完善文档
- 开发新模块

## 📄 开源协议

[MIT License](LICENSE)

