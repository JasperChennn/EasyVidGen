# 🎬 EasyVidGen

> **让视频生成训练像搭积木一样简单**  
> *Making video generation training as easy as building blocks.*

**[中文](#readme-zh)** · **[English](#readme-en)** · **[快速开始](#快速开始)**

---

<a id="readme-zh"></a>

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

- [ ] 支持主流视频生成模型（Wan、LTX等）一键训练
- [ ] 可视化训练日志与监控面板
- [ ] 多 GPU / 分布式训练支持
- [ ] 自定义模块插件体系

### 🔮 长期愿景

- [ ] 开箱即用/快速迭代
- [ ] 视频生成模型自动调优
- [ ] 丰富的预训练模型库与模板

---

## 🤝 贡献指南

欢迎所有形式的贡献！包括但不限于：

- 提交 Bug 反馈
- 提出功能建议
- 完善文档
- 开发新模块

## 📄 开源协议

[MIT License](LICENSE)

---

<a id="readme-en"></a>

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

### 🎯 Near term

- [ ] One-click training for mainstream video models (e.g. Wan, LTX)
- [ ] Training logs and monitoring
- [ ] Multi-GPU / distributed training
- [ ] Plugin-style custom modules

### 🔮 Long term

- [ ] Faster iteration out of the box
- [ ] Auto-tuning for video models
- [ ] More pretrained weights and templates

---

## 🤝 Contributing

We welcome bug reports, feature ideas, documentation improvements, and new modules.

## 📄 License

[MIT License](LICENSE)

---

<a id="快速开始"></a>

## 快速开始

详见仓库内训练与推理脚本（如 `train/`、`inference/` 目录）。配置与命令以各子目录说明为准。

For a quick start in English, see the `train/` and `inference/` folders and their scripts or comments.
