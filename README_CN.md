<h1 align="center">SnowLabel</h1>

<p align="center">
基于 SAM、DINO 特征记忆和 FastAPI 的渐进式 mask 自动标注工具。
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110%2B-009688">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

## 项目简介

SnowLabel 是一个面向实例 mask 的交互式自动标注工具。它使用 SAM 生成 mask，使用 DINO
特征编码用户教过的目标，并通过向量记忆在后续图片中预填相似目标的标签。标注过程中不重训模型；
每一次有效修正都会变成可检索的记忆。

本项目适用于类别边界需要逐步发现的视觉标注场景，例如工业检测、细粒度目标标注和数据集冷启动。

## 核心特性

- 基于 Canvas 的标注界面，支持点击、框选和全图分割。
- 渐进式向量记忆，复用用户修正过的 mask 预填后续标签。
- 两阶段检索，结合整体特征、局部特征和几何特征完成定类。
- FastAPI 服务同时提供网页 UI、标注 API、记忆库面板和旧 Label Studio ML 兼容接口。
- 支持本地 numpy 记忆后端，并可切换到 Milvus 持久化后端。
- 提供公网 Demo 启动脚本，可同时拉起 MariaDB、MinIO 和 Cloudflare 隧道。

## 环境依赖与安装

环境要求：

- Python 3.10 或更高版本。
- 推荐使用 NVIDIA GPU 运行 SAM 和 DINO 推理。
- 生产配置推荐准备本地 DINOv3 权重。

克隆并安装：

```powershell
git clone https://github.com/zhuwenmao/snowlabel.git
cd snowlabel
pip install -r requirements.txt
pip install -e .[dev]
pip install ultralytics transformers pymilvus
```

如需使用 GPU，请先按 PyTorch 官方说明安装匹配 CUDA 版本的 PyTorch 和 torchvision。

## 快速开始 / 使用示例

启动本地 FastAPI 服务：

```powershell
$env:SNOWLABEL_DINO="weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_MEMORY_BACKEND="milvus"
$env:SNOWLABEL_RES="512"
$env:SNOWLABEL_PARTS_GRID="4"
python -m snowlabel.server
```

打开应用：

```text
http://localhost:9090/
http://localhost:9090/try
http://localhost:9090/inspector
```

启动公网 Demo：

```powershell
powershell -ExecutionPolicy Bypass -File start_public_demo.ps1
```

固定公网地址：

```text
https://snowlabel.zhuwenmao.com
```

## 项目架构 / 目录结构

```text
|-- snowlabel/                 # 核心 Python 包与 FastAPI 服务
|-- snowlabel/static/          # 首页、Canvas 标注器和记忆库面板
|-- scripts/                   # 评估、冒烟测试和维护脚本
|-- eval/                      # 监督协议和质量门槛
|-- demo_images/               # 小型 Demo 图片集
|-- docs/                      # 设计说明和项目决策
|-- start_public_demo.ps1      # 公网 Demo 启动脚本
|-- requirements.txt           # 运行依赖
|-- pyproject.toml             # 包元数据和开发依赖
`-- README_CN.md               # 中文项目说明
```

## 常见问题

**必须使用 Label Studio 吗？**

不需要。当前主界面是 SnowLabel 自己的 FastAPI + Canvas 标注器。Label Studio 兼容接口仅用于旧流程集成。

**没有 DINOv3 权重怎么办？**

SnowLabel 可以回退到默认 DINOv2 配置。只有在所选 backbone 支持对应分辨率时，才应使用
DINOv3 专用的 `SNOWLABEL_RES=512` 配置。

**什么时候使用 `start_public_demo.ps1`？**

当需要完整公网 Demo 栈时使用，包括本地服务、MariaDB、MinIO 和 Cloudflare 隧道。

## 贡献指南

欢迎提交 Pull Request。请保持改动范围清晰，不要提交本地数据或模型权重，并根据改动范围运行对应检查。

常用验证命令：

```powershell
python scripts/test_protocol.py
python scripts/check_feature_space_guard.py
python scripts/check_api_edges.py
python scripts/check_durability.py
python scripts/smoke_webtool.py
```

如果改动涉及特征提取、检索、阈值、记忆后端、模型选择、分辨率或 part-grid 设置，请运行：

```powershell
python scripts/supervise.py --quick --tag <name> --compare
python scripts/supervise.py --tag <name> --compare
```

## 许可证

SnowLabel 使用 MIT License。详情见 `LICENSE`。

## 致谢与引用

SnowLabel 基于 Meta AI 的 SAM 与 DINO、FastAPI、Milvus 以及 Label Studio ML backend 协议构建。
分发衍生作品时，请遵守所有上游模型和开源库的许可证及使用条款。
