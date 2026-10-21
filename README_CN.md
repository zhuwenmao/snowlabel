<h1 align="center">SnowLabel</h1>

<p align="center">
基于 SAM、DINOv3 特征记忆和 FastAPI 的渐进式自动标注原型。
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.1-76B900">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.136-009688">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center"><a href="README.md">English</a> · 中文 · <a href="https://snowlabel.zhuwenmao.com">在线 Demo</a></p>

## 项目简介

SnowLabel 是一个面向实例 mask 的渐进式自动标注技术原型：SAM 给出 mask，DINOv3
编码用户确认过的区域，向量记忆复用之前的修正，为后续图片里的相似区域定类。
标注过程中模型不重训。

v0.1 的目标是为完整反馈闭环提供可复现的证据：分割 → 建议 → 接受或修正 → 存入记忆 →
安全复用。它不是成熟的商业标注平台，也还没有通用的标注导出流程。

![SnowLabel 架构](evidence/v0.1/architecture.png)

## 核心特性

- Canvas 标注界面，支持点击、框选和全图分割三种模式。
- 渐进式向量记忆，可预览、删除用户确认过的记忆样本。
- 两段式检索：整体特征召回，4×4 局部特征与几何特征精排。
- 置信度分档：自动标注 / 建议确认 / 静默拒识。
- 本地 numpy 或 Milvus 记忆后端；公网 Demo 支持多用户隔离。
- MinIO、MariaDB、Milvus 三库图片处理缓存，重复全分割直接命中缓存。
- FastAPI 同时提供 UI、API，并保留 Label Studio 兼容接口。

**可复现证据（2026-08-05）**

| 指标 | COCO 回归基准 | VisA 工业协议 |
|---|---:|---:|
| 范围 | 5 类，125 个已见样本 | 50 个已见类、300 个样本；12 个留出类、48 个探针 |
| 后程有效 mask 的自动标注精度 | 100%（39 次自动） | 100%（24 次自动） |
| 后程有效 mask 的自动覆盖率 | 61.9% | 16.0% |
| 自动标注污染率 | 0% | 0.33% |
| 开集误自动 | 0% | 总体 14.58%；23 个有效 mask 探针中为 0% |
| mask 成功率（IoU ≥ 0.5） | 84.8% | 49.33% |
| 后程人工省力比 | 66.67% | -0.33% |

VisA 结果并不说明已经具备工业可用性。它说明的是：分割可用时，自动标注依然精确；
微小缺陷定位和未知类建议仍是主要短板。在一张 1358×1104 的 VisA PCB 图上，真实
`/api/segment_all` 链路无缓存耗时 69.843 秒，命中缓存 1.039 秒，重复运行加速 67.24 倍。

COCO 与 VisA 的机器可读结果、固定的 VisA 样本清单、逐样本预测、曲线和失败案例可视化都在
[`evidence/v0.1`](evidence/v0.1)。

## 开发方式与我的角色

本仓库的大部分实现代码由 AI 编程助手（Claude Code 和 Codex）编写。我负责的是
agent 替代不了的部分：决定做什么、怎么衡量、接受哪些改动。

- **问题定义与指标。** 北极星指标是相对"只有 SAM、没有记忆"的纯手动基线的人工省力比，
  只把记忆闭环真正省下的工作算进来。贴错的自动标签会污染数据集，比沉默更糟，所以自动标注
  精度是硬门槛，记忆污染率单独统计；绝不靠放松阈值换覆盖率。
- **评测程序就是验收关卡。** 改动特征、检索、阈值、记忆、模型、分辨率或分块数，都必须通过
  [`scripts/supervise.py`](scripts/supervise.py)：硬门槛加上对 baseline 的噪声感知回归检查
  （[`eval/PROTOCOL.md`](eval/PROTOCOL.md)、[`eval/gates.json`](eval/gates.json)）。
  agent 写的代码就是靠它决定接受还是退回。
- **技术取舍**（理由和证据见 [`docs/DECISIONS.md`](docs/DECISIONS.md)）：
  - SAM 2.1 保持 tiny，不升 base。base 的 mIoU 更高（约 0.81），但更精细的 mask 让卡车比
    真正的汽车还"像汽车"，越过自动阈值、污染记忆。
  - 置信度改为绝对混合分，而不是随类别数变化的类间占比；阈值在两个开集种子上校准，
    要求零误自动。
  - 在 milvus-lite 前加一层内存镜像：交互式标注需要"写完马上能读到"，而 Windows 上的
    milvus-lite 做不到稳定可靠。
  - 只存修正和手动新增；被接受的预测只记遥测，不重复入库。
- **如实报告。** 上面的 VisA 结果包含失败部分：被接受的自动标签依然精确，但微小缺陷的 mask
  只有约一半可用，所以 VisA 上的省力比约等于零。瓶颈在分割，不在检索。
- **产品决策与验收。** 交互模型、新手引导、示例数据集；每项改动都经过审阅，并在运行中的应用里检查过。

## 环境依赖与安装

已验证环境：

- Windows 10/11、Python 3.11、CUDA 12.1、8 GB 显存的 NVIDIA GPU。
- 包元数据支持 Python 3.10+；发布的依赖锁定文件对应已验证的 Windows / Python 3.11 / CUDA 12.1。
- 复现公开结果需要 DINOv3-L 和 SAM2.1 tiny 权重。

```powershell
git clone https://github.com/zhuwenmao/snowlabel.git
cd snowlabel
pip install -r requirements.txt -c constraints/windows-py311-cu121.txt
pip install -e .[dev] -c constraints/windows-py311-cu121.txt
```

约束文件锁定了完整的已测试依赖树和 CUDA 12.1 的 PyTorch 源。也可以一键创建并验证干净的虚拟环境：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_verified_env.ps1
```

在 Hugging Face 申请 `facebook/dinov3-vitl16-pretrain-lvd1689m` 的访问权限，把快照放到
`weights/facebookdinov3-vitl16-pretrain-lvd1689m`，并把 `sam2.1_t.pt` 放在仓库根目录。
权重文件不进 Git。

```powershell
python scripts/verify_environment.py --require-weights
```

## 快速开始

启动本地服务：

```powershell
$env:SNOWLABEL_DINO="weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_MEMORY_BACKEND="numpy"
$env:SNOWLABEL_RES="512"
$env:SNOWLABEL_PARTS_GRID="4"
python -m snowlabel.server
```

打开 `http://localhost:9090/` 进入产品，`/try` 进入带引导的试用，`/inspector` 管理记忆。
完整公网栈的启动方式：

```powershell
powershell -ExecutionPolicy Bypass -File start_public_demo.ps1
```

启动脚本按顺序查找 Python：环境变量 `SNOWLABEL_PYTHON` → 名为 `snowlabel` 的 conda 环境 →
`PATH` 里的 `python`。固定公网 Demo：`https://snowlabel.zhuwenmao.com`。

复现工业证据：

```powershell
python scripts/download_visa.py
tar -xf data/datasets/visa/VisA_20220922.tar -C data/datasets/visa
python scripts/evaluate_visa.py --tag visa-v0.1 --publish-evidence
python scripts/benchmark_everything_api.py
python scripts/supervise.py --tag coco-v0.1 --compare
```

## 目录结构

```text
|-- snowlabel/                 # FastAPI 服务、模型、记忆和 UI
|-- eval/                      # COCO 与 VisA 评测协议
|-- scripts/                   # 环境、评测、冒烟和维护脚本
|-- constraints/               # 已验证的 Python 3.11 / CUDA 12.1 依赖锁定
|-- evidence/v0.1/             # 公开指标、样本清单、预测和图表
|-- demo_images/               # 小型演示图片集
|-- docs/                      # 架构说明和技术决策
|-- requirements.txt           # 运行依赖范围
|-- pyproject.toml             # 包元数据和开发依赖
`-- start_public_demo.ps1      # 完整公网 Demo 启动脚本
```

## 常见问题

**SnowLabel 是生产级标注平台吗？**

不是。v0.1 是渐进式自动标注的技术原型，用来演示和度量记忆反馈闭环。工业微小缺陷分割、
开集校准、导出、权限控制和运维工具都还需要更多工作。

**必须使用 Label Studio 吗？**

不需要。主界面是 SnowLabel 自己的 FastAPI + Canvas 应用，Label Studio 协议只保留给旧流程集成。

**为什么不附带权重？**

DINOv3 的访问权限和上游模型许可需要使用者自行处理，文件也太大不适合放进 Git。
环境检查脚本会在评测前检查两个预期位置。

**为什么 VisA 的分数比 COCO 差？**

COCO 测的是 5 个粗粒度物体类别；VisA 是细小、细粒度的工业缺陷，通用 SAM 用点提示时
经常把整个宿主物体而不是缺陷分割出来。发布的失败案例图明确展示了这个边界。

## 贡献指南

欢迎提交 Pull Request。请保持改动范围清晰，不要提交数据集、权重、缓存、上传图片、日志或本地数据库。
按改动范围运行对应检查：

```powershell
python scripts/test_protocol.py
python scripts/check_feature_space_guard.py
python scripts/check_api_edges.py
python scripts/check_durability.py
python scripts/smoke_webtool.py
```

改动特征、检索、阈值、记忆、模型、分辨率或分块数时，还必须通过：

```powershell
python scripts/supervise.py --quick --tag <name> --compare
python scripts/supervise.py --tag <name> --compare
```

## 许可证

SnowLabel 使用 MIT License，详见 `LICENSE`。

## 致谢与引用

SnowLabel 基于 Meta 的 SAM/SAM2 和 DINOv3、FastAPI、Milvus、MinIO 以及 Label Studio ML 协议构建。
工业基准使用 Amazon 的 VisA 数据集（CC BY 4.0）：

```bibtex
@article{zou2022spot,
  title={SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation},
  author={Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar},
  journal={arXiv preprint arXiv:2207.14315},
  year={2022}
}
```
