# SnowLabel Agent Guide

## 交互与接手原则
- 默认用中文回复用户。
- 这个仓库从 Claude Code 项目迁移而来；`CLAUDE.md` 仍是重要历史地图，但 Codex 接手时以本文件、源码、`snowlabel/config.py` 和最近文档为准。
- 接手前优先读：`docs/SESSION_2026-06-13.md`、`docs/DECISIONS.md`、`docs/FRONTEND.md`、`eval/PROTOCOL.md`、`results/supervise/FINDINGS.md`。
- 工作区可能有用户未提交改动，不要回滚、覆盖或“清理”非本任务修改。大权重、数据、缓存、上传图、日志和本地状态不要进 Git。

## 项目本质
SnowLabel 是渐进式自动标注工具：用户教过的 mask 会被编码进向量记忆，下一张相似图靠检索预填标签；模型不重训。当前主体验是 FastAPI + Canvas 标注器，而不是 Label Studio。

一个 FastAPI 进程 `snowlabel/server.py` 监听 9090，同时提供：
- `/`、`/label`：Canvas 标注 UI，前端在 `snowlabel/static/labeler.html`，后端 API 在 `snowlabel/web_api.py` 的 `/api/*`。
- `/inspector` 和 `/memory/*`：记忆库面板与管理接口。
- `/predict`、`/webhook`、`/preannotate`：保留的 Label Studio ML 协议和可选 SAM3 概念召回。

核心链路：`SnowPredictor` 单例串起 `models.py` 里的 SAM/DINO、`features.py` 的 mask 特征组、`predictor.py` 的两段式检索定类，以及 `memory.py` / `memory_milvus.py` 后端。

## 目录结构
- `snowlabel/`：Python 包与服务端核心。
- `snowlabel/static/labeler.html`、`inspector.html`：浏览器 UI。改 `labeler.html` 后通常只需浏览器硬刷新；改 `web_api.py` / `server.py` 需要重启服务。
- `snowlabel/imgcache/`：三库图片缓存，MinIO 存图片，MySQL/MariaDB 存 mask 元数据，Milvus 存 `v_whole` 向量。
- `eval/`：监督协议、指标、门槛和报告逻辑。
- `scripts/`：监督、诊断、冒烟和数据准备脚本。
- `.agents/skills/start-snowlabel/`：Codex 用的本项目启动 skill，替代迁移前 `.claude/skills/`。
- `demo_images/`：内置 demo 图。`data/`、`weights/`、`results/`、`*.pt` 等本地资产不要提交。

## 环境与启动
优先使用用户现有 conda 环境：

```powershell
%USERPROFILE%\miniconda3\envs\snowlabel\python.exe
```

不要假设裸 `python` 一定指向该环境。依赖安装：

```powershell
pip install -r requirements.txt
pip install -e .[dev]
```

本机当前生产配置：

```powershell
$env:SNOWLABEL_DINO="weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_MEMORY_BACKEND="milvus"
$env:SNOWLABEL_RES="512"; $env:SNOWLABEL_PARTS_GRID="4"
python -m snowlabel.server
```

公网 Demo / 电脑重启后恢复服务时，用 `.agents/skills/start-snowlabel` 的流程或直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File start_public_demo.ps1
```

它会启动 9090、MariaDB、MinIO 和 Cloudflare 命名隧道，固定公网地址是 `https://snowlabel.zhuwenmao.com`。`-Fresh` 会清空公网访客上传图和 `data\memory_public`，只有用户明确要求从零重置时才用。

`start_demo.ps1` 是旧 Label Studio 三窗口 demo 入口，里面仍带历史阈值和旧分辨率语义；不要把它当当前生产启动方式，除非先修正脚本。

## 关键实现事实
- SAM 默认保持 `sam2.1_t.pt` tiny。不要为了“mask 更细”擅自升 small/base；监督基准证明更大 SAM 会把近类推过 auto 阈值并污染记忆。
- DINOv3-L 生产配置必须显式 `SNOWLABEL_RES=512`，因为 patch16 下 512 可整除。默认 `RES=224` 是给 DINOv2 ViT-B/14 兜底。
- `features.py` 产出 `MaskFeatures`：`v_whole [D]`、`v_parts [parts_grid^2,D]`、`geom [4]`。`feature_space_id` 会把 DINO、分辨率、parts_grid、crop pad 和 dim 绑定进指纹。
- `predictor.py` 使用两段式检索：先按 `v_whole` 召回 top `rerank_n`，再按 `0.55*cos_whole + 0.35*parts_MaxSim + 0.10*geom_sim` 精排，并用 open-set、margin、support 分成 auto / suggest / silent。
- `confidence` 是 winner 的绝对混合分，不是类间 share。当前默认阈值来自 `config.py`：`SNOWLABEL_CONF_AUTO=0.68`、`SNOWLABEL_CONF_SUGGEST=0.55`、`SNOWLABEL_MARGIN_AUTO=0.08`。
- 绝不要带历史阈值 env，例如 `0.85/0.75/0.61` 或 `0.60/0.58/0.41`。起服务和跑分都用 `config.py` 默认，除非正在有意识地做阈值实验。
- Milvus 记忆后端是 RAM 镜像 + 写穿透：读全部走内存，Milvus 负责持久化。公网多租户 demo 必须用 numpy 后端，因为 Milvus schema 目前不持久化 `user_id`，不能做访客隔离。
- Everything 模式隐藏 silent 碎块；点空白可补漏检；右键负点只有 SAM 收嵌套 `points=[pts], labels=[labs]` 时才生效。
- 入库语义：`Space` 只是“加入本图”；`Enter` 只持久化手动新增或被用户改类的 mask。接受未修改的 auto 预测只写 telemetry，不重复写记忆。

## 测试与质量关卡
没有统一 pytest 主套件，仓库以脚本检查为主。根据改动范围选择：

```powershell
python scripts/test_protocol.py
python scripts/check_feature_space_guard.py
python scripts/check_api_edges.py
python scripts/check_durability.py
python scripts/smoke_webtool.py
python scripts/session_report.py
```

动到以下内容时，合入前必须跑监督程序并通过：`snowlabel/features.py`、`predictor.py`、`config.py` 阈值或模型配置、`memory*.py` 检索路径、SAM/DINO/分辨率/parts_grid 切换。跑之前停掉 9090，避免 GPU 放不下两套模型：

```powershell
python scripts/supervise.py --quick --tag <name> --compare
python scripts/supervise.py --tag <name> --compare
```

退出码：`0` 通过，`1` 硬门槛失败，`2` 相对 baseline 回归。阈值扫描用 `--sweep <run_dir>`，升 baseline 用 `--promote <run_dir>`。

## 代码风格
- Python 3.10+，4 空格缩进，必要处加类型提示，命名使用描述性 snake_case。
- 遵守现有模块边界：FastAPI 路由显式，ML 管线集中在 models/features/predictor/memory。
- `ruff` 是开发依赖；改 Python 时尽量对触及文件跑 lint。
- UI 改动要和 `docs/FRONTEND.md` 的交互模型一致，避免把标注器改成说明页或营销页。

## 安全与配置
- `snowlabel/config.py` 默认值是事实来源；README/历史文档可能落后。
- 不提交 `.env`、`.ls_state.json`、`weights/`、`data/`、`results/`、模型 checkpoint、访客上传、生成日志或本地数据库。
- 清库、`-Fresh`、删除访客数据、切换特征空间后重标，都属于有破坏性的动作；除非用户明确要求，不要主动执行。

## 提交与 PR
提交保持小而清晰，历史风格偏简洁，可用 `Fix:` 做定点修复。PR 说明要写用户可见变化、跑过的命令、模型/配置影响、相关 issue/docs，UI 改动附截图或短视频。
