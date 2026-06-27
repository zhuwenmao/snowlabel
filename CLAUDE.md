# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 交互规则

- **回复一律使用中文。**

## 项目本质

SnowLabel(工作目录名 LabelAnything):**Milvus 驱动的渐进式自动标注**。卖点是
"越标越准"——模型从不重训,只是把用户教过的 mask 存进向量记忆,下一张图靠检索
预测。简历/作品集项目,优先 demo 效果与可信数字,而非生产级鲁棒性。

## 延伸文档(`docs/`,深入细节看这里)

本文件给"地图";下面四篇给"细节",改相关部分前先读对应篇:

- [`docs/SESSION_2026-06-13.md`](docs/SESSION_2026-06-13.md) — **最近一次长会话的成果
  与状态交接**:用户依次的诉求、做了什么、当前配置/记忆库快照、待办。接手先看这篇。
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — **关键技术决策的"为什么"**:512+parts4、
  conf 绝对分语义+阈值校准、SAM 为何保持 tiny、指纹守卫、RAM 镜像、监督程序原则。
- [`docs/FRONTEND.md`](docs/FRONTEND.md) — **标注器前端交互与显示**(labeler.html):
  三种工具、everything 模式补漏检、细线轮廓显示、"加入本图/入库本图"语义、API 对应。
- [`eval/PROTOCOL.md`](eval/PROTOCOL.md) 与
  [`results/supervise/FINDINGS.md`](results/supervise/FINDINGS.md) — 监督协议与首轮结论。

## 环境与常用命令

- Python 解释器(conda env `snowlabel`,torch 2.5.1+cu121 / RTX 4060 8GB):
  `%USERPROFILE%\miniconda3\envs\snowlabel\python.exe`。直接用绝对路径调用,
  别假设 `python` 在 PATH 里指向它。
- 装依赖:`pip install -r requirements.txt`
- **启动服务器(当前正确的生产配置)**:
  ```powershell
  $env:SNOWLABEL_DINO="weights\facebookdinov3-vitl16-pretrain-lvd1689m"
  $env:SNOWLABEL_MEMORY_BACKEND="milvus"
  $env:SNOWLABEL_RES="512"; $env:SNOWLABEL_PARTS_GRID="4"
  python -m snowlabel.server      # → http://localhost:9090/ (标注 UI), /inspector (记忆面板)
  ```
  GPU 放不下两套模型,后台跑服务器时用 `Start-Process ... -RedirectStandardOutput/Error`
  写日志到 `data/cache/`,别用前台阻塞。

### 监督程序 —— 改动后的强制关卡

`scripts/supervise.py` 是唯一评估入口,回答"这次改动对真实用户是变好还是变坏"。
**动了 `snowlabel/features.py`、`predictor.py`、`config.py` 阈值、`memory*.py` 检索路径、
或换 SAM/DINO/分辨率,合入前必须跑并通过。** 跑之前先停掉 9090 服务器。

```powershell
python scripts/supervise.py --quick --tag <名字> --compare   # 迭代中(~35s)
python scripts/supervise.py --tag <名字> --compare            # 合入前正式(~75s)
python scripts/supervise.py --sweep <run_dir>                 # 零GPU:离线重放扫阈值
python scripts/supervise.py --promote <run_dir>              # 设为新 baseline
python scripts/supervise.py --classes "car,truck,bus" --n-per 20   # 自定义难类集
```
退出码:`0` 过 / `1` 硬门槛失败 / `2` 相对 baseline 回归。标准与成本表版本化在
`eval/gates.json`,完整协议见 `eval/PROTOCOL.md`,首轮结论见 `results/supervise/FINDINGS.md`。

### 检查脚本(`scripts/check_*.py`,大多需服务器在线)

`check_featuregroup`(端到端冒烟)、`check_api_edges`(API 边界)、`check_durability`
(Milvus 重启持久化)、`check_feature_space_guard`(指纹守卫,自包含)、`check_zh_label`
(中文往返)、`test_protocol`(LS 协议,无需 GPU)、`smoke_webtool`(接线,无需模型)、
`diag_coldstart`(tier 冷启动曲线)。**没有 pytest**;这些就是测试,各自 `python scripts\xxx.py`。

## 架构大图

一个 FastAPI 进程(`snowlabel/server.py`,端口 9090)同时提供:① 旧的 Label Studio
ML 协议(`/predict` `/webhook` `/preannotate`,保留不删);② 自建 Canvas 单页标注器
(`/`、`/label`,前端 `static/labeler.html`,后端 `web_api.py` 的 `/api/*`,mask 走 PNG
不走 RLE);③ 记忆面板(`/inspector` + `/memory/*`)。`SnowPredictor`(`predictor.py`)
是全局单例,串起 SAM + DINO + 记忆,一个进程一份。

**核心预测流水线**(读懂它要跨 `features.py` + `predictor.py` + `memory*.py`):

1. **SAM 出 mask**(`models.SamSegmenter`,ultralytics SAM2.1)。点提示用**嵌套格式**
   `points=[pts], labels=[labs]` 才能让负点(右键抑制)生效;扁平格式会把每个点当独立物体。
2. **crop-encode 提特征组**(`features.MaskFeatureExtractor`):按 mask 外接框 crop +
   letterbox 补正方形(不拉伸)→ resize 到 `SNOWLABEL_RES` → DINOv3-L 编码 →
   **软覆盖加权池化**(mask 缩到 patch 网格当权重)。产出一个 `MaskFeatures` 组:
   - `v_whole [1024]` 整体语义,ANN 检索键;
   - `v_parts [P²,1024]` 把 patch 网格切成 `parts_grid×parts_grid` 软池化,保留粗空间结构;
   - `geom [4]` 面积/长宽/extent/solidity。
   `features_batch` 把一张图所有 mask 的 crop 批量过一次 DINO(everything 模式靠它快)。
3. **两段式检索定类**(`predictor._predict_from_features`):
   stage-1 用 `v_whole` 余弦召回 top-`rerank_n`;**开集闸门**:最近邻余弦 < `open_set_min_sim`
   → silent(没学过的就闭嘴)。stage-2 逐候选精排
   `0.55·cos_whole + 0.35·parts_MaxSim + 0.10·geom_sim`(权重是 `predictor.py` 模块常量),
   按类取最高分,winner/runner-up 算 margin。
4. **分档**(`_tier_group`):`conf`(= winner 绝对混合分)+ margin + 该类样本支持度 →
   auto / suggest / silent。`cw`(原始最近邻余弦)随 `Prediction` 透出,供诊断/开集。

**记忆后端**(`memory.py` numpy / `memory_milvus.py`,接口一致,`get_memory` 按
`SNOWLABEL_MEMORY_BACKEND` 选)。Milvus 后端是 **RAM 镜像**架构:milvus-lite 在 Windows
上写后 ANN 可见有数秒延迟、`flush()` 还会 WinError 183 损坏数据——所以**所有读走内存**
(启动时 `_load_cache` 拉全量,`query` 是 numpy 暴力余弦),Milvus 只做写穿透落盘。
"Milvus 驱动"在叙事里要主动讲成"交互需要读己之所写、milvus-lite 不满足→RAM 镜像 +
Milvus 持久化,>10万条切回 HNSW"。

**监督闭环**(`eval/`):`protocol.py` 无头复跑真实标注循环(Phase A 雪球 5类×25,Phase B
开集探针),记录 IoU/latency/margin/cw/错误归因;`effort.py` 是人工省力比成本模型(北极星
指标);`report.py` 算指标+门槛+回归对比;`sweep.py` 离线重放扫阈值。

## 关键坑(本项目特有,踩过的)

- **`confidence` 是绝对混合分,不是类间 share。** 当前默认阈值
  `conf_auto=0.68 / conf_suggest=0.55 / margin_auto=0.08`(`config.py`,2026-06-13 双种子校准)。
  **绝不能再传历史 env `SNOWLABEL_CONF_AUTO=0.85/0.75/0.61` 或 `SUGGEST=0.60/0.58/0.41`**
  ——那些是旧 share 语义或旧校准,会让一切失准。跑分/起服务都别带这些 env,用 config 默认。
  注意 README 里写的 0.85/0.60 已过时,**`config.py` 默认值才是真相**。
- **`SNOWLABEL_RES=512` 只对 patch16 的 DINOv3 成立**(512÷16=32)。config 默认 `RES=224`
  是为了 patch14 的 DINOv2 兜底不报错(512÷14 除不尽会抛 ValueError);跑 DINOv3 必须显式带 512。
- **特征空间指纹守卫**:换 模型/分辨率/分块数/crop-pad 会让旧向量的余弦"还能算但没意义"。
  `features.feature_space_id` 是身份证,两后端加载非空旧库且指纹不符会**直接报错**
  (numpy 写 `index.json`、milvus 写 sidecar)。改了这些就得清库重标,不是 bug 是保护。
- **改完管线必跑 supervise**;`gates.json` 的回归判定对 auto 精度是噪声感知的(小于一个
  样本量的下降不算回归),p95 延迟只做绝对门槛不做相对比较(抖动大)。
- 文档双语(`README.md`/`README_CN.md` 等)且可能落后于代码,以代码与 `config.py` 为准。
  发布到 GitHub 前务必保持 `.gitignore` 完整(`weights/`、`data/`、`*.pt`、`.ls_state.json`、
  `.claude/` 都不进仓库)。
