# 关键技术决策与理由(ADR)

> CLAUDE.md 的"关键坑"记的是**结论**;这里记**为什么**和**证据**,方便日后回看或
> 推翻。每条:决策 / 理由 / 证据 / 影响。门槛与数字以 `config.py`、`eval/gates.json`、
> 最近一次 `results/supervise/.../metrics.json` 为准(文档可能落后于代码)。

## D1. 分辨率 512 + parts_grid 4(不是 224 + 2×2)

- **决策**:跑 DINOv3 时 `SNOWLABEL_RES=512`、`SNOWLABEL_PARTS_GRID=4`。
- **理由**:用户嫌 2×2 分块保留的空间结构太少;512÷16=32 的 patch 网格才够细到把
  v_parts 切成 4×4 软池化而每格仍有足够 patch。
- **证据/坑**:`RES=512` **只对 patch16 的 DINOv3 成立**;config 默认仍 224 是给
  patch14 的 DINOv2 兜底(512÷14 除不尽会抛 ValueError)。所以 512 必须**显式 env**,
  不写进默认值。
- **内存**:RAM 镜像每条约 68KB(v_whole 1024 + v_parts 16×1024 + geom),demo 规模
  (~万条内)不会顶爆——回答了用户"会不会把内存顶爆"。>10 万条再切 HNSW。

## D2. `confidence` = 绝对混合分(不是类间 share)+ 阈值双种子校准

- **决策**:`conf` 取 winner 的**绝对 rerank 混合分**
  `0.55·cos_whole + 0.35·parts_MaxSim + 0.10·geom_sim`;默认阈值
  `conf_auto=0.68 / conf_suggest=0.55 / margin_auto=0.08`。
- **理由**:旧的"类间 share"语义随类数变化、不可比;绝对分对类数不变,可跨实验对照。
- **证据**:`supervise.py` 在**两个**开集种子上扫阈值,取 strict 池(两次校准跑都
  零 false-auto);绑定难点是近类混淆(cup→bottle 在 ~0.68 附近)。
- **影响(必须记住)**:**历史 env `SNOWLABEL_CONF_AUTO=0.85/0.75/0.61`、
  `SUGGEST=0.60/0.58/0.41` 全部作废**——那是旧 share 语义或旧校准,带上会让一切失准。
  跑分、起服务都别带这些 env,用 `config.py` 默认。README 里的 0.85/0.60 已过时。

## D3. SAM 保持 tiny,不升级到 base ★ 最能体现"真正好用"

- **决策**:点击路径继续用 `sam2.1_t.pt`(tiny),**拒绝**升 small/base。
- **理由**:直觉是"更大的 SAM = 更好的 mask = 更好的识别",但监督基准否决了它。
- **证据**:base 的 mIoU 确实更高(~0.806),但更精细的 mask 让 **truck 的特征比真正的
  car 还更像 car**(truck 的 cw=0.783,高过真 car),于是 truck→car 越过 auto 阈值
  造成**数据污染**。没有任何余弦地板能分开"真正易混"的类(car/truck/bus)——提高
  开集阈值会把真 car 也一起拒掉。
- **影响**:这是"数据好看 vs 真正好用"的活教材:单看 mIoU 选 base 会让最终用户
  收到更多贴错的 auto 标签。**监督程序正确抓住了这点**(`results/supervise/` 里
  `sam-tiny` / `sam-base` 系列跑可对照),所以保留 tiny。详见
  [`../results/supervise/FINDINGS.md`](../results/supervise/FINDINGS.md)。

## D4. 特征空间指纹守卫(`features.feature_space_id`)

- **决策**:换 模型 / 分辨率 / 分块数 / crop-pad 时,两后端加载非空旧库且指纹不符
  **直接报错**,不静默加载。
- **理由**:这些一变,旧向量的余弦"还能算但已无意义"(不同特征空间),会悄悄毁掉
  检索质量而不报错——最坏的那种 bug。
- **实现**:指纹 = `{dino名}|res{R}|parts{P}|dim{D}|pad{CROP_PAD}|v1`;numpy 后端写
  `index.json`、milvus 后端写 sidecar `feature_space.json`;不符抛 RuntimeError。
  自包含验证:`scripts/check_feature_space_guard.py`(临时目录、可反复跑)。
- **影响**:改了这些就得**清库重标**——这是保护不是 bug。

## D5. Milvus 是 RAM 镜像 + 写穿透落盘(读全走内存)

- **决策**:milvus 后端启动时 `_load_cache` 拉全量进内存,`query` 用 numpy 暴力余弦;
  Milvus 只负责写穿透持久化。
- **理由**:milvus-lite 在 Windows 上 ① 写后 ANN 可见有数秒延迟,② `flush()` 会
  WinError 183 损坏数据。交互式标注必须"读己之所写"(刚入库的下一张就要能检索到),
  milvus-lite 满足不了。
- **叙事口径**:对外讲成"交互需要读己之所写、milvus-lite 不满足 → RAM 镜像 + Milvus
  持久化,>10 万条切回 HNSW",而不是"Milvus 装饰品"。
- **影响**:持久化由 `check_durability.py` 验证(重启不丢);记忆库随标注增长,
  服务器重启后内存镜像从落盘数据重建。

## D6. 监督程序的设计原则(北极星 = 人工省力比)

- **真正好用的量化**:北极星是**人工省力比**(effort_saving),对照组是"有 SAM 但
  无记忆"的纯手动——只把记忆系统省的力算进来,不蹭 SAM 的功劳(诚实归因)。
- **错误代价不对称**:auto 贴错 = 污染数据集,比沉默更糟 → auto 精度设**硬门槛**
  (后半程 ≥0.97)+ 单列 poison_rate,**绝不许靠调松阈值刷覆盖率**。
- **开集诚实**:没学过的类必须沉默 → openset_false_auto 硬门槛。
- **回归判定噪声感知**:auto 精度的回归红线取 `max(配置pp, 100/n_late_auto)`——
  小于一个样本粒度的下降不算回归;延迟回归用 p50(p95 抖动大,只做绝对门槛)。
- **离线扫阈值零 GPU**:教学由 GT 驱动,预测流与阈值无关 → 可离线重放(`--sweep`)
  在已跑过的预测上重算分档,免重跑模型。
- 完整协议:[`../eval/PROTOCOL.md`](../eval/PROTOCOL.md);门槛与成本表:
  [`../eval/gates.json`](../eval/gates.json)。
