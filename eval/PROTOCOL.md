# SnowLabel 监督协议

监督程序的唯一入口是 `scripts/supervise.py`。它回答一个问题:**这次改动之后,
工具对真实用户是变好了还是变坏了?** 所有评价标准(门槛、成本表、回归红线、
协议参数)都版本化在 [gates.json](gates.json) 里——改门槛必须改这个文件,留下痕迹。

## 何时必须跑

任何动到下列内容的改动,合入前必须跑一次并通过:

- `snowlabel/features.py`(特征提取/分块/crop 逻辑)
- `snowlabel/predictor.py`(精排权重、置信度语义、tier 闸门)
- `snowlabel/config.py` 中的阈值默认值(conf/margin/open_set/support/分辨率/分块数)
- `snowlabel/memory*.py` 的检索路径
- 换 SAM / DINO 模型或分辨率

```powershell
# 迭代中(~35s):
$env:SNOWLABEL_DINO="weights\facebookdinov3-vitl16-pretrain-lvd1689m"
$env:SNOWLABEL_RES="512"; $env:SNOWLABEL_PARTS_GRID="4"
python scripts\supervise.py --quick --tag 改动名 --compare

# 合入前(~75s,正式记录):
python scripts\supervise.py --tag 改动名 --compare
```

退出码:`0` 通过 / `1` 硬门槛失败 / `2` 相对 baseline 回归。
⚠️ 跑之前停掉 9090 服务器(GPU 放不下两套模型);跑分时**不要**带过期的
`SNOWLABEL_CONF_AUTO/SUGGEST` env(2026-06-12 起 conf 是绝对混合分语义;
2026-06-13 起默认 0.68/0.55/margin 0.08,双种子扫描校准,在 config.py)。

其他子命令:
- `--sweep <run_dir>`:对已存跑分离线重放 tier 闸门,扫 conf/margin 组合
  (零 GPU;教学是 GT 驱动的,预测流与阈值无关,所以重放有效)。
  阈值必须取 STRICT 解(校准跑零误放)并在**另一个种子**上验证后才能进 config。
- `--classes "car,truck,bus,cup,bowl"`:自定义类集(走全量标注+本地图过滤),
  用于难类/细粒度专项,**不要**与标准 baseline --compare。

## 怎么读 scorecard

| 指标 | 含义 | 为什么是它 |
|---|---|---|
| auto_precision_late(硬) | 后半程 auto 档、且 mask 有效(IoU≥0.5)时的正确率 | auto 错标=污染数据集,比沉默更糟 |
| poison_rate(硬) | 全程"mask 有效但 auto 贴错"占比 | 数据集质量底线 |
| openset_false_auto(硬) | 50 个未见类被 auto 的比率 | 没学过必须闭嘴 |
| effort_saving_late | 后半程相对"有SAM无记忆"纯手动的省力比 | **北极星**:真实价值就是省人工 |
| latency_p95_ms | 单击响应 p95 | 流畅是好用的前提 |
| late_auto_coverage | 后半程 auto 覆盖率(**无门槛**) | 防止为刷覆盖率调松阈值 |
| mask_miou / mask_fail_rate | SAM 切割质量(预警) | 切割差时,优化检索是无用功 |

成本模型(见 gates.json `effort_costs`):对照组 manual=2(点击+选类);
auto对=0,auto错=4,suggest对=1,suggest错=2,silent=2,**mask 切错(IoU<0.5)=3
(修 mask 覆盖一切 tier 收益)**。省的力只归功于记忆系统,不蹭 SAM 的。

## 协议设计要点(为什么可信)

- **教学仿真真人**:SAM mask 好(IoU≥0.5)就入它;切烂了入 GT mask(用户修过
  才会提交)——`taught` 列记录 sam/gt_fix。烂 mask 直接入库会污染评估记忆。
- **开集探针排除 Phase A 用过的图**:同图同物 crop 检索相似度≈1.0,那是泄漏
  不是开集能力。
- **点击点取 GT mask 内最靠质心的像素**,不是 bbox 中心(环形/人形物体的
  bbox 中心常落在背景上)。
- **metrics.json 必带完整配置快照**:不带配置的跑分没有可比性。

## baseline 管理

- 当前 baseline:`results/supervise/baseline/` = 2026-06-12 `calibrated-068`
  (阈值双种子校准后,全量 PASS:effort_saving_late=**0.667**,auto_prec=1.0,
  auto 覆盖 0.63,开集零误放)。
- 一次**有意的**改进通过全量跑后:`python scripts\supervise.py --promote <run_dir>`。
- 回归红线(gates.json `regression`):auto 精度降>1pp / 省力比降>5pp /
  **p50** 延迟升>50% → exit 2,不许合入。(p95 在 ~175 样本上受 GPU 状态扰动,
  2026-06-13 起只做绝对门槛不做相对比较——曾产生过假回归。)
- 趋势看 `results/supervise/history.csv`(每跑一行)。

## 真实使用遥测(最终裁判)

基准是仿真;真人用着顺不顺看遥测:网页工具每次入库会把"接受/纠正/手动加"
事件写进 `data/telemetry/events.jsonl`(识别对的不入库但**会上报 accept**;
全对的图按 Enter 记一条全对确认)。汇总:

```powershell
python scripts\session_report.py          # 真实接受率/纠正率/最常错的类对
```

基准分数涨而真实纠正率不降 → 在优化指标不是优化体验,回头检查协议假设。

## 已知缺口(诚实清单)

- 基准只有 5 个粗粒度 COCO 类:16 分块 vs 4 分块在这上面测不出差异(2026-06-13
  在难类集 car/truck/bus/cup/bowl 上实测逐类召回完全一致)。
- auto_precision 后半程样本量 ~40,1 个错≈2.5pp。回归判定已改噪声感知(小于"一个
  样本的量"的下降不算回归);连续两跑都低才算真失败。
- SAM2.1-tiny 的 mask_fail_rate≈15% 是当前省力比最大单一瓶颈(WARN)。但升 base
  (mIoU 0.806、失败率 11%)会把没学过的 truck 顶成 car 自动入库(开集误auto 0.04,
  seed23 FAIL)——更好的 mask 反而牺牲开集安全。**保持 tiny 为默认**,base 仅
  `SNOWLABEL_SAM=sam2.1_b.pt` opt-in。详见 `results/supervise/FINDINGS.md`。
- 近类混淆(truck/car cw=0.783、cup/bottle)无法用任何 cw/conf 阈值分开——卡车在
  DINO 空间里就是像车。正解是把该类也标进记忆,不是调阈值。
