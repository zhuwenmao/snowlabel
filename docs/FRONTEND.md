# 标注器前端交互与显示模型(`snowlabel/static/labeler.html`)

> 自建 Canvas 单页标注器,后端是 `web_api.py` 的 `/api/*`。**改 HTML 无需重启**
> (FastAPI 每请求从磁盘读 `labeler.html`),浏览器**硬刷新 Ctrl+Shift+R** 即可;
> **改 `web_api.py` 需重启服务器**(启动时 import)。mask 走 PNG,不走 RLE。

## 三种工具

| 工具 | 键 | 行为 |
|---|---|---|
| **point** 单点 | `p` | 左键=正点(＋)、右键=排除点(－),实时 SAM 出 draft |
| **box** 框选 | `b` | 拖拽外接框 → SAM 出 draft |
| **all** 全分割(everything) | `e` | 一次 SAM 全图分割 + 逐 mask 检索定类 |

切到 all 模式且本图还没有 auto mask 时会自动跑一次 `runSegmentAll()`。

## Everything 模式的交互(本次会话重点打磨)

`runSegmentAll()` 调 `/api/segment_all`(`max_masks:80`),对每个 mask 做检索预测:
- **空记忆冷启动**:保留 SAM 的无标签候选,只画中性轮廓、不画"?"标签;
  用户点选候选并指定第一个类别后即可入库。
- **已有记忆**:继续隐藏没有匹配类别的 silent 碎块,避免干扰。
- 底部提示区分"SAM 无候选""有候选但记忆未匹配""冷启动候选可选"三种状态。
- 重跑时 `masks=masks.filter(m=>m.source==="added")`:保留手动加的,只刷新 auto。

模式内的鼠标/键盘(`labeler.html` 的 click/contextmenu/keydown 监听):
- **点已识别的 mask** → `pickSetMaskAt()` 选中它,按数字键或点类别**改类**(改了才会入库)。
- **点空白处** → 进入补漏检:起一个新 draft,**完整单点流程**:
  - 继续左键加**正点**精修这个新 mask(`pts.push({label:1})` → `segmentDraft()`);
  - 右键加**负点**抑制(`contextmenu` 在 all 模式"正在画 draft 时"才生效:
    `tool==="point" || (tool==="all" && (pts.length||draft))`);
  - **Space** = `addDraft()`「完成此物体」:把当前 draft 收下,继续画下一个;
  - **C / Esc**:不在画 draft 时=重跑全分割;正在画时=取消当前 draft
    (`if(tool==="all" && !(pts.length||draft)){ runSegmentAll(); } else { clearDraft(); }`)。
- **Enter** = `commitImage()`「保存本图」:存进记忆并跳到下一张。

> 关键点:负点要生效,SAM 必须收**嵌套格式** `points=[pts], labels=[labs]`
> (后端 `models.SamSegmenter`);扁平格式会把每个点当独立物体,负点失效。

## 显示:细线轮廓(替代填充)

本次把半透明填充覆盖层换成**细线轮廓**(`drawContour`,labeler.html ~L275):
- 轮廓点来自**服务器** `web_api.mask_to_contour`(cv2 `findContours` + `approxPolyDP`
  简化),`/api/segment` 与 `/api/segment_all` 响应里带 `contour` 字段;前端兜底用
  `bboxContour(bbox)`(cv2 不可用或无轮廓时退化成矩形)。
- **绕质心向外扩 10%**:`k=1.10`,每点 `X=cx+(p[0]-cx)*k`。
- **按屏幕像素绘制**:画布按原图分辨率绘制再被 CSS 缩放,`dispScale()=cImg.width/显示宽度`。
  线宽 `1.6*dispScale()`(选中 3 倍)、标签字号 `12.5*dispScale()`、点半径 `6.5*dispScale()`,
  所以无论原图多大,屏幕上的线、字、点大小恒定;窗口 resize 时重画。
- **每类一色**:`strokeStyle=colorFor(label)`(预设色板命中即用,否则按类名 hash 成 HSL)。
- **标签位置 `tagPlace()`**:屏幕上 ≥56px 的物体,标签放在独占区域质心;更小的物体把标签
  挪到外接框上方(否则会盖住整个缺陷),< 28px 的微小物体再画一圈圆环 `drawRing()` 标出位置。
  草稿的标签贴在框顶,不压住用户刚点的点。标签互相重叠时 `drawTag()` 会上下挪开。
- **标签样式表达入库状态**:实心胶囊(类别色底)= 待保存(新增 / 新标 / 改过);
  深底描边 = 识别正确、无需保存。右侧「本图物体」顶部有图例。
- render 顺序:先描所有 mask 轮廓 → set 标签 → draft 轮廓 → 点 → 框选虚线框。
  ownership(`recomputeOwnership`)仍保留,只为算质心和入库时切互斥区,不再画填充。

## 入库语义:"完成此物体"(Space) vs "保存本图"(Enter)

只把**模型没做对的** mask 入库(labeler.html 注释 L159-162):
- `source==="added"`(你手动加的漏检) → 入库;
- `source==="auto"` 且 `label!==pred_label`(你改了类,说明模型分错) → 入库;
- `source==="auto"` 且没改 → 识别对了,**免入库**(不重复污染记忆)。

两个动作:
> 2026-10 改名:旧文案"加入本图 / 入库本图"两个都叫"…本图",用户容易以为 Space 也在存图。
> 现在 Space 叫**完成此物体**(完成对这一个实例的标注),Enter 叫**保存本图**。

- **完成此物体(Space / `addDraftBtn`)= `addDraft()`**:把当前 draft 收进本图的
  `masks[]`(`source:"added"`),**还没落库**,只是攒着。
- **保存本图(Enter / `commitImgBtn`)= `commitImage()`**:对每个 dirty mask 取其
  **互斥区** `ownedMaskPng(m)`(重叠裁掉,同一像素不会被两个标签存),POST
  `/api/commit_batch`。识别对、没动过的预测放进 `accepted`(**不存**,只回传给遥测
  统计真实接受率)。右侧「本图物体」只列已标类别的物体,未命名候选折叠成一行计数。

> 用户确认过的语义:**点对了直接 Enter 跳下一张,不会重复入库**——因为没 dirty
> mask,只记一条"全对确认"的遥测,记忆库不增长。这是有意的:记忆只长"教过的修正"。

## 示例数据集与推荐类别

- 注册表在 `snowlabel/sample_datasets.py`。`classes` 是按场景组织的**图片文件夹**
  (例如「缺件槽位」是一类图);`recommended` 才是图上要点选标注的**物体类别**
  (零件、空槽位、多余零件)。`/api/sample_datasets` 和 `/api/load_samples` 都返回 `recommended`。
- `featured=True` 的数据集(车辆、鸟类)在选择页置顶,带「首次推荐」标识;`level` 显示入门 / 进阶。
- 载入后 `setRecommended()` 把类别栏换成「推荐类别 + 记忆里已有的类别」,按 UID 存进
  `localStorage.snow_rec`,刷新页面不丢;「重新开始」会清掉。
- 推荐类别按顺序取色 `recColor(i)`(避开绿=自动、蓝=选中),同一数据集里各类颜色一定不同;
  其他类别仍按类名哈希取色。

## 与后端 API 的对应

| 前端动作 | 端点 | 关键返回 |
|---|---|---|
| `doDraft`(单点/框) | `POST /api/segment` | mask_png, bbox, label, tier, confidence, top_k, **contour** |
| `runSegmentAll` | `POST /api/segment_all` | masks[]:各含 mask_png/label/tier/**contour** |
| `commitImage` | `POST /api/commit_batch` | committed, memory_size;落 `data/telemetry/events.jsonl` |

遥测:`commit`/`commit_batch` 透传每个 mask 的 `pred_label/tier/confidence/source`,
`_commit_event` 记 `corrected = pred_label 存在 且 label!=pred_label`,
`scripts/session_report.py` 汇总真实接受/纠正/补充率。
