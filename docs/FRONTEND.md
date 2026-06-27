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

`runSegmentAll()` 调 `/api/segment_all`(`max_masks:20`),对每个 mask 做检索预测:
- **隐藏没认出的碎块**:`if(!m.label){ hidden++; continue; }`——silent 档(没学过)
  不显示,避免满屏"?"干扰。提示里报"隐藏 N 个没认出的碎块"。
- 重跑时 `masks=masks.filter(m=>m.source==="added")`:保留手动加的,只刷新 auto。

模式内的鼠标/键盘(`labeler.html` 的 click/contextmenu/keydown 监听):
- **点已识别的 mask** → `pickSetMaskAt()` 选中它,按数字键或点类别**改类**(改了才会入库)。
- **点空白处** → 进入补漏检:起一个新 draft,**完整单点流程**:
  - 继续左键加**正点**精修这个新 mask(`pts.push({label:1})` → `segmentDraft()`);
  - 右键加**负点**抑制(`contextmenu` 在 all 模式"正在画 draft 时"才生效:
    `tool==="point" || (tool==="all" && (pts.length||draft))`);
  - **Space** = `addDraft()` 把当前 draft 收下,继续画下一个;
  - **C / Esc**:不在画 draft 时=重跑全分割;正在画时=取消当前 draft
    (`if(tool==="all" && !(pts.length||draft)){ runSegmentAll(); } else { clearDraft(); }`)。
- **Enter** = `commitImage()` 入库本图。

> 关键点:负点要生效,SAM 必须收**嵌套格式** `points=[pts], labels=[labs]`
> (后端 `models.SamSegmenter`);扁平格式会把每个点当独立物体,负点失效。

## 显示:细线轮廓(替代填充)

本次把半透明填充覆盖层换成**细线轮廓**(`drawContour`,labeler.html ~L275):
- 轮廓点来自**服务器** `web_api.mask_to_contour`(cv2 `findContours` + `approxPolyDP`
  简化),`/api/segment` 与 `/api/segment_all` 响应里带 `contour` 字段;前端兜底用
  `bboxContour(bbox)`(cv2 不可用或无轮廓时退化成矩形)。
- **绕质心向外扩 10%**:`k=1.10`,每点 `X=cx+(p[0]-cx)*k`。
- **很细的线**:`lineWidth=Math.max(1.5, cImg.width/720)*(sel?2.2:1)`——选中的略粗。
- **每类一色**:`strokeStyle=colorFor(label)`(预设色板命中即用,否则按类名 hash 成 HSL)。
- **label 画在 mask 质心**、**文字色=轮廓色**(底下深色小药丸只为可读,不影响主色)。
- render 顺序:先描所有 mask 轮廓 → set 标签 → draft 轮廓 → 点 → 框选虚线框。
  ownership(`recomputeOwnership`)仍保留,只为算质心和入库时切互斥区,不再画填充。

## 入库语义:"加入本图" vs "入库本图"

只把**模型没做对的** mask 入库(labeler.html 注释 L159-162):
- `source==="added"`(你手动加的漏检) → 入库;
- `source==="auto"` 且 `label!==pred_label`(你改了类,说明模型分错) → 入库;
- `source==="auto"` 且没改 → 识别对了,**免入库**(不重复污染记忆)。

两个动作:
- **加入本图(Space / `addDraftBtn`)= `addDraft()`**:把当前 draft 收进本图的
  `masks[]`(`source:"added"`),**还没落库**,只是攒着。
- **入库本图(Enter / `commitImgBtn`)= `commitImage()`**:对每个 dirty mask 取其
  **互斥区** `ownedMaskPng(m)`(重叠裁掉,同一像素不会被两个标签存),POST
  `/api/commit_batch`。识别对、没动过的预测放进 `accepted`(**不存**,只回传给遥测
  统计真实接受率)。提示区 `queue` 实时显示"本图待入库"的类别 chip。

> 用户确认过的语义:**点对了直接 Enter 跳下一张,不会重复入库**——因为没 dirty
> mask,只记一条"全对确认"的遥测,记忆库不增长。这是有意的:记忆只长"教过的修正"。

## 与后端 API 的对应

| 前端动作 | 端点 | 关键返回 |
|---|---|---|
| `doDraft`(单点/框) | `POST /api/segment` | mask_png, bbox, label, tier, confidence, top_k, **contour** |
| `runSegmentAll` | `POST /api/segment_all` | masks[]:各含 mask_png/label/tier/**contour** |
| `commitImage` | `POST /api/commit_batch` | committed, memory_size;落 `data/telemetry/events.jsonl` |

遥测:`commit`/`commit_batch` 透传每个 mask 的 `pred_label/tier/confidence/source`,
`_commit_event` 记 `corrected = pred_label 存在 且 label!=pred_label`,
`scripts/session_report.py` 汇总真实接受/纠正/补充率。
