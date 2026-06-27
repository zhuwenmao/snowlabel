# SnowLabel —— 演示与录屏指南

录制 **点击 → 掩码 → 雪球 → 自动标注** 演示所需的一切。
后端已完整验证(见*后端已验证*一节);本指南聚焦需要人工 + 录屏工具的浏览器步骤。

[English](DEMO_GUIDE.md) | 中文

## 1. 启动所有服务

```powershell
cd <repo-root>
powershell -ExecutionPolicy Bypass -File start_demo.ps1
```

> `cd` 很关键——脚本在项目目录里,不在你的用户主目录。或者用完整路径运行:
> `powershell -ExecutionPolicy Bypass -File <repo-root>\start_demo.ps1`

会打开三个窗口 —— ML 后端(:9090)、图片服务(:8081)、Label Studio(:8080)。等约 40 秒,直到后端窗口打印 *Application startup complete*(它在加载 DINOv3-L)。检查一下:

```powershell
irm http://localhost:9090/stats     # 显示模型、后端、阈值
```

## 2. 打开 Label Studio

浏览器 → http://localhost:8080 → 登录 `demo@snowlabel.ai` / `snowlabel123` → 打开项目 **SnowLabel Demo**。20 个任务按类分组(cat×4、dog×4、……),所以标完一类的 4 张就能看到完整雪球。

> 录屏时让 Label Studio 窗口保持在**前台** —— 录屏工具(以及 CDP 截图接口)只能捕获可见窗口。

## 3. 录制雪球(最关键的镜头)

连续标注 4 张猫:

1. 点击 **Label All Tasks**(或打开 task 1)。
2. 打开左下角的 **Auto-Annotation** 开关。
3. 选中 **KeyPoint** 工具 —— 它是 smart 触发器(`smart="true"`)。
4. **在物体上点击。** SAM 分割出它,雪球记忆预测类别:
   - 猫 #1 → **silent**(记忆为空)—— 自己选 `cat` → **Submit**
   - 猫 #2–3 → **suggest**(支持度低)—— 确认 `cat` → **Submit**
   - 猫 #4 → **auto**(支持度 ≥ 3)—— 类别已预填 `cat` → 直接 **Submit**
5. 这个 **silent → suggest → auto** 的演进*就是*雪球。把它录下来。
   右侧 Regions 面板和 Choices 单选框会显示预测类别 + 置信度。

(想要更长的片段就接着标 dog/bird/……)

## 4. 展示 SAM 3 概念批量召回

一旦记忆认识了"cat",一次调用就能召回一张图里的**每一只**猫:

```powershell
$b = @{ tasks=@(@{data=@{image="http://localhost:8081/<某张猫图>.jpg"}}); labels=@("cat"); score_threshold=0.4 } | ConvertTo-Json -Depth 6
irm http://localhost:9090/preannotate -Method Post -Body $b -ContentType "application/json"
```

它会为每只检测到的猫返回一个 brush 掩码(meta 里的 `per_label_counts`)。

## 5. 展示 Memory Inspector

浏览器 → http://localhost:9090/inspector —— 实时查看雪球学到了什么:类分布柱状图 + 每一条记忆条目。边标注边刷新,看它增长。

## 后端已验证(无需浏览器)

| 内容 | 方式 | 结果 |
|------|------|------|
| 端到端雪球,**真实 SAM 掩码** | `scripts/demo_snowball_e2e.py` | 整体 86.0%、后半段 93.4%、**auto 档 95% 覆盖 80% 点击** —— `results/demo_e2e/snowball_e2e.png` |
| LS 协议层 | `scripts/test_protocol.py` | 提示解析 + 结果转换全部通过 |
| SAM 3 批量召回 | `scripts/sam3_batch_demo.py` | 5 个类全部验证 |

## 故障排查

- **点击只落了一个 keypoint,没出掩码** —— 确认 Auto-Annotation 已开,且选中的是 KeyPoint(smart)工具;后端窗口应打印 `Interactive predict: point`,而不是 `Auto predict (no prompt)`。
- **整体很慢 / 一直转圈** —— 后端被每个 task 的自动预测打满了。在提供的 LS 配置里它已关闭(`evaluate_predictions_automatically=false`);若又出现,到 LS 项目设置里把该项关掉即可。
- **录屏/截图是空白** —— LS 窗口必须可见且在前台。
- **停止所有服务**:
  ```powershell
  Get-NetTCPConnection -LocalPort 9090,8081,8080 -State Listen | %{ Stop-Process -Id $_.OwningProcess -Force }
  ```
