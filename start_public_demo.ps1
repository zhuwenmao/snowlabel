# ============================================================================
#  SnowLabel 公网 Demo 一键启动脚本（多租户隔离版）
#
#  作用：起一个"别人能用自己浏览器打开、上传自己图片试用"的在线 Demo。
#       每个访客的图片和记忆互相隔离，谁教的标签只有谁自己检索得到，
#       绝不污染你本地的车辆库（用的是独立的 data\memory_public 库）。
#
#  用法（在本文件所在目录）：
#       powershell -ExecutionPolicy Bypass -File start_public_demo.ps1
#  想顺便清空之前访客留下的图片/记忆、从零开始，加 -Fresh：
#       powershell -ExecutionPolicy Bypass -File start_public_demo.ps1 -Fresh
#
#  前提：① 本机开着科学上网代理（Clash/v2ray 等，脚本会自动找端口）——
#          因为 cloudflared 注册隧道的接口在国内被墙，必须走代理出去。
#        ② data\cache\cloudflared.exe 已存在（之前已下好）。
#  结束：关掉本脚本弹出的两个窗口即可；或运行 stop_public_demo.ps1。
# ============================================================================
param([switch]$Fresh)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py   = "$env:USERPROFILE\miniconda3\envs\snowlabel\python.exe"
$cf   = Join-Path $root "data\cache\cloudflared.exe"
Set-Location $root

function Say($m, $c="Gray") { Write-Host $m -ForegroundColor $c }

# --- 0) 前置检查 ---------------------------------------------------------
if (-not (Test-Path $py)) { Say "找不到 Python：$py" Red; exit 1 }
if (-not (Test-Path $cf)) { Say "找不到 cloudflared：$cf（请先下载放到 data\cache\）" Red; exit 1 }

# --- 0.5) 起三库基础设施(MariaDB + MinIO),幂等 --------------------------
Say "确保三库基础设施在跑(MariaDB + MinIO)…" Cyan
& "$root\start_infra.ps1"

# --- 1) 停掉占用 9090 的旧服务器 ----------------------------------------
$old = Get-NetTCPConnection -LocalPort 9090 -State Listen -ErrorAction SilentlyContinue
if ($old) {
  Say "停掉旧服务器 (PID $($old.OwningProcess -join ','))…" Yellow
  $old.OwningProcess | Select-Object -Unique | ForEach-Object {
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

# --- 2) 可选：清空访客数据，从零开始 ------------------------------------
if ($Fresh) {
  Say "清空 data\memory_public 和 data\uploads（-Fresh）…" Yellow
  Remove-Item "$root\data\memory_public\*" -Recurse -Force -ErrorAction SilentlyContinue
  Get-ChildItem "$root\data\uploads" -Directory -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force -Path "$root\data\uploads","$root\data\memory_public" | Out-Null

# --- 3) 起 SnowLabel 服务器（多租户 + 隔离库，绝不碰本地车库）----------
#  注意：公网 Demo 必须用 numpy 后端——只有它实现了 user_id 隔离；
#  milvus 后端只接收 user_id 参数不做过滤。RES 必须 512（DINOv3 patch16）。
#  绝不要带历史的 SNOWLABEL_CONF_AUTO/SUGGEST，用 config.py 默认 0.68/0.55。
$serverCmd = @"
Set-Location '$root'
Remove-Item Env:SNOWLABEL_CONF_AUTO    -ErrorAction SilentlyContinue
Remove-Item Env:SNOWLABEL_CONF_SUGGEST -ErrorAction SilentlyContinue
`$env:SNOWLABEL_DINO          = 'weights\facebookdinov3-vitl16-pretrain-lvd1689m'
`$env:SNOWLABEL_SAM           = 'sam2.1_t.pt'
`$env:SNOWLABEL_RES           = '512'
`$env:SNOWLABEL_PARTS_GRID    = '4'
`$env:SNOWLABEL_MEMORY_BACKEND= 'numpy'
`$env:SNOWLABEL_MULTIUSER     = '1'
`$env:SNOWLABEL_MEMORY_DIR    = 'data\memory_public'
`$env:SNOWLABEL_DATASET_DIR   = 'data\uploads'
`$env:SNOWLABEL_IMGCACHE      = '1'
`$env:NO_PROXY                = '127.0.0.1,localhost'
`$env:SNOWLABEL_PORT          = '9090'
& '$py' -m snowlabel.server
"@
Say "启动 SnowLabel 服务器（DINOv3-L 加载约 40 秒）…" Cyan
Start-Process powershell -ArgumentList "-NoExit","-Command",$serverCmd -WindowStyle Minimized

# 等 /stats 就绪
$ready = $false
for ($i=0; $i -lt 90; $i++) {
  Start-Sleep -Seconds 2
  try { Invoke-WebRequest "http://localhost:9090/stats" -UseBasicParsing -TimeoutSec 5 | Out-Null
        $ready = $true; break } catch {}
}
if (-not $ready) { Say "服务器 180 秒内没起来，看那个最小化的 PowerShell 窗口里的报错。" Red; exit 1 }
Say "服务器就绪 ✔" Green

# --- 4) 自动找本地代理端口 ----------------------------------------------
$proxy = $null
foreach ($p in 7897,7890,7891,10809,10808,2080,8889,1080,1087,8080) {
  $c = New-Object System.Net.Sockets.TcpClient
  try { $iar=$c.BeginConnect('127.0.0.1',$p,$null,$null)
        if ($iar.AsyncWaitHandle.WaitOne(250) -and $c.Connected) { $proxy="http://127.0.0.1:$p"; break } }
  catch {} finally { $c.Close() }
}
if ($proxy) { Say "检测到本地代理：$proxy" Green
              $env:HTTPS_PROXY=$proxy; $env:HTTP_PROXY=$proxy; $env:ALL_PROXY=$proxy }
else { Say "没检测到本地代理——cloudflared 直连国内大概率失败。请先打开你的科学上网工具再重跑。" Yellow }

# --- 5) 起 cloudflared 命名隧道（固定网址 snowlabel.zhuwenmao.com，永久不变）---
#  一次性配置已完成：~/.cloudflared 里有 cert.pem + <tunnelID>.json，DNS 已把
#  snowlabel.zhuwenmao.com CNAME 到隧道 snowlabel。这里只是把它跑起来，URL 不再变。
#  仍需代理：QUIC 走不通时 cloudflared 回退 HTTP/2(走代理)连 argotunnel。
$fixedUrl = "https://snowlabel.zhuwenmao.com"
$err = Join-Path $root "data\cache\cf.err.log"
$out = Join-Path $root "data\cache\cf.out.log"
Remove-Item $err,$out -ErrorAction SilentlyContinue
Say "启动命名隧道 snowlabel → $fixedUrl …" Cyan
Start-Process -FilePath $cf `
  -ArgumentList "tunnel","--edge-ip-version","4","--no-prechecks","--protocol","quic","--no-autoupdate","run","--url","http://localhost:9090","snowlabel" `
  -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Minimized
# 等隧道注册上连接
$connected = $false
for ($i=0; $i -lt 30; $i++) {
  Start-Sleep -Seconds 1
  if ((Get-Content $err -Raw -EA SilentlyContinue) -match 'Registered tunnel connection') { $connected=$true; break }
}

Write-Host ""
Say "============================================================" Green
Say "  固定公网链接（永久不变，发给别人直接打开）：" Green
Say "    $fixedUrl" Green
Say "  本机标注界面： http://localhost:9090/" Gray
Say "  记忆面板：     http://localhost:9090/inspector" Gray
Say "============================================================" Green
if ($connected) {
  Say "  隧道已连接 ✔  关闭：把弹出的两个最小化窗口关掉即可。" Gray
} else {
  Say "  隧道还在连接中——十几秒后访问上面网址即可；若长时间不通，多半是代理没开，" Yellow
  Say "  打开科学上网工具后重跑本脚本；或看 data\cache\cf.err.log 里的报错。" Yellow
}
