# SnowLabel demo — start all three services, each in its own window.
# Run:  powershell -ExecutionPolicy Bypass -File start_demo.ps1
$ErrorActionPreference = "Stop"
$py   = "$env:USERPROFILE\miniconda3\envs\snowlabel\python.exe"
$ls   = "$env:USERPROFILE\miniconda3\envs\snowlabel\Scripts\label-studio.exe"
$root = $PSScriptRoot

# 1) SnowLabel ML backend (:9090) — production stack: DINOv3-L + SAM2.1 + Milvus
$backend = @"
cd '$root'
`$env:SNOWLABEL_DINO='$root\weights\facebookdinov3-vitl16-pretrain-lvd1689m'
`$env:SNOWLABEL_SAM='sam2.1_t.pt'
`$env:SNOWLABEL_RES='224'
`$env:SNOWLABEL_MEMORY_BACKEND='milvus'
`$env:SNOWLABEL_CONF_AUTO='0.61'
`$env:SNOWLABEL_CONF_SUGGEST='0.41'
& '$py' -m snowlabel.server
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backend

# 2) CORS image server (:8081) — serves demo_images/ to LS + ML backend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; & '$py' scripts\serve_images.py"

# 3) Label Studio (:8080)
$lsCmd = @"
`$env:LABEL_STUDIO_USERNAME='demo@snowlabel.ai'
`$env:LABEL_STUDIO_PASSWORD='snowlabel123'
& '$ls' start --port 8080 --no-browser
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $lsCmd

Write-Host ""
Write-Host "Started 3 services in separate windows. Give the ML backend ~40s to load DINOv3-L." -ForegroundColor Green
Write-Host "  ML backend  : http://localhost:9090/stats   (Memory Inspector: http://localhost:9090/inspector)"
Write-Host "  Images      : http://localhost:8081/"
Write-Host "  Label Studio: http://localhost:8080   (login: demo@snowlabel.ai / snowlabel123)"
Write-Host ""
Write-Host "When done:  Get-NetTCPConnection -LocalPort 9090,8081,8080 -State Listen | %{ Stop-Process -Id `$_.OwningProcess -Force }"
