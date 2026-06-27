# 停掉公网 Demo：关闭 9090 服务器 + cloudflared 隧道。
#   powershell -ExecutionPolicy Bypass -File stop_public_demo.ps1
$srv = Get-NetTCPConnection -LocalPort 9090 -State Listen -ErrorAction SilentlyContinue
if ($srv) {
  $srv.OwningProcess | Select-Object -Unique | ForEach-Object {
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
  Write-Host "已停 9090 服务器。" -ForegroundColor Green
} else { Write-Host "9090 没在跑。" -ForegroundColor Gray }

$cf = Get-Process cloudflared -ErrorAction SilentlyContinue
if ($cf) { $cf | Stop-Process -Force -ErrorAction SilentlyContinue
           Write-Host "已停 cloudflared 隧道。" -ForegroundColor Green }
else { Write-Host "cloudflared 没在跑。" -ForegroundColor Gray }
