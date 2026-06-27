# ============================================================================
#  SnowLabel 三库基础设施一键启动:MariaDB(3306) + MinIO(9000/9001)。
#  幂等:已在跑则跳过。Milvus 是嵌入式 lite,随服务器进程,无需单独起。
#
#  用法:  powershell -ExecutionPolicy Bypass -File start_infra.ps1
#  从零重置(清空所有缓存数据,慎用):  ... start_infra.ps1 -Fresh
# ============================================================================
param([switch]$Fresh)
$ErrorActionPreference = "Stop"
$root      = $PSScriptRoot
$py        = "$env:USERPROFILE\miniconda3\envs\snowlabel\python.exe"
$mdb       = Join-Path $root "data\cache\mariadb-11.4.12-winx64"
$datadir   = Join-Path $root "data\mariadb_data"
$minio     = Join-Path $root "data\cache\minio.exe"
$miniodata = Join-Path $root "data\minio_data"

function PortUp($p){ [bool](Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue) }
function Say($m,$c="Gray"){ Write-Host $m -ForegroundColor $c }

if ($Fresh) {
  Say "[-Fresh] 清空所有缓存数据(MariaDB / MinIO / Milvus 缓存)…" Yellow
  foreach($p in 3306,9000){ Get-NetTCPConnection -LocalPort $p -State Listen -EA SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -EA SilentlyContinue } }
  Start-Sleep 2
  foreach($d in $datadir,$miniodata,(Join-Path $root "data\memory_public\snowlabel_imgcache.db")){
    if(Test-Path $d){ Remove-Item $d -Recurse -Force -EA SilentlyContinue } }
}

# --- MariaDB ---
if (PortUp 3306) { Say "MariaDB 已在 3306 运行 ✔" Green }
else {
  if (-not (Test-Path (Join-Path $datadir "mysql"))) {
    Say "首次初始化 MariaDB 数据目录…" Cyan
    & "$mdb\bin\mariadb-install-db.exe" --datadir="$datadir" 2>&1 | Out-Null
  }
  Say "启动 MariaDB(3306)…" Cyan
  Start-Process -FilePath "$mdb\bin\mysqld.exe" `
    -ArgumentList "--datadir=$datadir","--port=3306" `
    -WindowStyle Hidden -RedirectStandardError "$root\data\cache\mariadb.err.log"
}

# --- MinIO ---
if (PortUp 9000) { Say "MinIO 已在 9000 运行 ✔" Green }
else {
  New-Item -ItemType Directory -Force -Path $miniodata | Out-Null
  $env:MINIO_ROOT_USER = "minioadmin"; $env:MINIO_ROOT_PASSWORD = "minioadmin"
  Say "启动 MinIO(9000 API / 9001 控制台)…" Cyan
  Start-Process -FilePath $minio `
    -ArgumentList "server","$miniodata","--address",":9000","--console-address",":9001" `
    -WindowStyle Hidden `
    -RedirectStandardOutput "$root\data\cache\minio.out.log" `
    -RedirectStandardError  "$root\data\cache\minio.err.log"
}

# --- 等就绪 + 建库/桶(幂等) ---
Start-Sleep -Seconds 6
$env:NO_PROXY = "127.0.0.1,localhost"
& $py -c @"
import pymysql
c=pymysql.connect(host='127.0.0.1',port=3306,user='root',password='')
c.cursor().execute('CREATE DATABASE IF NOT EXISTS snowlabel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci'); c.close()
from minio import Minio
m=Minio('127.0.0.1:9000',access_key='minioadmin',secret_key='minioadmin',secure=False)
m.bucket_exists('snowlabel') or m.make_bucket('snowlabel')
print('  MariaDB db=snowlabel, MinIO bucket=snowlabel  OK')
"@
Say "============================================================" Green
Say "  三库基础设施就绪:" Green
Say "    MariaDB  : localhost:3306  (root / 无密码, 库 snowlabel)" Gray
Say "    MinIO    : localhost:9000  控制台 http://localhost:9001 (minioadmin/minioadmin)" Gray
Say "  接着起服务器(带缓存):  start_public_demo.ps1   (已自动开启 SNOWLABEL_IMGCACHE)" Gray
Say "============================================================" Green
