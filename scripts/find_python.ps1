# ============================================================================
#  定位运行 SnowLabel 的 Python 解释器(供启动脚本 dot-source 使用)。
#
#  查找顺序:
#    1. 环境变量 SNOWLABEL_PYTHON(显式指定,最高优先级)
#    2. conda 登记过的名为 snowlabel 的环境(~/.conda/environments.txt,
#       不依赖 conda 是否在 PATH 里,也不管环境装在哪个盘)
#    3. 当前 PATH 里的 python
#
#  用法:  . "$PSScriptRoot\scripts\find_python.ps1";  $py = Find-SnowlabelPython
# ============================================================================
function Find-SnowlabelPython {
  if ($env:SNOWLABEL_PYTHON -and (Test-Path $env:SNOWLABEL_PYTHON)) {
    return (Resolve-Path $env:SNOWLABEL_PYTHON).Path
  }
  $registry = Join-Path $env:USERPROFILE ".conda\environments.txt"
  if (Test-Path $registry) {
    foreach ($line in Get-Content $registry) {
      $dir = $line.Trim()
      if ($dir -match '[\\/]snowlabel$') {
        $candidate = Join-Path $dir "python.exe"
        if (Test-Path $candidate) { return $candidate }
      }
    }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  return $null
}
