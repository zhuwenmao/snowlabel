param(
    [string]$Venv = ".venv",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $Root $Venv
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$Constraints = Join-Path $Root "constraints\windows-py311-cu121.txt"

if (-not $Python) {
    $CondaPython = "D:\conda\envs\snowlabel\python.exe"
    if (Test-Path $CondaPython) {
        $Python = $CondaPython
    } else {
        $Python = (Get-Command python -ErrorAction Stop).Source
    }
} elseif (-not (Test-Path $Python)) {
    $Python = (Get-Command $Python -ErrorAction Stop).Source
}

$PythonVersion = & $Python -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not $PythonVersion.StartsWith("3.11.")) {
    throw "The published lock requires Python 3.11; found $PythonVersion at $Python"
}
if (-not (Test-Path $VenvPython)) {
    & $Python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $VenvPython -m pip install "pip==26.0.1" "setuptools==82.0.1" "wheel==0.46.3"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $VenvPython -m pip install -r (Join-Path $Root "requirements.txt") -c $Constraints
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $VenvPython -m pip install -e "${Root}[dev]" -c $Constraints
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $VenvPython (Join-Path $Root "scripts\verify_environment.py") --require-weights
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $VenvPython (Join-Path $Root "scripts\test_protocol.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Verified environment ready: $VenvPython"
