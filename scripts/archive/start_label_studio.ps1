param(
    [int]$Port = 8080,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Python executable not found in .venv: $pythonExe"
}

$labelStudioCheck = & $pythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('label_studio') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Label Studio is not installed in the project environment." -ForegroundColor Yellow
    Write-Host "Install it with:" -ForegroundColor Yellow
    Write-Host "  $pythonExe -m pip install label-studio" -ForegroundColor Cyan
    exit 1
}

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $repoRoot
$env:HOST = "http://$HostAddress`:$Port"

Write-Host "Starting Label Studio..." -ForegroundColor Green
Write-Host "Local files root: $repoRoot" -ForegroundColor Green
Write-Host "URL: http://$HostAddress`:$Port" -ForegroundColor Green

& $pythonExe -m label_studio.server start --internal-host $HostAddress --host "http://$HostAddress`:$Port" --port $Port