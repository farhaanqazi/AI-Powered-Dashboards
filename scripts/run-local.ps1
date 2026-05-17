# Local dev runner — starts the API and (if JOB_QUEUE_ENABLED) the Arq worker.
# Usage:  ./scripts/run-local.ps1
# Frontend runs separately:  cd frontend; npm run dev
#
# Reads .env if present. Requires a reachable Redis when JOB_QUEUE_ENABLED=true.

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$py = Join-Path $root 'venv/Scripts/python.exe'
$arq = Join-Path $root 'venv/Scripts/arq.exe'

# Load .env into the process environment (simple KEY=VALUE lines).
$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
  Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
      [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
  }
}

$queue = $env:JOB_QUEUE_ENABLED
$worker = $null
if ($queue -and $queue.ToLower() -in @('1', 'true', 'yes', 'on')) {
  if (-not (Test-Path $arq)) {
    Write-Error "arq not installed in venv. Run: ./venv/Scripts/python.exe -m pip install arq==0.26.0"
  }
  Write-Host '→ Starting Arq worker (out-of-process jobs)…' -ForegroundColor Cyan
  $worker = Start-Process -FilePath $arq -ArgumentList 'src.jobs.worker.WorkerSettings' -PassThru -NoNewWindow
} else {
  Write-Host '→ JOB_QUEUE_ENABLED not set: jobs run in-process (no worker).' -ForegroundColor Yellow
}

try {
  Write-Host '→ Starting API on http://localhost:8000 …' -ForegroundColor Cyan
  & $py -m uvicorn main:app --reload --port 8000
}
finally {
  if ($worker -and -not $worker.HasExited) {
    Write-Host '→ Stopping worker…' -ForegroundColor Cyan
    Stop-Process -Id $worker.Id -Force -ErrorAction SilentlyContinue
  }
}
