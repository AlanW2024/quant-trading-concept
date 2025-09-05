#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSCommandPath)

# 可選：載入 .env（若存在）
$envPath = Join-Path (Get-Location) ".env"
if (Test-Path $envPath) {
    Get-Content $envPath | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*$') { return }
        $kv = $_.Split('=', 2)
        if ($kv.Count -eq 2) {
            [Environment]::SetEnvironmentVariable($kv[0], $kv[1])
        }
    }
}

$OUT_DIR = if ($env:OUT) { $env:OUT } else { "out" }
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

python run_engine.py --out $OUT_DIR @args
