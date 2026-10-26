# run.ps1 - PowerShell entry point for Indic corpus curation pipeline
# Usage: .\run.ps1 [raw_dir] [out_dir]
param(
    [string]$RawDir = "data\raw",
    [string]$OutDir = "data\out"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $RawDir)) {
    Write-Error "Error: raw input directory '$RawDir' does not exist."
    exit 1
}

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

python -u src\pipeline.py $RawDir $OutDir
exit $LASTEXITCODE
