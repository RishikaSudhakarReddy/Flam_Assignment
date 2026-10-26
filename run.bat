@echo off
REM run.bat - Windows batch entry point for Indic corpus curation pipeline
REM Usage: run.bat [raw_dir] [out_dir]
setlocal enabledelayedexpansion

set "RAW_DIR=%~1"
set "OUT_DIR=%~2"

if "%RAW_DIR%"=="" set "RAW_DIR=data\raw"
if "%OUT_DIR%"=="" set "OUT_DIR=data\out"

if not exist "%RAW_DIR%" (
    echo Error: raw input directory "%RAW_DIR%" does not exist.
    exit /b 1
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

python -u src\pipeline.py "%RAW_DIR%" "%OUT_DIR%"
exit /b %ERRORLEVEL%
