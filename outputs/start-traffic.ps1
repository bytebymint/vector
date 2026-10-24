$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$TrafficPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TrafficPython)) {
    $TrafficPython = Join-Path $PSScriptRoot '..\work\video-runtime\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $TrafficPython)) { throw 'Run install-traffic.ps1 first.' }
& $TrafficPython -u app.py
