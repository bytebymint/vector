param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
$TrafficPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $TrafficPython -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
& $TrafficPython -m pip install -r requirements-traffic.txt
if ($LASTEXITCODE -ne 0) { throw 'Core dependency installation failed.' }
# Avoid a competing opencv-python distribution and the unnecessary cloud platform SDK.
& $TrafficPython -m pip install --no-deps ultralytics==8.4.183
if ($LASTEXITCODE -ne 0) { throw 'Ultralytics installation failed.' }
& $TrafficPython -c "from ultralytics import YOLO; YOLO('models/yolo26n.pt'); YOLO('models/yolo26s.pt')"
if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed.' }
Write-Host 'Ready. Run .\start-traffic.ps1 and open http://127.0.0.1:5000'
