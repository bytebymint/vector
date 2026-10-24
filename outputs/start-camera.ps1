param(
    [string]$PlayUrl,
    [ValidateSet('quality', 'fast')]
    [string]$Profile = 'quality'
)

$ErrorActionPreference = 'Stop'
$videoPython = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\work\video-runtime\Scripts\python.exe'))
if (-not (Test-Path -LiteralPath $videoPython)) {
    throw 'The local video runtime is missing. Recreate it using enhancement-notes.md.'
}
if (-not $PlayUrl) {
    $PlayUrl = Read-Host 'Paste the fresh RC-293 playUrl'
}
$modelScale = if ($Profile -eq 'fast') { 2 } else { 4 }
& $videoPython (Join-Path $PSScriptRoot 'camera.py') $PlayUrl --scale 5 --sr-model espcn --model-scale $modelScale --compute opencl --threads 4
