$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appFile = Join-Path $projectRoot "app.py"

$cloudflaredCandidates = @(
    (Join-Path $projectRoot "tools\cloudflared.exe"),
    "$env:USERPROFILE\Downloads\cloudflared-windows-amd64.exe"
)

$cloudflaredExe = $null
foreach ($candidate in $cloudflaredCandidates) {
    if (Test-Path $candidate) {
        $cloudflaredExe = $candidate
        break
    }
}

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment Python was not found:" -ForegroundColor Red
    Write-Host $venvPython
    exit 1
}

if (-not (Test-Path $appFile)) {
    Write-Host "app.py was not found:" -ForegroundColor Red
    Write-Host $appFile
    exit 1
}

if (-not $cloudflaredExe) {
    Write-Host "cloudflared.exe was not found." -ForegroundColor Red
    Write-Host "Please place cloudflared.exe in one of the following locations:"
    Write-Host "1. $projectRoot\tools\cloudflared.exe"
    Write-Host "or"
    Write-Host "2. $env:USERPROFILE\Downloads\cloudflared-windows-amd64.exe"
    exit 1
}

Write-Host "Project Root: $projectRoot"
Write-Host "Python: $venvPython"
Write-Host "App: $appFile"
Write-Host "Cloudflared: $cloudflaredExe"
Write-Host ""

# [Fix 1] Check and automatically kill old processes occupying port 8501 to ensure alignment with Cloudflare
$port8501 = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($port8501) {
    Write-Host "Warning: Port 8501 is already in use. Stopping old process to free the port..." -ForegroundColor Yellow
    $port8501 | Select-Object -First 1 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

Write-Host "Starting Streamlit..." -ForegroundColor Cyan

# [Fix 2] Force Streamlit to run on port 8501
Start-Process powershell -WorkingDirectory $projectRoot -ArgumentList @(
    "-NoExit",
    "-Command",
    "& '$venvPython' -m streamlit run '$appFile' --server.port 8501"
)

Write-Host "Waiting for Streamlit to start..." -ForegroundColor Cyan
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "Starting Cloudflare Tunnel..." -ForegroundColor Green
Write-Host "Please keep this window open. If you close it, the public URL will stop working." -ForegroundColor Yellow
Write-Host ""

# [Fix 3] Run Cloudflare in the original window, safely intercept the URL, and open the browser
$publicUrlOpened = $false

# [Fix 4] Use 127.0.0.1 instead of localhost to prevent IPv6 resolution issues
& $cloudflaredExe tunnel --url http://127.0.0.1:8501 2>&1 | ForEach-Object {
    $line = $_.ToString()
    Write-Host $line
    
    # Detect TryCloudflare's dynamic URL
    if (-not $publicUrlOpened -and $line -match 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com') {
        $publicUrl = $matches[0]
        Write-Host ""
        Write-Host "Public URL detected: $publicUrl" -ForegroundColor Green
        Write-Host "Waiting 4 seconds for Cloudflare tunnel to stabilize before opening..." -ForegroundColor Yellow
        
        # [Fix 5] Open browser asynchronously after a short delay so we don't block the tunnel setup
        Start-Process powershell -WindowStyle Hidden -ArgumentList "-Command", "Start-Sleep -Seconds 4; Start-Process '$publicUrl'"
        
        $publicUrlOpened = $true
    }
}