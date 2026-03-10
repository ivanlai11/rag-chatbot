@echo off
REM Launch the public Streamlit app with Cloudflare Tunnel
powershell -ExecutionPolicy Bypass -File "%~dp0start_public.ps1"
pause