@echo off
cd /d %~dp0
python main_pipeline.py --perception vlm --task "Navigate to the chair"
pause
