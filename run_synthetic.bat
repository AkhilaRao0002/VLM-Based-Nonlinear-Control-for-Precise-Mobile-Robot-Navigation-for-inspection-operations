@echo off
cd /d %~dp0
python synthetic_scene.py
python main_pipeline.py --perception synthetic
pause
