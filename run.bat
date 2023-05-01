@echo off
cmd /c python "%~dp0openpose.py" %* && exit || pause
