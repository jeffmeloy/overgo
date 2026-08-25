@echo off
rem overgo_gui.bat: double-click to run the workbench. Builds the server,
rem starts it on localhost:8080 over the repo store, and opens the GUI.
rem Pass a GGUF path to serve a different model:
rem   overgo_gui.bat D:\models\my-model.gguf
setlocal
cd /d "%~dp0"

set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=C:\Users\jeffm\adaptive_new\checkpoints\overgo-hfconvert\Qwen2.5-0.5B-f16.gguf"

echo Building the overgo server...
go build -o bin\overgo-server.exe .\cmd\server || goto :error

start "" http://localhost:8080/app.html
echo Serving %MODEL%
echo Close this window to stop the server.
bin\overgo-server.exe -listen 127.0.0.1:8080 -repo overgodb-store "%MODEL%"
goto :eof

:error
echo Build failed; the workbench did not start.
pause
