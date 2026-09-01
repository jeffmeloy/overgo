@echo off
rem overgo_gui.bat: double-click to run the workbench. Builds the server
rem and the model-swap proxy, starts the proxy on localhost:8080 over the
rem repo store, and opens the GUI. The model pill in the GUI switches the
rem served model live; no relaunch needed.
rem Pass a GGUF path (or a servable model name) to start on that model:
rem   overgo_gui.bat D:\models\my-model.gguf
rem With no argument the proxy starts without a default model; pick one from
rem the GUI's model pill, which lists every servable model in the store.
setlocal
cd /d "%~dp0"

set "MODEL=%~1"

echo Building the overgo server and swap proxy...
go build -o bin\overgo-server.exe .\cmd\server || goto :error
go build -o bin\overgo-swap.exe .\cmd\swap || goto :error

start "" http://localhost:8080/app.html
if "%MODEL%"=="" (
  echo Starting with no default model -- pick one from the GUI's model pill.
) else (
  echo Starting on %MODEL% -- switch models from the GUI's model pill.
)
echo Close this window to stop the server.
if "%MODEL%"=="" (
  bin\overgo-swap.exe -listen 127.0.0.1:8080 -repo overgodb-store -server bin\overgo-server.exe
) else (
  bin\overgo-swap.exe -listen 127.0.0.1:8080 -repo overgodb-store -server bin\overgo-server.exe -default "%MODEL%"
)
goto :eof

:error
echo Build failed; the workbench did not start.
pause
