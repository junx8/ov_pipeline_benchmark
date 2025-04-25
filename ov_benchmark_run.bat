@echo off
echo.
echo.
echo Start activate the virtual environment...

call ov_env\Scripts\activate.bat

echo.
echo.
echo Runing the benchmark...

python throughput_benchmark.py

echo.
echo.
echo Done!
echo Exit of the virtual environment...
call ov_env\Scripts\deactivate.bat
