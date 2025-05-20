@echo off
setlocal enabledelayedexpansion
echo.
echo.
echo Start activate the virtual environment...

call ov_env\Scripts\activate.bat

echo If you want run the Stress Test(default is No).
echo If no input is provided within 30 seconds, Will run the AI Performance test.
choice /C YN /T:30 /D:N /M "Do you want run the Stress Test?"

if errorlevel 2 (
    echo.
    echo.

    echo.
    echo.
    echo Runing the benchmark...
    python throughput_benchmark.py -d GPU


) else if errorlevel 1 (
    echo.
    echo.
    echo Starting run the Stress Test ...

    :input_loop
    set /p input_num=Please entry run hours for stress test:

    for /f "delims=0123456789" %%a in ("!input_num!") do (
        set "is_number=false"
        goto not_number
    )

    echo stress test for  !input_num! hours
    python throughput_benchmark.py -d GPU -t !input_num!
    goto end_script

    :not_number
    echo Invalid Number: !input_num!.
    goto input_loop

    :end_script
    endlocal
    pause
)


echo.
echo.
echo Done!
echo Exit of the virtual environment...
call ov_env\Scripts\deactivate.bat
