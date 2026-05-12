@echo off
:: MenoSCA-FBTS - Run All Experiments Script
:: Execution Order:
:: 1. SOTA Comparison
:: 2. Ablation Study
:: 3. Three Groups (Sleep-EDF)
:: 4. Three Groups (ISRUC)
:: 5. Three Groups (DREAMS)
:: 6. Generate Representative Hypnograms
:: 7. Comprehensive Analysis

setlocal

cd /d "%~dp0"

set "LOG_DIR=%~dp0experiment_results"
set "LOG_FILE=%LOG_DIR%\batch_run.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo ================================================
echo MenoSCA-FBTS - Run All Experiments
echo ================================================
echo Starting at: %date% %time%
echo Log file: %LOG_FILE%
echo ================================================

echo ================================================ > "%LOG_FILE%"
echo MenoSCA-FBTS - Run All Experiments >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo Starting at: %date% %time% >> "%LOG_FILE%"
echo Log file: %LOG_FILE% >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"

call :RUN_ONE "SOTA Comparison" "experiment_sota_comparison.py"
call :RUN_ONE "Ablation Study" "experiment_ablation_study.py"
call :RUN_ONE "Three Groups - Sleep-EDF" "experiment_three_groups_paper.py"
call :RUN_ONE "Three Groups - ISRUC" "experiment_three_groups_paper.py --dataset isruc"
call :RUN_ONE "Three Groups - DREAMS" "experiment_three_groups_paper.py --dataset dreams"
call :RUN_ONE "Generate Hypnograms" "generate_representative_hypnograms.py"
call :RUN_ONE "Comprehensive Analysis" "analysis_comprehensive.py"

echo.
echo ================================================
echo All experiments completed!
echo Finished at: %date% %time%
echo Log file: %LOG_FILE%
echo ================================================

echo. >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo All experiments completed! >> "%LOG_FILE%"
echo Finished at: %date% %time% >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"

pause
goto :EOF

:RUN_ONE
set "NAME=%~1"
set "CMD=%~2"

echo.
echo ================================================
echo [START] %NAME%
echo Command: python %CMD%
echo ================================================

echo. >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"
echo [START] %NAME% >> "%LOG_FILE%"
echo Command: python %CMD% >> "%LOG_FILE%"
echo ================================================ >> "%LOG_FILE%"

python %CMD% >> "%LOG_FILE%" 2>&1
set "CODE=%ERRORLEVEL%"

if %CODE% EQU 0 (
    echo [DONE] %NAME% (Exit Code: %CODE%)
    echo [DONE] %NAME% (Exit Code: %CODE%) >> "%LOG_FILE%"
) else (
    echo [WARNING] %NAME% (Exit Code: %CODE%)
    echo [WARNING] %NAME% (Exit Code: %CODE%) >> "%LOG_FILE%"
)

goto :EOF
