@echo off
echo Running tests with coverage...

pytest test/test_greedy_boruta.py -v --cov=greedy_boruta.GreedyBoruta --cov-report=term-missing --cov-report=xml

if %errorlevel% equ 0 (
    echo.
    echo Generating coverage badge...
    genbadge coverage -i coverage.xml -o coverage.svg
    
    if %errorlevel% equ 0 (
        echo.
        echo ✅ Coverage report and badge generated successfully!
        echo 📊 XML report: coverage.xml
        echo 🏅 Badge: coverage.svg
    ) else (
        echo ❌ Failed to generate badge
        exit /b 1
    )
) else (
    echo ❌ Tests failed
    exit /b 1
)