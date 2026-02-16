@echo off
REM ========================================
REM Clearissa EXE Builder - Robust Version
REM ========================================
REM This script builds a standalone executable for Clearissa.
REM It handles paths with spaces and special characters properly.
REM
REM Usage:
REM   build_clearissa.bat           - Build with automatic venv
REM   build_clearissa.bat --no-venv - Use current Python environment
REM   build_clearissa.bat --clean   - Clean previous builds first
REM ========================================

setlocal EnableDelayedExpansion

echo.
echo ========================================
echo  Clearissa EXE Builder
echo ========================================
echo.

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Store the root directory
set "ROOT_DIR=%CD%"
set "VENV_DIR=%ROOT_DIR%\.venv"
set "SPEC_FILE=%ROOT_DIR%\Clearissa.spec"
set "DIST_DIR=%ROOT_DIR%\dist"
set "BUILD_DIR=%ROOT_DIR%\build"

REM Default options
set "USE_VENV=1"
set "DO_CLEAN=0"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--no-venv" set "USE_VENV=0"
if /i "%~1"=="--clean" set "DO_CLEAN=1"
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
shift
goto parse_args

:show_help
echo.
echo Clearissa EXE Builder - Help
echo ========================================
echo.
echo Usage:
echo   build_clearissa.bat           - Build with automatic venv
echo   build_clearissa.bat --no-venv - Use current Python environment
echo   build_clearissa.bat --clean   - Clean previous builds first
echo   build_clearissa.bat --help    - Show this help message
echo.
echo This script builds a standalone Windows executable for Clearissa.
echo The output will be created at: dist\Clearissa\Clearissa.exe
echo.
exit /b 0

:args_done

REM Check if spec file exists
if not exist "%SPEC_FILE%" (
    echo ERROR: Clearissa.spec not found at: %SPEC_FILE%
    echo Please ensure you are running this script from the Clearissa project root.
    exit /b 1
)

REM Find Python interpreter
set "PYTHON_CMD="

REM Try py launcher with Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3.11"
    goto python_found
)

REM Try py launcher with any Python 3
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3"
    goto python_found
)

REM Try python command directly
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto python_found
)

REM Python not found
echo ERROR: Python not found. Please install Python 3.11 or later.
echo Visit: https://www.python.org/downloads/
exit /b 1

:python_found
echo [*] Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM Setup virtual environment if requested
if "%USE_VENV%"=="1" (
    echo [*] Setting up virtual environment...

    set "VENV_VALID=0"
    
    if exist "%VENV_DIR%\Scripts\python.exe" (
        echo     Checking existing virtual environment...
        
        REM Test if the venv's Python works
        "%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "VENV_VALID=1"
            echo     Existing virtual environment is valid
        ) else (
            echo     WARNING: Existing virtual environment is broken
            echo     The Python interpreter it was created with no longer exists
            echo     Removing and recreating virtual environment...
            
            REM Remove the broken venv
            rmdir /s /q "%VENV_DIR%" 2>nul
            
            REM Wait a moment for filesystem
            timeout /t 1 /nobreak >nul
            
            if exist "%VENV_DIR%" (
                echo     ERROR: Could not remove broken virtual environment
                echo     Please manually delete the .venv folder and try again
                exit /b 1
            )
        )
    )
    
    if !VENV_VALID! equ 0 (
        echo     Creating new virtual environment at: .venv
        %PYTHON_CMD% -m venv "%VENV_DIR%"
        if !errorlevel! neq 0 (
            echo ERROR: Failed to create virtual environment
            exit /b 1
        )
        
        REM Verify the new venv works
        "%VENV_DIR%\Scripts\python.exe" --version >nul 2>&1
        if !errorlevel! neq 0 (
            echo ERROR: Newly created virtual environment is not working
            exit /b 1
        )
        echo     New virtual environment created successfully
    )

    REM Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"
    if !errorlevel! neq 0 (
        echo ERROR: Failed to activate virtual environment
        exit /b 1
    )

    REM Verify Python works after activation
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo ERROR: Python not accessible after activating virtual environment
        exit /b 1
    )

    set "PYTHON_CMD=python"
    echo     Virtual environment activated
    echo.
) else (
    echo [*] Using current Python environment (--no-venv specified)
    echo.
)

REM Upgrade pip
echo [*] Upgrading pip...
%PYTHON_CMD% -m pip install --upgrade pip --quiet
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)
echo.

REM Install requirements
echo [*] Installing dependencies from requirements.txt...
if exist "%ROOT_DIR%\requirements.txt" (
    %PYTHON_CMD% -m pip install -r "%ROOT_DIR%\requirements.txt" --quiet
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install requirements
        goto cleanup_and_exit
    )
    echo     Dependencies installed successfully
) else (
    echo WARNING: requirements.txt not found
)
echo.

REM Verify optional plotting libraries used for ggplot export
echo [*] Verifying optional plotting libraries (plotnine, mizani)...
%PYTHON_CMD% -c "import plotnine; import mizani" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Optional plotting packages (plotnine, mizani^) are not available.
    echo          ggplot export will be disabled in the application unless you install them.
    echo          Install with: pip install plotnine mizani
    echo.
) else (
    echo     Optional plotting libraries are available.
    echo.
)

REM Install PyInstaller
echo [*] Installing PyInstaller...
%PYTHON_CMD% -m pip install "pyinstaller>=6.0" --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyInstaller
    goto cleanup_and_exit
)
echo     PyInstaller installed successfully
echo.

REM Clean Python cache files that might cause locking issues
echo [*] Cleaning Python cache files...
for /d /r "%ROOT_DIR%" %%d in (__pycache__) do @if exist "%%d" (
    echo     Removing: %%d
    rmdir /s /q "%%d" 2>nul
)
REM Also clean .pyc files
del /s /q "%ROOT_DIR%\*.pyc" >nul 2>&1
echo     Python cache cleaned
echo.

REM Terminate any running Clearissa.exe processes to prevent access denied errors
echo [*] Checking for running Clearissa.exe processes...
tasklist /FI "IMAGENAME eq Clearissa.exe" 2>nul | find /I "Clearissa.exe" >nul
if %errorlevel% equ 0 (
    echo     WARNING: Clearissa.exe is currently running
    echo     Attempting to terminate processes...
    taskkill /F /IM Clearissa.exe >nul 2>&1
    if %errorlevel% equ 0 (
        echo     Successfully terminated Clearissa.exe
        timeout /t 2 /nobreak >nul
    ) else (
        echo     ERROR: Could not terminate Clearissa.exe
        echo     Please close the application manually and try again.
        goto cleanup_and_exit
    )
) else (
    echo     No running Clearissa.exe processes found
)
echo.

REM Clean previous builds (always clean for reliable builds)
echo [*] Cleaning previous builds...
if exist "%BUILD_DIR%" (
    echo     Attempting to remove: build\
    REM Use retry logic for Windows file locking issues
    for /L %%i in (1,1,3) do (
        rmdir /s /q "%BUILD_DIR%" 2>nul
        if not exist "%BUILD_DIR%" goto build_removed
        echo     Retry %%i: Waiting for file locks to release...
        timeout /t 2 /nobreak >nul
    )
    :build_removed
    if exist "%BUILD_DIR%" (
        echo WARNING: Could not fully remove build directory.
        echo          Attempting to continue with manual cleanup...
        REM Try removing specific problematic subdirectories
        if exist "%BUILD_DIR%\Clearissa" (
            REM Remove attributes and try again
            attrib -r -s -h "%BUILD_DIR%\Clearissa\*.*" /s /d >nul 2>&1
            rmdir /s /q "%BUILD_DIR%\Clearissa" 2>nul
        )
    )
)
if exist "%DIST_DIR%" (
    echo     Removing: dist\
    rmdir /s /q "%DIST_DIR%" 2>nul
)
echo     Clean completed
echo.

REM Set environment variable for PyQtGraph
set "PYQTGRAPH_QT_LIB=PyQt5"

REM Run PyInstaller
echo [*] Building executable with PyInstaller...
echo     This may take several minutes...
echo.
%PYTHON_CMD% -m PyInstaller --clean --noconfirm "%SPEC_FILE%"
if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo  BUILD FAILED
    echo ========================================
    echo.
    echo PyInstaller encountered an error.
    echo Check the output above for details.
    goto cleanup_and_exit
)

REM Check if executable was created
set "EXE_PATH=%DIST_DIR%\Clearissa\Clearissa.exe"
if not exist "%EXE_PATH%" (
    echo.
    echo ========================================
    echo  BUILD FAILED
    echo ========================================
    echo.
    echo Executable not found at expected location:
    echo %EXE_PATH%
    goto cleanup_and_exit
)

REM Success!
echo.
echo ========================================
echo  BUILD SUCCESSFUL!
echo ========================================
echo.
echo Executable created at:
echo %EXE_PATH%
echo.
echo Size:
for %%A in ("%EXE_PATH%") do echo %%~zA bytes
echo.
echo You can now run the application by double-clicking:
echo dist\Clearissa\Clearissa.exe
echo.
echo To create a distributable package, compress the entire
echo "dist\Clearissa" folder into a ZIP file.
echo.

REM Deactivate venv if we used one
if "%USE_VENV%"=="1" (
    if exist "%VENV_DIR%\Scripts\deactivate.bat" (
        call "%VENV_DIR%\Scripts\deactivate.bat" 2>nul
    )
)

exit /b 0

:cleanup_and_exit
if "%USE_VENV%"=="1" (
    if exist "%VENV_DIR%\Scripts\deactivate.bat" (
        call "%VENV_DIR%\Scripts\deactivate.bat" 2>nul
    )
)
exit /b 1