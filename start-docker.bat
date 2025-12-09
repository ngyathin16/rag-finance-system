@echo off
REM =============================================================================
REM RAG Finance System - Docker Quick Start (Windows)
REM =============================================================================
REM This batch file helps you quickly start the RAG Finance System on Windows
REM =============================================================================

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo RAG Finance System - Docker Quick Start
echo ================================================================================
echo.

REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows first
    echo https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo [OK] Docker is installed

REM Check if Docker is running
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)
echo [OK] Docker is running

REM Check if .env file exists
if not exist .env (
    echo.
    echo [WARNING] .env file not found
    echo.
    set /p OPENAI_KEY="Enter your OpenAI API key: "
    (
        echo # RAG Finance System - Environment Variables
        echo OPENAI_API_KEY=!OPENAI_KEY!
        echo VECTOR_STORE_MODE=chroma
        echo MAX_CORRECTIONS=2
        echo LOG_LEVEL=INFO
    ) > .env
    echo [OK] .env file created
) else (
    echo [OK] .env file exists
)

echo.
echo ================================================================================
echo Building and Starting Services
echo ================================================================================
echo.
echo This may take a few minutes on first run...
echo.

REM Build and start services
docker-compose up -d --build

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start services
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo [OK] All services started successfully!
echo.

REM Wait for services to be ready
echo ================================================================================
echo Waiting for Services to be Ready
echo ================================================================================
echo.

echo Waiting for RAG API to be healthy...
set RETRY=0
:WAIT_API
timeout /t 2 /nobreak >nul
curl -f http://localhost:8000/health >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] RAG API is ready!
    goto API_READY
)
set /a RETRY+=1
if %RETRY% LSS 30 goto WAIT_API
echo [WARNING] RAG API may not be ready yet
echo Check logs with: docker-compose logs rag-api
:API_READY

timeout /t 3 /nobreak >nul

REM Check other services
curl -f http://localhost:16686 >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Jaeger UI is ready!
) else (
    echo [WARNING] Jaeger UI may not be ready yet
)

curl -f http://localhost:9090/-/healthy >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Prometheus is ready!
) else (
    echo [WARNING] Prometheus may not be ready yet
)

curl -f http://localhost:3000/api/health >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Grafana is ready!
) else (
    echo [WARNING] Grafana may not be ready yet
)

echo.
echo ================================================================================
echo Access Information
echo ================================================================================
echo.
echo Services are now running!
echo.
echo RAG Finance API:
echo   URL:  http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo.
echo Jaeger (Distributed Tracing):
echo   URL:  http://localhost:16686
echo.
echo Prometheus (Metrics):
echo   URL:  http://localhost:9090
echo.
echo Grafana (Dashboards):
echo   URL:      http://localhost:3000
echo   Username: admin
echo   Password: admin
echo.
echo ================================================================================
echo Quick Test
echo ================================================================================
echo.
echo Test the API with:
echo.
echo curl -X POST http://localhost:8000/query ^
echo   -H "Content-Type: application/json" ^
echo   -d "{\"query\": \"What was the revenue in Q4 2024?\"}"
echo.
echo ================================================================================
echo Useful Commands
echo ================================================================================
echo.
echo View logs:           docker-compose logs -f
echo View specific logs:  docker-compose logs -f rag-api
echo Stop services:       docker-compose down
echo Restart services:    docker-compose restart
echo Check status:        docker-compose ps
echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo Press any key to exit...
pause >nul

