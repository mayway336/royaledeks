@echo off
chcp 65001 >nul
echo ================================================
echo Clash Royale Deck Generator - Установка
echo ================================================
echo.

echo [1/4] Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден! Установите Python 3.9+
    pause
    exit /b 1
)
echo OK: Python установлен
echo.

echo [2/4] Создание виртуального окружения...
if not exist venv (
    python -m venv venv
    echo OK: Виртуальное окружение создано
) else (
    echo OK: Виртуальное окружение уже существует
)
echo.

echo [3/4] Установка зависимостей...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo ОШИБКА: Не удалось установить зависимости
    pause
    exit /b 1
)
echo OK: Зависимости установлены
echo.

echo [4/4] Инициализация проекта...
python init_project.py
echo.

echo ================================================
echo Установка завершена!
echo ================================================
echo.
echo Следующие шаги:
echo 1. Проверьте парсер: python scripts\check_api.py
echo 2. Запустите ETL Pipeline: python scripts\etl_pipeline.py
echo 3. Обучите модель: python scripts\train_model.py
echo 4. Запустите веб-приложение: python scripts\run_web.py
echo.
pause
