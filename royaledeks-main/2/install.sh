#!/bin/bash

echo "================================================"
echo Clash Royale Deck Generator - Установка
echo "================================================"
echo ""

echo "[1/4] Проверка Python..."
if ! command -v python3 &> /dev/null; then
    echo "ОШИБКА: Python не найден! Установите Python 3.9+"
    exit 1
fi
echo "OK: Python установлен $(python3 --version)"
echo ""

echo "[2/4] Создание виртуального окружения..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "OK: Виртуальное окружение создано"
else
    echo "OK: Виртуальное окружение уже существует"
fi
echo ""

echo "[3/4] Установка зависимостей..."
source venv/bin/activate
pip install --upgrade pip > /dev/null
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ОШИБКА: Не удалось установить зависимости"
    exit 1
fi
echo "OK: Зависимости установлены"
echo ""

echo "[4/4] Инициализация проекта..."
python init_project.py
echo ""

echo "================================================"
echo Установка завершена!
echo "================================================"
echo ""
echo "Следующие шаги:"
echo "1. Проверьте парсер: python scripts/check_api.py"
echo "2. Запустите ETL Pipeline: python scripts/etl_pipeline.py"
echo "3. Обучите модель: python scripts/train_model.py"
echo "4. Запустите веб-приложение: python scripts/run_web.py"
echo ""
