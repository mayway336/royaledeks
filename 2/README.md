# Clash Royale Deck Generator

ML-генератор колод для Clash Royale на основе архитектуры Transformer Decoder.

## 📋 Описание

Проект использует машинное обучение для генерации валидных колод карт, учитывая:
- Правила слотов (Эволюции, Герои, Чемпионы)
- Баланс эликсира
- Актуальную мету (данные из открытых источников)

## 🏗️ Архитектура

```
clash_royale_generator/
├── data/               # ETL Pipeline, парсинг, предобработка
├── model/              # Transformer Decoder архитектура
├── rule_engine/        # Динамическое маскирование и валидация
├── train/              # Модуль обучения
├── eval/               # Оценка качества модели
├── web/                # Веб-приложение (FastAPI)
├── utils/              # Утилиты
├── config.py           # Конфигурация
└── requirements.txt    # Зависимости
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск ETL Pipeline

```bash
python scripts/etl_pipeline.py
```

Этот скрипт:
- Парсит данные из Clash Royale Fandom Wiki (карты)
- Парсит данные из DeckShop.pro (колоды)
- Сохраняет всё в SQLite базу данных

### 3. Обучение модели

```bash
python scripts/train_model.py
```

### 4. Запуск веб-приложения

```bash
python scripts/run_web.py
```

Откройте в браузере: `http://localhost:8000`

## 📊 Компоненты системы

### ETL Pipeline (Веб-парсинг)
- **Карты**: Clash Royale Fandom Wiki
- **Колоды**: DeckShop.pro, RoyaleAPI (публичные данные)
- Автоматическая фильтрация по количеству игр
- Сохранение в SQLite БД

### ML Model (Transformer Decoder)
- Embedding карт с признаками (эликсир, тип, редкость)
- Masked Self-Attention для авторегрессионной генерации
- Предсказание следующей карты в последовательности

### Rule Engine
- Динамическое маскирование на каждом шаге генерации
- Учёт 3 специальных слотов (Evolution, Hero, Wild)
- Исключение дубликатов карт

### Веб-приложение
- REST API для генерации колод
- Интерактивный UI
- Визуализация статистики

## 🎯 Конфигурация

Основные параметры в `.env`:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `EMBEDDING_DIM` | Размерность эмбеддинга карт | 128 |
| `NUM_HEADS` | Количество голов внимания | 8 |
| `NUM_LAYERS` | Количество слоёв Transformer | 6 |
| `BATCH_SIZE` | Размер батча при обучении | 32 |
| `LEARNING_RATE` | Скорость обучения | 0.0001 |
| `TOP_K` | Параметр сэмплирования | 50 |
| `REQUEST_DELAY` | Задержка между запросами (сек) | 1.0 |

## 📈 Метрики

- **Validity Rate**: Доля валидных колод
- **Diversity**: Уникальность генераций
- **Meta-Similarity**: Сходство с топ-колодами

## 🔧 Источники данных

Проект использует **веб-парсинг** вместо API:

| Источник | Данные | URL |
|----------|--------|-----|
| Clash Royale Fandom Wiki | Карты | https://clashroyale.fandom.com |
| DeckShop.pro | Колоды | https://www.deckshop.pro |
| RoyaleAPI | Колоды (публично) | https://royaleapi.com |

**Преимущества:**
- Не требуется API ключ
- Нет ограничений на количество запросов
- Данные всегда актуальны

## 📝 Лицензия

MIT

## 🙏 Благодарности

- [Clash Royale Fandom Wiki](https://clashroyale.fandom.com/)
- [DeckShop.pro](https://www.deckshop.pro/)
- [RoyaleAPI](https://royaleapi.com/)
