# Руководство по использованию

## 📦 Установка

### Windows
```bash
install.bat
```

### Linux/Mac
```bash
chmod +x install.sh
./install.sh
```

### Вручную
```bash
# Создание виртуального окружения
python -m venv venv

# Активация
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Инициализация
python init_project.py
```

## ⚙️ Настройка

Проект **не требует API ключей** - все данные собираются через веб-парсинг из открытых источников.

При необходимости настройте параметры в `.env`:

```
REQUEST_DELAY=1.0      # Задержка между запросами (увеличьте при проблемах)
MAX_RETRIES=3          # Количество повторных попыток
TIMEOUT=30             # Таймаут запроса (сек)
```

## 🚀 Быстрый старт

### Шаг 1: Проверка парсера
```bash
python scripts/check_api.py
```

Проверяет возможность парсинга сайтов-источников.

### Шаг 2: Загрузка данных (ETL Pipeline)
```bash
python scripts/etl_pipeline.py
```

Загружает:
- Все карты из Clash Royale Fandom Wiki
- Топ колод из DeckShop.pro

**Время выполнения:** 5-15 минут (зависит от количества данных)

### Шаг 3: Обучение модели
```bash
python scripts/train_model.py
```

Параметры в `.env`:
- `NUM_EPOCHS` - количество эпох (по умолчанию 100)
- `BATCH_SIZE` - размер батча (32)
- `LEARNING_RATE` - скорость обучения (0.0001)
- `EARLY_STOPPING_PATIENCE` - ранняя остановка (10)

**Время выполнения:** 10-30 минут (зависит от GPU/CPU)

### Шаг 4: Оценка качества (опционально)
```bash
python scripts/evaluate_model.py --num-decks 100
```

Генерирует отчёт в `models/evaluation_report.txt`

### Шаг 5: Запуск веб-приложения
```bash
python scripts/run_web.py
```

Откройте в браузере: http://localhost:8000

## 🎮 Использование веб-приложения

### Параметры генерации:

| Параметр | Описание | Рекомендуемое значение |
|----------|----------|------------------------|
| Количество колод | Сколько колод сгенерировать | 1-10 |
| Температура | Креативность модели (0.1-2.0) | 1.0 |
| Top-K | Ограничение выбора (1-100) | 50 |
| Seed | Фиксация случайности | - |
| Rule Engine | Валидация правил | Включено |

### Интерпретация результатов:

- **✓ Валидна** - колода соответствует правилам слотов
- **✗ [причина]** - колода невалидна (дубликаты, превышение лимитов)

## 📊 Архитектура модели

```
Transformer Decoder:
├── Card Embedding (128 dim)
├── Positional Encoding
├── 6x Transformer Decoder Layers
│   ├── Masked Self-Attention (8 heads)
│   └── Feed-Forward Network
└── Output Linear Layer
```

**Размер словаря:** ~40-100 карт  
**Параметров модели:** ~5M

## 🔧 Продвинутое использование

### Генерация через API

```python
import requests

response = requests.post(
    'http://localhost:8000/api/generate',
    json={
        'num_decks': 5,
        'temperature': 0.8,
        'top_k': 30,
        'use_rule_engine': True
    }
)

decks = response.json()['decks']
for deck in decks:
    print(f"Колода: {deck['cards']}")
    print(f"Средний эликсир: {deck['avg_elixir']}")
    print(f"Валидна: {deck['is_valid']}")
```

### Прямая генерация в Python

```python
import torch
from model.transformer import DeckGeneratorModel
from data.preprocessor import DataPreprocessor
from rule_engine.rule_engine import RuleEngine

# Загрузка
preprocessor = DataPreprocessor.load('data/preprocessor')
model = DeckGeneratorModel(...)
model.load_state_dict(torch.load('models/best_model.pt'))
rule_engine = RuleEngine(...)

# Генерация
card_features = torch.FloatTensor([...])
with torch.no_grad():
    generated = model.generate(
        card_features=card_features,
        rule_engine=rule_engine,
        temperature=1.0,
        top_k=50
    )
```

## 📈 Метрики качества

| Метрика | Описание | Хорошее значение |
|---------|----------|------------------|
| Validity Rate | Доля валидных колод | >90% |
| Diversity | Уникальность генераций | >80% |
| Meta-Similarity | Сходство с топ-колодами | >0.7 |

## 🐛 Решение проблем

### Ошибка "Модель не загружена"
- Запустите `scripts/train_model.py`
- Проверьте наличие `models/best_model.pt`

### Ошибка "База данных пуста"
- Запустите `scripts/etl_pipeline.py`
- Проверьте интернет-соединение

### Парсинг не работает
- Увеличьте `REQUEST_DELAY` в `.env` (до 2-3 сек)
- Проверьте интернет-соединение
- Некоторые сайты могут блокировать запросы

### Низкое качество генерации
- Увеличьте количество данных в БД
- Увеличьте `NUM_EPOCHS`
- Попробуйте разные значения `temperature` и `top_k`

### Ошибка парсинга карт
- Структура Fandom Wiki могла измениться
- Запустите ETL ещё раз - будут созданы карты по умолчанию

## 📝 Структура проекта

```
clash_royale_generator/
├── data/               # ETL и предобработка
│   ├── database.py     # SQLite БД
│   ├── parser.py       # Веб-парсер (BeautifulSoup)
│   └── preprocessor.py # Векторизация
├── model/              # ML модель
│   ├── transformer.py  # Transformer Decoder
│   └── embeddings.py   # Эмбеддинги
├── rule_engine/        # Правила
│   └── rule_engine.py  # Маскирование
├── train/              # Обучение
│   ├── trainer.py      # Trainer класс
│   └── dataset.py      # DataLoader
├── eval/               # Оценка
│   └── metrics.py      # Метрики
├── web/                # Веб-приложение
│   ├── app.py          # FastAPI
│   ├── templates/      # HTML шаблоны
│   └── static/         # CSS/JS
├── scripts/            # Скрипты запуска
│   ├── check_api.py    # Проверка парсера
│   ├── etl_pipeline.py # Загрузка данных
│   ├── train_model.py  # Обучение
│   ├── evaluate_model.py # Оценка
│   └── run_web.py      # Веб-сервер
├── models/             # Сохранённые модели
├── data/               # БД и препроцессор
├── logs/               # Логи
└── .env                # Конфигурация
```

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в `logs/`
2. Убедитесь, что все зависимости установлены
3. Проверьте интернет-соединение
4. Убедитесь, что есть данные в БД
