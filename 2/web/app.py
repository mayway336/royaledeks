"""
Web Application - FastAPI приложение для генерации колод
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import uvicorn
from pathlib import Path

from config import (
    MODELS_DIR, DATA_DIR, EMBEDDING_DIM, NUM_HEADS,
    NUM_LAYERS, DROPOUT, MAX_SEQ_LEN, TOP_K, TEMPERATURE
)
from utils.logger import logger

# Создание FastAPI приложения
app = FastAPI(
    title="Clash Royale Deck Generator",
    description="ML-генератор колод для Clash Royale на основе Transformer",
    version="0.1.0"
)

# Директории
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Создание директорий
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Монтирование статики
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Шаблоны
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Глобальные переменные для модели
model = None
preprocessor = None
rule_engine = None
card_metadata = None


class GenerationRequest(BaseModel):
    """
    Запрос на генерацию колоды
    """
    num_decks: int = 1
    temperature: float = TEMPERATURE
    top_k: int = TOP_K
    use_rule_engine: bool = True
    seed: Optional[int] = None


class DeckResponse(BaseModel):
    """
    Ответ с сгенерированной колодой
    """
    cards: List[str]
    card_ids: List[int]
    avg_elixir: float
    is_valid: bool
    validation_message: str
    special_cards: Dict[str, int]


class GenerationResponse(BaseModel):
    """
    Ответ с несколькими колодами
    """
    decks: List[DeckResponse]
    generation_time: float
    model_info: Dict[str, Any]


def load_model():
    """Загрузка обученной модели и препроцессора"""
    global model, preprocessor, rule_engine, card_metadata
    
    try:
        import pickle
        from model.transformer import DeckGeneratorModel
        from data.parser import DEFAULT_CARDS
        from rule_engine.rule_engine import RuleEngine
        
        # Загрузка метаданных и словаря напрямую
        preprocess_path = DATA_DIR / "preprocessor"
        if preprocess_path.exists():
            # Загрузка словаря
            with open(preprocess_path / "vocabulary.pkl", "rb") as f:
                vocab_data = pickle.load(f)
            
            # Загрузка метаданных
            with open(preprocess_path / "metadata.pkl", "rb") as f:
                card_metadata = pickle.load(f)
            
            # Загрузка энкодера
            with open(preprocess_path / "encoder.pkl", "rb") as f:
                encoder_data = pickle.load(f)
            
            # Создание простого препроцессора с загруженными данными
            from data.preprocessor import CardVocabulary, CardFeatureEncoder
            
            preprocessor = type('Preprocessor', (), {})()
            preprocessor.vocab = type('Vocab', (), {})()
            preprocessor.vocab.card_to_idx = vocab_data['card_to_idx']
            preprocessor.vocab.idx_to_card = vocab_data['idx_to_card']
            preprocessor.vocab.card_to_name = vocab_data['card_to_name']
            preprocessor.vocab.name_to_card = vocab_data['name_to_card']
            preprocessor.vocab.size = vocab_data['size']
            
            preprocessor.encoder = type('Encoder', (), {})()
            preprocessor.encoder.card_features = encoder_data['card_features']
            preprocessor.encoder.feature_dim = encoder_data['feature_dim']
            
            preprocessor.card_info = card_metadata.get('card_info', {})
            
            # Инициализация Rule Engine
            rule_engine = RuleEngine(
                vocab_size=preprocessor.vocab.size,
                evolveable_cards=card_metadata.get('evolveable_cards', set()),
                hero_cards=card_metadata.get('hero_cards', set()),
                champion_cards=card_metadata.get('champion_cards', set())
            )
        
        # Загрузка модели
        model_path = MODELS_DIR / "best_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')

            # Инициализация модели
            model = DeckGeneratorModel(
                vocab_size=preprocessor.vocab.size,
                feature_dim=preprocessor.encoder.feature_dim,
                embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT,
                max_seq_len=MAX_SEQ_LEN
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            logger.info("Модель загружена успешно")
        else:
            logger.warning("Модель не найдена. Требуется обучение.")

    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Событие при запуске приложения"""
    logger.info("Запуск приложения...")
    load_model()


@app.get("/", response_model=None)
async def root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": model is not None
    })


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_deck(req: GenerationRequest):
    """
    Генерация колод (оригинальная версия через model.generate)

    Args:
        req: Запрос с параметрами генерации

    Returns:
        Список сгенерированных колод
    """
    import time
    import traceback

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Требуется обучение."
        )

    start_time = time.time()

    try:
        # Установка seed если указан
        if req.seed is not None:
            torch.manual_seed(req.seed)

        # Подготовка признаков
        import numpy as np
        
        idx_to_card = preprocessor.vocab.idx_to_card
        card_features_list = []
        for idx in range(preprocessor.vocab.size):
            card_id = idx_to_card.get(idx)
            if card_id is not None and card_id in preprocessor.encoder.card_features:
                card_features_list.append(preprocessor.encoder.card_features[card_id])
            else:
                card_features_list.append(np.zeros(preprocessor.encoder.feature_dim))
        
        # [vocab_size, feature_dim]
        card_features = torch.FloatTensor(np.array(card_features_list))

        # Генерация через model.generate() - оригинальная версия
        batch_size = req.num_decks
        
        with torch.no_grad():
            generated = model.generate(
                card_features=card_features,
                rule_engine=rule_engine if req.use_rule_engine else None,
                temperature=float(req.temperature) if req.temperature else 1.0,
                top_k=int(req.top_k) if req.top_k else 50
            )
            # generated shape: [batch_size, 8]

        # Конвертация в названия карт
        decks_response = []

        for deck_idx in range(batch_size):
            deck_indices = generated[deck_idx].tolist()

            # Конвертация индексов в названия
            card_names = []
            card_ids = []
            for idx in deck_indices:
                card_id = preprocessor.vocab.idx_to_card.get(idx, idx)
                card_name = preprocessor.vocab.card_to_name.get(card_id, f"Unknown_{idx}")
                card_names.append(card_name)
                card_ids.append(card_id)

            # Вычисление среднего эликсира
            elixirs = [
                preprocessor.card_info.get(cid, {}).get('elixir_cost', 0)
                for cid in card_ids
            ]
            avg_elixir = sum(elixirs) / len(elixirs) if elixirs else 0

            # Валидация
            if rule_engine and req.use_rule_engine:
                is_valid, validation_msg = rule_engine.validate_generated_deck(
                    deck_indices, preprocessor.vocab.idx_to_card
                )
            else:
                is_valid = True
                validation_msg = "OK"

            # Подсчёт специальных карт
            special_cards = {'evolutions': 0, 'heroes': 0, 'champions': 0}
            if card_metadata:
                for cid in card_ids:
                    if cid in card_metadata.get('evolveable_cards', set()):
                        special_cards['evolutions'] += 1
                    if cid in card_metadata.get('hero_cards', set()):
                        special_cards['heroes'] += 1
                    if cid in card_metadata.get('champion_cards', set()):
                        special_cards['champions'] += 1

            decks_response.append(DeckResponse(
                cards=card_names,
                card_ids=card_ids,
                avg_elixir=round(avg_elixir, 2),
                is_valid=is_valid,
                validation_message=validation_msg,
                special_cards=special_cards
            ))

        generation_time = time.time() - start_time

        return GenerationResponse(
            decks=decks_response,
            generation_time=round(generation_time, 3),
            model_info={
                "vocab_size": preprocessor.vocab.size,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS
            }
        )
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cards")
async def get_cards():
    """Получение списка всех карт"""
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    cards = []
    for card_id, name in preprocessor.vocab.card_to_name.items():
        card_info = preprocessor.card_info.get(card_id, {})
        cards.append({
            "id": card_id,
            "name": name,
            "elixir": card_info.get('elixir_cost', 0),
            "rarity": card_info.get('rarity', 'Unknown'),
            "type": card_info.get('type', 'Unknown'),
            "is_evolveable": card_info.get('is_evolveable', False),
            "is_hero": card_info.get('is_hero', False)
        })
    
    return {"cards": cards}


@app.get("/api/stats")
async def get_stats():
    """Получение статистики"""
    if preprocessor is None:
        return {"status": "model_not_loaded"}
    
    return {
        "status": "ok",
        "vocab_size": preprocessor.vocab.size,
        "model_loaded": model is not None,
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_seq_len": MAX_SEQ_LEN
        }
    }


@app.get("/api/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Запуск сервера
    
    Args:
        host: Хост для прослушивания
        port: Порт
        reload: Автоперезагрузка при изменениях
    """
    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()
