"""Web Application - FastAPI приложение для генерации колод."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from config import (
    DATA_DIR,
    EMBEDDING_DIM,
    MAX_SEQ_LEN,
    NUM_HEADS,
    NUM_LAYERS,
)
from data.database import Database
from utils.logger import logger


app = FastAPI(
    title="Clash Royale Deck Generator",
    description="Генератор колод Clash Royale с учётом правил слотов",
    version="0.2.0",
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

cards_by_id: Dict[int, Dict[str, Any]] = {}
all_cards: List[Dict[str, Any]] = []
deck_corpus: List[List[int]] = []


class GenerationRequest(BaseModel):
    num_decks: int = Field(default=3, ge=1, le=20)
    input_cards: List[int] = Field(default_factory=list, description="Префикс 0-7 карт")


class DeckResponse(BaseModel):
    cards: List[str]
    card_ids: List[int]
    avg_elixir: float
    is_valid: bool
    validation_message: str
    special_cards: Dict[str, int]


class GenerationResponse(BaseModel):
    decks: List[DeckResponse]
    generation_time: float
    model_info: Dict[str, Any]


def _is_hero(card: Dict[str, Any]) -> bool:
    return bool(card.get("is_hero", False) or card.get("is_champion", False) or card.get("rarity") == "Champion")


def _is_evolution(card: Dict[str, Any]) -> bool:
    return bool(card.get("is_evolution", False))


def _is_special(card: Dict[str, Any]) -> bool:
    return _is_hero(card) or _is_evolution(card)


def _slot_allows(slot_idx: int, card: Dict[str, Any]) -> bool:
    hero = _is_hero(card)
    evo = _is_evolution(card)

    if slot_idx == 0:
        return evo or not hero
    if slot_idx == 1:
        return hero or not evo
    if slot_idx == 2:
        return True
    return not hero and not evo


def _validate_slot_structure(card_ids: List[int]) -> tuple[bool, str]:
    if len(card_ids) != 8:
        return False, "Колода должна содержать ровно 8 карт"
    if len(set(card_ids)) != 8:
        return False, "В колоде есть дубликаты"

    for i, cid in enumerate(card_ids):
        card = cards_by_id.get(cid)
        if not card:
            return False, f"Неизвестная карта id={cid}"
        if not _slot_allows(i, card):
            return False, f"Карта {card['name']} не подходит для слота {i+1}"

    return True, "OK"


def _score_deck(deck: List[int], prefix: List[int]) -> float:
    if not prefix:
        return 1.0
    score = 0.0
    deck_set = set(deck)
    for idx, cid in enumerate(prefix):
        if idx < len(deck) and deck[idx] == cid:
            score += 3.0
        elif cid in deck_set:
            score += 1.0
    return score


def _generate_one(prefix: List[int]) -> List[int]:
    prefix = prefix[:7]
    for i, cid in enumerate(prefix):
        card = cards_by_id.get(cid)
        if not card:
            raise HTTPException(status_code=400, detail=f"Неизвестная карта: {cid}")
        if not _slot_allows(i, card):
            raise HTTPException(status_code=400, detail=f"Карта {card['name']} не подходит для слота {i+1}")

    candidates = sorted(deck_corpus, key=lambda d: _score_deck(d, prefix), reverse=True)
    if not candidates:
        raise HTTPException(status_code=503, detail="Нет данных колод в БД")

    base = candidates[0]
    used = set(prefix)
    result = list(prefix)

    for slot in range(len(prefix), 8):
        pick = None

        # сначала из лучшей подходящей колоды
        for cid in base:
            if cid in used:
                continue
            card = cards_by_id.get(cid)
            if card and _slot_allows(slot, card):
                pick = cid
                break

        # затем по глобальной частоте
        if pick is None:
            freq = Counter()
            for deck in candidates[:120]:
                for cid in deck:
                    if cid not in used:
                        freq[cid] += 1

            for cid, _ in freq.most_common():
                card = cards_by_id.get(cid)
                if card and _slot_allows(slot, card):
                    pick = cid
                    break

        if pick is None:
            # крайний fallback — любой валидный id
            for card in all_cards:
                cid = card["card_id"]
                if cid not in used and _slot_allows(slot, card):
                    pick = cid
                    break

        if pick is None:
            raise HTTPException(status_code=500, detail=f"Не удалось заполнить слот {slot + 1}")

        result.append(pick)
        used.add(pick)

    return result


def load_data() -> None:
    global cards_by_id, all_cards, deck_corpus

    db = Database(str(DATA_DIR / "clash_royale.db"))
    db.connect()
    cards = db.get_all_cards()
    decks = db.get_filtered_decks(min_games=1, months_back=120)
    db.disconnect()

    cards_by_id = {c["card_id"]: c for c in cards}
    all_cards = sorted(cards, key=lambda x: x["name"])

    valid_decks = []
    for d in decks:
        card_ids = d.get("cards", [])
        is_valid, _ = _validate_slot_structure(card_ids)
        if is_valid:
            valid_decks.append(card_ids)

    deck_corpus = valid_decks
    logger.info(f"Web: загружено карт {len(all_cards)}, валидных колод {len(deck_corpus)}")


@app.on_event("startup")
async def startup_event() -> None:
    load_data()


@app.get("/", response_model=None)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_loaded": bool(deck_corpus)})


@app.post("/api/generate", response_model=GenerationResponse)
async def generate_deck(req: GenerationRequest):
    import time

    if len(req.input_cards) > 7:
        raise HTTPException(status_code=400, detail="Можно передать от 0 до 7 карт")

    start = time.time()
    decks: List[DeckResponse] = []

    for _ in range(req.num_decks):
        generated = _generate_one(req.input_cards)
        is_valid, msg = _validate_slot_structure(generated)

        names = [cards_by_id[cid]["name"] for cid in generated]
        avg_elixir = sum(float(cards_by_id[cid].get("elixir_cost", 0) or 0) for cid in generated) / 8

        special = {
            "evolutions": sum(1 for cid in generated if _is_evolution(cards_by_id[cid])),
            "heroes": sum(1 for cid in generated if bool(cards_by_id[cid].get("is_hero", False))),
            "champions": sum(1 for cid in generated if bool(cards_by_id[cid].get("is_champion", False))),
        }

        decks.append(
            DeckResponse(
                cards=names,
                card_ids=generated,
                avg_elixir=round(avg_elixir, 2),
                is_valid=is_valid,
                validation_message=msg,
                special_cards=special,
            )
        )

    return GenerationResponse(
        decks=decks,
        generation_time=round(time.time() - start, 3),
        model_info={
            "mode": "hybrid-corpus-generator",
            "vocab_size": len(all_cards),
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_seq_len": MAX_SEQ_LEN,
        },
    )


@app.get("/api/cards")
async def get_cards():
    return {"cards": all_cards}


@app.get("/api/stats")
async def get_stats():
    return {
        "status": "ok" if deck_corpus else "model_not_loaded",
        "vocab_size": len(all_cards),
        "model_loaded": bool(deck_corpus),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_seq_len": MAX_SEQ_LEN,
        },
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model_loaded": bool(deck_corpus)}


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    uvicorn.run("web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()
