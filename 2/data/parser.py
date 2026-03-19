"""
Парсеры данных Clash Royale:
1) Карты из Clash Royale Fandom (через MediaWiki API + HTML таблицы)
2) Колоды из RoyaleAPI и DeckShop с унифицированным форматом
"""
from __future__ import annotations

import re
import time
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:  # pragma: no cover
    HAS_CLOUDSCRAPER = False

from config import REQUEST_DELAY, MAX_RETRIES, TIMEOUT, MIN_GAMES_PLAYED
from utils.logger import logger


CARD_PAGE = "https://clashroyale.fandom.com/wiki/Cards"
CARD_API_PARSE = "https://clashroyale.fandom.com/api.php?action=parse&page=Cards&format=json"
ROYALEAPI_POPULAR = "https://royaleapi.com/decks/popular"
DECKSHOP_BEST = "https://www.deckshop.pro/best-decks/"
DECKSHOP_PRO = "https://www.deckshop.pro/deck/list"
ROYALEAPI_TOP = "https://royaleapi.com/decks/top"
OPEN_CRT = "https://open-crt.com/decks"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _slug_to_name(slug: str) -> str:
    name = slug.strip().lower()
    name = re.sub(r"-(ev\d+|evo|hero)$", "", name)
    return " ".join(part.capitalize() for part in name.split("-"))


def _extract_number(value: str) -> Optional[float]:
    value = value.replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group(0))


class ClashRoyaleAPI:
    """Сборщик данных для карт и колод."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
        )
        self.scraper = cloudscraper.create_scraper() if HAS_CLOUDSCRAPER else None
        logger.info("ClashRoyaleAPI инициализирован")

    def _request(self, url: str, use_cloudscraper: bool = False) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if use_cloudscraper and self.scraper:
                    response = self.scraper.get(url, timeout=TIMEOUT)
                else:
                    response = self.session.get(url, timeout=TIMEOUT)
                response.raise_for_status()
                time.sleep(REQUEST_DELAY)
                return response.text
            except Exception as exc:  # pragma: no cover
                last_error = exc
                logger.warning(f"Ошибка запроса {url} (попытка {attempt}/{MAX_RETRIES}): {exc}")
                time.sleep(REQUEST_DELAY * attempt)

        raise RuntimeError(f"Не удалось загрузить {url}: {last_error}")

    def _parse_fandom_cards(self, html: str) -> List[Dict[str, Any]]:
        """Парсинг карт из таблиц страницы Cards."""
        soup = BeautifulSoup(html, "html.parser")
        cards: Dict[str, Dict[str, Any]] = {}

        def get_or_create_card(name: str, *, is_evolution: bool = False) -> Dict[str, Any]:
            key = name.lower()
            if key not in cards:
                cards[key] = {
                    "name": name,
                    "elixir_cost": 0,
                    "rarity": "Common",
                    "type": "Troop",
                    "is_evolveable": False,
                    "is_hero": False,
                    "is_champion": False,
                    "is_evolution": is_evolution,
                    "base_name": None,
                    "icon_url": None,
                    "source": "fandom",
                }
            return cards[key]

        # Основные таблицы с Card + Cost
        for table in soup.select("table.wikitable"):
            header_cells = [
                _normalize_text(th.get_text(" ", strip=True)).lower() for th in table.select("tr th")
            ]
            if "card" not in header_cells:
                continue

            cost_idx = None
            for idx, header in enumerate(header_cells):
                if "cost" in header:
                    cost_idx = idx
                    break

            for row in table.select("tr")[1:]:
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue

                # Обычно вторая колонка — ссылка на карту
                anchor = cells[1].find("a")
                card_name = _normalize_text(anchor.get_text(" ", strip=True) if anchor else cells[1].get_text(" ", strip=True))
                if not card_name or card_name.lower() == "card":
                    continue

                if "card level" in card_name.lower() or "wiki" in card_name.lower():
                    continue

                card = get_or_create_card(card_name)

                if cost_idx is not None and cost_idx < len(cells):
                    parsed_cost = _extract_number(cells[cost_idx].get_text(" ", strip=True))
                    if parsed_cost is not None:
                        card["elixir_cost"] = parsed_cost

                href = anchor.get("href") if anchor else None
                if href and href.startswith("/"):
                    card["icon_url"] = f"https://clashroyale.fandom.com{href}"

                # Грубая типизация по таблице
                text_table = " ".join(header_cells)
                if "spell" in text_table:
                    card["type"] = "Spell"
                elif "building" in text_table or "spawner" in text_table:
                    card["type"] = "Building"

        # Блок эволюций (отдельные сущности)
        evo_header = soup.find(id="Card_Evolution")
        if evo_header:
            node: Optional[Tag] = evo_header.parent
            while node and (node := node.find_next_sibling()):
                if node.name == "h2":
                    break
                for anchor in node.select("a"):
                    text = _normalize_text(anchor.get_text(" ", strip=True))
                    if not text or len(text) < 3:
                        continue
                    if "evolution" in text.lower() or "evolved" in text.lower() or text.startswith("Evo "):
                        base_name = re.sub(r"(?i)evolved\s+", "", text)
                        base_name = re.sub(r"(?i)evolution\s+", "", base_name)
                        base_name = re.sub(r"(?i)^evo\s+", "", base_name).strip()
                        evo_name = f"Evolution {base_name}"
                        evo = get_or_create_card(evo_name, is_evolution=True)
                        evo["is_evolution"] = True
                        evo["is_evolveable"] = True
                        evo["base_name"] = base_name
                        if base_name.lower() in cards:
                            evo["elixir_cost"] = cards[base_name.lower()].get("elixir_cost", 0)

        # Чемпионы и герои
        champions = {
            "Golden Knight",
            "Archer Queen",
            "Mighty Miner",
            "Skeleton King",
            "Monk",
            "Little Prince",
            "Goblinstein",
        }
        for champion in champions:
            if champion.lower() in cards:
                cards[champion.lower()]["rarity"] = "Champion"
                cards[champion.lower()]["is_champion"] = True
                cards[champion.lower()]["is_hero"] = True

        # Маркировка evolveable базовых карт (если найдена их эволюция)
        for card in list(cards.values()):
            base_name = card.get("base_name")
            if base_name and base_name.lower() in cards:
                cards[base_name.lower()]["is_evolveable"] = True

        # Присваиваем card_id стабильно по имени
        prepared: List[Dict[str, Any]] = []
        for card in sorted(cards.values(), key=lambda x: x["name"].lower()):
            stable_id = int(hashlib.md5(card["name"].encode("utf-8")).hexdigest()[:8], 16)
            card["card_id"] = stable_id
            prepared.append(card)

        return prepared

    def get_all_cards(self) -> List[Dict[str, Any]]:
        """Получить все карты из Fandom (включая эволюции и героев/чемпионов)."""
        logger.info("Загрузка карт из Fandom MediaWiki API...")
        raw = self._request(CARD_API_PARSE, use_cloudscraper=False)
        json_match = json.loads(raw)
        html = json_match["parse"]["text"]["*"]

        cards = self._parse_fandom_cards(html)
        if not cards:
            raise RuntimeError("Парсер карт вернул пустой список")

        logger.info(f"Получено карт: {len(cards)}")
        return cards

    def _parse_royaleapi_decks(self) -> List[Dict[str, Any]]:
        logger.info("Парсинг колод из RoyaleAPI...")
        decks: List[Dict[str, Any]] = []
        seen_global = set()
        
        # Парсим несколько страниц/разделов RoyaleAPI
        urls_to_parse = [
            ROYALEAPI_POPULAR,
            ROYALEAPI_TOP,
        ]
        
        for url in urls_to_parse:
            try:
                html = self._request(url, use_cloudscraper=True)
                soup = BeautifulSoup(html, "html.parser")
                
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "/decks/stats/" not in href and "/decks/top/" not in href:
                        continue
                    
                    # Извлекаем карты из URL
                    slug_part = href.split("/decks/stats/")[-1].split("/decks/top/")[-1].split("?")[0]
                    slug_cards = [s for s in slug_part.split(",") if s]
                    if len(slug_cards) != 8:
                        continue
                    
                    key = tuple(sorted(slug_cards))
                    if key in seen_global:
                        continue
                    seen_global.add(key)
                    
                    cards = [_slug_to_name(slug) for slug in slug_cards]
                    deck_id = f"royaleapi_{hashlib.md5(','.join(sorted(slug_cards)).encode('utf-8')).hexdigest()[:16]}"
                    
                    decks.append(
                        {
                            "deck_id": deck_id,
                            "cards": cards,
                            "avg_elixir": 0.0,
                            "win_rate": 0.5,
                            "games_played": max(1000, MIN_GAMES_PLAYED),
                            "trophy_limit": None,
                            "season": None,
                            "source": "royaleapi",
                            "timestamp": datetime.now(),
                        }
                    )
            except Exception as e:
                logger.warning(f"Ошибка при парсинге {url}: {e}")
        
        logger.info(f"RoyaleAPI: найдено колод {len(decks)}")
        return decks

    def _parse_deckshop_decks(self) -> List[Dict[str, Any]]:
        logger.info("Парсинг колод из DeckShop...")
        decks: List[Dict[str, Any]] = []
        seen_global = set()
        
        # Парсим несколько страниц DeckShop
        urls_to_parse = [
            DECKSHOP_BEST,
            DECKSHOP_PRO,
        ]
        
        for url in urls_to_parse:
            try:
                html = self._request(url, use_cloudscraper=False)
                soup = BeautifulSoup(html, "html.parser")
                
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "/deck/detail/" not in href and "/best-decks/" not in href:
                        continue
                    
                    slug_part = href.split("/deck/detail/")[-1].split("/best-decks/")[-1].split("?")[0].strip("/")
                    slug_cards = [s for s in slug_part.split(",") if s]
                    if len(slug_cards) != 8:
                        continue
                    
                    key = tuple(sorted(slug_cards))
                    if key in seen_global:
                        continue
                    seen_global.add(key)
                    
                    cards = [_slug_to_name(slug) for slug in slug_cards]
                    deck_id = f"deckshop_{hashlib.md5(','.join(sorted(slug_cards)).encode('utf-8')).hexdigest()[:16]}"
                    decks.append(
                        {
                            "deck_id": deck_id,
                            "cards": cards,
                            "avg_elixir": 0.0,
                            "win_rate": 0.5,
                            "games_played": max(700, MIN_GAMES_PLAYED),
                            "trophy_limit": None,
                            "season": None,
                            "source": "deckshop",
                            "timestamp": datetime.now(),
                        }
                    )
            except Exception as e:
                logger.warning(f"Ошибка при парсинге {url}: {e}")
        
        logger.info(f"DeckShop: найдено колод {len(decks)}")
        return decks

    def get_top_decks(self, limit: int = 500, min_games: int = MIN_GAMES_PLAYED) -> List[Dict[str, Any]]:
        """Получить объединенные колоды из всех источников с глобальной проверкой уникальности."""
        decks = self._parse_royaleapi_decks() + self._parse_deckshop_decks()

        # Глобальная дедупликация по составу карт (независимо от порядка)
        # Также проверяем uniqueness по deck_id для избежания полных дубликатов
        uniq: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        seen_deck_ids = set()
        
        for deck in decks:
            if deck["games_played"] < min_games:
                continue
            
            # Проверка на уникальный deck_id
            if deck["deck_id"] in seen_deck_ids:
                continue
            seen_deck_ids.add(deck["deck_id"])
            
            # Сигнатура колоды - отсортированный набор карт (case-insensitive)
            signature = tuple(sorted(c.lower() for c in deck["cards"]))
            
            # Если такая комбинация карт уже есть, пропускаем
            if signature not in uniq:
                uniq[signature] = deck
            else:
                # Если есть дубликат по картам, оставляем запись с большим games_played
                if deck["games_played"] > uniq[signature]["games_played"]:
                    uniq[signature] = deck

        merged = list(uniq.values())[:limit]
        logger.info(f"Итоговое количество колод (global dedup): {len(merged)}")
        return merged

    def get_popular_decks(self, limit: int = 200, min_games: int = MIN_GAMES_PLAYED) -> List[Dict[str, Any]]:
        return self.get_top_decks(limit=limit, min_games=min_games)


def main() -> None:
    api = ClashRoyaleAPI()

    cards = api.get_all_cards()
    print(f"Карты: {len(cards)}")
    print(cards[0])

    decks = api.get_top_decks(limit=20, min_games=1)
    print(f"Колоды: {len(decks)}")
    print(decks[0])


if __name__ == "__main__":
    main()
