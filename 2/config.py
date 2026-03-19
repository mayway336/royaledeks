"""
Конфигурация проекта Clash Royale Deck Generator
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Базовый путь проекта
BASE_DIR = Path(__file__).parent

# ==================== Database Configuration ====================
DATABASE_PATH = os.getenv("DATABASE_PATH", BASE_DIR / "data" / "clash_royale.db")

# ==================== Model Architecture ====================
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "8"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "6"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "8"))

# ==================== Training Configuration ====================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "50"))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "10"))

# ==================== Data Filtering ====================
MIN_GAMES_PLAYED = int(os.getenv("MIN_GAMES_PLAYED", "100"))
SEASON_MONTHS_BACK = int(os.getenv("SEASON_MONTHS_BACK", "3"))

# ==================== Generation Configuration ====================
TOP_K = int(os.getenv("TOP_K", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))

# ==================== Scraping Configuration ====================
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

# ==================== Paths ====================
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Создание директорий
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ==================== Card Types & Rarities ====================
CARD_RARITIES = ["Common", "Rare", "Epic", "Legendary", "Champion"]
CARD_TYPES = ["Troop", "Spell", "Building"]
SPECIAL_CARD_TYPES = ["Evolution", "Hero", "Champion"]

# ==================== Slot Rules ====================
MAX_SPECIAL_SLOTS = 3

# ==================== Sources ====================
SOURCES = {
    'cards': 'https://clashroyale.fandom.com/wiki/Cards',
    'decks': 'https://royaleapi.com/decks',
    'deckshop': 'https://www.deckshop.pro/deck/list'
}
