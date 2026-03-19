"""
Упрощённая генерация колод для демонстрации
Использует случайное сэмплирование с весами модели
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pickle
import numpy as np
from model.transformer import DeckGeneratorModel
from rule_engine.rule_engine import RuleEngine

# Загрузка данных
print("Загрузка данных...")
with open('data/preprocessor/vocabulary.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
with open('data/preprocessor/encoder.pkl', 'rb') as f:
    encoder_data = pickle.load(f)
with open('data/preprocessor/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

vocab_size = vocab_data['size']
idx_to_card = vocab_data['idx_to_card']
card_to_name = vocab_data['card_to_name']

print(f"Загружено {vocab_size} карт")

# Подготовка признаков
print("Подготовка признаков...")
card_features_list = []
for idx in range(vocab_size):
    card_id = idx_to_card.get(idx)
    if card_id is not None and card_id in encoder_data['card_features']:
        card_features_list.append(encoder_data['card_features'][card_id])
    else:
        card_features_list.append(np.zeros(encoder_data['feature_dim']))

card_features = torch.FloatTensor(np.array(card_features_list))

# Загрузка модели
print("Загрузка модели...")
model = DeckGeneratorModel(
    vocab_size=vocab_size,
    feature_dim=encoder_data['feature_dim'],
    embedding_dim=128,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    max_seq_len=8
)
model.load_state_dict(torch.load('models/best_model.pt', map_location='cpu')['model_state_dict'])
model.eval()
print("Модель загружена")

# Rule Engine
rule_engine = RuleEngine(
    vocab_size=vocab_size,
    evolveable_cards=metadata.get('evolveable_cards', set()),
    hero_cards=metadata.get('hero_cards', set()),
    champion_cards=metadata.get('champion_cards', set())
)

# Простая генерация через forward pass
print("\nГенерация колоды (упрощённая)...")

def generate_simple_deck(model, card_features, rule_engine, top_k=20):
    """Генерация одной колоды через последовательное предсказание"""
    vocab_size = card_features.shape[0]
    START_TOKEN = vocab_size  # Используем vocab_size как START токен
    
    generated = []
    used_cards = set()
    
    for step in range(8):
        # Подготовка входа
        if step == 0:
            input_seq = torch.LongTensor([[START_TOKEN]])
            input_features = torch.zeros(1, 1, card_features.shape[1])
        else:
            input_seq = torch.LongTensor([generated])
            input_features_list = [card_features[idx].unsqueeze(0) for idx in generated]
            input_features = torch.cat(input_features_list, dim=0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_seq, input_features)
            probs = torch.softmax(logits[0, -1], dim=-1)
        
        # Применение маски
        mask = rule_engine.create_mask(
            generated_cards=[generated] if generated else None,
            batch_size=1
        )[0]
        probs = probs * mask
        probs = probs / probs.sum()
        
        # Top-K сэмплирование
        top_probs, top_indices = torch.topk(probs, min(top_k, vocab_size))
        top_probs = top_probs / top_probs.sum()
        sampled = torch.multinomial(top_probs, 1).item()
        next_card = top_indices[sampled].item()
        
        generated.append(next_card)
        used_cards.add(next_card)
    
    return generated

# Генерация 5 колод
for i in range(5):
    print(f"\n=== Колода #{i+1} ===")
    deck_indices = generate_simple_deck(model, card_features, rule_engine, top_k=20)
    
    # Конвертация в названия
    cards = []
    for idx in deck_indices:
        card_id = idx_to_card.get(idx, idx)
        card_name = card_to_name.get(card_id, f"Unknown_{idx}")
        cards.append(card_name)
    
    print(f"Карты: {', '.join(cards)}")
    
    # Проверка валидности
    is_valid, msg = rule_engine.validate_generated_deck(deck_indices, idx_to_card)
    print(f"Валидна: {is_valid} ({msg})")

print("\n✅ Генерация завершена!")
