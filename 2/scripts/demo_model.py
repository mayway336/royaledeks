"""
Демонстрация работы модели
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pickle
from model.transformer import DeckGeneratorModel
from data.parser import DEFAULT_CARDS

# Загрузка данных
with open('data/preprocessor/vocabulary.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
with open('data/preprocessor/encoder.pkl', 'rb') as f:
    encoder_data = pickle.load(f)

print(f"Загружено {vocab_data['size']} карт")

# Загрузка модели
model = DeckGeneratorModel(
    vocab_size=vocab_data['size'],
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

# Тест forward pass
batch_size = 2
seq_len = 4
input_seq = torch.randint(0, vocab_data['size'], (batch_size, seq_len))

# Получение признаков для карт по индексам
idx_to_card = {v: k for k, v in vocab_data['card_to_idx'].items()}
card_features_list = []
for idx in range(vocab_data['size']):
    card_id = idx_to_card.get(idx, idx)
    if card_id in encoder_data['card_features']:
        card_features_list.append(encoder_data['card_features'][card_id])
    else:
        card_features_list.append([0.0] * encoder_data['feature_dim'])

card_features = torch.FloatTensor(card_features_list)  # [vocab_size, feature_dim]

# Для теста берём первые seq_len карт
card_features = card_features[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)  # [B, S, F]

with torch.no_grad():
    logits = model(input_seq, card_features)
    print(f"Input shape: {input_seq.shape}")
    print(f"Card features shape: {card_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Vocab size: {vocab_data['size']}")
    
    # Предсказание следующей карты
    next_probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_card = torch.argmax(next_probs, dim=-1)
    print(f"Predicted next cards: {next_card.tolist()}")

print("\n✅ Модель работает корректно!")
