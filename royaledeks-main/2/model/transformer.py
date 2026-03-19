"""
Архитектура Transformer Decoder для генерации колод
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, MAX_SEQ_LEN
from utils.logger import logger


class MaskedSelfAttention(nn.Module):
    """
    Маскированный слой самовнимания
    
    Prevents attending to future positions (causal masking)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Инициализация слоя внимания
        
        Args:
            embedding_dim: Размерность эмбеддинга
            num_heads: Количество голов внимания
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        assert embedding_dim % num_heads == 0, "embedding_dim должен делиться на num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Линейные проекции для Q, K, V
        self.qkv_projection = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входной тензор [batch_size, seq_len, embedding_dim]
            mask: Маска для внимания [batch_size, 1, seq_len, seq_len] или None
            
        Returns:
            Выходной тензор [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V проекции
        qkv = self.qkv_projection(x)  # [B, S, 3E]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, head_dim]
        q, k, v = qkv.unbind(0)  # Каждый: [B, H, S, head_dim]
        
        # Вычисление внимания
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        
        # Применение маски
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax и dropout
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, S, S]
        attn_probs = self.dropout(attn_probs)
        
        # Применение внимания к value
        attn_output = torch.matmul(attn_probs, v)  # [B, H, S, head_dim]
        
        # Конкатенация голов и проекция
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embedding_dim)
        output = self.output_projection(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Полносвязный слой (Feed-Forward Network)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        """
        Инициализация FFN
        
        Args:
            embedding_dim: Размерность эмбеддинга
            ff_dim: Размерность скрытого слоя
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Один слой Transformer Decoder
    
    Состоит из:
    - Masked Self-Attention
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        """
        Инициализация слоя decoder
        
        Args:
            embedding_dim: Размерность эмбеддинга
            num_heads: Количество голов внимания
            ff_dim: Размерность скрытого слоя FFN
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        self.self_attn = MaskedSelfAttention(embedding_dim, num_heads, dropout)
        self.ffn = FeedForward(embedding_dim, ff_dim, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входной тензор [batch_size, seq_len, embedding_dim]
            mask: Маска для внимания
            
        Returns:
            Выходной тензор
        """
        # Self-attention с residual и norm
        attn_output = self.self_attn(self.layer_norm1(x), mask)
        x = x + attn_output
        
        # FFN с residual и norm
        ffn_output = self.ffn(self.layer_norm2(x))
        x = x + ffn_output
        
        return x


class TransformerDecoder(nn.Module):
    """
    Полная архитектура Transformer Decoder
    
    Состоит из N слоёв decoder с маскированным вниманием
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = None,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN
    ):
        """
        Инициализация Transformer Decoder
        
        Args:
            embedding_dim: Размерность эмбеддинга
            num_heads: Количество голов внимания
            num_layers: Количество слоёв decoder
            ff_dim: Размерность FFN (по умолчанию 4 * embedding_dim)
            dropout: Коэффициент dropout
            max_seq_len: Максимальная длина последовательности
        """
        super().__init__()
        
        if ff_dim is None:
            ff_dim = 4 * embedding_dim
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Слои decoder
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Финальная нормализация
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # causal mask для предотвращения заглядывания вперёд
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )
        
        logger.info(
            f"Transformer Decoder инициализирован: "
            f"embedding_dim={embedding_dim}, num_heads={num_heads}, "
            f"num_layers={num_layers}"
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            x: Входной тензор [batch_size, seq_len, embedding_dim]
            mask: Дополнительная маска (например, для паддинга)
            
        Returns:
            Выходной тензор [batch_size, seq_len, embedding_dim]
        """
        # Комбинирование causal mask с пользовательской маской
        combined_mask = self.causal_mask[:, :, :x.size(1), :x.size(1)]
        
        if mask is not None:
            combined_mask = combined_mask & mask
        
        # Проход через слои
        for layer in self.layers:
            x = layer(x, combined_mask)
        
        # Финальная нормализация
        x = self.layer_norm(x)
        
        return x


class DeckGeneratorModel(nn.Module):
    """
    Полная модель для генерации колод
    
    Архитектура:
    1. Card Embedding + Positional Encoding
    2. Transformer Decoder
    3. Output Layer (Linear + Softmax)
    """
    
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embedding_dim: int = EMBEDDING_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        max_seq_len: int = MAX_SEQ_LEN
    ):
        """
        Инициализация модели
        
        Args:
            vocab_size: Размер словаря карт
            feature_dim: Размерность вектора признаков карты
            embedding_dim: Размерность эмбеддинга
            num_heads: Количество голов внимания
            num_layers: Количество слоёв Transformer
            dropout: Коэффициент dropout
            max_seq_len: Максимальная длина последовательности
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Эмбеддинг карт
        self.card_embedding = CardEmbedding(
            vocab_size, feature_dim, embedding_dim, dropout
        )
        
        # Позиционное кодирование
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_seq_len, dropout
        )
        
        # Transformer Decoder
        self.transformer_decoder = TransformerDecoder(
            embedding_dim, num_heads, num_layers, dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Выходной слой для предсказания следующей карты
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Инициализация весов
        self._init_weights()
        
        logger.info(f"Модель DeckGenerator инициализирована: vocab_size={vocab_size}")
    
    def _init_weights(self):
        """Инициализация весов модели"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        card_indices: torch.Tensor,
        card_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            card_indices: Токены карт [batch_size, seq_len]
            card_features: Признаки карт [batch_size, seq_len, feature_dim]
            
        Returns:
            Логиты для каждой карты [batch_size, seq_len, vocab_size]
        """
        # Эмбеддинг + позиционное кодирование
        x = self.card_embedding(card_indices, card_features)
        x = self.positional_encoding(x)
        
        # Transformer Decoder
        x = self.transformer_decoder(x)
        
        # Выходной слой
        logits = self.output_layer(x)
        
        return logits
    
    def predict_step(
        self,
        card_indices: torch.Tensor,
        card_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Предсказание на одном шаге генерации
        
        Args:
            card_indices: Текущие токены карт [batch_size, current_seq_len]
            card_features: Признаки карт [batch_size, current_seq_len, feature_dim]
            mask: Маска для исключения карт [batch_size, vocab_size]
            
        Returns:
            Вероятности для каждой карты [batch_size, vocab_size]
        """
        # Получение логитов
        logits = self.forward(card_indices, card_features)
        
        # Берём логиты только для последней позиции
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Применение маски (если есть)
        if mask is not None:
            last_logits = last_logits.masked_fill(mask == 0, -1e9)
        
        # Softmax для получения вероятностей
        probs = F.softmax(last_logits, dim=-1)
        
        return probs
    
    @torch.no_grad()
    def generate(
        self,
        card_features: torch.Tensor,
        rule_engine: Optional['RuleEngine'] = None,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Генерация полной колоды (8 карт)

        Args:
            card_features: Признаки всех карт [vocab_size, feature_dim]
            rule_engine: RuleEngine для динамического маскирования
            temperature: Температура сэмплирования
            top_k: Параметр Top-K сэмплирования

        Returns:
            Сгенерированные индексы карт [batch_size, 8]
        """
        from rule_engine.rule_engine import RuleEngine

        # Определяем batch_size
        if card_features.dim() == 2:
            # [vocab_size, feature_dim] - генерируем 1 колоду
            batch_size = 1
            vocab_size, feature_dim = card_features.shape
        else:
            # [batch_size, vocab_size, feature_dim]
            batch_size = card_features.shape[0]
            vocab_size = card_features.shape[1]
            feature_dim = card_features.shape[2]
        
        # START токен
        START_TOKEN = self.vocab_size

        # Инициализация последовательности
        generated = torch.full(
            (batch_size, self.max_seq_len),
            START_TOKEN,
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        
        # Разворачиваем card_features для batch
        if card_features.dim() == 2:
            card_features = card_features.unsqueeze(0).expand(batch_size, -1, -1)  # [B, V, F]

        # Инициализация RuleEngine
        if rule_engine is None:
            rule_engine = RuleEngine(self.vocab_size)

        # Авторегрессионная генерация
        for step in range(self.max_seq_len):
            # Подготовка входа (все сгенерированные токены до текущего)
            input_seq = generated[:, :step + 1]  # [B, step+1]
            
            # Для каждой позиции в последовательности нужны признаки соответствующей карты
            # На позиции step мы предсказываем карту, используя признаки уже выбранных карт
            # Для START токена используем нулевые признаки
            
            # Создаём тензор признаков для входной последовательности
            # input_features shape: [B, step+1, feature_dim]
            input_features = []
            for b in range(batch_size):
                seq_features = []
                for i in range(step + 1):
                    token_id = generated[b, i].item()
                    if token_id == START_TOKEN or token_id >= vocab_size:
                        # START токен или ещё не выбранная карта - нулевые признаки
                        seq_features.append(torch.zeros(feature_dim, device=generated.device))
                    else:
                        # Берём признаки карты по индексу
                        seq_features.append(card_features[b, token_id])
                input_features.append(torch.stack(seq_features))
            input_features = torch.stack(input_features)  # [B, step+1, F]

            # Создание маски от RuleEngine
            # mask shape: [batch_size, vocab_size]
            if step > 0:
                # generated[:, :step] shape: [batch_size, step]
                # .tolist() returns List[List[int]]
                generated_cards = generated[:, :step].tolist()
            else:
                generated_cards = None
            
            mask = rule_engine.create_mask(
                generated_cards=generated_cards,
                batch_size=batch_size
            )

            # Предсказание следующей карты
            probs = self.predict_step(input_seq, input_features, mask)

            # Сэмплирование
            if temperature != 1.0:
                probs = probs ** (1.0 / temperature)

            next_card = self._sample_top_k(probs, top_k)

            # Добавление в последовательность
            generated[:, step] = next_card

        return generated
    
    def _sample_top_k(self, probs: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Top-K сэмплирование

        Args:
            probs: Вероятности [batch_size, vocab_size]
            top_k: Количество топ кандидатов

        Returns:
            Индексы выбранных карт [batch_size]
        """
        # Гарантируем что top_k — int
        top_k = int(top_k)
        
        # Ограничиваем top_k размером словаря
        vocab_size = probs.shape[-1]
        top_k = min(top_k, vocab_size)
        
        # Получение топ-K вероятностей
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        # Нормализация топ-K вероятностей
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        # Сэмплирование из топ-K
        sampled_indices = torch.multinomial(top_probs, num_samples=1)
        sampled_indices = torch.gather(top_indices, 1, sampled_indices).squeeze(-1)

        return sampled_indices


# Импорт CardEmbedding и PositionalEncoding для __all__
from .embeddings import CardEmbedding, PositionalEncoding

__all__ = [
    "MaskedSelfAttention",
    "FeedForward",
    "TransformerDecoderLayer",
    "TransformerDecoder",
    "DeckGeneratorModel",
    "CardEmbedding",
    "PositionalEncoding"
]
