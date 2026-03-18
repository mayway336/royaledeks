"""
Модуль эмбеддингов карт
"""
import torch
import torch.nn as nn


class CardEmbedding(nn.Module):
    """
    Эмбеддинг карт с добавлением признаков
    
    Комбинирует:
    - Обучаемый вектор карты (embedding)
    - Вектор признаков (эликсир, тип, редкость, флаги)
    """
    
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embedding_dim: int,
        dropout: float = 0.1
    ):
        """
        Инициализация эмбеддинга
        
        Args:
            vocab_size: Размер словаря карт
            feature_dim: Размерность вектора признаков
            embedding_dim: Размерность выходного эмбеддинга
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        # Обучаемый эмбеддинг карты
        self.token_embedding = nn.Embedding(vocab_size + 1, embedding_dim)  # +1 для START токена
        
        # Проекция признаков
        self.feature_projection = nn.Linear(feature_dim, embedding_dim)
        
        # Комбинирование и финальная проекция
        self.combined_projection = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, card_indices: torch.Tensor, card_features: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход
        
        Args:
            card_indices: Токены карт [batch_size, seq_len]
            card_features: Признаки карт [batch_size, seq_len, feature_dim]
            
        Returns:
            Эмбеддинги [batch_size, seq_len, embedding_dim]
        """
        # Токен эмбеддинг
        token_emb = self.token_embedding(card_indices)  # [B, S, E]
        
        # Проекция признаков
        feature_emb = self.feature_projection(card_features)  # [B, S, E]
        
        # Конкатенация и проекция
        combined = torch.cat([token_emb, feature_emb], dim=-1)  # [B, S, 2E]
        output = self.combined_projection(combined)  # [B, S, E]
        
        # Dropout и нормализация
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Позиционное кодирование
    
    Добавляет информацию о позиции карты в последовательности
    """
    
    def __init__(self, embedding_dim: int, max_len: int = 8, dropout: float = 0.1):
        """
        Инициализация позиционного кодирования
        
        Args:
            embedding_dim: Размерность эмбеддинга
            max_len: Максимальная длина последовательности
            dropout: Коэффициент dropout
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        
        # Создание позиционных кодирований
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(6.93 / embedding_dim))
        
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Регистрация буфера (не участвует в градиентах)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, E]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Добавление позиционного кодирования
        
        Args:
            x: Входной тензор [batch_size, seq_len, embedding_dim]
            
        Returns:
            Тензор с позиционным кодированием
        """
        # x shape: [B, S, E]
        # pe shape: [1, max_len, E]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
