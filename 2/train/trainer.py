"""
Модуль обучения модели
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, MODELS_DIR, WEIGHT_DECAY, GRAD_CLIP, WARMUP_STEPS
)
from utils.logger import logger


class Trainer:
    """
    Тренер для обучения модели генерации колод
    
    Использует:
    - Cross-Entropy Loss с маскированием
    - Teacher Forcing
    - Early Stopping
    - Learning Rate Scheduling (Cosine Annealing with Warmup)
    - Gradient Clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = LEARNING_RATE,
        device: Optional[str] = None
    ):
        """
        Инициализация тренера
        
        Args:
            model: Модель для обучения
            learning_rate: Начальная скорость обучения
            device: Устройство (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Loss функция
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Оптимизатор с weight decay для регуляризации
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=WEIGHT_DECAY
        )
        
        # Scheduler - Cosine Annealing с теплым стартом
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Early stopping параметры
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        
        # История обучения
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []
        
        # Параметры warmup
        self.warmup_steps = WARMUP_STEPS
        self.grad_clip = GRAD_CLIP
        
        logger.info(f"Trainer инициализирован: device={self.device}, lr={learning_rate}, warmup_steps={self.warmup_steps}")
    
    def train_epoch(
        self,
        dataloader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Обучение за одну эпоху
        
        Args:
            dataloader: DataLoader для train данных
            epoch: Номер эпохи
            
        Returns:
            (average_loss, perplexity)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (input_seq, target_seq, features) in enumerate(pbar):
            # Перемещение на устройство
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            features = features.to(self.device)
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(input_seq, features)
            
            # Reshape для loss
            # logits: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
            # target: [batch, seq_len] -> [batch * seq_len]
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            target_flat = target_seq.view(-1)
            
            # Вычисление loss
            loss = self.criterion(logits_flat, target_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping с настраиваемым порогом
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            
            # Шаг оптимизатора
            self.optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            num_batches += 1
            
            # Обновление прогресс-бара
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def validate(
        self,
        dataloader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Валидация модели

        Args:
            dataloader: DataLoader для validation данных
            epoch: Номер эпохи

        Returns:
            (average_loss, perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Проверка на пустой dataloader
        if len(dataloader) == 0:
            logger.warning("Пустой validation dataloader, пропускаем валидацию")
            return 0.0, 1.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

        with torch.no_grad():
            for batch_idx, (input_seq, target_seq, features) in enumerate(pbar):
                # Перемещение на устройство
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                features = features.to(self.device)

                # Forward pass
                logits = self.model(input_seq, features)

                # Reshape для loss
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)
                target_flat = target_seq.view(-1)

                # Вычисление loss
                loss = self.criterion(logits_flat, target_flat)

                # Статистика
                total_loss += loss.item()
                num_batches += 1

                # Обновление прогресс-бара
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if num_batches == 0:
            logger.warning("Не было валидационных батчей")
            return 0.0, 1.0

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity
    
    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = NUM_EPOCHS,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Полный цикл обучения
        
        Args:
            train_loader: DataLoader для train
            val_loader: DataLoader для validation
            num_epochs: Количество эпох
            save_path: Путь для сохранения лучшей модели
            
        Returns:
            История обучения
        """
        if save_path is None:
            save_path = os.path.join(MODELS_DIR, "best_model.pt")
        
        logger.info(f"Начало обучения: {num_epochs} эпох")
        logger.info(f"Сохранение лучшей модели в: {save_path}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_perplexity = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_perplexity = self.validate(val_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Логирование
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train PPL: {train_perplexity:.2f} | "
                f"Val Loss: {val_loss:.4f} | Val PPL: {val_perplexity:.2f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Сохранение истории
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'perplexity': train_perplexity
            })
            self.val_history.append({
                'epoch': epoch + 1,
                'loss': val_loss,
                'perplexity': val_perplexity
            })
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Early Stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Сохранение лучшей модели
                self.save_checkpoint(save_path, epoch + 1, val_loss)
                logger.info(f"✓ Новая лучшая модель сохранена (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"Early stopping counter: {self.patience_counter}/{self.early_stopping_patience}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Ранняя остановка на эпохе {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Обучение завершено за {total_time:.1f}s")
        logger.info(f"Лучший val_loss: {self.best_val_loss:.4f}")
        
        return {
            'train_loss': [h['loss'] for h in self.train_history],
            'val_loss': [h['loss'] for h in self.val_history],
            'train_perplexity': [h['perplexity'] for h in self.train_history],
            'val_perplexity': [h['perplexity'] for h in self.val_history]
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float
    ) -> None:
        """
        Сохранение чекпоинта модели
        
        Args:
            path: Путь для сохранения
            epoch: Текущая эпоха
            val_loss: Текущий validation loss
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Чекпоинт сохранён: {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Загрузка чекпоинта модели
        
        Args:
            path: Путь к чекпоинту
            
        Returns:
            Данные чекпоинта
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        logger.info(f"Чекпоинт загружен: {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint
    
    def get_model(self) -> nn.Module:
        """Получение модели"""
        return self.model
