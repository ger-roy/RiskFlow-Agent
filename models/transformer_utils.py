import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np

class TransformerNLP:
    """
    Работа с transformer-based моделями для NLP задач
    (Показывает понимание архитектуры трансформеров)
    """
    
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        """
        Используем маленькую ruBERT для демонстрации
        """
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # режим инференса
        
        print(f"Загружена трансформер модель: {model_name}")
        print(f"Размер эмбеддингов: {self.model.config.hidden_size}")
        print(f"Количество слоев: {self.model.config.num_hidden_layers}")
        print(f"Количество голов внимания: {self.model.config.num_attention_heads}")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Получение эмбеддингов текста
        """
        # Токенизация
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Прогон через модель
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # [CLS] токен как представление всего текста
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings[0]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Вычисление семантической близости текстов
        """
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        
        # Косинусная близость
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
    
    def explain_architecture(self):
        """
        Метод для демонстрации понимания архитектуры
        (на собеседовании можно рассказать)
        """
        explanation = {
            "input": "Токенизация -> эмбеддинги -> позиционные кодировки",
            "attention": "Multi-head self-attention для учета контекста",
            "encoder": "Стеки из feed-forward + attention + residual connections",
            "output": "Контекстуализированные эмбеддинги токенов"
        }
        return explanation

# Пример использования
def demo_transformer():
    nlp = TransformerNLP()
    
    # Тестовые тексты
    texts = [
        "Компания показывает стабильный рост прибыли",
        "Прибыль компании растет стабильными темпами",
        "Компания обанкротилась"
    ]
    
    print("\nСемантическая близость:")
    sim1 = nlp.compute_similarity(texts[0], texts[1])
    sim2 = nlp.compute_similarity(texts[0], texts[2])
    
    print(f"Текст 1 и 2: {sim1:.3f} (похожие)")
    print(f"Текст 1 и 3: {sim2:.3f} (разные)")
    
    print("\nАрхитектура трансформера:")
    for k, v in nlp.explain_architecture().items():
        print(f"{k}: {v}")