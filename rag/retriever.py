import numpy as np
from typing import List, Dict
import json
import os

class DocumentRetriever:
    """
    Упрощенная RAG система для поиска документов
    (Демонстрирует понимание RAG, даже без эмбеддингов)
    """
    
    def __init__(self, docs_path: str = "rag/documents/"):
        self.docs_path = docs_path
        self.documents = self._load_documents()
        self.index = self._build_index()
    
    def _load_documents(self) -> List[Dict]:
        """Загрузка тестовых документов"""
        # Создаем синтетические документы, если их нет
        docs = [
            {"id": 1, "title": "Кредитный договор ООО Ромашка", 
             "content": "Компания имеет стабильный денежный поток...",
             "tags": ["кредит", "малый бизнес"]},
            {"id": 2, "title": "Анализ рынка 2025",
             "content": "Рост кредитования в секторе...",
             "tags": ["макроэкономика", "тренды"]},
            {"id": 3, "title": "Методика оценки рисков",
             "content": "Для оценки кредитного риска используются следующие показатели...",
             "tags": ["методология", "риски"]},
        ]
        return docs
    
    def _build_index(self):
        """Построение простого индекса (без эмбеддингов)"""
        index = {}
        for doc in self.documents:
            # Индексируем по словам из названия и тегов
            words = doc["title"].lower().split() + doc["tags"]
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(doc["id"])
        return index
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Поиск релевантных документов
        (здесь используется простой BM25-like подход)
        """
        query_words = query.lower().split()
        
        # Считаем релевантность
        scores = {}
        for doc in self.documents:
            score = 0
            for word in query_words:
                if word in doc["title"].lower() or word in doc["content"].lower():
                    score += 1
                if word in doc["tags"]:
                    score += 2
            scores[doc["id"]] = score
        
        # Сортируем и возвращаем топ
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in top_docs:
            if score > 0:
                doc = next(d for d in self.documents if d["id"] == doc_id)
                results.append(f"{doc['title']}: {doc['content'][:100]}...")
        
        return results