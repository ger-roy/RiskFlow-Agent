# Agent  | Мультиагентная система для анализа кредитных рисков

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20-green)](https://github.com/langchain-ai/langgraph)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

##  О проекте

**RiskFlow Agent** — учебный проект, созданный для демонстрации компетенций в области современных ML/NLP технологий. 
Представляет собой мультиагентную систему для автоматизации анализа кредитных рисков с использованием LLM, RAG и классического ML.


##  Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                    │
│                         (main.py)                            │
│              Управляет потоком данных между агентами         │
└───────────┬───────────────────────────────┬─────────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────┐      ┌───────────────────────┐
│   Research Pipeline   │      │   Analysis Pipeline    │
├───────────────────────┤      ├───────────────────────┤
│ 1. Analyst Agent      │      │ 1. Risk Agent         │
│    (analyst_agent.py) │      │    (risk_agent.py)    │
│    • Поиск данных     │      │    • Генерация отчёта │
│    • Сбор информации  │      │    • Вызов LLM        │
│                       │      │                       │
│ 2. RAG System         │      │ 2. Critic Agent       │
│    (utils/data_loader)│      │    (critic_agent.py)  │
│    • Загрузка данных  │      │    • Проверка отчёта  │
│    • Поиск контекста  │      │    • Валидация        │
└───────────┬───────────┘      └───────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
            ┌─────────────────────────────────┐
            │         ML & LLM Models         │
            │           (models/)              │
            ├─────────────────────────────────┤
            │ • LoRA fine-tuning               │
            │   (models/lora_adapter.py)       │
            │ • Credit Scoring Model           │
            │   (models/scoring_model.pkl)     │
            └────────────────┬─────────────────┘
                             │
                            ▼
            ┌─────────────────────────────────┐
            │      Parallel Processing        │
            │           (utils/)               │
            ├─────────────────────────────────┤
            │ • Async data loading             │
            │ • Threading for I/O              │
            │   (utils/parallel_runner.py)     │
            └─────────────────────────────────┘
```

### Компоненты архитектуры:

| Компонент | Файл | Назначение |
|-----------|------|------------|
| **Orchestrator** | `main.py` | Координирует работу агентов через LangGraph |
| **Analyst Agent** | `agents/analyst_agent.py` | Собирает и структурирует данные о компании |
| **Risk Agent** | `agents/risk_agent.py` | Генерирует отчет с использованием LLM |
| **Critic Agent** | `agents/critic_agent.py` | Проверяет отчет на корректность |
| **RAG System** | `utils/data_loader.py` | Загружает и индексирует документы |
| **LoRA Adapter** | `models/lora_adapter.py` | Адаптация LLM под финансовый домен |
| **Scoring Model** | `models/scoring_model.pkl` | Классическая ML модель для скоринга |
| **Parallel Runner** | `utils/parallel_runner.py` | Asyncio + threading оптимизация |

### Поток данных:

1. **Analyst Agent** получает запрос и собирает данные о компании
2. **RAG System** загружает релевантные документы через `data_loader`
3. **Risk Agent** вызывает **LoRA-tuned LLM** для генерации отчета
4. **Scoring Model** вычисляет кредитный скоринг
5. **Critic Agent** проверяет отчет и отправляет на доработку или финал
6. **Parallel Runner** оптимизирует все I/O операции через asyncio
---

