# Agent 🤖 | Мультиагентная система для анализа кредитных рисков

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20-green)](https://github.com/langchain-ai/langgraph)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📌 О проекте

**RiskFlow Agent** — учебный проект, созданный для демонстрации компетенций в области современных ML/NLP технологий. 
Представляет собой мультиагентную систему для автоматизации анализа кредитных рисков с использованием LLM, RAG и классического ML.


## 🏗 Архитектура системы
                ┌─────────────────────────────────────┐
                │         LangGraph Orchestrator      │
                │  (Управляет потоком между агентами) │
                └───────────────┬─────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    ▼                           ▼                           ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Research Agent│◄────────►│ Analysis Agent│◄────────►│ Report Gen │
│ (Поиск инфо) │ │ (Оценка рисков│ │ (Формирование │
└───────┬───────┘ └───────┬───────┘ │ отчета) │
│ │ └───────────────┘
▼ ▼ ▲
┌───────────────┐ ┌───────────────┐ │
│ RAG System │ │ LoRA-tuned │───────────────────┘
│(Поиск по базе │ │ LLM │
│ документов) │ │(Финансовый │
└───────────────┘ │ экспертный │
│ домен) │
└───────────────┘

---

## ✨ Ключевые компоненты

### 1. 🦜 Мультиагентная система на LangGraph
Реализована оркестрация трех специализированных агентов с состоянием и условными переходами.

```python
# Пример из кода
workflow.add_node("retrieve_docs", self._retrieve_documents)
workflow.add_node("research", self.research_agent.run)
workflow.add_node("analyze", self.analysis_agent.run)
workflow.add_conditional_edges("analyze", self._should_continue, {...})