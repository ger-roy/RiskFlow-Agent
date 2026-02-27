# Agent  | Мультиагентная система для анализа кредитных рисков

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20-green)](https://github.com/langchain-ai/langgraph)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

##  О проекте

**RiskFlow Agent** — учебный проект, созданный для демонстрации компетенций в области современных ML/NLP технологий. 
Представляет собой мультиагентную систему для автоматизации анализа кредитных рисков с использованием LLM, RAG и классического ML.


##  Архитектура системы
                Архитектура системы

```mermaid
graph TD
    A[LangGraph Orchestrator] --> B[Research Agent]
    A --> C[Analysis Agent]
    A --> D[Report Generator]
    
    B <--> C
    C <--> D
    
    B --> E[RAG System<br/>Document Search]
    C --> F[LoRA-tuned LLM<br/>Financial Expert]
    F --> D
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:1px
    style C fill:#bbf,stroke:#333,stroke-width:1px
    style D fill:#bbf,stroke:#333,stroke-width:1px
    style E fill:#bfb,stroke:#333,stroke-width:1px
    style F fill:#bfb,stroke:#333,stroke-width:1px
---

##  Ключевые компоненты

### 1. Мультиагентная система на LangGraph
Реализована оркестрация трех специализированных агентов с состоянием и условными переходами.

```python
# Пример из кода
workflow.add_node("retrieve_docs", self._retrieve_documents)
workflow.add_node("research", self.research_agent.run)
workflow.add_node("analyze", self.analysis_agent.run)
workflow.add_conditional_edges("analyze", self._should_continue, {...})
