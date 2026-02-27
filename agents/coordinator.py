from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from rag.retriever import DocumentRetriever

class AgentState(TypedDict):
    company_name: str
    query: str
    retrieved_docs: List[str]
    risk_analysis: str
    final_report: str
    status: str

class AgentCoordinator:
    """
    Оркестратор мультиагентной системы на LangGraph
    """
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.retriever = DocumentRetriever()
        
        # Строим граф
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Построение графа взаимодействия агентов"""
        workflow = StateGraph(AgentState)
        
        # Определяем узлы
        workflow.add_node("retrieve_docs", self._retrieve_documents)
        workflow.add_node("research", self.research_agent.run)
        workflow.add_node("analyze", self.analysis_agent.run)
        workflow.add_node("generate_report", self._generate_report)
        
        # Строим связи
        workflow.set_entry_point("retrieve_docs")
        workflow.add_edge("retrieve_docs", "research")
        workflow.add_edge("research", "analyze")
        workflow.add_edge("analyze", "generate_report")
        workflow.add_conditional_edges(
            "generate_report",
            self._should_continue,
            {"continue": "research", "end": END}
        )
        
        return workflow.compile()
    
    def _retrieve_documents(self, state: AgentState):
        """Поиск релевантных документов (RAG)"""
        docs = self.retriever.search(state["query"], top_k=3)
        state["retrieved_docs"] = docs
        return state
    
    def _generate_report(self, state: AgentState):
        """Финальная генерация отчета"""
        state["final_report"] = f"""
        Отчет по компании {state['company_name']}:
        
        Найденные документы: {len(state['retrieved_docs'])}
        Анализ рисков: {state['risk_analysis']}
        
        Статус: Завершено
        """
        state["status"] = "completed"
        return state
    
    def _should_continue(self, state: AgentState):
        """Условие для цикла"""
        if state["status"] == "completed":
            return "end"
        return "continue"
    
    async def run(self, company: str, query: str):
        """Запуск агента (с поддержкой asyncio)"""
        initial_state = {
            "company_name": company,
            "query": query,
            "retrieved_docs": [],
            "risk_analysis": "",
            "final_report": "",
            "status": "started"
        }
        
        # Асинхронный запуск графа
        result = await self.graph.arun(initial_state)
        return result