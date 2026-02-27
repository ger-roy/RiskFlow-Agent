import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Callable, Any, Dict
import queue

class AsyncRunner:
    """
    Демонстрация работы с asyncio, threading, multiprocessing
    (Прямое выполнение требований из вакансии)
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
    
    async def run_async_tasks(self, tasks: List[Callable]) -> List[Any]:
        """
        Asyncio: для I/O-bound задач
        (ожидание ответов от API, запросы к БД)
        """
        print(f"Запуск {len(tasks)} асинхронных задач...")
        start = time.time()
        
        # Создаем корутины
        coroutines = [task() if asyncio.iscoroutinefunction(task) 
                      else asyncio.to_thread(task) for task in tasks]
        
        # Запускаем параллельно
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        elapsed = time.time() - start
        print(f"Асинхронное выполнение заняло: {elapsed:.2f}с")
        
        return results
    
    def run_threading_tasks(self, tasks: List[Callable]) -> List[Any]:
        """
        Threading: для параллельного выполнения I/O операций
        (когда asyncio не подходит)
        """
        print(f"Запуск {len(tasks)} задач в потоках...")
        start = time.time()
        
        # Запускаем в thread pool
        futures = [self.thread_pool.submit(task) for task in tasks]
        results = [f.result() for f in futures]
        
        elapsed = time.time() - start
        print(f"Многопоточное выполнение заняло: {elapsed:.2f}с")
        
        return results
    
    def run_multiprocessing_tasks(self, tasks: List[Callable]) -> List[Any]:
        """
        Multiprocessing: для CPU-bound задач
        (обучение моделей, обработка больших данных)
        """
        print(f"Запуск {len(tasks)} задач в процессах...")
        start = time.time()
        
        # Запускаем в process pool
        futures = [self.process_pool.submit(task) for task in tasks]
        results = [f.result() for f in futures]
        
        elapsed = time.time() - start
        print(f"Многопроцессное выполнение заняло: {elapsed:.2f}с")
        
        return results
    
    def demonstrate_differences(self):
        """
        Демонстрация понимания когда что использовать
        (важно для собеседования)
        """
        demo = {
            "asyncio": {
                "when": "I/O bound, много соединений",
                "example": "Одновременные запросы к 100 API",
                "why": "Один поток, но не блокируется"
            },
            "threading": {
                "when": "I/O bound, блокирующие библиотеки",
                "example": "Работа с файловой системой",
                "why": "Несколько потоков, разделяют память"
            },
            "multiprocessing": {
                "when": "CPU bound, вычисления",
                "example": "Обучение нескольких моделей",
                "why": "Отдельные процессы, обходят GIL"
            }
        }
        return demo

# Примеры задач для демонстрации
def io_bound_task(task_id: int):
    """Имитация I/O операции"""
    time.sleep(1)  # вместо реального запроса
    return f"Task {task_id} completed"

async def async_io_task(task_id: int):
    """Асинхронная I/O операция"""
    await asyncio.sleep(1)  # не блокирует
    return f"Async task {task_id} completed"

def cpu_bound_task(task_id: int):
    """Имитация CPU вычислений"""
    result = 0
    for i in range(10**7):  # нагружаем CPU
        result += i ** 0.5
    return f"CPU task {task_id} finished"

# Демонстрация
async def main():
    runner = AsyncRunner()
    
    print("="*50)
    print("ДЕМОНСТРАЦИЯ ПАРАЛЛЕЛЬНОГО ВЫПОЛНЕНИЯ")
    print("="*50)
    
    # Asyncio демо
    async_tasks = [lambda i=i: async_io_task(i) for i in range(5)]
    async_results = await runner.run_async_tasks(async_tasks)
    
    # Threading демо
    thread_tasks = [lambda i=i: io_bound_task(i) for i in range(5)]
    thread_results = runner.run_threading_tasks(thread_tasks)
    
    # Multiprocessing демо (опционально, может быть долго)
    # cpu_tasks = [lambda i=i: cpu_bound_task(i) for i in range(2)]
    # cpu_results = runner.run_multiprocessing_tasks(cpu_tasks)
    
    print("\n" + "="*50)
    print("КОГДА ЧТО ИСПОЛЬЗОВАТЬ:")
    for method, info in runner.demonstrate_differences().items():
        print(f"\n{method}:")
        for k, v in info.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())