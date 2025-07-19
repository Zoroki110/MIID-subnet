# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Yanez - MIID Team

"""
Performance Optimization Module

This module provides performance optimization utilities including:
- Caching mechanisms
- Batch processing utilities
- Performance monitoring
- Memory management
- Async optimization helpers
"""

import asyncio
import functools
import time
import threading
import gc
from typing import Any, Callable, Dict, List, Optional, Union
from collections import OrderedDict
import bittensor as bt
import psutil
import weakref


class LRUCache:
    """Thread-safe LRU Cache implementation with size and time limits."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def get(self, key: Any) -> Any:
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self.ttl_seconds and key in self.timestamps:
                if time.time() - self.timestamps[key] > self.ttl_seconds:
                    self._remove_key(key)
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def put(self, key: Any, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                self._remove_key(oldest_key)
            
            self.cache[key] = value
            if self.ttl_seconds:
                self.timestamps[key] = time.time()
    
    def _remove_key(self, key: Any) -> None:
        """Remove key from cache and timestamps."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)


def lru_cache_with_ttl(max_size: int = 128, ttl_seconds: Optional[float] = None):
    """Decorator for LRU caching with optional TTL."""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size, ttl_seconds)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: {"size": cache.size(), "max_size": cache.max_size}
        
        return wrapper
    return decorator


class BatchProcessor:
    """Utility for batch processing with configurable batch size and timeout."""
    
    def __init__(self, 
                 batch_size: int = 10, 
                 timeout_seconds: float = 1.0,
                 processor_func: Optional[Callable] = None):
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.processor_func = processor_func
        self.batch = []
        self.last_process_time = time.time()
        self.lock = threading.Lock()
    
    def add_item(self, item: Any) -> List[Any]:
        """Add item to batch and process if conditions are met."""
        with self.lock:
            self.batch.append(item)
            
            should_process = (
                len(self.batch) >= self.batch_size or
                time.time() - self.last_process_time >= self.timeout_seconds
            )
            
            if should_process:
                return self._process_batch()
            
            return []
    
    def flush(self) -> List[Any]:
        """Process remaining items in batch."""
        with self.lock:
            if self.batch:
                return self._process_batch()
            return []
    
    def _process_batch(self) -> List[Any]:
        """Process current batch and return results."""
        if not self.batch:
            return []
        
        current_batch = self.batch.copy()
        self.batch.clear()
        self.last_process_time = time.time()
        
        if self.processor_func:
            try:
                return self.processor_func(current_batch)
            except Exception as e:
                bt.logging.error(f"Error processing batch: {e}")
                return []
        
        return current_batch


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        with self.lock:
            self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a performance timer and return duration."""
        with self.lock:
            if name not in self.start_times:
                bt.logging.warning(f"Timer '{name}' was not started")
                return 0.0
            
            duration = time.time() - self.start_times.pop(name)
            
            if name not in self.metrics:
                self.metrics[name] = []
            
            self.metrics[name].append(duration)
            return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = self.metrics[name]
            return {
                'count': len(values),
                'total': sum(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        with self.lock:
            return {name: self.get_stats(name) for name in self.metrics}
    
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self.start_times.clear()


def performance_timer(name: Optional[str] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable) -> Callable:
        timer_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                bt.logging.debug(f"Function '{timer_name}' took {duration:.4f} seconds")
        
        return wrapper
    return decorator


def async_performance_timer(name: Optional[str] = None):
    """Decorator to time async function execution."""
    def decorator(func: Callable) -> Callable:
        timer_name = name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                bt.logging.debug(f"Async function '{timer_name}' took {duration:.4f} seconds")
        
        return wrapper
    return decorator


class MemoryManager:
    """Memory management utilities."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident set size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    @staticmethod
    def force_garbage_collection() -> int:
        """Force garbage collection and return number of objects collected."""
        return gc.collect()
    
    @staticmethod
    def memory_monitor(threshold_mb: float = 500.0):
        """Decorator to monitor memory usage and warn if threshold exceeded."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                initial_memory = MemoryManager.get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    final_memory = MemoryManager.get_memory_usage()
                    memory_increase = final_memory['rss_mb'] - initial_memory['rss_mb']
                    
                    if memory_increase > threshold_mb:
                        bt.logging.warning(
                            f"Function '{func.__name__}' increased memory usage by {memory_increase:.2f} MB "
                            f"(threshold: {threshold_mb} MB)"
                        )
            
            return wrapper
        return decorator


class AsyncSemaphoreManager:
    """Manage concurrent operations with semaphores."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_operations = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire semaphore and track active operations."""
        await self.semaphore.acquire()
        async with self.lock:
            self.active_operations += 1
    
    def release(self):
        """Release semaphore and update active operations count."""
        self.semaphore.release()
        asyncio.create_task(self._decrement_operations())
    
    async def _decrement_operations(self):
        async with self.lock:
            self.active_operations -= 1
    
    async def get_active_count(self) -> int:
        """Get number of active operations."""
        async with self.lock:
            return self.active_operations


def throttle_concurrent_calls(max_concurrent: int = 10):
    """Decorator to limit concurrent async function calls."""
    semaphore_manager = AsyncSemaphoreManager(max_concurrent)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await semaphore_manager.acquire()
            try:
                return await func(*args, **kwargs)
            finally:
                semaphore_manager.release()
        
        return wrapper
    return decorator


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def optimize_similarity_calculations():
    """Optimization suggestions for similarity calculations."""
    return {
        "use_caching": "Cache phonetic algorithm results for repeated names",
        "batch_processing": "Process multiple names simultaneously",
        "early_termination": "Stop calculation if similarity threshold is met",
        "algorithm_selection": "Use faster algorithms for initial filtering",
        "vectorization": "Use numpy operations where possible"
    }