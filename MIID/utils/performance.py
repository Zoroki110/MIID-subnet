# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Yanez - MIID Team

"""
Performance Optimization Module for MIID Subnet

Provides performance optimization utilities including:
- LRU caching with TTL support
- Function execution timing
- Memory monitoring
- Batch processing utilities
"""

import functools
import time
import threading
from typing import Any, Callable, Dict, List, Optional
from collections import OrderedDict
import bittensor as bt


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
    """
    Decorator for LRU caching with optional TTL.
    
    Args:
        max_size: Maximum cache size
        ttl_seconds: Time to live for cached entries
    """
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
    """
    Decorator to time function execution.
    
    Args:
        name: Optional name for the timer (defaults to function name)
    """
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
    """
    Decorator to time async function execution.
    
    Args:
        name: Optional name for the timer (defaults to function name)
    """
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


# Global performance monitor instance
performance_monitor = PerformanceMonitor()