# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Yanez - MIID Team

"""
Error Handling Module for MIID Subnet

Provides comprehensive error handling utilities including:
- Custom exception classes
- Error handling decorators
- Retry mechanisms
- Safe execution utilities
"""

import functools
import time
import traceback
import asyncio
from typing import Any, Callable, Optional, Type, Union
import bittensor as bt


class MIIDError(Exception):
    """Base exception class for MIID subnet errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class MIIDValidationError(MIIDError):
    """Exception raised for validation-related errors."""
    pass


class MIIDMinerError(MIIDError):
    """Exception raised for miner-related errors."""
    pass


class MIIDNetworkError(MIIDError):
    """Exception raised for network-related errors."""
    pass


class MIIDConfigurationError(MIIDError):
    """Exception raised for configuration-related errors."""
    pass


class MIIDSecurityError(MIIDError):
    """Exception raised for security-related errors."""
    pass


def with_error_handling(
    exception_type: Type[Exception] = Exception,
    default_return: Any = None,
    log_errors: bool = True
):
    """
    Decorator to add error handling to functions.
    
    Args:
        exception_type: Type of exception to catch
        default_return: Default value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                if log_errors:
                    bt.logging.error(f"Error in {func.__name__}: {str(e)}")
                    bt.logging.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


def with_async_error_handling(
    exception_type: Type[Exception] = Exception,
    default_return: Any = None,
    log_errors: bool = True
):
    """
    Decorator to add error handling to async functions.
    
    Args:
        exception_type: Type of exception to catch
        default_return: Default value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                if log_errors:
                    bt.logging.error(f"Error in {func.__name__}: {str(e)}")
                    bt.logging.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Factor to multiply delay by after each failure
        exceptions: Exception types to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        bt.logging.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    bt.logging.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    
        return wrapper
    return decorator


def async_retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    Decorator to retry async function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Factor to multiply delay by after each failure
        exceptions: Exception types to catch and retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        bt.logging.error(f"Async function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    bt.logging.warning(
                        f"Async function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
                    
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, log_errors: bool = True, **kwargs):
    """
    Safely execute a function and return a default value on error.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            bt.logging.error(f"Error executing {func.__name__}: {str(e)}")
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")
        return default_return


async def safe_async_execute(func: Callable, *args, default_return: Any = None, log_errors: bool = True, **kwargs):
    """
    Safely execute an async function and return a default value on error.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        default_return: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            bt.logging.error(f"Error executing async {func.__name__}: {str(e)}")
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")
        return default_return


class ErrorContext:
    """Context manager for enhanced error handling and logging."""
    
    def __init__(self, operation_name: str, log_success: bool = True):
        self.operation_name = operation_name
        self.log_success = log_success
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        bt.logging.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            if self.log_success:
                bt.logging.info(f"Operation '{self.operation_name}' completed successfully in {duration:.2f}s")
        else:
            bt.logging.error(f"Operation '{self.operation_name}' failed after {duration:.2f}s: {exc_val}")
            bt.logging.debug(f"Traceback: {traceback.format_exc()}")
        
        return False  # Don't suppress exceptions