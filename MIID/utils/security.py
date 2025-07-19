# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Yanez - MIID Team

"""
Security Enhancement Module

This module provides security utilities for the MIID subnet including:
- Input validation and sanitization
- Rate limiting
- API key management
- Request authentication
- Anti-abuse measures
"""

import re
import time
import hashlib
import secrets
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Union
from collections import defaultdict, deque
import bittensor as bt
from datetime import datetime, timedelta

from MIID.utils.error_handling import MIIDSecurityError, with_error_handling


class InputValidator:
    """Comprehensive input validation for MIID subnet."""
    
    # Safe name patterns (letters, spaces, hyphens, apostrophes)
    SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z\s\-'\.]{1,100}$")
    
    # SQL injection patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"union\s+select", r"drop\s+table", r"delete\s+from",
        r"insert\s+into", r"update\s+set", r"exec\s*\(",
        r"<script", r"javascript:", r"onclick\s*="
    ]
    
    @classmethod
    def validate_name(cls, name: str) -> bool:
        """
        Validate that a name contains only safe characters.
        
        Args:
            name: Name string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(name, str):
            return False
        
        if not name or not name.strip():
            return False
        
        # Check length
        if len(name.strip()) > 100:
            return False
        
        # Check pattern
        return bool(cls.SAFE_NAME_PATTERN.match(name.strip()))
    
    @classmethod
    def validate_names_list(cls, names: List[str], max_count: int = 50) -> bool:
        """
        Validate a list of names.
        
        Args:
            names: List of name strings
            max_count: Maximum allowed number of names
            
        Returns:
            True if all names are valid, False otherwise
        """
        if not isinstance(names, list):
            return False
        
        if len(names) == 0 or len(names) > max_count:
            return False
        
        return all(cls.validate_name(name) for name in names)
    
    @classmethod
    def validate_query_template(cls, template: str) -> bool:
        """
        Validate query template for safety.
        
        Args:
            template: Query template string
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(template, str):
            return False
        
        if not template or not template.strip():
            return False
        
        # Check length
        if len(template) > 5000:  # Reasonable template length limit
            return False
        
        # Check for malicious patterns
        template_lower = template.lower()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, template_lower):
                return False
        
        # Must contain {name} placeholder
        if "{name}" not in template:
            return False
        
        # Should not contain multiple {name} placeholders
        if template.count("{name}") > 1:
            return False
        
        return True
    
    @classmethod
    def sanitize_string(cls, text: str, max_length: int = 1000) -> str:
        """
        Sanitize string by removing potentially dangerous characters.
        
        Args:
            text: Text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
        
        # Truncate if too long
        text = text[:max_length]
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Remove potential script tags and javascript
        text = re.sub(r'<script.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        return text.strip()


class RateLimiter:
    """Rate limiting implementation with sliding window."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for given identifier.
        
        Args:
            identifier: Unique identifier (e.g., IP address, hotkey)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        
        with self.lock:
            # Clean old requests outside the window
            cutoff_time = current_time - self.window_seconds
            while (self.requests[identifier] and 
                   self.requests[identifier][0] < cutoff_time):
                self.requests[identifier].popleft()
            
            # Check if under limit
            if len(self.requests[identifier]) < self.max_requests:
                self.requests[identifier].append(current_time)
                return True
            
            return False
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get number of remaining requests for identifier."""
        current_time = time.time()
        
        with self.lock:
            # Clean old requests
            cutoff_time = current_time - self.window_seconds
            while (self.requests[identifier] and 
                   self.requests[identifier][0] < cutoff_time):
                self.requests[identifier].popleft()
            
            return max(0, self.max_requests - len(self.requests[identifier]))
    
    def clear_identifier(self, identifier: str) -> None:
        """Clear rate limit data for identifier."""
        with self.lock:
            self.requests.pop(identifier, None)


class APIKeyManager:
    """Manage API keys and authentication."""
    
    def __init__(self):
        self.valid_keys: Set[str] = set()
        self.key_permissions: Dict[str, Set[str]] = {}
        self.key_usage: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def generate_api_key(self, permissions: Optional[Set[str]] = None) -> str:
        """
        Generate a new API key.
        
        Args:
            permissions: Set of permissions for the key
            
        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        
        with self.lock:
            self.valid_keys.add(api_key)
            self.key_permissions[api_key] = permissions or set()
        
        return api_key
    
    def validate_api_key(self, api_key: str, required_permission: Optional[str] = None) -> bool:
        """
        Validate API key and check permissions.
        
        Args:
            api_key: API key to validate
            required_permission: Required permission to check
            
        Returns:
            True if valid and has permission, False otherwise
        """
        with self.lock:
            if api_key not in self.valid_keys:
                return False
            
            if required_permission:
                if required_permission not in self.key_permissions.get(api_key, set()):
                    return False
            
            # Record usage
            self.key_usage[api_key].append(time.time())
            
            return True
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if key was revoked, False if key didn't exist
        """
        with self.lock:
            if api_key in self.valid_keys:
                self.valid_keys.remove(api_key)
                self.key_permissions.pop(api_key, None)
                self.key_usage.pop(api_key, None)
                return True
            
            return False


class RequestAuthenticator:
    """Authenticate and validate requests."""
    
    def __init__(self):
        self.blocked_hotkeys: Set[str] = set()
        self.trusted_hotkeys: Set[str] = set()
        self.lock = threading.Lock()
    
    def is_hotkey_blocked(self, hotkey: str) -> bool:
        """Check if hotkey is blocked."""
        with self.lock:
            return hotkey in self.blocked_hotkeys
    
    def block_hotkey(self, hotkey: str, reason: str = "") -> None:
        """Block a hotkey."""
        with self.lock:
            self.blocked_hotkeys.add(hotkey)
        
        bt.logging.warning(f"Blocked hotkey {hotkey}: {reason}")
    
    def unblock_hotkey(self, hotkey: str) -> None:
        """Unblock a hotkey."""
        with self.lock:
            self.blocked_hotkeys.discard(hotkey)
    
    def trust_hotkey(self, hotkey: str) -> None:
        """Add hotkey to trusted list."""
        with self.lock:
            self.trusted_hotkeys.add(hotkey)
    
    def is_hotkey_trusted(self, hotkey: str) -> bool:
        """Check if hotkey is trusted."""
        with self.lock:
            return hotkey in self.trusted_hotkeys


class AbuseDetector:
    """Detect potential abuse patterns."""
    
    def __init__(self):
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_request(self, identifier: str, request_data: Dict[str, Any]) -> None:
        """Record request for pattern analysis."""
        current_time = time.time()
        
        with self.lock:
            self.request_patterns[identifier].append(current_time)
            
            # Keep only recent requests (last hour)
            cutoff_time = current_time - 3600
            self.request_patterns[identifier] = [
                t for t in self.request_patterns[identifier] if t > cutoff_time
            ]
    
    def detect_abuse(self, identifier: str) -> bool:
        """
        Detect if identifier shows abusive patterns.
        
        Args:
            identifier: Identifier to check
            
        Returns:
            True if abuse detected, False otherwise
        """
        current_time = time.time()
        
        with self.lock:
            requests = self.request_patterns.get(identifier, [])
            
            if not requests:
                return False
            
            # Check for burst patterns (too many requests in short time)
            recent_requests = [t for t in requests if t > current_time - 60]  # Last minute
            if len(recent_requests) > 50:  # More than 50 requests per minute
                self.suspicious_patterns[identifier] += 1
                return True
            
            # Check for sustained high volume
            if len(requests) > 500:  # More than 500 requests per hour
                self.suspicious_patterns[identifier] += 1
                return True
            
            return False


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """
    Decorator for rate limiting function calls.
    
    Args:
        max_requests: Maximum requests allowed in window
        window_seconds: Time window in seconds
    """
    rate_limiter = RateLimiter(max_requests, window_seconds)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Try to extract identifier from arguments
            identifier = "global"  # Default identifier
            
            # Look for common identifier patterns in arguments
            for arg in args:
                if hasattr(arg, 'hotkey'):
                    identifier = str(arg.hotkey)
                    break
                elif isinstance(arg, str) and len(arg) == 48:  # Looks like a hotkey
                    identifier = arg
                    break
            
            if not rate_limiter.is_allowed(identifier):
                bt.logging.warning(f"Rate limit exceeded for {identifier}")
                raise MIIDSecurityError(f"Rate limit exceeded for {identifier}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_input(**validation_rules):
    """
    Decorator for input validation.
    
    Args:
        **validation_rules: Validation rules for function arguments
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments based on rules
            for arg_name, rule in validation_rules.items():
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    
                    if rule == 'name' and not InputValidator.validate_name(value):
                        raise MIIDSecurityError(f"Invalid name: {arg_name}")
                    elif rule == 'names_list' and not InputValidator.validate_names_list(value):
                        raise MIIDSecurityError(f"Invalid names list: {arg_name}")
                    elif rule == 'query_template' and not InputValidator.validate_query_template(value):
                        raise MIIDSecurityError(f"Invalid query template: {arg_name}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global instances
default_rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
request_authenticator = RequestAuthenticator()
abuse_detector = AbuseDetector()