# =============================================================================
# File: utils/logger.py
"""Logging utilities for debugging"""

import logging
import streamlit as st
from datetime import datetime
from typing import Any, Dict


class AppLogger:
    """Centralized logging for the application"""

    def __init__(self, name: str = "TabularDataApp"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Console handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Log info message"""
        if extra_data:
            message += f" | Data: {extra_data}"
        self.logger.info(message)

    def error(self, message: str, exception: Exception = None):
        """Log error message"""
        if exception:
            message += f" | Exception: {str(exception)}"
        self.logger.error(message)

        # Also show in Streamlit for user visibility
        if hasattr(st, 'error'):
            st.error(f"Error: {message}")

    def debug(self, message: str, data: Any = None):
        """Log debug message"""
        if data:
            message += f" | Debug data: {data}"
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        if hasattr(st, 'warning'):
            st.warning(message)
