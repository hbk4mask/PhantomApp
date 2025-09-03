# =============================================================================
# File: utils/error_handlers.py
"""Error handling utilities"""

import streamlit as st
import traceback
from typing import Callable, Any
from functools import wraps


def handle_errors(logger=None):
    """Decorator for handling errors in functions"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"

                if logger:
                    logger.error(error_msg, e)
                else:
                    print(error_msg)
                    print(traceback.format_exc())

                # Show user-friendly error in Streamlit
                if hasattr(st, 'error'):
                    st.error(f"An error occurred: {str(e)}")

                return None

        return wrapper

    return decorator


class ErrorHandler:
    """Centralized error handling"""

    def __init__(self, logger=None):
        self.logger = logger

    def handle_file_upload_error(self, e: Exception, filename: str = "file"):
        """Handle file upload errors"""
        error_msg = f"Failed to read {filename}: {str(e)}"

        if "UnicodeDecodeError" in str(type(e)):
            user_msg = f"File encoding error. Please ensure {filename} is saved as UTF-8."
        elif "EmptyDataError" in str(type(e)):
            user_msg = f"The uploaded file {filename} appears to be empty."
        elif "ParserError" in str(type(e)):
            user_msg = f"Could not parse {filename}. Please check the CSV format."
        else:
            user_msg = f"Error reading {filename}: {str(e)}"

        if self.logger:
            self.logger.error(error_msg, e)

        st.error(user_msg)
        return None

    def handle_api_error(self, e: Exception, api_name: str = "API"):
        """Handle API errors"""
        error_msg = f"{api_name} error: {str(e)}"

        if "401" in str(e) or "Unauthorized" in str(e):
            user_msg = f"Authentication failed. Please check your {api_name} key."
        elif "429" in str(e) or "rate limit" in str(e).lower():
            user_msg = f"Rate limit exceeded for {api_name}. Please try again later."
        elif "timeout" in str(e).lower():
            user_msg = f"Request to {api_name} timed out. Please try again."
        else:
            user_msg = f"{api_name} error: {str(e)}"

        if self.logger:
            self.logger.error(error_msg, e)

        st.error(user_msg)
        return None

    def handle_data_generation_error(self, e: Exception):
        """Handle data generation errors"""
        error_msg = f"Data generation failed: {str(e)}"

        if "memory" in str(e).lower():
            user_msg = "Not enough memory to generate data. Try reducing the number of rows."
        elif "constraint" in str(e).lower():
            user_msg = "Data generation failed due to conflicting constraints. Please review your specifications."
        else:
            user_msg = f"Data generation failed: {str(e)}"

        if self.logger:
            self.logger.error(error_msg, e)

        st.error(user_msg)
        st.info("💡 Try reducing the number of rows or simplifying the schema.")
        return None