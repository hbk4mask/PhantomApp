# =============================================================================
# File: utils/performance_monitor.py
"""Performance monitoring utilities"""

import time
import streamlit as st
from functools import wraps
from typing import Callable, Any


class PerformanceMonitor:
    """Monitor performance of operations"""

    def __init__(self):
        self.timings = {}

    def time_operation(self, operation_name: str):
        """Decorator to time operations"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    duration = end_time - start_time

                    self.timings[operation_name] = duration

                    # Show timing in Streamlit if duration > 1 second
                    if duration > 1.0:
                        st.info(f"⏱️ {operation_name} completed in {duration:.2f} seconds")

                    return result

                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    st.error(f"❌ {operation_name} failed after {duration:.2f} seconds")
                    raise e

            return wrapper

        return decorator

    def get_timing_report(self) -> str:
        """Get a report of all operation timings"""
        if not self.timings:
            return "No operations timed yet."

        report = "⏱️ **Performance Report:**\n"
        for operation, duration in self.timings.items():
            report += f"- {operation}: {duration:.2f}s\n"

        return report
