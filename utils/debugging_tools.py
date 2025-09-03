# =============================================================================
# File: utils/debugging_tools.py
"""Debugging tools and utilities"""

import streamlit as st
import pandas as pd
import json
from typing import Any, Dict


class DebugTools:
    """Tools for debugging the application"""

    @staticmethod
    def show_debug_info(show_debug: bool = False):
        """Show debug information in sidebar"""
        if not show_debug:
            return

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🐛 Debug Info")

            if st.button("Show Session State"):
                st.json(dict(st.session_state))

            if st.button("Clear Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    @staticmethod
    def inspect_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
        """Create an inspector for dataframes"""
        with st.expander(f"🔍 Inspect {name}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Shape:**", df.shape)
                st.write("**Memory Usage:**", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                st.write("**Data Types:**")
                st.write(df.dtypes)

            with col2:
                st.write("**Null Values:**")
                st.write(df.isnull().sum())
                st.write("**Duplicate Rows:**", df.duplicated().sum())

            st.write("**Sample Data:**")
            st.dataframe(df.head(), use_container_width=True)

    @staticmethod
    def log_function_call(func_name: str, args: Dict[str, Any] = None):
        """Log function calls for debugging"""
        if 'debug_log' not in st.session_state:
            st.session_state.debug_log = []

        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'function': func_name,
            'args': args or {}
        }

        st.session_state.debug_log.append(log_entry)

        # Keep only last 50 entries
        if len(st.session_state.debug_log) > 50:
            st.session_state.debug_log = st.session_state.debug_log[-50:]

    @staticmethod
    def show_debug_log():
        """Show the debug log"""
        if 'debug_log' not in st.session_state:
            st.info("No debug log available")
            return

        with st.expander("📝 Debug Log"):
            for entry in reversed(st.session_state.debug_log[-10:]):  # Show last 10
                st.code(json.dumps(entry, indent=2))
