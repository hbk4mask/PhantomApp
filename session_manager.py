# =============================================================================
# File: session_manager.py
"""Session state management - Enhanced with LLM method tracking and OpenAI support"""

import streamlit as st
from typing import Any


class SessionManager:
    """Class to handle Streamlit session state management"""

    # @staticmethod
    # def initialize_session_state():
    #     """Initialize all session state variables - FIXED WITH PROPER DEFAULTS"""
    #     defaults = {
    #         # API Keys - Initialize empty but don't overwrite existing values
    #         "claude_api_key": "",
    #         "openai_api_key": "",
    #         "groq_api_key": "",
    #
    #         # Models
    #         "groq_model": "llama-3.3-70b-versatile",
    #         "openai_model": "gpt-4o",
    #
    #         # Method selections
    #         "analysis_method": "code",
    #         "prompt_method": "code",
    #         "comparison_method": "code",
    #
    #         # Method tracking
    #         "analysis_method_used": None,
    #         "prompt_method_used": None,
    #         "comparison_method_used": None,
    #         "generation_api_used": None,
    #
    #         # Data and analysis
    #         "analysis": None,
    #         "upload_prompt": "",
    #         "schema_prompt": "",
    #         "schema": [],
    #         "generation_mode": "upload",
    #         "schema_finalized": False,
    #         "column_counter": 0,
    #         "generate_data": False,
    #         "synthetic_analysis": None,
    #         "comparison_results": None,
    #
    #         # LLM-specific analysis storage
    #         "llm_analysis": None,
    #         "llm_synthetic_analysis": None,
    #         "llm_comparison_results": None,
    #
    #         # Data storage
    #         "original_data": None,
    #         "generated_synthetic_data": None,
    #         "uploaded_synthetic_data": None,
    #         "generated_synthetic_analysis": None,
    #         "uploaded_synthetic_analysis": None,
    #
    #         # Relationship filtering
    #         "filtered_analysis": None,
    #         "relationship_classifications": None,
    #         "integrity_excluded_fields": set(),
    #
    #         # Row count selection
    #         "selected_num_rows": None,
    #
    #         # UI state tracking - NEW
    #         "ui_initialized": False,
    #         "sidebar_configured": False
    #     }
    #
    #     # Only set defaults for keys that don't exist
    #     # This prevents overwriting user inputs
    #     for key, default_value in defaults.items():
    #         if key not in st.session_state:
    #             st.session_state[key] = default_value
    #
    #     # Mark as initialized
    #     st.session_state.ui_initialized = True

    @staticmethod
    def initialize_session_state():
        import streamlit as st

        defaults = {
            # Analysis state
            "parsed_before": None,
            "parsed_result": None,
            "parsed_trace": [],
            "quick_before": None,
            "quick_result": None,
            "quick_trace": [],

            # Upload / workflow state
            "upload_prompt": "",
            "prompt_method_used": None,
            "generated_synthetic_data": None,
            "generation_api_used": None,
            "generate_data": False,
            "selected_num_rows": None,
            "original_data": None,
            "upload_code_profile": None,
            "upload_code_numeric_summary": None,
            "generation_api_choice": "auto",
            # Step tracker for reruns
            "upload_step": "upload",  # can be: upload | analysis_done | prompt_ready | generated
        }

        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    @staticmethod
    def ensure_api_keys_initialized(config):
        """Ensure API keys are properly initialized from config - NEW METHOD"""
        # Only initialize from config if not already set by user
        if not st.session_state.get("claude_api_key") and hasattr(config, 'CLAUDE_API_KEY'):
            st.session_state.claude_api_key = config.CLAUDE_API_KEY or ""

        if not st.session_state.get("openai_api_key") and hasattr(config, 'OPENAI_API_KEY'):
            st.session_state.openai_api_key = config.OPENAI_API_KEY or ""

        if not st.session_state.get("groq_api_key") and hasattr(config, 'GROQ_API_KEY'):
            st.session_state.groq_api_key = config.GROQ_API_KEY or ""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state"""
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any):
        """Set value in session state"""
        st.session_state[key] = value

    @staticmethod
    def is_initialized() -> bool:
        """Check if session state is properly initialized - NEW METHOD"""
        return st.session_state.get("ui_initialized", False)

    @staticmethod
    def clear_analysis():
        """Clear analysis-related session state"""
        keys_to_clear = [
            "analysis", "upload_prompt", "synthetic_analysis",
            "comparison_results", "generate_data", "llm_analysis",
            "llm_synthetic_analysis", "llm_comparison_results",
            "analysis_method_used", "prompt_method_used",
            "comparison_method_used", "filtered_analysis",
            "relationship_classifications"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def clear_schema():
        """Clear schema-related session state"""
        keys_to_clear = [
            "schema", "schema_prompt", "schema_finalized",
            "column_counter", "generate_data", "prompt_method_used"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                if key == "schema":
                    st.session_state[key] = []
                elif key in ["schema_finalized", "generate_data"]:
                    st.session_state[key] = False
                elif key == "column_counter":
                    st.session_state[key] = 0
                else:
                    del st.session_state[key]

    @staticmethod
    def clear_generated_data():
        """Clear generated synthetic data and related analysis"""
        keys_to_clear = [
            "generated_synthetic_data", "generated_synthetic_analysis",
            "generate_data", "generation_api_used"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def clear_uploaded_data():
        """Clear uploaded synthetic data and related analysis"""
        keys_to_clear = [
            "uploaded_synthetic_data", "uploaded_synthetic_analysis"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def get_method_summary():
        """Get summary of methods used"""
        return {
            "analysis_method": st.session_state.get("analysis_method_used"),
            "prompt_method": st.session_state.get("prompt_method_used"),
            "comparison_method": st.session_state.get("comparison_method_used"),
            "generation_api": st.session_state.get("generation_api_used")
        }

    @staticmethod
    def has_llm_capabilities():
        """Check if LLM capabilities are available"""
        return bool(st.session_state.get("claude_api_key"))

    @staticmethod
    def has_openai_capabilities():
        """Check if OpenAI capabilities are available"""
        return bool(st.session_state.get("openai_api_key"))

    @staticmethod
    def has_groq_capabilities():
        """Check if Groq capabilities are available"""
        return bool(st.session_state.get("groq_api_key"))

    @staticmethod
    def get_available_analysis_methods():
        """Get list of available analysis methods based on API keys"""
        methods = ["code"]
        if SessionManager.has_llm_capabilities() or SessionManager.has_openai_capabilities():
            methods.append("llm")
        return methods

    @staticmethod
    def get_available_generation_apis():
        """Get list of available data generation APIs"""
        apis = []
        if SessionManager.has_llm_capabilities():
            apis.append("claude")
        if SessionManager.has_openai_capabilities():
            apis.append("openai")
        if SessionManager.has_groq_capabilities():
            apis.append("groq")
        return apis

    @staticmethod
    def get_generation_status():
        """Get comprehensive generation status"""
        return {
            "claude_available": SessionManager.has_llm_capabilities(),
            "openai_available": SessionManager.has_openai_capabilities(),
            "groq_available": SessionManager.has_groq_capabilities(),
            "any_available": any([
                SessionManager.has_llm_capabilities(),
                SessionManager.has_openai_capabilities(),
                SessionManager.has_groq_capabilities()
            ]),
            "last_used_api": st.session_state.get("generation_api_used"),
            "available_apis": SessionManager.get_available_generation_apis()
        }