# =============================================================================
# File: debug_main_app.py
"""Debug version of main app with additional monitoring"""

import streamlit as st
from main_app import TabularDataApp
from utils.logger import AppLogger
from utils.performance_monitor import PerformanceMonitor
from utils.debugging_tools import DebugTools
from utils.error_handlers import ErrorHandler


class DebugTabularDataApp(TabularDataApp):
    """Debug version of the main app with enhanced monitoring"""

    def __init__(self):
        super().__init__()
        self.logger = AppLogger("DebugApp")
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler(self.logger)

        # Enable debug mode
        self.debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)

    def run(self):
        """Run the application with debug features"""
        try:
            # Show debug tools
            DebugTools.show_debug_info(self.debug_mode)

            # Log application start
            self.logger.info("Application started")
            DebugTools.log_function_call("app.run")

            # Run main application with performance monitoring
            with self.performance_monitor.time_operation("full_app_run"):
                super().run()

            # Show performance report in debug mode
            if self.debug_mode:
                with st.sidebar:
                    st.markdown("---")
                    st.markdown(self.performance_monitor.get_timing_report())
                    DebugTools.show_debug_log()

        except Exception as e:
            self.error_handler.handle_data_generation_error(e)
            self.logger.error("Application error", e)


if __name__ == "__main__":
    app = DebugTabularDataApp()
    app.run()