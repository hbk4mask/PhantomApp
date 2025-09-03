# =============================================================================
# File: main_app.py (COMPLETE WITH OPENAI INTEGRATION AND FIXED PROMPT BUTTON)
"""Main Streamlit application with detailed comparison reporting"""

import streamlit as st
import pandas as pd

# Import all our custom modules
from config import Config
from data_analysis import DataAnalyzer
from llm_analyzer import LLMAnalyzer
from prompt_generator import PromptGenerator
from data_generator import DataGenerator
from comparison import DataComparator
from ui_components import UIComponents
from session_manager import SessionManager
from relationship_filter import integrate_relationship_filtering


class TabularDataApp:
    """Main application class - FIXED VERSION"""

    def __init__(self):
        self.config = Config()
        self.analyzer = DataAnalyzer(self.config)
        self.llm_analyzer = LLMAnalyzer(self.config)
        self.prompt_generator = PromptGenerator(self.config)
        self.data_generator = DataGenerator(self.config)
        self.comparator = DataComparator(self.config)
        self.ui = UIComponents()
        self.session = SessionManager()

    def run(self):
        """Run the main application - FIXED VERSION"""

        # CRITICAL: Setup page FIRST
        self.ui.setup_page()

        # CRITICAL: Initialize session state FIRST
        self.session.initialize_session_state()

        # CRITICAL: Ensure API keys are initialized from config
        self.session.ensure_api_keys_initialized(self.config)

        # CRITICAL: Setup sidebar with proper state management
        try:
            claude_api_key, openai_api_key, groq_api_key = self.ui.setup_sidebar(self.config)
        except Exception as e:
            st.error(f"Sidebar setup error: {str(e)}")
            # Fallback to session state values
            claude_api_key = st.session_state.get('claude_api_key', '')
            openai_api_key = st.session_state.get('openai_api_key', '')
            groq_api_key = st.session_state.get('groq_api_key', '')

        # Mode selection
        generation_mode = self.ui.display_mode_selection()

        # Main content based on mode
        if generation_mode == "upload":
            self._handle_upload_mode()
        else:
            self._handle_schema_mode()

        # Synthetic data upload section
        self._handle_synthetic_upload_section()

        # Comprehensive comparison section
        self._handle_comprehensive_comparison_section()

        # Clean footer
        self._display_clean_footer()



    def _handle_upload_mode(self):
        """Upload → Analyze → Prompt → Generate flow with stable state across reruns."""
        import pandas as pd
        import streamlit as st

        st.header("📂 Upload & Analyze Existing Data")

        # --- EARLY ROUTING: restore correct screen after any Streamlit rerun ---
        step = self.session.get("upload_step") or "upload"

        # If generation already happened, show the generated data first
        gen_df = self.session.get("generated_synthetic_data")
        if step == "generated" and gen_df is not None:
            st.subheader("🎲 Generated Synthetic Data")
            if hasattr(self, "ui") and hasattr(self.ui, "display_synthetic_data"):
                self.ui.display_synthetic_data(gen_df)
            else:
                st.dataframe(gen_df, use_container_width=True)

            # Always show synthetic analysis controls on the generated screen
            if hasattr(self, "_handle_synthetic_analysis"):
                self._handle_synthetic_analysis(gen_df)
            return

        # If we already have a prompt, jump straight to the prompt section
        saved_prompt = self.session.get("upload_prompt")
        if step == "prompt_ready" and saved_prompt:
            self._handle_prompt_section("upload_prompt")
            return

        # --- Normal uploader path (stable key prevents flicker that resets flow) ---
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="upload_csv")
        if not uploaded_file:
            st.info("Upload a CSV to begin.")
            return

        # Read CSV safely
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        # Persist original data
        self.session.set("original_data", df)
        st.success(
            f"Uploaded **{getattr(uploaded_file, 'name', 'file')}** "
            f"with {len(df)} rows × {len(df.columns)} columns."
        )

        with st.expander("Preview (first 100 rows)", expanded=False):
            st.dataframe(df.head(100), use_container_width=True, height=400)

        st.markdown("---")

        # === Analyze (code method) ===
        if st.button("🔍 Analyze Data (code method)", key="analyze_upload_code"):
            with st.spinner("Analyzing data (code method)..."):
                try:
                    analysis = self.analyzer.analyze_data(df)  # correct API
                    self.session.set("analysis", analysis)
                    self.session.set("analysis_method_used", "code")
                    self.session.set("upload_step", "analysis_done")
                    st.success("✅ Code-based analysis completed!")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    return

            # Show results via your UI component (expects full analysis dict)
            if hasattr(self.ui, "display_analysis_results"):
                self.ui.display_analysis_results(analysis, self.analyzer)

            # Optional: relationship filtering
            try:
                from relationship_filter import integrate_relationship_filtering
                filtered = integrate_relationship_filtering(df, analysis)
                if filtered != analysis:
                    self.session.set("analysis", filtered)
            except Exception:
                pass  # non-fatal

        # === Row count selector (before prompt generation) ===
        rows_selected = st.number_input(
            "Target number of synthetic rows",
            min_value=self.config.MIN_ROWS,
            max_value=self.config.MAX_ROWS,
            value=int(self.session.get("selected_num_rows") or self.config.DEFAULT_ROWS),
            key="target_rows_upload",
        )
        self.session.set("selected_num_rows", int(rows_selected))

        # === Generate Prompt (code method) ===
        # Only show once analysis exists and no prompt yet
        if self.session.get("analysis") and not self.session.get("upload_prompt"):
            if st.button("🧾 Generate Prompt (code method)", key="gen_prompt_upload_code"):
                from prompt_generator import PromptGenerator
                prompt_gen = PromptGenerator()
                try:
                    prompt = prompt_gen.generate_from_analysis(
                        self.session.get("analysis"),
                        num_rows=int(self.session.get("selected_num_rows") or self.config.DEFAULT_ROWS),
                    )
                except Exception as e:
                    st.error(f"Prompt generation failed: {e}")
                    return

                # Persist + route to prompt view
                self.session.set("upload_prompt", prompt)
                self.session.set("prompt_method_used", "code")
                self.session.set("upload_step", "prompt_ready")

                # Render immediately; avoid bouncing back on rerun
                self._handle_prompt_section("upload_prompt")
                st.stop()

    def _handle_schema_mode(self):
        """Handle custom schema mode"""
        st.header("🛠️ Define Custom Schema")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Schema builder
            self.ui.display_schema_builder()

            # Schema management
            if self.session.get("schema"):
                self._handle_schema_management()

        with col2:
            st.write("")  # Placeholder

        # Generate prompts section
        if self.session.get("schema"):
            self._handle_schema_prompts()

    def _handle_schema_management(self):
        """Handle schema finalization and management"""
        st.markdown("---")
        st.subheader("✅ Schema Confirmation")

        if self.session.get("schema_finalized"):
            st.write("**Finalized Columns:**")
            for i, col_def in enumerate(self.session.get("schema"), 1):
                col_info = f"{i}. **{col_def['name']}** ({col_def['type']})"
                if col_def.get('unique'):
                    col_info += " - Unique"
                if col_def.get('null_percentage', 0) > 0:
                    col_info += f" - {col_def['null_percentage']}% nulls"
                st.write(col_info)

            if st.button("🗑️ Clear All Columns", type="secondary"):
                self.session.clear_schema()
                st.rerun()
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🎯 Finalize Schema", type="primary"):
                    self.session.set("schema_finalized", True)
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear All Columns", type="secondary"):
                    self.session.clear_schema()
                    st.rerun()

    def _handle_schema_prompts(self):
        """Handle schema-based prompt generation - UPDATED WITH ROW COUNT SELECTOR"""
        st.markdown("---")
        st.subheader("🚀 Generate Prompts")

        # Row count selector
        col1, col2 = st.columns([2, 1])

        with col1:
            num_rows = st.number_input(
                "Number of Rows",
                min_value=self.config.MIN_ROWS,
                max_value=self.config.MAX_ROWS,
                value=self.config.DEFAULT_ROWS,
                step=10,
                help="Number of synthetic records to generate",
                key="schema_num_rows_input"
            )

        with col2:
            st.markdown("**Quick Presets:**")
            common_sizes = [50, 100, 500, 1000]
            for size in common_sizes:
                if size <= self.config.MAX_ROWS:
                    if st.button(f"{size} rows", key=f"schema_preset_{size}"):
                        st.session_state["schema_num_rows_input"] = size
                        st.rerun()

        # Show prompt generation method info
        if not self.ui.display_prompt_generation_info():
            return

        prompt_method = st.session_state.get('prompt_method', 'code')
        prompt_button_text = f"📋 Generate Prompt ({prompt_method.title()} Method)"

        if st.button(prompt_button_text, type="primary"):
            if self.session.get("schema_finalized"):
                with st.spinner(f"Generating prompt using {prompt_method} method..."):
                    try:
                        if prompt_method == 'llm':
                            claude_api_key = st.session_state.get('claude_api_key')
                            openai_api_key = st.session_state.get('openai_api_key')

                            if not claude_api_key and not openai_api_key:
                                st.error("Claude or OpenAI API key required for LLM prompt generation")
                                return

                            # For schema-based LLM prompt generation, we need to convert schema to analysis format
                            st.warning(
                                "LLM prompt generation from schema is not yet fully implemented. Using code-based method.")
                            prompt = self.prompt_generator.generate_from_schema(
                                self.session.get("schema"),
                                num_rows
                            )
                        else:
                            prompt = self.prompt_generator.generate_from_schema(
                                self.session.get("schema"),
                                num_rows
                            )

                        self.session.set("schema_prompt", prompt)
                        self.session.set("prompt_method_used", prompt_method)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Prompt generation failed: {str(e)}")
            elif self.session.get("schema"):
                st.warning("⚠️ Please finalize your schema first.")
            else:
                st.error("❌ Please add at least one column.")

        # Handle prompt section
        self._handle_prompt_section("schema_prompt")


    # def _handle_prompt_section(self, prompt_key: str):
    #     """Render a saved/created prompt and controls for generation."""
    #     import streamlit as st
    #     import re
    #
    #     prompt = self.session.get(prompt_key)
    #     if not prompt:
    #         return
    #
    #     # Stay on the prompt screen across reruns triggered by inputs/radio
    #     self.session.set("upload_step", "prompt_ready")
    #
    #     st.subheader("📝 Generated Prompt")
    #     st.text_area("Prompt", prompt, height=300)
    #
    #     # ---- Provider choice (Auto / OpenAI / Claude / Grok) ----
    #     choices = ["Auto", "OpenAI", "Claude", "Grok"]
    #     stored = (self.session.get("generation_api_choice") or "auto").lower()
    #     idx_map = {"auto": 0, "openai": 1, "claude": 2, "grok": 3, "groq": 3}
    #     idx = idx_map.get(stored, 0)  # safe default to Auto if unknown
    #
    #     api_choice = st.radio(
    #         "Provider",
    #         choices,
    #         index=idx,
    #         horizontal=True,
    #         key=f"api_choice_{prompt_key}",
    #     )
    #     # Persist normalized value ('grok' even if label is 'Grok')
    #     self.session.set("generation_api_choice", api_choice.lower() if api_choice != "Grok" else "grok")
    #
    #     # ---- Row count (synced with session) ----
    #     # num_rows = st.number_input(
    #     #     "Number of synthetic rows",
    #     #     min_value=self.config.MIN_ROWS,
    #     #     max_value=self.config.MAX_ROWS,
    #     #     value=int(self.session.get("selected_num_rows") or self.config.DEFAULT_ROWS),
    #     #     key=f"num_rows_{prompt_key}",
    #     # )
    #     # self.session.set("selected_num_rows", int(num_rows))
    #
    #     col1, col2 = st.columns(2)
    #
    #     with col1:
    #         if st.button("🎲 Generate Synthetic Data", type="primary", key=f"generate_{prompt_key}"):
    #             effective_rows = int(self.session.get("selected_num_rows") or num_rows)
    #
    #             # Rewrite embedded row-counts in the prompt to the chosen value
    #             prompt_to_use = prompt
    #             # e.g., "EXACTLY 50 rows" -> "EXACTLY {effective_rows} rows"
    #             prompt_to_use = re.sub(r'(?i)(exactly\s*)\d+(\s*rows?)', rf'\g<1>{effective_rows}\2', prompt_to_use)
    #             # e.g., "with 50 rows" or "with exactly 50 rows" -> chosen value
    #             prompt_to_use = re.sub(r'(?i)(with\s+)(?:exactly\s*)?\d+(\s*rows?)', rf'\1{effective_rows}\2',
    #                                    prompt_to_use)
    #
    #             # Strong override in case the model still hesitates
    #             prompt_to_use += (
    #                 f"\n\nCRITICAL: Output EXACTLY {effective_rows} rows. "
    #                 f"If any instruction conflicts, this requirement takes precedence."
    #             )
    #
    #             self.session.set("selected_num_rows", effective_rows)
    #             self.session.set("generate_data", True)
    #
    #             # Run generation now and stop so the UI doesn’t bounce
    #             self._handle_data_generation(prompt_to_use, effective_rows)
    #             st.stop()
    #
    #     with col2:
    #         if st.button("❌ Clear Prompt", key=f"clear_{prompt_key}"):
    #             self.session.set(prompt_key, "")
    #             self.session.set("upload_step", "analysis_done")  # return to analysis screen
    #             st.stop()
    def _handle_prompt_section(self, prompt_key: str):
        """Render a saved/created prompt and controls for generation (no post-prompt row input)."""
        import streamlit as st
        import re

        prompt = self.session.get(prompt_key)
        if not prompt:
            return

        # Stay on the prompt screen across reruns triggered by inputs/radio
        self.session.set("upload_step", "prompt_ready")

        st.subheader("📝 Generated Prompt")
        st.text_area("Prompt", prompt, height=300)

        # ---- Provider choice (Auto / OpenAI / Claude / Grok) ----
        choices = ["Auto", "OpenAI", "Claude", "Grok"]
        stored = (self.session.get("generation_api_choice") or "auto").lower()
        idx_map = {"auto": 0, "openai": 1, "claude": 2, "grok": 3, "groq": 3}
        api_choice_idx = idx_map.get(stored, 0)

        api_choice = st.radio(
            "Provider",
            choices,
            index=api_choice_idx,
            horizontal=True,
            key=f"api_choice_{prompt_key}",
        )
        # Persist normalized value ('grok' even if label is 'Grok')
        self.session.set("generation_api_choice", api_choice.lower() if api_choice != "Grok" else "grok")

        # Use the row count selected earlier (before prompt generation)
        effective_rows = int(self.session.get("selected_num_rows") or self.config.DEFAULT_ROWS)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎲 Generate Synthetic Data", type="primary", key=f"generate_{prompt_key}"):
                # Rewrite embedded row-counts in the prompt to the chosen value
                prompt_to_use = prompt
                # e.g., "EXACTLY 50 rows" -> chosen value
                prompt_to_use = re.sub(r'(?i)(exactly\s*)\d+(\s*rows?)', rf'\g<1>{effective_rows}\2', prompt_to_use)
                # e.g., "with 50 rows" or "with exactly 50 rows" -> chosen value
                prompt_to_use = re.sub(r'(?i)(with\s+)(?:exactly\s*)?\d+(\s*rows?)', rf'\1{effective_rows}\2',
                                       prompt_to_use)
                # Strong override in case the model hesitates
                prompt_to_use += (
                    f"\n\nCRITICAL: Output EXACTLY {effective_rows} rows. "
                    f"If any instruction conflicts, this requirement takes precedence."
                )

                self.session.set("generate_data", True)
                self._handle_data_generation(prompt_to_use, effective_rows)
                st.stop()

        with col2:
            if st.button("❌ Clear Prompt", key=f"clear_{prompt_key}"):
                self.session.set(prompt_key, "")
                self.session.set("upload_step", "analysis_done")  # return to analysis screen
                st.stop()

    def _update_prompt_row_count(self, prompt: str, new_row_count: int) -> str:
        """Update the row count in the prompt"""
        import re

        # Replace various patterns for row count
        patterns = [
            r'Generate synthetic tabular data with EXACTLY \d+ rows',
            r'Generate synthetic tabular data with \d+ rows',
            r'generate exactly \d+ rows',
            r'Generate \d+ rows',
            r'EXACTLY \d+ rows',
            r'\d+ rows'
        ]

        updated_prompt = prompt
        for pattern in patterns:
            if re.search(pattern, updated_prompt, re.IGNORECASE):
                if 'EXACTLY' in pattern:
                    replacement = f'Generate synthetic tabular data with EXACTLY {new_row_count} rows'
                else:
                    replacement = f'Generate synthetic tabular data with {new_row_count} rows'

                updated_prompt = re.sub(pattern, replacement, updated_prompt, count=1, flags=re.IGNORECASE)
                break

        # Also update any references in generation instructions
        updated_prompt = re.sub(r'Generate EXACTLY \d+ rows \(no more, no less\)',
                                f'Generate EXACTLY {new_row_count} rows (no more, no less)',
                                updated_prompt, flags=re.IGNORECASE)

        updated_prompt = re.sub(r'Row count = \d+ ✓',
                                f'Row count = {new_row_count} ✓',
                                updated_prompt)

        return updated_prompt

    # def _handle_data_generation(self, prompt: str, num_rows: int = None):
    #     """Generate synthetic data using configured LLM(s); keep view stable across reruns."""
    #     import streamlit as st
    #     import pandas as pd
    #     from data_generator import DataGenerator
    #
    #     if not prompt:
    #         st.error("No prompt available for generation.")
    #         return
    #
    #     rows = int(num_rows or self.config.DEFAULT_ROWS)
    #     choice = (self.session.get("generation_api_choice") or "auto").lower()
    #
    #     # Simple provider-key presence check
    #     def _ok(v):
    #         return bool(v) and not str(v).lower().startswith(("your_", "paste_", "sk-xxxx"))
    #
    #     has_any_key = any([
    #         _ok(getattr(self.config, "OPENAI_API_KEY", None)),
    #         _ok(getattr(self.config, "CLAUDE_API_KEY", None)),
    #         _ok(getattr(self.config, "GROQ_API_KEY", None)),
    #         _ok(getattr(self.config, "GEMINI_API_KEY", None)),
    #     ])
    #     if not has_any_key and choice == "auto":
    #         st.error("No valid API key configured. Set at least one provider key in your environment.")
    #         self.session.set("upload_step", "prompt_ready")
    #         return
    #
    #     with st.spinner("Generating synthetic data..."):
    #         try:
    #             generator = DataGenerator(self.config)
    #
    #             # helper to normalize return types from generators:
    #             def _normalize(result, default_used_api: str):
    #                 # Some generators return just df, others (df, used_api)
    #                 if isinstance(result, tuple):
    #                     if len(result) >= 2:
    #                         return result[0], result[1]
    #                     elif len(result) == 1:
    #                         return result[0], default_used_api
    #                     else:
    #                         return None, default_used_api
    #                 else:
    #                     return result, default_used_api
    #
    #             # --- primary generation call per provider ---
    #             if choice == "openai":
    #                 result = generator.generate_with_openai(
    #                     prompt,
    #                     rows,
    #                     getattr(self.config, "OPENAI_API_KEY", ""),
    #                     getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
    #                 )
    #                 df_synth, used_api = _normalize(result, "OpenAI")
    #
    #             elif choice == "claude":
    #                 result = generator.generate_with_claude(
    #                     prompt,
    #                     rows,
    #                     getattr(self.config, "CLAUDE_API_KEY", ""),
    #                 )
    #                 df_synth, used_api = _normalize(result, "Claude")
    #
    #             elif choice in ("grok", "groq"):
    #                 result = generator.generate_with_groq(
    #                     prompt,
    #                     rows,
    #                     getattr(self.config, "GROQ_API_KEY", ""),
    #                     getattr(self.config, "DEFAULT_GROQ_MODEL", getattr(self.config, "DEFAULT_MODEL", None)),
    #                 )
    #                 df_synth, used_api = _normalize(result, "Grok")
    #
    #             else:
    #                 # Auto fallback (already returns (df, used_api))
    #                 df_synth, used_api = generator.generate_with_auto_fallback(
    #                     prompt,
    #                     rows,
    #                     claude_key=getattr(self.config, "CLAUDE_API_KEY", None),
    #                     openai_key=getattr(self.config, "OPENAI_API_KEY", None),
    #                     groq_key=getattr(self.config, "GROQ_API_KEY", None),
    #                     openai_model=getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
    #                     groq_model=getattr(self.config, "DEFAULT_GROQ_MODEL", None),
    #                 )
    #
    #             # ---- Top-up loop: if the model returned fewer rows, continue the CSV ----
    #             def _continue_prompt(base_prompt: str, header: str, already: int, need_more: int) -> str:
    #                 return (
    #                     f"{base_prompt}\n\n"
    #                     f"CONTINUE THE CSV:\n"
    #                     f"- Add EXACTLY {need_more} more rows.\n"
    #                     f"- DO NOT repeat the header.\n"
    #                     f"- Use the SAME column order: {header}\n"
    #                     f"- Output ONLY the additional CSV rows (no prose, no header, no markdown).\n"
    #                 )
    #
    #             if df_synth is not None and not df_synth.empty and len(df_synth) < rows:
    #                 header = ",".join(list(df_synth.columns))
    #                 attempts = 0
    #                 while len(df_synth) < rows and attempts < 3:
    #                     remaining = rows - len(df_synth)
    #                     cont = _continue_prompt(prompt, header, len(df_synth), remaining)
    #
    #                     if choice == "openai":
    #                         more = generator.generate_with_openai(
    #                             cont,
    #                             remaining,
    #                             getattr(self.config, "OPENAI_API_KEY", ""),
    #                             getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
    #                         )
    #                         more_df, _ = _normalize(more, "OpenAI")
    #                     elif choice == "claude":
    #                         more = generator.generate_with_claude(
    #                             cont,
    #                             remaining,
    #                             getattr(self.config, "CLAUDE_API_KEY", ""),
    #                         )
    #                         more_df, _ = _normalize(more, "Claude")
    #                     elif choice in ("grok", "groq"):
    #                         more = generator.generate_with_groq(
    #                             cont,
    #                             remaining,
    #                             getattr(self.config, "GROQ_API_KEY", ""),
    #                             getattr(self.config, "DEFAULT_GROQ_MODEL", getattr(self.config, "DEFAULT_MODEL", None)),
    #                         )
    #                         more_df, _ = _normalize(more, "Grok")
    #                     else:
    #                         more_df, _ = generator.generate_with_auto_fallback(
    #                             cont,
    #                             remaining,
    #                             claude_key=getattr(self.config, "CLAUDE_API_KEY", None),
    #                             openai_key=getattr(self.config, "OPENAI_API_KEY", None),
    #                             groq_key=getattr(self.config, "GROQ_API_KEY", None),
    #                             openai_model=getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
    #                             groq_model=getattr(self.config, "DEFAULT_GROQ_MODEL", None),
    #                         )
    #
    #                     # append only if columns align
    #                     if isinstance(more_df, pd.DataFrame) and list(more_df.columns) == list(df_synth.columns):
    #                         df_synth = pd.concat([df_synth, more_df], ignore_index=True)
    #                     else:
    #                         break  # don't corrupt structure
    #                     attempts += 1
    #
    #         except Exception as e:
    #             st.error(f"Synthetic data generation failed.\n\n**Provider choice**: {choice}\n\n**Error**: {e}")
    #             self.session.set("upload_step", "prompt_ready")
    #             return
    #
    #     if df_synth is None or getattr(df_synth, "empty", False):
    #         st.error("Synthetic data generation returned no rows.")
    #         self.session.set("upload_step", "prompt_ready")
    #         return
    #
    #     # Persist results and pin UI to the generated view
    #     self.session.set("generated_synthetic_data", df_synth)
    #     self.session.set("generation_api_used", used_api)
    #     self.session.set("generate_data", False)
    #     self.session.set("upload_step", "generated")
    #
    #     requested = int(self.session.get("selected_num_rows") or rows)
    #     try:
    #         if len(df_synth) != requested:
    #             st.warning(
    #                 f"Requested {requested} rows, received {len(df_synth)}. "
    #                 f"The model may have truncated; you can try regenerating or switch provider."
    #             )
    #     except Exception:
    #         pass
    #
    #     st.success(f"Synthetic data generated using {used_api}.")
    #     if hasattr(self, "ui") and hasattr(self.ui, "display_synthetic_data"):
    #         self.ui.display_synthetic_data(df_synth)
    #     else:
    #         st.dataframe(df_synth, use_container_width=True)
    #
    #     if hasattr(self, "_handle_synthetic_analysis"):
    #         self._handle_synthetic_analysis(df_synth)
    def _handle_data_generation(self, prompt: str, num_rows: int = None):
        """Generate synthetic data using configured provider(s); keep view stable and top up if short."""
        import streamlit as st
        import pandas as pd
        from data_generator import DataGenerator

        if not prompt:
            st.error("No prompt available for generation.")
            return

        rows = int(num_rows or self.session.get("selected_num_rows") or self.config.DEFAULT_ROWS)
        choice = (self.session.get("generation_api_choice") or "auto").lower()

        # Simple provider-key presence check (for Auto path)
        def _ok(v):
            return bool(v) and not str(v).lower().startswith(("your_", "paste_", "sk-xxxx"))

        has_any_key = any([
            _ok(getattr(self.config, "OPENAI_API_KEY", None)),
            _ok(getattr(self.config, "CLAUDE_API_KEY", None)),
            _ok(getattr(self.config, "GROQ_API_KEY", None)),
            _ok(getattr(self.config, "GEMINI_API_KEY", None)),
        ])
        if not has_any_key and choice == "auto":
            st.error("No valid API key configured. Set at least one provider key in your environment.")
            self.session.set("upload_step", "prompt_ready")
            return

        with st.spinner("Generating synthetic data..."):
            try:
                generator = DataGenerator(self.config)

                # Normalize different return types (some methods return df, others (df, used_api))
                def _normalize(result, default_used_api: str):
                    if isinstance(result, tuple):
                        if len(result) >= 2:
                            return result[0], result[1]
                        elif len(result) == 1:
                            return result[0], default_used_api
                        return None, default_used_api
                    return result, default_used_api

                # --- Primary generation call per provider ---
                if choice == "openai":
                    result = generator.generate_with_openai(
                        prompt,
                        rows,
                        getattr(self.config, "OPENAI_API_KEY", ""),
                        getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
                    )
                    df_synth, used_api = _normalize(result, "OpenAI")

                elif choice == "claude":
                    result = generator.generate_with_claude(
                        prompt,
                        rows,
                        getattr(self.config, "CLAUDE_API_KEY", ""),
                    )
                    df_synth, used_api = _normalize(result, "Claude")

                elif choice in ("grok", "groq"):
                    result = generator.generate_with_groq(
                        prompt,
                        rows,
                        getattr(self.config, "GROQ_API_KEY", ""),
                        getattr(self.config, "DEFAULT_GROQ_MODEL", getattr(self.config, "DEFAULT_MODEL", None)),
                    )
                    df_synth, used_api = _normalize(result, "Grok")

                else:
                    # Auto fallback (already returns (df, used_api))
                    df_synth, used_api = generator.generate_with_auto_fallback(
                        prompt,
                        rows,
                        claude_key=getattr(self.config, "CLAUDE_API_KEY", None),
                        openai_key=getattr(self.config, "OPENAI_API_KEY", None),
                        groq_key=getattr(self.config, "GROQ_API_KEY", None),
                        openai_model=getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
                        groq_model=getattr(self.config, "DEFAULT_GROQ_MODEL", None),
                    )

                # ---- Top-up loop: if the model returned fewer rows, continue the CSV ----
                def _continue_prompt(base_prompt: str, header: str, need_more: int) -> str:
                    return (
                        f"{base_prompt}\n\n"
                        f"CONTINUE THE CSV:\n"
                        f"- Add EXACTLY {need_more} more rows.\n"
                        f"- DO NOT repeat the header.\n"
                        f"- Use the SAME column order: {header}\n"
                        f"- Output ONLY the additional CSV rows (no prose, no header, no markdown)."
                    )

                if df_synth is not None and not df_synth.empty and len(df_synth) < rows:
                    header = ",".join(list(df_synth.columns))
                    attempts = 0
                    while len(df_synth) < rows and attempts < 3:
                        remaining = rows - len(df_synth)
                        cont = _continue_prompt(prompt, header, remaining)

                        if choice == "openai":
                            more = generator.generate_with_openai(
                                cont,
                                remaining,
                                getattr(self.config, "OPENAI_API_KEY", ""),
                                getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
                            )
                            more_df, _ = _normalize(more, "OpenAI")
                        elif choice == "claude":
                            more = generator.generate_with_claude(
                                cont,
                                remaining,
                                getattr(self.config, "CLAUDE_API_KEY", ""),
                            )
                            more_df, _ = _normalize(more, "Claude")
                        elif choice in ("grok", "groq"):
                            more = generator.generate_with_groq(
                                cont,
                                remaining,
                                getattr(self.config, "GROQ_API_KEY", ""),
                                getattr(self.config, "DEFAULT_GROQ_MODEL", getattr(self.config, "DEFAULT_MODEL", None)),
                            )
                            more_df, _ = _normalize(more, "Grok")
                        else:
                            more_df, _ = generator.generate_with_auto_fallback(
                                cont,
                                remaining,
                                claude_key=getattr(self.config, "CLAUDE_API_KEY", None),
                                openai_key=getattr(self.config, "OPENAI_API_KEY", None),
                                groq_key=getattr(self.config, "GROQ_API_KEY", None),
                                openai_model=getattr(self.config, "DEFAULT_OPENAI_MODEL", None),
                                groq_model=getattr(self.config, "DEFAULT_GROQ_MODEL", None),
                            )

                        if isinstance(more_df, pd.DataFrame) and list(more_df.columns) == list(df_synth.columns):
                            df_synth = pd.concat([df_synth, more_df], ignore_index=True)
                        else:
                            break
                        attempts += 1

            except Exception as e:
                st.error(f"Synthetic data generation failed.\n\n**Provider choice**: {choice}\n\n**Error**: {e}")
                self.session.set("upload_step", "prompt_ready")
                return

        if df_synth is None or getattr(df_synth, "empty", False):
            st.error("Synthetic data generation returned no rows.")
            self.session.set("upload_step", "prompt_ready")
            return

        # Persist results and pin UI to the generated view
        self.session.set("generated_synthetic_data", df_synth)
        self.session.set("generation_api_used", used_api)
        self.session.set("generate_data", False)
        self.session.set("upload_step", "generated")

        # Inform if row count still mismatched
        requested = int(self.session.get("selected_num_rows") or rows)
        try:
            if len(df_synth) != requested:
                st.warning(
                    f"Requested {requested} rows, received {len(df_synth)}. "
                    f"The model may have truncated; try regenerating or switch provider."
                )
        except Exception:
            pass

        st.success(f"Synthetic data generated using {used_api}.")
        if hasattr(self, "ui") and hasattr(self.ui, "display_synthetic_data"):
            self.ui.display_synthetic_data(df_synth)
        else:
            st.dataframe(df_synth, use_container_width=True)

        if hasattr(self, "_handle_synthetic_analysis"):
            self._handle_synthetic_analysis(df_synth)

    def _handle_synthetic_analysis(self, df_synthetic: pd.DataFrame):
        """Handle analysis of synthetic data with options"""

        st.markdown("---")
        st.markdown("### 🔬 Choose Analysis Type")

        # Analysis type selection
        analysis_type = st.radio(
            "What type of analysis would you like to run?",
            ["basic", "comparison"],
            format_func=lambda
                x: "📊 Basic Data Analysis (individual dataset stats)" if x == "basic" else "🔄 Comprehensive Comparison (original vs synthetic)",
            horizontal=True,
            key="analysis_type_selection"
        )

        if analysis_type == "basic":
            # Basic analysis code
            analysis_method = st.session_state.get('analysis_method', 'code')
            button_text = f"🔍 Analyze Synthetic Data ({analysis_method.title()} Method)"

            if st.button(button_text, type="primary"):
                with st.spinner(f"Analyzing synthetic data using {analysis_method} method..."):
                    try:
                        if analysis_method == 'llm':
                            claude_api_key = st.session_state.get('claude_api_key')
                            openai_api_key = st.session_state.get('openai_api_key')  # NEW
                            openai_model = st.session_state.get('openai_model')  # NEW

                            if not claude_api_key and not openai_api_key:
                                st.error("Claude or OpenAI API key required for LLM analysis")
                                return

                            synthetic_analysis = self.llm_analyzer.analyze_data_with_llm(
                                df_synthetic, claude_api_key, openai_api_key, openai_model  # NEW PARAMETERS
                            )
                            st.success("✅ LLM analysis of synthetic data completed!")
                        else:
                            synthetic_analysis = self.analyzer.analyze_data(df_synthetic)
                            st.success("✅ Code-based analysis of synthetic data completed!")

                        self.session.set("generated_synthetic_analysis", synthetic_analysis)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error analyzing synthetic data: {str(e)}")

            # Display synthetic analysis if available
            if self.session.get("generated_synthetic_analysis"):
                st.markdown("---")
                analysis_method_used = st.session_state.get('analysis_method', 'code')
                method_indicator = "🤖 LLM Analysis" if analysis_method_used == 'llm' else "⚡ Code Analysis"
                st.markdown(f"### 🔬 Generated Synthetic Data Analysis ({method_indicator})")

                self.ui.display_analysis_results(
                    self.session.get("generated_synthetic_analysis"),
                    self.analyzer
                )

        elif analysis_type == "comparison":
            # Direct comparison without requiring basic analysis first
            if not self.session.get("analysis"):
                st.warning(
                    "⚠️ Original data analysis required for comparison. Please analyze your original data first.")
                return

            comparison_method = st.session_state.get('comparison_method', 'code')
            button_text = f"🔄 Run Comprehensive Comparison ({comparison_method.title()} Method)"

            if st.button(button_text, type="primary"):
                with st.spinner("Running comprehensive comparison..."):
                    try:
                        # Analyze synthetic data quietly (without showing results)
                        analysis_method = st.session_state.get('analysis_method', 'code')

                        if analysis_method == 'llm':
                            claude_api_key = st.session_state.get('claude_api_key')
                            openai_api_key = st.session_state.get('openai_api_key')
                            openai_model = st.session_state.get('openai_model')

                            if not claude_api_key and not openai_api_key:
                                st.error("Claude or OpenAI API key required for LLM analysis")
                                return

                            synthetic_analysis = self.llm_analyzer.analyze_data_with_llm(
                                df_synthetic, claude_api_key, openai_api_key, openai_model
                            )
                        else:
                            synthetic_analysis = self.analyzer.analyze_data(df_synthetic)

                        # Store for potential future use
                        self.session.set("generated_synthetic_analysis", synthetic_analysis)

                        # Run comparison immediately
                        self.ui.display_comprehensive_comparison(
                            self.session.get("analysis"),
                            synthetic_analysis,
                            self.comparator,
                            "Original vs Generated Synthetic Data"
                        )

                    except Exception as e:
                        st.error(f"Comparison failed: {str(e)}")

        # Clear buttons (common for both types)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Generated Data"):
                self.session.set("generate_data", False)
                if "generated_synthetic_analysis" in st.session_state:
                    del st.session_state["generated_synthetic_analysis"]
                if "generated_synthetic_data" in st.session_state:
                    del st.session_state["generated_synthetic_data"]
                st.rerun()

        with col2:
            if self.session.get("generated_synthetic_analysis") and st.button("🗑️ Clear Analysis"):
                if "generated_synthetic_analysis" in st.session_state:
                    del st.session_state["generated_synthetic_analysis"]
                st.rerun()

    def _handle_synthetic_upload_section(self):
        """Handle synthetic data upload section"""
        uploaded_synthetic = self.ui.display_synthetic_upload_section()

        # Display uploaded synthetic analysis if available
        if self.session.get("uploaded_synthetic_analysis"):
            st.markdown("---")
            analysis_method_used = st.session_state.get('analysis_method', 'code')
            method_indicator = "🤖 LLM Analysis" if analysis_method_used == 'llm' else "⚡ Code Analysis"
            st.markdown(f"### 🔬 Uploaded Synthetic Data Analysis ({method_indicator})")

            self.ui.display_analysis_results(
                self.session.get("uploaded_synthetic_analysis"),
                self.analyzer
            )

    def _handle_comprehensive_comparison_section(self):
        """Handle comprehensive comparison section with detailed reporting - ONLY SHOW WHEN DATA EXISTS"""

        # Check what data is available for comparison
        original_analysis = self.session.get("analysis")
        generated_analysis = self.session.get("generated_synthetic_analysis")
        uploaded_analysis = self.session.get("uploaded_synthetic_analysis")

        comparison_options = []
        if original_analysis and generated_analysis:
            comparison_options.append("Original vs Generated Synthetic")
        if original_analysis and uploaded_analysis:
            comparison_options.append("Original vs Uploaded Synthetic")
        if generated_analysis and uploaded_analysis:
            comparison_options.append("Generated vs Uploaded Synthetic")

        # ONLY SHOW THE SECTION IF THERE ARE COMPARISON OPTIONS AVAILABLE
        if not comparison_options:
            # Don't show anything - no section header, no info message
            return

        # Only show the comparison section when we have data to compare
        st.markdown("---")
        st.header("🔄 Comprehensive Data Comparison")

        # Show comparison method info
        if not self.ui.display_comparison_method_info():
            return

        # Comparison selection
        st.markdown("### 🎯 Select Comparison")
        selected_comparison = st.selectbox(
            "Choose datasets to compare:",
            comparison_options,
            key="comparison_selector"
        )

        comparison_method = st.session_state.get('comparison_method', 'code')
        button_text = f"🔄 Run Comprehensive Comparison ({comparison_method.title()} Method)"

        if st.button(button_text, type="primary"):
            with st.spinner(f"Running comprehensive comparison using {comparison_method} method..."):
                try:
                    if selected_comparison == "Original vs Generated Synthetic":
                        self.ui.display_comprehensive_comparison(
                            original_analysis,
                            generated_analysis,
                            self.comparator,
                            "Original vs Generated Synthetic Data"
                        )
                    elif selected_comparison == "Original vs Uploaded Synthetic":
                        self.ui.display_comprehensive_comparison(
                            original_analysis,
                            uploaded_analysis,
                            self.comparator,
                            "Original vs Uploaded Synthetic Data"
                        )
                    elif selected_comparison == "Generated vs Uploaded Synthetic":
                        self.ui.display_comprehensive_comparison(
                            generated_analysis,
                            uploaded_analysis,
                            self.comparator,
                            "Generated vs Uploaded Synthetic Data"
                        )
                except Exception as e:
                    st.error(f"Comparison failed: {str(e)}")

        # Quick comparison metrics if data is available
        if len(comparison_options) > 0:
            st.markdown("### 📊 Quick Comparison Overview")

            # Create quick comparison table
            if original_analysis:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Original Data**")
                    st.metric("Rows", f"{original_analysis['row_count']:,}")
                    st.metric("Columns", len(original_analysis["columns"]))
                    st.metric("Quality Score", f"{original_analysis['data_quality_score']}/100")
                    st.metric("Relationships", len(original_analysis.get("relationships", {}).get("one_to_many", [])))

                with col2:
                    if generated_analysis:
                        st.markdown("**Generated Synthetic**")
                        st.metric("Rows", f"{generated_analysis['row_count']:,}")
                        st.metric("Columns", len(generated_analysis["columns"]))
                        st.metric("Quality Score", f"{generated_analysis['data_quality_score']}/100")
                        st.metric("Relationships",
                                  len(generated_analysis.get("relationships", {}).get("one_to_many", [])))

                with col3:
                    if uploaded_analysis:
                        st.markdown("**Uploaded Synthetic**")
                        st.metric("Rows", f"{uploaded_analysis['row_count']:,}")
                        st.metric("Columns", len(uploaded_analysis["columns"]))
                        st.metric("Quality Score", f"{uploaded_analysis['data_quality_score']}/100")
                        st.metric("Relationships",
                                  len(uploaded_analysis.get("relationships", {}).get("one_to_many", [])))

    def _display_clean_footer(self):
        """Display clean application footer without enhanced features"""
        st.markdown("---")

        # Debug info in expander (only show if needed)
        with st.expander("🔧 Debug Info"):
            st.write("**Current Method Settings:**")
            st.write(f"Analysis Method: {st.session_state.get('analysis_method', 'code')}")
            st.write(f"Prompt Method: {st.session_state.get('prompt_method', 'code')}")
            st.write(f"Comparison Method: {st.session_state.get('comparison_method', 'code')}")

            # Show API status
            st.write("**API Status:**")
            st.write(f"Claude API: {'✅ Available' if st.session_state.get('claude_api_key') else '❌ Not configured'}")
            st.write(f"OpenAI API: {'✅ Available' if st.session_state.get('openai_api_key') else '❌ Not configured'}")
            st.write(f"Groq API: {'✅ Available' if st.session_state.get('groq_api_key') else '❌ Not configured'}")

            if st.button("Clear All Session State"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    app = TabularDataApp()
    app.run()