# =============================================================================
# File: ui_components.py (ENHANCED WITH DETAILED REPORTING)
"""UI components for Streamlit app with detailed comparison reports"""

import streamlit as st
import pandas as pd
import io
from typing import Dict, Any, List
from typing import Dict, Any, List, Tuple, Optional, Set  # Add Set here

# Import the Config class
from config import Config


class UIComponents:
    """Class to handle UI components and display functions"""

    @staticmethod
    def setup_page():
        """Set up the main page configuration"""
        st.set_page_config(page_title="Phantom Data", layout="wide")
        st.title("📊 Phantom Data")

    @staticmethod
    def setup_sidebar(config: Config):
        """Set up the sidebar with API configuration - FIXED VERSION"""
        with st.sidebar:
            st.header("🔑 API Configuration")

            # Initialize session state for API keys if not exists
            if 'claude_api_key' not in st.session_state:
                st.session_state.claude_api_key = config.CLAUDE_API_KEY
            if 'openai_api_key' not in st.session_state:
                st.session_state.openai_api_key = config.OPENAI_API_KEY
            if 'groq_api_key' not in st.session_state:
                st.session_state.groq_api_key = config.GROQ_API_KEY

            # Claude API Key input - FIXED
            claude_api_key = st.text_input(
                "Claude API Key",
                value=st.session_state.claude_api_key,
                type="password",
                help="Required for LLM-based analysis, prompt generation, and comparison",
                key="claude_api_key_input"  # Fixed: Added stable key
            )

            # Update session state only if value changed
            if claude_api_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = claude_api_key

            # Show status
            if st.session_state.claude_api_key:
                st.success("✅ Claude API Key configured")

            # OpenAI API Key input - FIXED
            openai_api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                help="For ChatGPT-based data generation",
                key="openai_api_key_input"  # Fixed: Added stable key
            )

            # Update session state only if value changed
            if openai_api_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = openai_api_key

            # Show status
            if st.session_state.openai_api_key:
                st.success("✅ OpenAI API Key configured")

            # Groq API Key input - FIXED
            groq_api_key = st.text_input(
                "Groq API Key (Optional)",
                value=st.session_state.groq_api_key,
                type="password",
                help="Optional: For Groq-based data generation",
                key="groq_api_key_input"  # Fixed: Added stable key
            )

            # Update session state only if value changed
            if groq_api_key != st.session_state.groq_api_key:
                st.session_state.groq_api_key = groq_api_key

            # Show status
            if st.session_state.groq_api_key:
                st.success("✅ Groq API Key configured")

            st.markdown("---")

            # Analysis method selection - FIXED
            st.markdown("### 🔬 Analysis Method")

            # Initialize if not exists
            if 'analysis_method' not in st.session_state:
                st.session_state.analysis_method = 'code'

            analysis_options = ["Code-based", "LLM-based"]
            current_index = 0 if st.session_state.analysis_method == 'code' else 1

            analysis_method = st.radio(
                "Choose analysis approach:",
                analysis_options,
                index=current_index,
                key="analysis_method_radio",
                help="Code-based: Fast, deterministic. LLM-based: More comprehensive, may find subtle patterns"
            )

            # Update session state
            st.session_state.analysis_method = "code" if analysis_method == "Code-based" else "llm"

            # Prompt generation method - FIXED
            st.markdown("### 📝 Prompt Generation")

            # Initialize if not exists
            if 'prompt_method' not in st.session_state:
                st.session_state.prompt_method = 'code'

            prompt_options = ["Code-based", "LLM-based"]
            prompt_current_index = 0 if st.session_state.prompt_method == 'code' else 1

            prompt_method = st.radio(
                "Choose prompt generation:",
                prompt_options,
                index=prompt_current_index,
                key="prompt_method_radio",
                help="Code-based: Template-based. LLM-based: Contextual and adaptive"
            )

            # Update session state
            st.session_state.prompt_method = "code" if prompt_method == "Code-based" else "llm"

            # Comparison method - FIXED
            st.markdown("### 📊 Comparison Method")

            # Initialize if not exists
            if 'comparison_method' not in st.session_state:
                st.session_state.comparison_method = 'code'

            comparison_options = ["Code-based", "LLM-based"]
            comparison_current_index = 0 if st.session_state.comparison_method == 'code' else 1

            comparison_method = st.radio(
                "Choose comparison approach:",
                comparison_options,
                index=comparison_current_index,
                key="comparison_method_radio",
                help="Code-based: Statistical metrics. LLM-based: Semantic understanding"
            )

            # Update session state
            st.session_state.comparison_method = "code" if comparison_method == "Code-based" else "llm"

            st.markdown("---")

            # Model Selection Section - FIXED
            st.markdown("### 🤖 Model Selection")

            # Initialize model selections if not exists
            if 'openai_model' not in st.session_state:
                st.session_state.openai_model = config.DEFAULT_OPENAI_MODEL
            if 'groq_model' not in st.session_state:
                st.session_state.groq_model = config.DEFAULT_MODEL

            # OpenAI Model Selection - FIXED
            if st.session_state.openai_api_key:
                st.markdown("**OpenAI Model:**")
                try:
                    current_model_index = config.OPENAI_MODELS.index(st.session_state.openai_model)
                except ValueError:
                    current_model_index = 0

                openai_model = st.selectbox(
                    "Select ChatGPT Model",
                    config.OPENAI_MODELS,
                    index=current_model_index,
                    key="openai_model_select"
                )
                st.session_state.openai_model = openai_model

            # Groq Model Selection - FIXED
            if st.session_state.groq_api_key:
                st.markdown("**Groq Model:**")
                try:
                    current_groq_index = config.AVAILABLE_MODELS.index(st.session_state.groq_model)
                except ValueError:
                    current_groq_index = 0

                groq_model = st.selectbox(
                    "Select Groq Model",
                    config.AVAILABLE_MODELS,
                    index=current_groq_index,
                    key="groq_model_select"
                )
                st.session_state.groq_model = groq_model

            st.markdown("---")
            st.markdown("### 🤖 Available Models")
            st.markdown("""
                    **Claude API:**
                    - claude-3-5-sonnet-20241022 (Recommended for analysis)

                    **OpenAI API:**
                    - gpt-5 (Latest & Most Advanced) 🆕
                    - gpt-4o (Excellent performance)
                    - gpt-4o-mini (Faster, cost-effective)
                    - gpt-4-turbo
                    - gpt-3.5-turbo (Legacy)

                    **Groq API (Optional):**
                    - llama-3.3-70b-versatile (Recommended)
                    - llama-3.1-8b-instant (Faster)
                    - mixtral-8x7b-32768
                    """)

            # Data Generation Priority
            st.markdown("### 🎯 Generation Strategy")
            st.markdown("""
                    **Auto-fallback order:**
                    1. Claude (Highest quality analysis)
                    2. OpenAI GPT-5 (Latest & most capable) 🆕
                    3. Groq (Fast fallback)

                    The system will automatically try the next API if one fails.
                    """)

            # Return the current values from session state
            return (
                st.session_state.claude_api_key,
                st.session_state.openai_api_key,
                st.session_state.groq_api_key
            )

    @staticmethod
    def display_mode_selection():
        """Display mode selection for upload vs schema"""
        st.markdown("### 🎯 Choose Your Approach")

        mode = st.radio(
            "How would you like to start?",
            ["upload", "schema"],
            format_func=lambda x: "📂 Upload & Analyze Existing Data" if x == "upload" else "🛠️ Define Custom Schema",
            horizontal=True,
            key="generation_mode_radio"
        )

        # Store in session state
        st.session_state.generation_mode = mode

        return mode

    @staticmethod
    def display_analysis_method_info():
        """Display information about selected analysis method"""
        analysis_method = st.session_state.get('analysis_method', 'code')

        if analysis_method == 'llm':
            if not st.session_state.get('claude_api_key'):
                st.warning("⚠️ Claude API key required for LLM-based analysis")
                return False
            st.info("🤖 Using LLM-based analysis - More comprehensive but takes longer")
        else:
            st.info("⚡ Using Code-based analysis - Fast and deterministic")

        return True

    @staticmethod
    def display_prompt_generation_info():
        """Display information about selected prompt generation method"""
        prompt_method = st.session_state.get('prompt_method', 'code')

        if prompt_method == 'llm':
            if not st.session_state.get('claude_api_key'):
                st.warning("⚠️ Claude API key required for LLM-based prompt generation")
                return False
            st.info("🤖 Using LLM-based prompt generation - Adaptive and contextual")
        else:
            st.info("⚡ Using Code-based prompt generation - Template-based")

        return True

    @staticmethod
    def display_comparison_method_info():
        """Display information about selected comparison method"""
        comparison_method = st.session_state.get('comparison_method', 'code')

        if comparison_method == 'llm':
            if not st.session_state.get('claude_api_key'):
                st.warning("⚠️ Claude API key required for LLM-based comparison")
                return False
            st.info("🤖 Using LLM-based comparison - Semantic understanding")
        else:
            st.info("⚡ Using Code-based comparison - Statistical metrics")

        return True

    @staticmethod
    def display_synthetic_data(df_synthetic: pd.DataFrame):
        """Display generated synthetic data"""
        st.success(f"✅ Generated {len(df_synthetic)} rows with {len(df_synthetic.columns)} columns")

        # Create display-friendly version
        data_display = df_synthetic.copy()
        for col in data_display.columns:
            if data_display[col].dtype == 'object':
                data_display[col] = data_display[col].astype(str)
                data_display[col] = data_display[col].replace('nan', '')

        # Display preview
        st.subheader("📋 Generated Synthetic Data Preview")
        st.dataframe(data_display.head(10), use_container_width=True)

        # Download button
        csv_data = df_synthetic.to_csv(index=False)
        st.download_button(
            label="📥 Download Synthetic Data (CSV)",
            data=csv_data,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )

    @staticmethod
    def display_synthetic_upload_section():
        """Display section for uploading synthetic data for comparison"""
        st.markdown("---")
        st.header("📤 Upload Synthetic Data for Comparison")
        st.info(
            "💡 Upload a synthetic dataset to compare with your original data or previously generated synthetic data")

        uploaded_synthetic = st.file_uploader(
            "Upload Synthetic CSV file",
            type=["csv"],
            key="synthetic_upload"
        )

        if uploaded_synthetic:
            try:
                synthetic_data = pd.read_csv(uploaded_synthetic)
                st.success(
                    f"Uploaded synthetic data: {uploaded_synthetic.name} with {len(synthetic_data)} rows and {len(synthetic_data.columns)} columns")

                # Create display-friendly version
                data_display = synthetic_data.copy()
                for col in data_display.columns:
                    if data_display[col].dtype == 'object':
                        data_display[col] = data_display[col].astype(str)
                        data_display[col] = data_display[col].replace('nan', '')

                # Display preview
                st.subheader("📋 Uploaded Synthetic Data Preview")
                st.dataframe(data_display.head(10), use_container_width=True)

                # Store in session state
                st.session_state.uploaded_synthetic_data = synthetic_data

                # Analyze uploaded synthetic data with method selection
                analysis_method = st.session_state.get('analysis_method', 'code')
                button_text = f"🔍 Analyze Uploaded Synthetic Data ({analysis_method.title()})"

                if st.button(button_text):
                    with st.spinner(f"Analyzing uploaded synthetic data using {analysis_method} method..."):
                        if analysis_method == 'llm':
                            claude_api_key = st.session_state.get('claude_api_key')
                            if not claude_api_key:
                                st.error("Claude API key required for LLM analysis")
                                return None

                            from llm_analyzer import LLMAnalyzer
                            llm_analyzer = LLMAnalyzer()
                            try:
                                uploaded_analysis = llm_analyzer.analyze_data_with_llm(synthetic_data, claude_api_key)
                                st.session_state.uploaded_synthetic_analysis = uploaded_analysis
                                st.success("✅ LLM analysis completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"LLM analysis failed: {str(e)}")
                        else:
                            from data_analysis import DataAnalyzer
                            from config import Config
                            analyzer = DataAnalyzer(Config())
                            uploaded_analysis = analyzer.analyze_data(synthetic_data)
                            st.session_state.uploaded_synthetic_analysis = uploaded_analysis
                            st.rerun()

                return synthetic_data

            except Exception as e:
                st.error(f"Error reading synthetic file: {str(e)}")
                return None

        return None

    @staticmethod
    def display_comprehensive_comparison(original_analysis: Dict[str, Any],
                                         synthetic_analysis: Dict[str, Any],
                                         comparator,
                                         comparison_title: str = "Original vs Synthetic Data"):
        """Display comprehensive comparison with detailed report format and configurable integrity checking"""
        st.markdown(f"### 📊 {comparison_title}")

        # Add integrity configuration UI - NEW
        integrity_excluded_fields = UIComponents.display_integrity_configuration(original_analysis)

        # Check if LLM comparison is selected and available
        comparison_method = st.session_state.get('comparison_method', 'code')

        if comparison_method == 'llm':
            claude_api_key = st.session_state.get('claude_api_key')
            if not claude_api_key:
                st.error("Claude API key required for LLM-based comparison")
                return

            # SAFELY get original dataframes for LLM comparison
            try:
                original_df = st.session_state.get('original_data')
                synthetic_df = None

                # Safely get synthetic dataframe
                if st.session_state.get('generated_synthetic_data') is not None:
                    synthetic_df = st.session_state.get('generated_synthetic_data')
                elif st.session_state.get('uploaded_synthetic_data') is not None:
                    synthetic_df = st.session_state.get('uploaded_synthetic_data')

                # Check if we have the required dataframes
                if original_df is None or synthetic_df is None:
                    st.error("Original and synthetic dataframes required for LLM comparison")
                    st.info("Falling back to code-based comparison...")
                else:
                    with st.spinner("Running LLM-based comparison..."):
                        try:
                            from llm_analyzer import LLMAnalyzer
                            llm_analyzer = LLMAnalyzer()
                            llm_comparison = llm_analyzer.compare_data_with_llm(
                                original_df, synthetic_df, original_analysis, synthetic_analysis, claude_api_key
                            )
                            UIComponents._display_llm_comparison_results(llm_comparison)
                            return
                        except Exception as e:
                            st.error(f"LLM comparison failed: {str(e)}")
                            st.info("Falling back to code-based comparison...")

            except Exception as e:
                st.error(f"Error accessing dataframes for LLM comparison: {str(e)}")
                st.info("Falling back to code-based comparison...")

        # CODE-BASED COMPARISON - NO DATAFRAME OPERATIONS - UPDATED WITH INTEGRITY EXCLUSIONS
        try:
            # Generate detailed report using analysis data only - NO DATAFRAMES - WITH EXCLUSIONS
            detailed_report = comparator.generate_detailed_report(
                original_analysis, synthetic_analysis, None, None, integrity_excluded_fields
            )

            # Display the detailed report
            UIComponents._display_detailed_comparison_report(detailed_report)

        except Exception as e:
            st.error(f"Code-based comparison failed: {str(e)}")
            import traceback
            st.error("Full error trace:")
            st.code(traceback.format_exc())

    @staticmethod
    def _display_detailed_comparison_report(report: Dict[str, Any]):
        """Display the detailed comparison report matching the UI format"""

        # Overall Comparison Scores
        st.markdown("#### 🎯 Overall Comparison Scores")
        scores = report["overall_scores"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{scores['overall_score']:.2%}")
        with col2:
            st.metric("Column Similarity", f"{scores['column_similarity']:.2%}")
        with col3:
            st.metric("Relationship Preservation", f"{scores['relationship_preservation']:.2%}")
        with col4:
            st.metric("Pattern Similarity", f"{scores['pattern_similarity']:.2%}")

        # Row Count Information Section
        st.markdown("#### 📊 Row Count Information")

        # Get row counts from the basic metrics
        basic_metrics = report["basic_metrics"]
        orig_rows = None
        synth_rows = None

        for metric_info in basic_metrics:
            if metric_info["metric"] == "Row Count":
                orig_rows = metric_info["original"]
                synth_rows = metric_info["synthetic"]
                break

        if orig_rows is not None and synth_rows is not None:
            UIComponents._display_row_count_info_box(orig_rows, synth_rows)

        # Basic Metrics Comparison
        st.markdown("#### 📈 Basic Metrics Comparison")
        basic_metrics = report["basic_metrics"]

        basic_df_data = []
        for metric_info in basic_metrics:
            basic_df_data.append([
                metric_info["metric"],
                str(metric_info["original"]),
                str(metric_info["synthetic"]),
                str(metric_info["difference"]),
                metric_info["similarity"]
            ])

        if basic_df_data:
            basic_df = pd.DataFrame(
                basic_df_data,
                columns=["Metric", "Original", "Synthetic", "Difference", "Similarity"]
            )
            for col in basic_df.columns:
                basic_df[col] = basic_df[col].astype(str)
            st.dataframe(basic_df, use_container_width=True)

        # Relationship Comparison
        if "relationship_comparison" in report and report["relationship_comparison"].get("one_to_many_relationships"):
            st.markdown("#### 🔗 Relationship Comparison")

            rel_data = report["relationship_comparison"]["one_to_many_relationships"]
            rel_score = report["relationship_comparison"]["relationship_preservation_score"]

            st.markdown(f"**Relationship Preservation Score: {rel_score:.2%}**")

            rel_df_data = []
            for rel in rel_data:
                rel_df_data.append([
                    rel["relationship"],
                    f"{rel['original_ratio']:.2f}",
                    f"{rel['synthetic_ratio']:.2f}",
                    f"{rel['difference']:.2f}",
                    rel["similarity"]
                ])

            if rel_df_data:
                rel_df = pd.DataFrame(
                    rel_df_data,
                    columns=["Relationship", "Original Ratio", "Synthetic Ratio", "Difference", "Similarity"]
                )
                for col in rel_df.columns:
                    rel_df[col] = rel_df[col].astype(str)
                st.dataframe(rel_df, use_container_width=True)

        # Column-by-Column Detailed Comparison
        st.markdown("#### 🔍 Column-by-Column Detailed Comparison")

        column_comparisons = report["column_comparison"]

        col_df_data = []
        for col_comp in column_comparisons:
            # Calculate uniqueness similarity percentage
            orig_unique = col_comp.get("original_unique", 0)
            synth_unique = col_comp.get("synthetic_unique", 0)
            uniqueness_sim = col_comp.get("uniqueness_similarity", 0)

            col_df_data.append([
                col_comp["column"],
                col_comp["original_type"],
                col_comp["synthetic_type"],
                f"{orig_unique} ({(orig_unique / max(orig_unique, synth_unique, 1) * 100):.0f}%)" if orig_unique > 0 else "0",
                f"{synth_unique} ({(synth_unique / max(orig_unique, synth_unique, 1) * 100):.0f}%)" if synth_unique > 0 else "0",
                f"{uniqueness_sim:.2%}",
                col_comp.get("pattern_quality", "🟡 Unknown")
            ])

        if col_df_data:
            col_df = pd.DataFrame(
                col_df_data,
                columns=["Column", "Original Type", "Synthetic Type", "Original Unique",
                         "Synthetic Unique", "Uniqueness Similarity", "Pattern Quality"]
            )
            for col in col_df.columns:
                col_df[col] = col_df[col].astype(str)
            st.dataframe(col_df, use_container_width=True)

        # Pattern Comparison
        st.markdown("#### 🧠 Pattern Comparison")
        pattern_analysis = report["pattern_analysis"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Common Patterns:**")
            common_patterns = pattern_analysis.get("common_patterns", [])
            if common_patterns:
                for pattern in common_patterns:
                    st.write(f"✅ {pattern}")
            else:
                st.write("No common patterns found")

            st.markdown("**Only in Original:**")
            orig_only = pattern_analysis.get("only_in_original", [])
            if orig_only:
                for pattern in orig_only:
                    st.write(f"⚠️ {pattern}")
            else:
                st.write("No patterns exclusive to original")

        with col2:
            st.markdown("**Only in Synthetic:**")
            synth_only = pattern_analysis.get("only_in_synthetic", [])
            if synth_only:
                for pattern in synth_only:
                    st.write(f"📊 {pattern}")
            else:
                st.write("No patterns exclusive to synthetic")

            pattern_similarity = pattern_analysis.get("pattern_similarity", 0)
            st.metric("Pattern Similarity", f"{pattern_similarity:.2%}")

        # Value Distribution Analysis
        if "value_distribution" in report and report["value_distribution"]:
            st.markdown("#### 📊 Value Distribution Analysis")

            dist_data = []
            for col, analysis in report["value_distribution"].items():
                dist_data.append([
                    col,
                    analysis["original_unique"],
                    analysis["synthetic_unique"],
                    f"{analysis['diversity_preservation']:.2%}",
                    f"{analysis['overlap_score']:.2%}",
                    UIComponents._get_quality_indicator(analysis['diversity_preservation'])
                ])

            if dist_data:
                dist_df = pd.DataFrame(
                    dist_data,
                    columns=["Column", "Original Unique", "Synthetic Unique",
                             "Diversity Preservation", "Value Overlap", "Quality"]
                )
                for col in dist_df.columns:
                    dist_df[col] = dist_df[col].astype(str)
                st.dataframe(dist_df, use_container_width=True)

        # Data Integrity Check - UPDATED TO HANDLE EXCLUSIONS
        if "data_integrity" in report and report["data_integrity"]:
            st.markdown("#### 🔄 Data Integrity Check")

            # Show exclusion info if fields were excluded
            if 'integrity_excluded_fields' in st.session_state and st.session_state.integrity_excluded_fields:
                excluded_fields = st.session_state.integrity_excluded_fields
                st.info(
                    f"ℹ️ {len(excluded_fields)} field(s) excluded from integrity check: {', '.join(sorted(excluded_fields))}")

            integrity_data = []
            for col, check in report["data_integrity"].items():
                reuse_pct = check["reuse_percentage"]
                status = "🟢 Perfect" if reuse_pct == 0 else "🟡 Minor concern" if reuse_pct < 0.1 else "🔴 Significant reuse"

                integrity_data.append([
                    col,
                    f"{check['reused_values']}/{check['total_synthetic']}",
                    f"{reuse_pct:.1%}",
                    status
                ])

            if integrity_data:
                integrity_df = pd.DataFrame(
                    integrity_data,
                    columns=["Column", "Reused Values", "Reuse Percentage", "Status"]
                )
                for col in integrity_df.columns:
                    integrity_df[col] = integrity_df[col].astype(str)
                st.dataframe(integrity_df, use_container_width=True)
            else:
                st.info("All fields have been excluded from integrity checking.")

        # Summary & Recommendations
        st.markdown("#### 📋 Summary & Recommendations")

        recommendations = report["summary_recommendations"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🟢 Strengths:**")
            strengths = recommendations.get("strengths", [])
            if strengths:
                for strength in strengths:
                    st.write(f"✅ {strength}")
            else:
                st.write("No major strengths identified")

        with col2:
            st.markdown("**🔴 Critical Issues:**")
            issues = recommendations.get("critical_issues", [])
            if issues:
                for issue in issues:
                    st.write(f"🚨 {issue}")
            else:
                st.write("No critical issues found")

        with col3:
            st.markdown("**🔧 Improvements:**")
            improvements = recommendations.get("improvements", [])
            if improvements:
                for improvement in improvements:
                    st.write(f"🔧 {improvement}")
            else:
                st.write("No specific improvements needed")

        # Overall Assessment
        st.markdown("#### 🎯 Overall Assessment")

        grade = recommendations.get("overall_grade", "C")
        assessment = recommendations.get("overall_assessment", "Assessment not available")
        overall_score = recommendations.get("overall_score", 0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Grade", grade)
            st.metric("Overall Score", f"{overall_score:.1%}")

        with col2:
            # Color code the assessment based on grade
            if grade in ['A', 'B']:
                st.success(f"**Assessment:** {assessment}")
            elif grade == 'C':
                st.warning(f"**Assessment:** {assessment}")
            else:
                st.error(f"**Assessment:** {assessment}")

    # ALSO ADD THIS HELPER METHOD TO YOUR UIComponents CLASS
    @staticmethod
    def _display_row_count_info_box(orig_rows: int, synth_rows: int):
        """Display informative box about row count differences"""

        if orig_rows == synth_rows:
            st.success(f"✅ Row counts match: {orig_rows:,} rows in both datasets")
        else:
            difference = synth_rows - orig_rows
            percentage_change = (difference / orig_rows) * 100 if orig_rows > 0 else 0

            if difference > 0:
                st.info(f"ℹ️ Synthetic data has **{difference:,} more rows** than original ({percentage_change:+.1f}%)")
            else:
                st.info(
                    f"ℹ️ Synthetic data has **{abs(difference):,} fewer rows** than original ({percentage_change:.1f}%)")

            st.caption("💡 Row count differences are intentional and don't affect data quality assessment")

    @staticmethod
    def display_integrity_configuration(analysis: Dict[str, Any]) -> Set[str]:
        """Display data integrity configuration interface"""

        if "columns" not in analysis:
            return set()

        st.markdown("#### ⚙️ Data Integrity Check Configuration")

        with st.expander("Configure Fields to Check for Value Reuse", expanded=False):
            st.info("""
            💡 **About Data Integrity Checking**: This identifies values reused between original and synthetic data.

            • **Categorical fields** (DeviceType, DeviceOS) can legitimately reuse values
            • **PII fields** (EmailID, phone numbers) should NOT reuse values
            • **Identifier fields** (IDs, UUIDs) should be unique
            """)

            # Get default exclusions
            default_excluded = set()
            field_categories = {}

            for col_name, col_info in analysis["columns"].items():
                category = UIComponents._classify_field_for_integrity(col_name, col_info)
                field_categories[col_name] = category

                # Default exclude categorical fields
                if category == 'categorical':
                    default_excluded.add(col_name)

            # Initialize session state
            if 'integrity_excluded_fields' not in st.session_state:
                st.session_state.integrity_excluded_fields = default_excluded

            # Group fields by category
            categories = {
                'pii': {'label': '🔐 Personal Information', 'desc': 'Should not reuse values'},
                'identifier': {'label': '🆔 Unique Identifiers', 'desc': 'Should be unique'},
                'categorical': {'label': '📂 Categorical Values', 'desc': 'Can reuse values'},
                'coordinate': {'label': '📍 Coordinates', 'desc': 'Geographic data'},
                'technical': {'label': '⚙️ Technical Data', 'desc': 'IP addresses, URLs'},
                'other': {'label': '📄 Other Fields', 'desc': 'Other data types'}
            }

            col1, col2 = st.columns([3, 1])

            with col1:
                for category, info in categories.items():
                    fields_in_category = [col for col, cat in field_categories.items() if cat == category]
                    if not fields_in_category:
                        continue

                    st.markdown(f"**{info['label']}** - {info['desc']}")

                    # Individual field checkboxes
                    for field in fields_in_category:
                        field_included = field not in st.session_state.integrity_excluded_fields
                        new_included = st.checkbox(
                            f"{field}",
                            value=field_included,
                            key=f"integrity_field_{field}"
                        )

                        if new_included != field_included:
                            if new_included:
                                st.session_state.integrity_excluded_fields.discard(field)
                            else:
                                st.session_state.integrity_excluded_fields.add(field)

                    st.markdown("")

            with col2:
                st.markdown("**Quick Actions:**")

                if st.button("🎯 Smart Defaults"):
                    st.session_state.integrity_excluded_fields = default_excluded.copy()
                    st.rerun()

                if st.button("✅ Include All"):
                    st.session_state.integrity_excluded_fields = set()
                    st.rerun()

                # Show status
                excluded_count = len(st.session_state.integrity_excluded_fields)
                total_count = len(analysis["columns"])

                st.metric("Fields Checked", total_count - excluded_count)
                st.metric("Fields Excluded", excluded_count)

        return st.session_state.integrity_excluded_fields

    @staticmethod
    def _classify_field_for_integrity(col_name: str, col_info: Dict[str, Any]) -> str:
        """Classify field category for integrity UI"""
        col_name_lower = col_name.lower()

        # PII fields
        if any(keyword in col_name_lower for keyword in ['email', 'phone', 'name', 'ssn', 'passport']):
            return 'pii'

        # Identifiers
        if any(keyword in col_name_lower for keyword in ['id', 'uuid', 'key', 'identifier']):
            return 'identifier'

        # Categorical
        if any(keyword in col_name_lower for keyword in
               ['type', 'category', 'status', 'os', 'device', 'brand', 'model']):
            return 'categorical'

        # Coordinates
        if any(keyword in col_name_lower for keyword in ['lat', 'lon', 'latitude', 'longitude']):
            return 'coordinate'

        # Technical
        if any(keyword in col_name_lower for keyword in ['ip', 'url', 'address', 'host']):
            return 'technical'

        unique_pct = col_info["stats"].get("unique_percent", 0)
        if unique_pct > 90:
            return 'identifier'
        elif unique_pct < 20:
            return 'categorical'

        return 'other'
    @staticmethod
    def _get_quality_indicator(score: float) -> str:
        """Get quality indicator based on score"""
        if score >= 0.9:
            return "🟢 Excellent"
        elif score >= 0.7:
            return "🟡 Good"
        elif score >= 0.5:
            return "🟡 Moderate"
        else:
            return "🔴 Poor"

    @staticmethod
    def _display_llm_comparison_results(llm_comparison: Dict[str, Any]):
        """Display LLM-based comparison results"""
        st.markdown("#### 🤖 LLM-Based Comparison Results")

        # Overall Scores
        if "overall_scores" in llm_comparison:
            scores = llm_comparison["overall_scores"]
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Overall Similarity", f"{scores.get('overall_similarity', 0):.2%}")
            with col2:
                st.metric("Structural Fidelity", f"{scores.get('structural_fidelity', 0):.2%}")
            with col3:
                st.metric("Statistical Fidelity", f"{scores.get('statistical_fidelity', 0):.2%}")
            with col4:
                st.metric("Relationship Preservation", f"{scores.get('relationship_preservation', 0):.2%}")
            with col5:
                st.metric("Pattern Fidelity", f"{scores.get('pattern_fidelity', 0):.2%}")

        # Basic Metrics
        if "basic_metrics" in llm_comparison:
            st.markdown("#### 📊 Structural Comparison")
            basic = llm_comparison["basic_metrics"]

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Structural Matches:**")
                st.write(f"✅ Row count match: {basic.get('row_count_match', False)}")
                st.write(f"✅ Column count match: {basic.get('column_count_match', False)}")
            with col2:
                st.write(f"✅ Column names match: {basic.get('column_names_match', False)}")
                st.write(f"✅ Data types match: {basic.get('data_types_match', False)}")

        # Column Comparisons
        if "column_comparisons" in llm_comparison:
            st.markdown("#### 🔍 Column-by-Column Analysis")

            column_data = []
            for col_comp in llm_comparison["column_comparisons"]:
                issues_str = "; ".join(col_comp.get("issues", []))
                recommendations_str = "; ".join(col_comp.get("recommendations", []))

                column_data.append([
                    str(col_comp.get("column", "")),
                    "✅" if col_comp.get("type_match", False) else "❌",
                    f"{col_comp.get('statistical_similarity', 0):.2%}",
                    f"{col_comp.get('distribution_similarity', 0):.2%}",
                    str(issues_str) if issues_str else "None",
                    str(recommendations_str) if recommendations_str else "None"
                ])

            if column_data:
                df_columns = pd.DataFrame(
                    column_data,
                    columns=["Column", "Type Match", "Statistical Similarity",
                             "Distribution Similarity", "Issues", "Recommendations"]
                )
                for col in df_columns.columns:
                    df_columns[col] = df_columns[col].astype(str)
                st.dataframe(df_columns, use_container_width=True)

        # Additional LLM comparison sections can be added here following the same pattern...

    @staticmethod
    def display_analysis_results(analysis: Dict[str, Any], analyzer):
        """Display data analysis results"""
        analysis_method = st.session_state.get('analysis_method', 'code')
        method_indicator = "🤖 LLM Analysis" if analysis_method == 'llm' else "⚡ Code Analysis"

        st.markdown(f"### 📈 Data Analysis Results ({method_indicator})")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Quality Score", f"{analysis['data_quality_score']}/100")
        with col2:
            st.metric("Total Rows", analysis['row_count'])
        with col3:
            st.metric("Total Nulls", analysis['total_nulls'])

        # Column Summary
        st.markdown("### 📊 Column Summary")
        summary_data = []
        for col, info in analysis["columns"].items():
            if analysis_method == 'llm':
                # For LLM analysis, format stats differently
                key_stats = UIComponents._format_llm_stats(info)
            else:
                # Use existing analyzer method for code-based analysis
                key_stats = analyzer.get_key_stats(info)

            row = [str(col), str(info["type"]), str(info["stats"]["count"]), str(key_stats)]
            summary_data.append(row)

        df_summary = pd.DataFrame(
            summary_data,
            columns=["Column", "Type", "Non-Null Count", "Key Statistics"]
        )

        # Ensure all columns are string type for Arrow compatibility
        for col in df_summary.columns:
            df_summary[col] = df_summary[col].astype(str)

        st.dataframe(df_summary, use_container_width=True)

        # Display relationships and patterns (same logic for both methods)
        UIComponents._display_relationships_and_patterns(analysis)

    @staticmethod
    def _format_llm_stats(info: Dict[str, Any]) -> str:
        """Format statistics from LLM analysis for display"""
        type_ = info["type"]
        stats = info["stats"]
        key_stats = ""

        if type_ in ["integer", "float"]:
            min_val = stats.get('min', 'N/A')
            max_val = stats.get('max', 'N/A')
            mean_val = stats.get('mean', 'N/A')
            key_stats = f"Range: {min_val} - {max_val}, Mean: {mean_val}"
        elif type_ == "datetime":
            min_val = stats.get('min', 'N/A')
            max_val = stats.get('max', 'N/A')
            key_stats = f"Range: {min_val} - {max_val}"
        elif type_ == "boolean":
            true_pct = stats.get('true_percentage', 'N/A')
            key_stats = f"True: {true_pct}%"
        else:  # string
            unique = stats.get('unique', 'N/A')
            unique_pct = stats.get('unique_percent', 'N/A')
            key_stats = f"Unique: {unique} ({unique_pct}%)"
            if stats.get("most_common") and len(stats["most_common"]) >= 2:
                mc = stats["most_common"]
                key_stats += f", Most common: {mc[0]} ({mc[1]})"

        null_pct = stats.get('null_percentage', 0)
        if null_pct > 0:
            key_stats += f", Nulls: {null_pct}%"

        return key_stats

    @staticmethod
    def _display_relationships_and_patterns(analysis: Dict[str, Any]):
        """Display relationships and patterns section"""
        # NEW: Display Column Relationships
        if analysis.get("relationships"):
            relationships = analysis["relationships"]

            # One-to-many relationships
            if relationships.get("one_to_many"):
                st.markdown("### 🔗 One-to-Many Relationships")
                for rel in relationships["one_to_many"]:
                    st.info(f"📧 **{rel['parent']}** → **{rel['child']}**: {rel['description']}")

            # Many-to-many relationships
            if relationships.get("many_to_many"):
                st.markdown("### 🔄 Many-to-Many Relationships")
                for rel in relationships["many_to_many"]:
                    st.info(f"🔄 **{rel['col1']}** ↔ **{rel['col2']}**: {rel['description']}")

            # Functional dependencies
            if relationships.get("functional_dependencies"):
                st.markdown("### ⚙️ Functional Dependencies")
                for rel in relationships["functional_dependencies"]:
                    st.info(f"⚙️ {rel['description']}")

            # Value patterns
            if relationships.get("value_mappings"):
                st.markdown("### 🎯 Value Patterns")
                for col, mappings in relationships["value_mappings"].items():
                    if mappings:
                        st.write(f"**{col}:**")
                        for pattern_type, pattern_info in mappings.items():
                            if pattern_type == "email_domains":
                                with st.expander(f"📧 Email Domain Analysis"):
                                    st.write(f"**Total domains:** {pattern_info['total_domains']}")
                                    st.write("**Top domains:**")
                                    for domain, count in pattern_info["top_domains"].items():
                                        st.write(f"  - {domain}: {count} occurrences")
                            elif pattern_type == "phone_patterns":
                                with st.expander(f"📞 Phone Pattern Analysis"):
                                    st.write(f"**Total patterns:** {pattern_info['total_patterns']}")
                                    st.write("**Top area/country codes:**")
                                    for pattern, count in pattern_info["top_patterns"].items():
                                        st.write(f"  - {pattern}: {count} occurrences")
                            elif pattern_type == "uuid_like":
                                st.write(
                                    f"🆔 Contains UUID-like identifiers (avg length: {pattern_info['avg_length']:.1f})")
                            elif pattern_type == "common_prefixes":
                                with st.expander(f"🏷️ Common Prefix Analysis"):
                                    st.write("**Common prefixes:**")
                                    for prefix, count in pattern_info["top_prefixes"].items():
                                        st.write(f"  - {prefix}_*: {count} occurrences")

        # Existing patterns section
        st.markdown("### 🧠 Detected Patterns")
        if analysis["patterns"]:
            with st.container():
                st.markdown("""
                <div style='background-color: #fffbea; padding: 1rem; border-radius: 0.5rem; border: 1px solid #ffe58f; color: black;'>
                <ul style='margin: 0; padding-left: 1.25rem;'>
                {} 
                </ul>
                </div>
                """.format("\n".join([f"<li>{p}</li>" for p in analysis["patterns"]])),
                            unsafe_allow_html=True)
        else:
            st.info("No specific patterns detected in the data.")

        # Correlations
        if analysis.get("correlations", {}).get("high_correlations"):
            st.markdown("### 📊 High Correlations")
            for col1, col2, corr_val in analysis["correlations"]["high_correlations"]:
                relationship = "positive" if corr_val > 0 else "negative"
                st.write(f"**{col1}** ↔ **{col2}**: {relationship} correlation ({corr_val})")

    @staticmethod
    def display_schema_builder():
        """Display the schema builder interface"""
        st.subheader("📋 Schema Builder")

        # Display existing columns
        if st.session_state.get('schema', []):
            st.write("**Added Columns:**")
            for i, col_def in enumerate(st.session_state.schema, 1):
                col_info = f"{i}. **{col_def['name']}** ({col_def['type']})"
                if col_def.get('unique'):
                    col_info += " - Unique"
                st.write(col_info)
            st.markdown("---")

        return UIComponents._schema_form()

    @staticmethod
    def _schema_form():
        """Schema form for adding columns"""
        st.write("**Add New Column**")
        col_name = st.text_input(
            "Column Name",
            placeholder="e.g., user_id, name, age",
            key=f"col_name_{st.session_state.get('column_counter', 0)}"
        )

        with st.form("add_column_form", clear_on_submit=True):
            col_type = st.selectbox("Data Type", ["integer", "float", "string", "boolean", "datetime"])

            # Type-specific options
            type_params = {}

            if col_type in ["integer", "float"]:
                type_params['min'] = st.number_input("Minimum Value (optional)", value=None)
                type_params['max'] = st.number_input("Maximum Value (optional)", value=None)
                type_params['mean'] = st.number_input("Mean (optional)", value=None)
                type_params['std'] = st.number_input("Standard Deviation (optional)", value=None)

            elif col_type == "string":
                type_params['unique_percent'] = st.slider("Uniqueness %", 0, 100, 50)
                type_params['categories'] = st.text_input("Categories (comma-separated, optional)")
                type_params['pattern'] = st.text_input("Pattern (optional, e.g., email, phone)")

            elif col_type == "boolean":
                type_params['true_percentage'] = st.slider("True %", 0, 100, 50)

            elif col_type == "datetime":
                type_params['start_date'] = st.date_input("Start Date")
                type_params['end_date'] = st.date_input("End Date")

            # Common options
            null_percentage = st.slider("Null %", 0, 100, 0)
            unique_constraint = st.checkbox("Unique constraint")

            submitted = st.form_submit_button("➕ Add Column")

            if submitted and col_name:
                # Prepare column definition
                col_def = {
                    'name': col_name,
                    'type': col_type,
                    'null_percentage': null_percentage,
                    'unique': unique_constraint,
                    **type_params
                }

                # Process categories for string type
                if col_type == "string" and type_params.get('categories'):
                    col_def['categories'] = [cat.strip() for cat in type_params['categories'].split(',')]

                # Add to schema
                if 'schema' not in st.session_state:
                    st.session_state.schema = []

                st.session_state.schema.append(col_def)
                st.session_state.column_counter = st.session_state.get('column_counter', 0) + 1

                st.success(f"✅ Added column: {col_name}")
                st.rerun()

            elif submitted:
                st.error("❌ Please enter a column name")

        return None

    @staticmethod
    def _display_row_count_info_box(orig_rows: int, synth_rows: int):
        """Display informative box about row count differences"""

        if orig_rows == synth_rows:
            st.success(f"✅ Row counts match: {orig_rows} rows in both datasets")
        else:
            difference = synth_rows - orig_rows
            percentage_change = (difference / orig_rows) * 100 if orig_rows > 0 else 0

            if difference > 0:
                st.info(f"ℹ️ Synthetic data has **{difference} more rows** than original ({percentage_change:+.1f}%)")
            else:
                st.info(
                    f"ℹ️ Synthetic data has **{abs(difference)} fewer rows** than original ({percentage_change:.1f}%)")

            st.caption("💡 Row count differences are intentional and don't affect data quality assessment")