# =============================================================================
# File: llm_analyzer.py
"""LLM-based data analysis utilities - COMPLETE WITH OPENAI AND GPT-5 SUPPORT"""

import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional
import anthropic
from openai import OpenAI
from config import Config


class LLMAnalyzer:
    """Class to handle LLM-based data analysis using Claude or OpenAI API"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def analyze_data_with_llm(self, df: pd.DataFrame, claude_api_key: str = None,
                              openai_api_key: str = None, openai_model: str = None) -> Optional[Dict[str, Any]]:
        """Analyze data using Claude or OpenAI API with automatic fallback"""

        # Try Claude first if available
        if claude_api_key:
            try:
                return self._analyze_with_claude(df, claude_api_key)
            except Exception as claude_error:
                print(f"Claude analysis failed: {claude_error}")

                # Try OpenAI as fallback
                if openai_api_key:
                    print("Falling back to OpenAI...")
                    return self._analyze_with_openai(df, openai_api_key, openai_model)
                else:
                    raise claude_error

        # Try OpenAI if Claude not available
        elif openai_api_key:
            return self._analyze_with_openai(df, openai_api_key, openai_model)

        else:
            raise ValueError("Either Claude or OpenAI API key is required for LLM analysis")

    def _analyze_with_claude(self, df: pd.DataFrame, api_key: str) -> Dict[str, Any]:
        """Analyze data using Claude API"""
        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Prepare data sample for analysis
            data_sample = self._prepare_data_sample(df)

            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(df, data_sample)

            # Make API call
            response = client.messages.create(
                model=self.config.CLAUDE_MODEL,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ]
            )

            # Parse response
            analysis_result = self._parse_llm_analysis_response(response.content[0].text)
            return analysis_result

        except Exception as e:
            raise Exception(f"Error analyzing data with Claude: {str(e)}")

    def _analyze_with_openai(self, df: pd.DataFrame, api_key: str, model: str = None) -> Dict[str, Any]:
        """Analyze data using OpenAI API - UPDATED WITH TOKEN PARAMETER FIX"""
        if model is None:
            model = self.config.DEFAULT_OPENAI_MODEL

        try:
            client = OpenAI(api_key=api_key)

            # Prepare data sample for analysis
            data_sample = self._prepare_data_sample(df)

            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(df, data_sample)

            # Handle different token parameter for different models
            completion_params = {
                "model": model,
                "temperature": 0.1,  # Lower temperature for more consistent analysis
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data analyst expert. Analyze tabular datasets and provide comprehensive analysis in the exact JSON format requested. Be precise and thorough."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ]
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if model in self.config.NEWER_OPENAI_MODELS:
                completion_params["max_completion_tokens"] = 4000
            else:
                completion_params["max_tokens"] = 4000

            # Make API call
            response = client.chat.completions.create(**completion_params)

            # Parse response
            analysis_result = self._parse_llm_analysis_response(response.choices[0].message.content)
            return analysis_result

        except Exception as e:
            raise Exception(f"Error analyzing data with OpenAI: {str(e)}")

    def generate_prompt_with_llm(self, analysis: Dict[str, Any], num_rows: int,
                                 claude_api_key: str = None, openai_api_key: str = None,
                                 openai_model: str = None) -> Optional[str]:
        """Generate data generation prompt using Claude or OpenAI API"""

        # Try Claude first if available
        if claude_api_key:
            try:
                return self._generate_prompt_with_claude(analysis, num_rows, claude_api_key)
            except Exception as claude_error:
                print(f"Claude prompt generation failed: {claude_error}")

                # Try OpenAI as fallback
                if openai_api_key:
                    print("Falling back to OpenAI for prompt generation...")
                    return self._generate_prompt_with_openai(analysis, num_rows, openai_api_key, openai_model)
                else:
                    raise claude_error

        # Try OpenAI if Claude not available
        elif openai_api_key:
            return self._generate_prompt_with_openai(analysis, num_rows, openai_api_key, openai_model)

        else:
            raise ValueError("Either Claude or OpenAI API key is required for LLM prompt generation")

    def _generate_prompt_with_claude(self, analysis: Dict[str, Any], num_rows: int, api_key: str) -> str:
        """Generate prompt using Claude API"""
        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Create prompt generation request
            prompt_request = self._create_prompt_generation_request(analysis, num_rows)

            # Make API call
            response = client.messages.create(
                model=self.config.CLAUDE_MODEL,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_request
                    }
                ]
            )

            return response.content[0].text.strip()

        except Exception as e:
            raise Exception(f"Error generating prompt with Claude: {str(e)}")

    def _generate_prompt_with_openai(self, analysis: Dict[str, Any], num_rows: int,
                                     api_key: str, model: str = None) -> str:
        """Generate prompt using OpenAI API - UPDATED WITH TOKEN PARAMETER FIX"""
        if model is None:
            model = self.config.DEFAULT_OPENAI_MODEL

        try:
            client = OpenAI(api_key=api_key)

            # Create prompt generation request
            prompt_request = self._create_prompt_generation_request(analysis, num_rows)

            # Handle different token parameter for different models
            completion_params = {
                "model": model,
                "temperature": 0.3,  # Slightly higher temperature for creative prompt generation
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in synthetic data generation. Create detailed, comprehensive prompts that will generate high-quality synthetic data preserving all relationships and statistical properties."
                    },
                    {
                        "role": "user",
                        "content": prompt_request
                    }
                ]
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if model in self.config.NEWER_OPENAI_MODELS:
                completion_params["max_completion_tokens"] = 4000
            else:
                completion_params["max_tokens"] = 4000

            # Make API call
            response = client.chat.completions.create(**completion_params)

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Error generating prompt with OpenAI: {str(e)}")

    def compare_data_with_llm(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                              original_analysis: Dict[str, Any], synthetic_analysis: Dict[str, Any],
                              claude_api_key: str = None, openai_api_key: str = None,
                              openai_model: str = None) -> Optional[Dict[str, Any]]:
        """Compare datasets using Claude or OpenAI API"""

        # Try Claude first if available
        if claude_api_key:
            try:
                return self._compare_with_claude(original_df, synthetic_df, original_analysis,
                                                 synthetic_analysis, claude_api_key)
            except Exception as claude_error:
                print(f"Claude comparison failed: {claude_error}")

                # Try OpenAI as fallback
                if openai_api_key:
                    print("Falling back to OpenAI for comparison...")
                    return self._compare_with_openai(original_df, synthetic_df, original_analysis,
                                                     synthetic_analysis, openai_api_key, openai_model)
                else:
                    raise claude_error

        # Try OpenAI if Claude not available
        elif openai_api_key:
            return self._compare_with_openai(original_df, synthetic_df, original_analysis,
                                             synthetic_analysis, openai_api_key, openai_model)

        else:
            raise ValueError("Either Claude or OpenAI API key is required for LLM comparison")

    def _compare_with_claude(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                             original_analysis: Dict[str, Any], synthetic_analysis: Dict[str, Any],
                             api_key: str) -> Dict[str, Any]:
        """Compare datasets using Claude API"""
        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Create comparison prompt
            comparison_prompt = self._create_comparison_prompt(
                original_df, synthetic_df, original_analysis, synthetic_analysis
            )

            # Make API call
            response = client.messages.create(
                model=self.config.CLAUDE_MODEL,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": comparison_prompt
                    }
                ]
            )

            # Parse comparison response
            comparison_result = self._parse_comparison_response(response.content[0].text)
            return comparison_result

        except Exception as e:
            raise Exception(f"Error comparing data with Claude: {str(e)}")

    def _compare_with_openai(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                             original_analysis: Dict[str, Any], synthetic_analysis: Dict[str, Any],
                             api_key: str, model: str = None) -> Dict[str, Any]:
        """Compare datasets using OpenAI API - UPDATED WITH TOKEN PARAMETER FIX"""
        if model is None:
            model = self.config.DEFAULT_OPENAI_MODEL

        try:
            client = OpenAI(api_key=api_key)

            # Create comparison prompt
            comparison_prompt = self._create_comparison_prompt(
                original_df, synthetic_df, original_analysis, synthetic_analysis
            )

            # Handle different token parameter for different models
            completion_params = {
                "model": model,
                "temperature": 0.1,  # Low temperature for consistent analysis
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a data quality expert. Compare original and synthetic datasets providing detailed analysis in the exact JSON format requested. Be thorough and accurate in your assessment."
                    },
                    {
                        "role": "user",
                        "content": comparison_prompt
                    }
                ]
            }

            # Use max_completion_tokens for newer models, max_tokens for older ones
            if model in self.config.NEWER_OPENAI_MODELS:
                completion_params["max_completion_tokens"] = 4000
            else:
                completion_params["max_tokens"] = 4000

            # Make API call
            response = client.chat.completions.create(**completion_params)

            # Parse comparison response
            comparison_result = self._parse_comparison_response(response.choices[0].message.content)
            return comparison_result

        except Exception as e:
            raise Exception(f"Error comparing data with OpenAI: {str(e)}")

    def _prepare_data_sample(self, df: pd.DataFrame) -> str:
        """Prepare a representative sample of the data for LLM analysis"""
        # Take first 10 rows for analysis
        sample_df = df.head(10)

        # Convert to CSV string
        csv_sample = sample_df.to_csv(index=False)

        # Add basic info
        info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "sample_data": csv_sample
        }

        return json.dumps(info, indent=2)

    def _create_analysis_prompt(self, df: pd.DataFrame, data_sample: str) -> str:
        """Create comprehensive analysis prompt for Claude"""
        prompt = f"""
You are a data analyst expert. Analyze the following tabular dataset and provide a comprehensive analysis in JSON format.

Dataset Information:
{data_sample}

Please analyze the data and return a JSON response with the following structure:

{{
    "columns": {{
        "column_name": {{
            "type": "string|integer|float|boolean|datetime",
            "stats": {{
                // For numeric columns:
                "min": number,
                "max": number,
                "mean": number,
                "median": number,
                "std": number,
                "q25": number,
                "q75": number,
                "skewness": number,
                "kurtosis": number,
                "outlier_count": number,
                "outlier_percentage": number,

                // For string columns:
                "unique": number,
                "unique_percent": number,
                "most_common": ["value", count],
                "entropy": number,

                // For boolean columns:
                "true_percentage": number,

                // For datetime columns:
                "min": "date_string",
                "max": "date_string",
                "range_days": number,

                // Common for all:
                "count": number,
                "null_count": number,
                "null_percentage": number
            }}
        }}
    }},
    "relationships": {{
        "one_to_many": [
            {{
                "parent": "column_name",
                "child": "column_name",
                "avg_children": number,
                "description": "string"
            }}
        ],
        "many_to_many": [
            {{
                "col1": "column_name",
                "col2": "column_name",
                "avg_ratio": number,
                "description": "string"
            }}
        ],
        "functional_dependencies": [
            {{
                "description": "string"
            }}
        ],
        "value_mappings": {{
            "column_name": {{
                "email_domains": {{
                    "top_domains": {{"domain": count}},
                    "total_domains": number,
                    "description": "string"
                }},
                "phone_patterns": {{
                    "top_patterns": {{"pattern": count}},
                    "total_patterns": number,
                    "description": "string"
                }},
                "uuid_like": {{
                    "description": "string",
                    "avg_length": number
                }},
                "common_prefixes": {{
                    "top_prefixes": {{"prefix": count}},
                    "description": "string"
                }}
            }}
        }}
    }},
    "patterns": [
        "list of detected patterns as strings"
    ],
    "correlations": {{
        "high_correlations": [
            ["col1", "col2", correlation_value]
        ]
    }},
    "row_count": number,
    "total_nulls": number,
    "data_quality_score": number
}}

Instructions:
1. Analyze each column carefully and determine its appropriate type
2. Calculate relevant statistics based on the column type
3. Detect relationships between columns (one-to-many, many-to-many)
4. Identify value patterns (email domains, phone patterns, etc.)
5. Find correlations between numeric columns
6. Assign a data quality score (0-100) based on completeness, consistency, and structure
7. List any interesting patterns you observe
8. Ensure all numeric values are properly formatted
9. Return ONLY the JSON response, no additional text

Focus on being accurate and thorough in your analysis. Pay special attention to:
- Column relationships and cardinalities
- Data patterns and structures
- Statistical distributions
- Data quality issues
"""
        return prompt

    def _parse_llm_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM analysis response"""
        try:
            # Clean response - remove any markdown formatting
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parse JSON
            analysis = json.loads(cleaned_response)

            # Validate structure
            self._validate_analysis_structure(analysis)

            return analysis

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM analysis response as JSON: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing LLM analysis response: {str(e)}")

    def _validate_analysis_structure(self, analysis: Dict[str, Any]):
        """Validate the structure of LLM analysis response"""
        required_keys = ['columns', 'relationships', 'patterns', 'correlations', 'row_count', 'total_nulls',
                         'data_quality_score']

        for key in required_keys:
            if key not in analysis:
                raise ValueError(f"Missing required key in analysis: {key}")

        # Validate columns structure
        if not isinstance(analysis['columns'], dict):
            raise ValueError("'columns' must be a dictionary")

        # Validate relationships structure
        relationships = analysis['relationships']
        if not isinstance(relationships, dict):
            raise ValueError("'relationships' must be a dictionary")

        expected_rel_keys = ['one_to_many', 'many_to_many', 'functional_dependencies', 'value_mappings']
        for key in expected_rel_keys:
            if key not in relationships:
                relationships[key] = [] if key != 'value_mappings' else {}

    def _create_prompt_generation_request(self, analysis: Dict[str, Any], num_rows: int) -> str:
        """Create prompt for generating data generation instructions"""

        # Convert numpy types to native Python types for JSON serialization
        analysis_cleaned = self._clean_analysis_for_json(analysis)

        prompt = f"""
You are an expert in synthetic data generation. Based on the following data analysis, create a comprehensive prompt that can be used to generate {num_rows} rows of synthetic data that preserves all the patterns, relationships, and statistical properties of the original dataset.

Original Data Analysis:
{json.dumps(analysis_cleaned, indent=2)}

Create a detailed prompt that includes:

1. **Column Specifications**: Detailed requirements for each column including:
   - Data type and statistical properties
   - Value ranges and distributions
   - Uniqueness constraints
   - Null value percentages

2. **Relationship Preservation**: Exact instructions for maintaining:
   - One-to-many relationships with specific ratios
   - Many-to-many relationships
   - Functional dependencies
   - Cross-column constraints

3. **Pattern Preservation**: Instructions for maintaining:
   - Value patterns (email domains, phone formats, etc.)
   - Statistical distributions
   - Correlation structures

4. **Generation Strategy**: Step-by-step instructions for:
   - Order of column generation
   - Relationship-first approach
   - Validation requirements

5. **Quality Assurance**: Specific criteria for:
   - Ratio validation
   - Statistical property verification
   - Pattern consistency checks

The prompt should be detailed enough that an LLM can generate high-quality synthetic data that closely matches the original dataset's characteristics.

Generate a comprehensive prompt that will produce exactly {num_rows} rows of synthetic data in CSV format.
"""
        return prompt

    def _clean_analysis_for_json(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Clean analysis data to make it JSON serializable"""

        def convert_item(item):
            """Recursively convert numpy/pandas types to native Python types"""
            if isinstance(item, dict):
                return {k: convert_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert_item(i) for i in item]
            elif isinstance(item, tuple):
                return tuple(convert_item(i) for i in item)
            elif isinstance(item, (np.integer, np.int64, np.int32)):
                return int(item)
            elif isinstance(item, (np.floating, np.float64, np.float32)):
                return float(item)
            elif isinstance(item, np.bool_):
                return bool(item)
            elif isinstance(item, np.ndarray):
                return item.tolist()
            elif pd.isna(item):
                return None
            elif hasattr(item, 'isoformat'):  # datetime objects
                return item.isoformat()
            else:
                return item

        return convert_item(analysis)

    def _create_comparison_prompt(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                  original_analysis: Dict[str, Any], synthetic_analysis: Dict[str, Any]) -> str:
        """Create comprehensive comparison prompt"""

        # Clean analyses for JSON serialization
        original_cleaned = self._clean_analysis_for_json(original_analysis)
        synthetic_cleaned = self._clean_analysis_for_json(synthetic_analysis)

        prompt = f"""
You are a data quality expert. Compare the original and synthetic datasets based on their analyses and provide a comprehensive comparison report in JSON format.

Original Dataset Analysis:
{json.dumps(original_cleaned, indent=2)}

Synthetic Dataset Analysis:
{json.dumps(synthetic_cleaned, indent=2)}

Dataset Summaries:
- Original: {len(original_df)} rows, {len(original_df.columns)} columns
- Synthetic: {len(synthetic_df)} rows, {len(synthetic_df.columns)} columns

Provide a detailed comparison in the following JSON format:

{{
    "overall_scores": {{
        "overall_similarity": number (0-1),
        "structural_fidelity": number (0-1),
        "statistical_fidelity": number (0-1),
        "relationship_preservation": number (0-1),
        "pattern_fidelity": number (0-1)
    }},
    "basic_metrics": {{
        "row_count_match": boolean,
        "column_count_match": boolean,
        "column_names_match": boolean,
        "data_types_match": boolean
    }},
    "column_comparisons": [
        {{
            "column": "string",
            "type_match": boolean,
            "statistical_similarity": number (0-1),
            "distribution_similarity": number (0-1),
            "issues": ["list of issues"],
            "recommendations": ["list of recommendations"]
        }}
    ],
    "relationship_analysis": {{
        "one_to_many_preserved": number (0-1),
        "many_to_many_preserved": number (0-1),
        "functional_deps_preserved": number (0-1),
        "missing_relationships": ["list"],
        "extra_relationships": ["list"]
    }},
    "pattern_analysis": {{
        "value_patterns_preserved": number (0-1),
        "correlation_structure_preserved": number (0-1),
        "statistical_distributions_preserved": number (0-1),
        "pattern_issues": ["list of issues"]
    }},
    "quality_assessment": {{
        "synthetic_quality_score": number (0-100),
        "improvement_areas": ["list"],
        "strengths": ["list"],
        "overall_recommendation": "string"
    }},
    "detailed_findings": {{
        "critical_issues": ["list"],
        "minor_issues": ["list"],
        "excellent_preservation": ["list"]
    }}
}}

Instructions:
1. Compare each aspect thoroughly and assign accurate similarity scores
2. Identify specific issues and provide actionable recommendations
3. Focus on relationship and pattern preservation quality
4. Provide an overall assessment of synthetic data quality
5. Be specific about what was preserved well and what needs improvement
6. Return ONLY the JSON response, no additional text

Pay special attention to:
- Relationship ratio accuracy
- Statistical property preservation
- Value pattern consistency
- Data quality metrics
"""
        return prompt

    def _parse_comparison_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM comparison response"""
        try:
            # Clean response
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parse JSON
            comparison = json.loads(cleaned_response)

            return comparison

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM comparison response as JSON: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing LLM comparison response: {str(e)}")