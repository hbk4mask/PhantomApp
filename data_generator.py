# =============================================================================
# File: data_generator.py
"""Data generation utilities - Updated with Claude, OpenAI, and Groq API support"""

import pandas as pd
import io
import re
import anthropic
from groq import Groq
from openai import OpenAI  # NEW
from typing import Optional
from config import Config


class DataGenerator:
    """Class to handle synthetic data generation"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def generate_with_claude(self, prompt: str, num_rows: int, api_key: str) -> Optional[pd.DataFrame]:
        """Generate synthetic data using Claude API"""
        if not api_key:
            raise ValueError("Claude API key is required")

        try:
            client = anthropic.Anthropic(api_key=api_key)

            enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS: 
- Output ONLY the CSV data, nothing else
- Include headers as the first row
- Do not include any explanations, markdown formatting, or code blocks
- Generate exactly {num_rows} rows of data (plus header row)
- Ensure all data follows the specified constraints and relationships
- Maintain exact statistical properties and relationship ratios as specified
- Use proper CSV formatting with commas as delimiters
- No quotes around values unless they contain commas or special characters

Output format: Pure CSV data starting immediately with the header row."""

            response = client.messages.create(
                model=self.config.CLAUDE_MODEL,
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ]
            )

            csv_data = response.content[0].text.strip()
            csv_data = self._clean_csv_response(csv_data)

            return pd.read_csv(io.StringIO(csv_data))

        except Exception as e:
            raise Exception(f"Error generating data with Claude: {str(e)}")

    def generate_with_openai(self, prompt: str, num_rows: int, api_key: str, model: str = None) -> Optional[
        pd.DataFrame]:
        """Generate synthetic data using OpenAI API - NEW"""
        if not api_key:
            raise ValueError("OpenAI API key is required")

        if model is None:
            model = self.config.DEFAULT_OPENAI_MODEL

        try:
            client = OpenAI(api_key=api_key)

            enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS: 
- Output ONLY the CSV data, nothing else
- Include headers as the first row
- Do not include any explanations, markdown formatting, or code blocks
- Generate exactly {num_rows} rows of data (plus header row)
- Ensure all data follows the specified constraints and relationships
- Maintain exact statistical properties and relationship ratios as specified
- Use proper CSV formatting with commas as delimiters
- No quotes around values unless they contain commas or special characters

Output format: Pure CSV data starting immediately with the header row."""

            # Handle different token parameter for different models
            # inside generate_with_openai(...)
            completion_params = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise data generator. Output only CSV."
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                # Deterministic decoding:
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                # Some SDKs support this; harmless if ignored:
                "seed": 1337,
            }
            # keep your token handling:
            if model in self.config.NEWER_OPENAI_MODELS:
                completion_params["max_completion_tokens"] = 8000
            else:
                completion_params["max_tokens"] = 8000

            # completion_params = {
            #     "model": model,
            #     "messages": [
            #         {
            #             "role": "system",
            #             "content": "You are a data generation expert. Generate synthetic tabular data in CSV format based on the given specifications. Output only CSV data, no explanations or formatting."
            #         },
            #         {
            #             "role": "user",
            #             "content": enhanced_prompt
            #         }
            #     ],
            #     "temperature": 0.7,
            # }
            #
            # # Use max_completion_tokens for newer models, max_tokens for older ones
            # if model in self.config.NEWER_OPENAI_MODELS:
            #     completion_params["max_completion_tokens"] = 8000
            # else:
            #     completion_params["max_tokens"] = 8000

            response = client.chat.completions.create(**completion_params)

            csv_data = response.choices[0].message.content.strip()
            csv_data = self._clean_csv_response(csv_data)

            return pd.read_csv(io.StringIO(csv_data))

        except Exception as e:
            raise Exception(f"Error generating data with OpenAI: {str(e)}")

    def generate_with_groq(self, prompt: str, num_rows: int, api_key: str, model: str) -> Optional[pd.DataFrame]:
        """Generate synthetic data using Groq API"""
        if not api_key:
            raise ValueError("Groq API key is required")

        try:
            client = Groq(api_key=api_key)

            enhanced_prompt = f"""{prompt}

IMPORTANT: 
- Output ONLY the CSV data, nothing else
- Include headers as the first row
- Do not include any explanations or markdown formatting
- Generate exactly {num_rows} rows of data (plus header row)
- Ensure all data follows the specified constraints"""

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data generation expert. Generate synthetic tabular data in CSV format based on the given specifications. Output only CSV data, no explanations."
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                model=model,
                temperature=0.7,
                max_tokens=8000,
            )

            csv_data = chat_completion.choices[0].message.content.strip()
            csv_data = self._clean_csv_response(csv_data)

            return pd.read_csv(io.StringIO(csv_data))

        except Exception as e:
            raise Exception(f"Error generating data with Groq: {str(e)}")

    def generate_with_auto_fallback(self, prompt: str, num_rows: int,
                                    claude_key: str = None, openai_key: str = None, groq_key: str = None,
                                    openai_model: str = None, groq_model: str = None) -> tuple[
        Optional[pd.DataFrame], str]:
        """Generate data with automatic fallback between APIs - NEW"""

        # Try APIs in order of preference
        errors = []

        # Try Claude first (highest quality)
        if claude_key:
            try:
                df = self.generate_with_claude(prompt, num_rows, claude_key)
                return df, "claude"
            except Exception as e:
                errors.append(f"Claude: {str(e)}")

        # Try OpenAI second
        if openai_key:
            try:
                df = self.generate_with_openai(prompt, num_rows, openai_key, openai_model)
                return df, "openai"
            except Exception as e:
                errors.append(f"OpenAI: {str(e)}")

        # Try Groq last
        if groq_key and groq_model:
            try:
                df = self.generate_with_groq(prompt, num_rows, groq_key, groq_model)
                return df, "groq"
            except Exception as e:
                errors.append(f"Groq: {str(e)}")

        # If all failed
        error_msg = "All APIs failed:\n" + "\n".join(errors)
        raise Exception(error_msg)

    def _clean_csv_response(self, csv_data: str) -> str:
        """Clean CSV response from LLM"""
        # Remove markdown code blocks
        csv_data = re.sub(r'^```.*\n', '', csv_data, flags=re.MULTILINE)
        csv_data = re.sub(r'\n```$', '', csv_data, flags=re.MULTILINE)
        csv_data = re.sub(r'^```', '', csv_data, flags=re.MULTILINE)
        csv_data = re.sub(r'```$', '', csv_data, flags=re.MULTILINE)

        # Remove any leading/trailing whitespace
        csv_data = csv_data.strip()

        # Remove any text before the actual CSV header (look for common column patterns)
        lines = csv_data.split('\n')
        csv_start_idx = 0

        for i, line in enumerate(lines):
            # Look for a line that looks like a CSV header (contains commas and alphanumeric content)
            if ',' in line and any(c.isalnum() for c in line):
                csv_start_idx = i
                break

        if csv_start_idx > 0:
            csv_data = '\n'.join(lines[csv_start_idx:])

        return csv_data.strip()