# =============================================================================
# File: utils/validators.py
"""Data validation utilities"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class DataValidator:
    """Class to validate data and schemas"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate uploaded dataframe"""
        errors = []

        if df is None:
            errors.append("DataFrame is None")
            return False, errors

        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors

        if len(df.columns) == 0:
            errors.append("DataFrame has no columns")
            return False, errors

        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
        if unnamed_cols:
            errors.append(f"Found unnamed columns: {unnamed_cols}")

        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            errors.append(f"Duplicate column names found: {duplicates}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_schema(schema: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate custom schema"""
        errors = []

        if not schema:
            errors.append("Schema is empty")
            return False, errors

        column_names = []
        for i, col_def in enumerate(schema):
            # Check required fields
            if 'name' not in col_def:
                errors.append(f"Column {i}: Missing 'name' field")
                continue

            if 'type' not in col_def:
                errors.append(f"Column {i}: Missing 'type' field")
                continue

            name = col_def['name']
            col_type = col_def['type']

            # Check duplicate names
            if name in column_names:
                errors.append(f"Duplicate column name: {name}")
            column_names.append(name)

            # Validate column name
            if not name or not isinstance(name, str):
                errors.append(f"Column {i}: Invalid name '{name}'")

            # Validate type
            valid_types = ['integer', 'float', 'string', 'boolean', 'datetime']
            if col_type not in valid_types:
                errors.append(f"Column {name}: Invalid type '{col_type}'. Must be one of {valid_types}")

            # Type-specific validations
            if col_type in ['integer', 'float']:
                min_val = col_def.get('min')
                max_val = col_def.get('max')
                if min_val is not None and max_val is not None and min_val > max_val:
                    errors.append(f"Column {name}: min value ({min_val}) > max value ({max_val})")

            elif col_type == 'string':
                unique_pct = col_def.get('unique_percent', 50)
                if not 0 <= unique_pct <= 100:
                    errors.append(f"Column {name}: unique_percent must be between 0 and 100")

            elif col_type == 'boolean':
                true_pct = col_def.get('true_percentage', 50)
                if not 0 <= true_pct <= 100:
                    errors.append(f"Column {name}: true_percentage must be between 0 and 100")

            elif col_type == 'datetime':
                start_date = col_def.get('start_date')
                end_date = col_def.get('end_date')
                if start_date and end_date:
                    try:
                        start = pd.to_datetime(start_date)
                        end = pd.to_datetime(end_date)
                        if start > end:
                            errors.append(f"Column {name}: start_date > end_date")
                    except:
                        errors.append(f"Column {name}: Invalid date format")

            # Validate null percentage
            null_pct = col_def.get('null_percentage', 0)
            if not 0 <= null_pct <= 100:
                errors.append(f"Column {name}: null_percentage must be between 0 and 100")

        return len(errors) == 0, errors

    @staticmethod
    def validate_generated_data(df: pd.DataFrame, expected_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate generated synthetic data"""
        errors = []

        # Basic validation
        is_valid, basic_errors = DataValidator.validate_dataframe(df)
        errors.extend(basic_errors)

        if not is_valid:
            return False, errors

        # Check expected columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing expected columns: {missing_cols}")

        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            errors.append(f"Unexpected columns: {extra_cols}")

        return len(errors) == 0, errors
