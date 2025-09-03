# =============================================================================
# File: tests/test_data_analysis.py
"""Unit tests for data analysis module"""

import unittest
import pandas as pd
import numpy as np
from data_analysis import DataAnalyzer
from config import Config


class TestDataAnalyzer(unittest.TestCase):
    """Test cases for DataAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = DataAnalyzer(Config())

        # Create test data
        self.test_df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'boolean': [True, False, True, True, False],
            'datetime': pd.date_range('2024-01-01', periods=5),
            'with_nulls': [1, 2, None, 4, None]
        })

    def test_analyze_data_structure(self):
        """Test basic data analysis structure"""
        result = self.analyzer.analyze_data(self.test_df)

        # Check required keys
        required_keys = ['columns', 'patterns', 'correlations', 'row_count', 'total_nulls', 'data_quality_score']
        for key in required_keys:
            self.assertIn(key, result)

        # Check row count
        self.assertEqual(result['row_count'], 5)

        # Check columns analysis
        self.assertEqual(len(result['columns']), 6)

        # Check data types detection
        self.assertEqual(result['columns']['numeric_int']['type'], 'integer')
        self.assertEqual(result['columns']['numeric_float']['type'], 'float')
        self.assertEqual(result['columns']['categorical']['type'], 'string')
        self.assertEqual(result['columns']['boolean']['type'], 'boolean')
        self.assertEqual(result['columns']['datetime']['type'], 'datetime')

    def test_numeric_analysis(self):
        """Test numeric column analysis"""
        result = self.analyzer.analyze_data(self.test_df)

        int_stats = result['columns']['numeric_int']['stats']
        self.assertEqual(int_stats['min'], 1)
        self.assertEqual(int_stats['max'], 5)
        self.assertEqual(int_stats['mean'], 3.0)

        float_stats = result['columns']['numeric_float']['stats']
        self.assertEqual(float_stats['min'], 1.1)
        self.assertEqual(float_stats['max'], 5.5)

    def test_null_detection(self):
        """Test null value detection"""
        result = self.analyzer.analyze_data(self.test_df)

        null_stats = result['columns']['with_nulls']['stats']
        self.assertEqual(null_stats['null_count'], 2)
        self.assertEqual(null_stats['null_percentage'], 40.0)

    def test_pattern_detection(self):
        """Test pattern detection"""
        # Create data with unique identifier
        df_with_id = self.test_df.copy()
        df_with_id['unique_id'] = range(len(df_with_id))

        result = self.analyzer.analyze_data(df_with_id)
        patterns = result['patterns']

        # Should detect unique identifier
        id_patterns = [p for p in patterns if 'unique identifier' in p]
        self.assertTrue(len(id_patterns) > 0)

    def test_data_quality_score(self):
        """Test data quality score calculation"""
        result = self.analyzer.analyze_data(self.test_df)
        score = result['data_quality_score']

        # Should be between 0 and 100
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

        # Should be reduced due to null values
        self.assertLess(score, 100)