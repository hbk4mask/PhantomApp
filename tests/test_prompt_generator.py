# =============================================================================
# File: tests/test_prompt_generator.py
"""Unit tests for prompt generator module"""

import unittest
from prompt_generator import PromptGenerator
from config import Config


class TestPromptGenerator(unittest.TestCase):
    """Test cases for PromptGenerator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = PromptGenerator(Config())

        # Mock analysis data
        self.mock_analysis = {
            'columns': {
                'age': {
                    'type': 'integer',
                    'stats': {'min': 18, 'max': 65, 'mean': 35.5, 'null_percentage': 0}
                },
                'name': {
                    'type': 'string',
                    'stats': {'unique': 100, 'unique_percent': 95, 'null_percentage': 0}
                }
            },
            'patterns': ['age appears to be numeric'],
            'correlations': {'high_correlations': []},
            'row_count': 100
        }

        # Mock schema
        self.mock_schema = [
            {
                'name': 'user_id',
                'type': 'integer',
                'min': 1,
                'max': 1000,
                'unique': True,
                'null_percentage': 0
            },
            {
                'name': 'category',
                'type': 'string',
                'categories': ['A', 'B', 'C'],
                'unique_percent': 30,
                'null_percentage': 5
            }
        ]

    def test_generate_from_analysis(self):
        """Test prompt generation from analysis"""
        prompt = self.generator.generate_from_analysis(self.mock_analysis, 500)

        # Check basic structure
        self.assertIn('500 rows', prompt)
        self.assertIn('age (integer)', prompt)
        self.assertIn('name (string)', prompt)
        self.assertIn('Range: 18 to 65', prompt)
        self.assertIn('Instructions:', prompt)
        self.assertIn('CSV format', prompt)

    def test_generate_from_schema(self):
        """Test prompt generation from schema"""
        prompt = self.generator.generate_from_schema(self.mock_schema, 200)

        # Check basic structure
        self.assertIn('200 rows', prompt)
        self.assertIn('user_id (integer)', prompt)
        self.assertIn('category (string)', prompt)
        self.assertIn('Unique constraint', prompt)
        self.assertIn('Categories: A, B, C', prompt)

    def test_correlation_handling(self):
        """Test correlation handling in prompts"""
        analysis_with_corr = self.mock_analysis.copy()
        analysis_with_corr['correlations'] = {
            'high_correlations': [('age', 'salary', 0.8)]
        }

        prompt = self.generator.generate_from_analysis(analysis_with_corr)
        self.assertIn('Correlation Constraints', prompt)
        self.assertIn('age and salary', prompt)