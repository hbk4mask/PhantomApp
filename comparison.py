# =============================================================================
# File: comparison.py (COMPLETE FILE - ANALYSIS-BASED APPROACH WITH ROW COUNT EXCLUSION)
"""Data comparison utilities with detailed analysis reporting - NO DATAFRAME OPERATIONS"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance
from typing import Dict, Any, List, Tuple
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional, Set  # Add Set here
# Import the Config class
from config import Config


class DataComparator:
    """Class to handle comparison between original and synthetic data"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def compare_datasets(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive comparison between datasets"""
        comparison = {}

        for col in original_df.columns:
            if col in synthetic_df.columns:
                comparison[col] = self._compare_columns(
                    original_df[col],
                    synthetic_df[col]
                )

        return {
            "column_comparisons": comparison,
            "overall_similarity": self._calculate_overall_similarity(comparison),
            "missing_columns": set(original_df.columns) - set(synthetic_df.columns),
            "extra_columns": set(synthetic_df.columns) - set(original_df.columns)
        }

    def compare_analyses(self, original_analysis: Dict[str, Any], synthetic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare detailed analyses including relationships and patterns"""
        comparison = {
            "basic_metrics": self._compare_basic_metrics(original_analysis, synthetic_analysis),
            "column_details": self._compare_column_details(original_analysis, synthetic_analysis),
            "relationships": self._compare_relationships(original_analysis, synthetic_analysis),
            "patterns": self._compare_patterns(original_analysis, synthetic_analysis),
            "correlations": self._compare_correlations(original_analysis, synthetic_analysis),
            "overall_scores": self._calculate_overall_comparison_scores(original_analysis, synthetic_analysis)
        }
        return comparison

    def generate_detailed_report(self, original_analysis: Dict[str, Any], synthetic_analysis: Dict[str, Any],
                                 original_df: pd.DataFrame = None, synthetic_df: pd.DataFrame = None,
                                 integrity_excluded_fields: Set[str] = None) -> Dict[str, Any]:
        """Generate detailed comparison report with configurable integrity checking"""
        # Enhanced analysis using ONLY analysis dictionaries
        detailed_report = {
            "overall_scores": self._calculate_enhanced_overall_scores(original_analysis, synthetic_analysis),
            "basic_metrics": self._generate_basic_metrics_table(original_analysis, synthetic_analysis),
            "column_comparison": self._generate_column_comparison_table(original_analysis, synthetic_analysis),
            "relationship_comparison": self._generate_relationship_comparison_table(original_analysis,
                                                                                    synthetic_analysis),
            "pattern_analysis": self._generate_pattern_analysis(original_analysis, synthetic_analysis),

            # Analysis-based methods instead of DataFrame operations
            "value_distribution": self._analyze_value_distributions_from_analysis(original_analysis,
                                                                                  synthetic_analysis),
            "data_integrity": self._check_data_integrity_from_analysis(
                original_analysis, synthetic_analysis, integrity_excluded_fields),  # Add this parameter
            "summary_recommendations": self._generate_enhanced_recommendations_from_analysis(original_analysis,
                                                                                             synthetic_analysis)
        }

        return detailed_report

    # =============================================================================
    # UPDATED METHODS - SKIP ROW COUNT IN SIMILARITY CALCULATIONS
    # =============================================================================

    def _generate_basic_metrics_table(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic metrics comparison table - UPDATED TO SKIP ROW COUNT SIMILARITY"""

        orig_rows = orig.get("row_count", 0)
        synth_rows = synth.get("row_count", 0)
        # Don't calculate row similarity - user may intentionally want different counts
        row_similarity = "N/A (Intentional)"

        orig_cols = len(orig.get("columns", {}))
        synth_cols = len(synth.get("columns", {}))
        col_similarity = 1.0 if orig_cols == synth_cols else 0.0

        orig_nulls = orig.get("total_nulls", 0)
        synth_nulls = synth.get("total_nulls", 0)

        orig_quality = orig.get("data_quality_score", 0)
        synth_quality = synth.get("data_quality_score", 0)
        quality_similarity = 1.0 - abs(orig_quality - synth_quality) / 100 if orig_quality > 0 else 0

        return [
            {
                "metric": "Row Count",
                "original": orig_rows,
                "synthetic": synth_rows,
                "difference": synth_rows - orig_rows,
                "similarity": row_similarity  # Show as "N/A (Intentional)"
            },
            {
                "metric": "Total Nulls",
                "original": orig_nulls,
                "synthetic": synth_nulls,
                "difference": synth_nulls - orig_nulls,
                "similarity": "100.00%" if orig_nulls == synth_nulls else "N/A"
            },
            {
                "metric": "Data Quality Score",
                "original": orig_quality,
                "synthetic": synth_quality,
                "difference": synth_quality - orig_quality,
                "similarity": f"{quality_similarity:.2%}"
            },
            {
                "metric": "Column Count",
                "original": orig_cols,
                "synthetic": synth_cols,
                "difference": synth_cols - orig_cols,
                "similarity": f"{col_similarity:.2%}"
            }
        ]

    def _compare_basic_metrics(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare basic dataset metrics - UPDATED TO SKIP ROW COUNT SIMILARITY"""

        orig_rows = orig["row_count"]
        synth_rows = synth["row_count"]

        # Skip row count similarity calculation
        return {
            "row_count": {
                "original": orig_rows,
                "synthetic": synth_rows,
                "difference": synth_rows - orig_rows,
                "similarity_score": None  # Don't calculate similarity for intentionally different row counts
            },
            "total_nulls": {
                "original": orig["total_nulls"],
                "synthetic": synth["total_nulls"],
                "difference": synth["total_nulls"] - orig["total_nulls"]
            },
            "data_quality_score": {
                "original": orig["data_quality_score"],
                "synthetic": synth["data_quality_score"],
                "difference": synth["data_quality_score"] - orig["data_quality_score"]
            },
            "column_count": {
                "original": len(orig["columns"]),
                "synthetic": len(synth["columns"]),
                "difference": len(synth["columns"]) - len(orig["columns"])
            }
        }

    def _calculate_overall_comparison_scores(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall comparison scores - UPDATED TO EXCLUDE ROW COUNT"""

        # Basic metrics score - EXCLUDE row count similarity
        basic_metrics = self._compare_basic_metrics(orig, synth)

        # Calculate basic score based on column count and quality only
        col_count_orig = len(orig["columns"])
        col_count_synth = len(synth["columns"])
        col_similarity = 1.0 if col_count_orig == col_count_synth else 0.0

        orig_quality = orig["data_quality_score"]
        synth_quality = synth["data_quality_score"]
        quality_similarity = 1.0 - abs(orig_quality - synth_quality) / 100 if orig_quality > 0 else 0

        # Basic score based on structure and quality, NOT row count
        basic_score = (col_similarity + quality_similarity) / 2

        # Column similarity score
        column_details = self._compare_column_details(orig, synth)
        column_similarities = []
        for col_comp in column_details:
            if col_comp.get("overall_numeric_similarity"):
                column_similarities.append(col_comp["overall_numeric_similarity"])
            elif col_comp.get("overall_string_similarity"):
                column_similarities.append(col_comp["overall_string_similarity"])
            elif col_comp.get("overall_boolean_similarity"):
                column_similarities.append(col_comp["overall_boolean_similarity"])
            elif col_comp.get("overall_datetime_similarity"):
                column_similarities.append(col_comp["overall_datetime_similarity"])

        column_score = sum(column_similarities) / len(column_similarities) if column_similarities else 0

        # Relationship preservation score
        relationships = self._compare_relationships(orig, synth)
        rel_score = relationships.get("one_to_many", {}).get("relationship_preservation_score", 0)

        # Pattern similarity score
        patterns = self._compare_patterns(orig, synth)
        pattern_score = patterns.get("pattern_similarity", 0)

        # Overall weighted score - EXCLUDE row count similarity
        overall_score = (basic_score * 0.3 + column_score * 0.4 + rel_score * 0.2 + pattern_score * 0.1)

        return {
            "basic_metrics_score": basic_score,
            "column_similarity_score": column_score,
            "relationship_preservation_score": rel_score,
            "pattern_similarity_score": pattern_score,
            "overall_score": overall_score
        }

    def _calculate_enhanced_overall_scores(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced overall scores for the report - UPDATED TO EXCLUDE ROW COUNT"""

        # Basic scores from existing method (now excludes row count)
        basic_scores = self._calculate_overall_comparison_scores(orig, synth)

        # Enhanced pattern similarity
        pattern_similarity = self._calculate_pattern_similarity_score(orig, synth)

        # Enhanced relationship preservation
        relationship_score = self._calculate_relationship_preservation_score(orig, synth)

        # Column similarity (weighted by importance)
        column_similarity = self._calculate_weighted_column_similarity(orig, synth)

        # Overall weighted score - focuses on data quality, not quantity
        overall_score = (
                basic_scores["basic_metrics_score"] * 0.2 +  # Structure + quality (no row count)
                column_similarity * 0.3 +  # Column-level similarity
                relationship_score * 0.3 +  # Relationship preservation
                pattern_similarity * 0.2  # Pattern similarity
        )

        return {
            "overall_score": overall_score,
            "column_similarity": column_similarity,
            "relationship_preservation": relationship_score,
            "pattern_similarity": pattern_similarity
        }

    # =============================================================================
    # ANALYSIS-BASED METHODS (NO DATAFRAMES)
    # =============================================================================

    def _analyze_value_distributions_from_analysis(self, orig_analysis: Dict[str, Any],
                                                   synth_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze value distributions using pre-calculated analysis data - NO DATAFRAMES"""

        analysis = {}
        orig_columns = orig_analysis.get("columns", {})
        synth_columns = synth_analysis.get("columns", {})

        for col in orig_columns:
            if col in synth_columns:
                orig_col = orig_columns[col]
                synth_col = synth_columns[col]

                # Only analyze string columns that have value distribution data
                if (orig_col["type"] == "string" and synth_col["type"] == "string" and
                        "value_distribution" in orig_col["stats"] and "value_distribution" in synth_col["stats"]):
                    orig_stats = orig_col["stats"]
                    synth_stats = synth_col["stats"]

                    # Get value distributions
                    orig_values = set(orig_stats["value_distribution"].keys())
                    synth_values = set(synth_stats["value_distribution"].keys())

                    # Calculate overlap
                    common_values = orig_values & synth_values
                    total_values = orig_values | synth_values
                    overlap_score = len(common_values) / len(total_values) if total_values else 0

                    # Get unique counts
                    orig_unique = orig_stats.get("unique", 0)
                    synth_unique = synth_stats.get("unique", 0)

                    analysis[col] = {
                        "original_unique": orig_unique,
                        "synthetic_unique": synth_unique,
                        "common_values": len(common_values),
                        "overlap_score": overlap_score,
                        "diversity_preservation": synth_unique / orig_unique if orig_unique > 0 else 1.0,

                        # Email domain analysis if available
                        "email_analysis": self._compare_email_domains_from_analysis(
                            orig_stats.get("email_domains", {}),
                            synth_stats.get("email_domains", {})
                        ) if orig_stats.get("email_domains") else {},

                        # Phone pattern analysis if available
                        "phone_analysis": self._compare_phone_patterns_from_analysis(
                            orig_stats.get("phone_patterns", {}),
                            synth_stats.get("phone_patterns", {})
                        ) if orig_stats.get("phone_patterns") else {}
                    }

        return analysis

    def _check_data_integrity_from_analysis(self, orig_analysis: Dict[str, Any],
                                            synth_analysis: Dict[str, Any],
                                            excluded_fields: Set[str] = None) -> Dict[str, Any]:
        """Check data integrity using analysis data with field exclusions - UPDATED"""

        if excluded_fields is None:
            excluded_fields = set()

        integrity_checks = {}
        orig_columns = orig_analysis.get("columns", {})
        synth_columns = synth_analysis.get("columns", {})

        for col in orig_columns:
            # Skip excluded fields
            if col in excluded_fields:
                continue

            if col in synth_columns:
                orig_col = orig_columns[col]
                synth_col = synth_columns[col]

                # Check for value reuse in string columns
                if (orig_col["type"] == "string" and synth_col["type"] == "string" and
                        "all_unique_values" in orig_col["stats"] and "all_unique_values" in synth_col["stats"]):

                    orig_values = set(orig_col["stats"]["all_unique_values"])
                    synth_values = set(synth_col["stats"]["all_unique_values"])

                    reused_values = orig_values & synth_values
                    total_synthetic = len(synth_values)
                    reuse_percentage = len(reused_values) / total_synthetic if total_synthetic > 0 else 0

                    integrity_checks[col] = {
                        "reused_values": len(reused_values),
                        "reuse_percentage": reuse_percentage,
                        "total_synthetic": total_synthetic,
                        "sample_reused": list(reused_values)[:5]  # Show first 5 reused values
                    }

                # Check for value reuse in numeric columns (useful for IDs, coordinates)
                elif (orig_col["type"] in ["integer", "float"] and synth_col["type"] in ["integer", "float"] and
                      "sample_unique_values" in orig_col["stats"] and "sample_unique_values" in synth_col["stats"]):

                    orig_values = set(
                        map(str, orig_col["stats"]["sample_unique_values"]))  # Convert to string for comparison
                    synth_values = set(map(str, synth_col["stats"]["sample_unique_values"]))

                    reused_values = orig_values & synth_values
                    total_synthetic = synth_col["stats"].get("unique_values_count", 0)
                    reuse_percentage = len(reused_values) / total_synthetic if total_synthetic > 0 else 0

                    integrity_checks[col] = {
                        "reused_values": len(reused_values),
                        "reuse_percentage": reuse_percentage,
                        "total_synthetic": total_synthetic,
                        "sample_reused": list(reused_values)[:5]
                    }

        return integrity_checks

    def _classify_field_category(self, col_name: str, col_info: Dict[str, Any]) -> str:
        """Classify field category for integrity checking context"""
        col_name_lower = col_name.lower()

        # PII fields that should not be reused
        if any(keyword in col_name_lower for keyword in ['email', 'phone', 'name', 'ssn', 'passport']):
            return 'pii'

        # Identifier fields that should be unique
        if any(keyword in col_name_lower for keyword in ['id', 'uuid', 'key', 'identifier']):
            return 'identifier'

        # Categorical fields that can be legitimately reused
        if any(keyword in col_name_lower for keyword in
               ['type', 'category', 'status', 'os', 'device', 'brand', 'model']):
            return 'categorical'

        # Coordinate fields
        if any(keyword in col_name_lower for keyword in ['lat', 'lon', 'latitude', 'longitude']):
            return 'coordinate'

        # Technical fields
        if any(keyword in col_name_lower for keyword in ['ip', 'url', 'address', 'host']):
            return 'technical'

        # Check uniqueness to infer category
        unique_pct = col_info["stats"].get("unique_percent", 0)
        if unique_pct > 90:
            return 'identifier'
        elif unique_pct < 20:
            return 'categorical'

        return 'other'
    def _compare_email_domains_from_analysis(self, orig_domains: Dict, synth_domains: Dict) -> Dict[str, Any]:
        """Compare email domains from analysis data"""
        if not orig_domains or not synth_domains:
            return {}

        orig_domain_set = set(orig_domains.get("all_domains", []))
        synth_domain_set = set(synth_domains.get("all_domains", []))

        common_domains = orig_domain_set & synth_domain_set
        total_domains = orig_domain_set | synth_domain_set

        return {
            "original_domain_count": orig_domains.get("total_domains", 0),
            "synthetic_domain_count": synth_domains.get("total_domains", 0),
            "common_domains": len(common_domains),
            "domain_overlap_score": len(common_domains) / len(total_domains) if total_domains else 0,
            "diversity_preservation": synth_domains.get("total_domains", 0) / orig_domains.get("total_domains", 1),
            "missing_domains": list(orig_domain_set - synth_domain_set)[:5],  # First 5 missing domains
            "new_domains": list(synth_domain_set - orig_domain_set)[:5]  # First 5 new domains
        }

    def _compare_phone_patterns_from_analysis(self, orig_patterns: Dict, synth_patterns: Dict) -> Dict[str, Any]:
        """Compare phone patterns from analysis data"""
        if not orig_patterns or not synth_patterns:
            return {}

        orig_pattern_set = set(orig_patterns.get("all_patterns", []))
        synth_pattern_set = set(synth_patterns.get("all_patterns", []))

        common_patterns = orig_pattern_set & synth_pattern_set
        total_patterns = orig_pattern_set | synth_pattern_set

        return {
            "original_pattern_count": orig_patterns.get("total_patterns", 0),
            "synthetic_pattern_count": synth_patterns.get("total_patterns", 0),
            "common_patterns": len(common_patterns),
            "pattern_overlap_score": len(common_patterns) / len(total_patterns) if total_patterns else 0,
            "diversity_preservation": synth_patterns.get("total_patterns", 0) / orig_patterns.get("total_patterns", 1),
            "missing_patterns": list(orig_pattern_set - synth_pattern_set),
            "new_patterns": list(synth_pattern_set - orig_pattern_set)
        }

    def _generate_enhanced_recommendations_from_analysis(self, orig_analysis: Dict[str, Any],
                                                         synth_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced recommendations using analysis data - NO DATAFRAMES"""

        strengths = []
        critical_issues = []
        improvements = []

        # Analyze overall scores
        overall_scores = self._calculate_enhanced_overall_scores(orig_analysis, synth_analysis)

        if overall_scores["relationship_preservation"] > 0.9:
            strengths.append("Excellent relationship preservation")
        elif overall_scores["relationship_preservation"] < 0.5:
            critical_issues.append("Poor relationship preservation")

        if overall_scores["pattern_similarity"] < 0.6:
            critical_issues.append("Significant pattern loss")
            improvements.append("Enhance pattern preservation in generation prompts")

        # Analyze column-specific issues from analysis data
        orig_columns = orig_analysis.get("columns", {})
        synth_columns = synth_analysis.get("columns", {})

        for col in orig_columns:
            if col in synth_columns:
                orig_col = orig_columns[col]
                synth_col = synth_columns[col]

                # Check uniqueness preservation
                orig_unique = orig_col["stats"].get("unique", 0)
                synth_unique = synth_col["stats"].get("unique", 0)

                if synth_unique < orig_unique * 0.5:
                    critical_issues.append(f"Poor uniqueness in {col} column ({synth_unique} vs {orig_unique})")
                    improvements.append(f"Improve {col} value diversity")

                # Check email domain diversity
                if "email_domains" in orig_col["stats"] and "email_domains" in synth_col["stats"]:
                    orig_domains = orig_col["stats"]["email_domains"].get("total_domains", 0)
                    synth_domains = synth_col["stats"]["email_domains"].get("total_domains", 0)

                    if synth_domains < orig_domains * 0.5:
                        critical_issues.append(f"Significant email domain diversity loss in {col}")
                        improvements.append(f"Preserve more email domains in {col} generation")

                # Check phone pattern preservation
                if "phone_patterns" in orig_col["stats"] and "phone_patterns" in synth_col["stats"]:
                    orig_patterns = orig_col["stats"]["phone_patterns"].get("total_patterns", 0)
                    synth_patterns = synth_col["stats"]["phone_patterns"].get("total_patterns", 0)

                    if synth_patterns < orig_patterns * 0.7:
                        critical_issues.append(f"Phone pattern diversity reduced in {col}")
                        improvements.append(f"Maintain more phone number patterns in {col}")

        # Assess data quality
        orig_quality = orig_analysis.get("data_quality_score", 0)
        synth_quality = synth_analysis.get("data_quality_score", 0)

        if abs(orig_quality - synth_quality) < 5:
            strengths.append("Consistent data quality scores")

        # Overall assessment
        overall_score = overall_scores["overall_score"]
        if overall_score >= 0.8:
            grade = "A"
            assessment = "Excellent synthetic data quality"
        elif overall_score >= 0.7:
            grade = "B"
            assessment = "Good synthetic data quality"
        elif overall_score >= 0.6:
            grade = "C"
            assessment = "Moderate synthetic data quality"
        else:
            grade = "D"
            assessment = "Poor synthetic data quality - significant improvements needed"

        return {
            "strengths": strengths,
            "critical_issues": critical_issues,
            "improvements": improvements,
            "overall_grade": grade,
            "overall_assessment": assessment,
            "overall_score": overall_score
        }

    # =============================================================================
    # ENHANCED SCORING METHODS
    # =============================================================================

    def _calculate_pattern_similarity_score(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> float:
        """Calculate enhanced pattern similarity score"""
        orig_patterns = set(orig.get("patterns", []))
        synth_patterns = set(synth.get("patterns", []))

        if not orig_patterns:
            return 1.0

        common = orig_patterns & synth_patterns
        return len(common) / len(orig_patterns)

    def _calculate_relationship_preservation_score(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> float:
        """Calculate relationship preservation score"""
        orig_rel = orig.get("relationships", {})
        synth_rel = synth.get("relationships", {})

        orig_otm = orig_rel.get("one_to_many", [])
        synth_otm = synth_rel.get("one_to_many", [])

        if not orig_otm:
            return 1.0

        synth_rel_map = {f"{r['parent']} → {r['child']}": r for r in synth_otm}

        similarities = []
        for orig_rel_item in orig_otm:
            key = f"{orig_rel_item['parent']} → {orig_rel_item['child']}"
            orig_ratio = orig_rel_item.get('avg_children', 0)

            if key in synth_rel_map:
                synth_ratio = synth_rel_map[key].get('avg_children', 0)
                similarity = max(0, 1 - abs(orig_ratio - synth_ratio) / max(orig_ratio, 1))
                similarities.append(similarity)
            else:
                similarities.append(0)

        return sum(similarities) / len(similarities) if similarities else 0

    def _calculate_weighted_column_similarity(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> float:
        """Calculate weighted column similarity"""
        column_details = self._compare_column_details(orig, synth)

        similarities = []
        for col_comp in column_details:
            if col_comp.get("overall_numeric_similarity"):
                similarities.append(col_comp["overall_numeric_similarity"])
            elif col_comp.get("overall_string_similarity"):
                similarities.append(col_comp["overall_string_similarity"])
            elif col_comp.get("overall_boolean_similarity"):
                similarities.append(col_comp["overall_boolean_similarity"])
            elif col_comp.get("type_match"):
                similarities.append(0.5)  # Partial credit for type match

        return sum(similarities) / len(similarities) if similarities else 0

    def _generate_column_comparison_table(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed column comparison table"""

        column_comparisons = []
        orig_columns = orig.get("columns", {})
        synth_columns = synth.get("columns", {})

        all_columns = set(orig_columns.keys()) | set(synth_columns.keys())

        for col_name in all_columns:
            if col_name in orig_columns and col_name in synth_columns:
                orig_col = orig_columns[col_name]
                synth_col = synth_columns[col_name]

                # Type match
                type_match = orig_col["type"] == synth_col["type"]

                # Calculate similarities based on type
                if orig_col["type"] in ["integer", "float"] and synth_col["type"] in ["integer", "float"]:
                    stat_sim, dist_sim = self._calculate_numeric_similarities(orig_col["stats"], synth_col["stats"])
                elif orig_col["type"] == "string" and synth_col["type"] == "string":
                    stat_sim, dist_sim = self._calculate_string_similarities(orig_col["stats"], synth_col["stats"])
                else:
                    stat_sim, dist_sim = 0.5, 0.5  # Default for other types

                # Pattern quality assessment
                pattern_quality = self._assess_column_pattern_quality(col_name, orig_col, synth_col)

                column_comparisons.append({
                    "column": col_name,
                    "original_type": orig_col["type"],
                    "synthetic_type": synth_col["type"],
                    "type_match": type_match,
                    "original_unique": orig_col["stats"].get("unique", orig_col["stats"].get("count", 0)),
                    "synthetic_unique": synth_col["stats"].get("unique", synth_col["stats"].get("count", 0)),
                    "uniqueness_similarity": self._calculate_uniqueness_similarity(orig_col, synth_col),
                    "statistical_similarity": stat_sim,
                    "distribution_similarity": dist_sim,
                    "pattern_quality": pattern_quality
                })

        return column_comparisons

    def _generate_relationship_comparison_table(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Generate relationship comparison analysis"""

        orig_rel = orig.get("relationships", {})
        synth_rel = synth.get("relationships", {})

        # One-to-many relationships comparison
        one_to_many_comparison = []
        orig_otm = orig_rel.get("one_to_many", [])
        synth_otm = synth_rel.get("one_to_many", [])

        # Create relationship mapping
        synth_rel_map = {}
        for rel in synth_otm:
            key = f"{rel['parent']} → {rel['child']}"
            synth_rel_map[key] = rel

        for orig_rel_item in orig_otm:
            key = f"{orig_rel_item['parent']} → {orig_rel_item['child']}"
            orig_ratio = orig_rel_item.get('avg_children', 0)

            if key in synth_rel_map:
                synth_ratio = synth_rel_map[key].get('avg_children', 0)
                difference = abs(orig_ratio - synth_ratio)
                similarity = max(0, 1 - difference / max(orig_ratio, 1)) if orig_ratio > 0 else 0
            else:
                synth_ratio = 0
                difference = orig_ratio
                similarity = 0

            one_to_many_comparison.append({
                "relationship": key,
                "original_ratio": orig_ratio,
                "synthetic_ratio": synth_ratio,
                "difference": difference,
                "similarity": f"{similarity:.2%}"
            })

        # Calculate overall relationship preservation score
        total_similarities = [float(item["similarity"].rstrip('%')) / 100 for item in one_to_many_comparison]
        relationship_preservation_score = sum(total_similarities) / len(
            total_similarities) if total_similarities else 1.0

        return {
            "one_to_many_relationships": one_to_many_comparison,
            "relationship_preservation_score": relationship_preservation_score
        }

    def _generate_pattern_analysis(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pattern analysis comparison"""

        orig_patterns = set(orig.get("patterns", []))
        synth_patterns = set(synth.get("patterns", []))

        common_patterns = orig_patterns & synth_patterns
        only_original = orig_patterns - synth_patterns
        only_synthetic = synth_patterns - orig_patterns

        pattern_similarity = len(common_patterns) / len(
            orig_patterns | synth_patterns) if orig_patterns | synth_patterns else 1.0

        return {
            "common_patterns": list(common_patterns),
            "only_in_original": list(only_original),
            "only_in_synthetic": list(only_synthetic),
            "pattern_similarity": pattern_similarity
        }

    # =============================================================================
    # HELPER METHODS FOR ENHANCED CALCULATIONS
    # =============================================================================

    def _calculate_numeric_similarities(self, orig_stats: Dict, synth_stats: Dict) -> Tuple[float, float]:
        """Calculate numeric column similarities"""
        # Statistical similarity
        orig_mean = orig_stats.get("mean", 0)
        synth_mean = synth_stats.get("mean", 0)
        orig_std = orig_stats.get("std", 1)
        synth_std = synth_stats.get("std", 1)

        mean_sim = max(0, 1 - abs(orig_mean - synth_mean) / max(abs(orig_mean), 1))
        std_sim = max(0, 1 - abs(orig_std - synth_std) / max(orig_std, 1))

        statistical_similarity = (mean_sim + std_sim) / 2

        # Distribution similarity (simplified)
        distribution_similarity = statistical_similarity * 0.9  # Placeholder

        return statistical_similarity, distribution_similarity

    def _calculate_string_similarities(self, orig_stats: Dict, synth_stats: Dict) -> Tuple[float, float]:
        """Calculate string column similarities"""
        orig_unique = orig_stats.get("unique", 0)
        synth_unique = synth_stats.get("unique", 0)

        uniqueness_sim = min(orig_unique, synth_unique) / max(orig_unique, synth_unique) if max(orig_unique,
                                                                                                synth_unique) > 0 else 1.0

        return uniqueness_sim, uniqueness_sim

    def _calculate_uniqueness_similarity(self, orig_col: Dict, synth_col: Dict) -> float:
        """Calculate uniqueness similarity between columns"""
        orig_unique = orig_col["stats"].get("unique", orig_col["stats"].get("count", 0))
        synth_unique = synth_col["stats"].get("unique", synth_col["stats"].get("count", 0))

        if max(orig_unique, synth_unique) == 0:
            return 1.0

        return min(orig_unique, synth_unique) / max(orig_unique, synth_unique)

    def _assess_column_pattern_quality(self, col_name: str, orig_col: Dict, synth_col: Dict) -> str:
        """Assess pattern quality for a column"""
        orig_unique_pct = orig_col["stats"].get("unique_percent", 0)
        synth_unique_pct = synth_col["stats"].get("unique_percent", 0)

        uniqueness_ratio = synth_unique_pct / orig_unique_pct if orig_unique_pct > 0 else 1.0

        if uniqueness_ratio >= 0.9:
            return "🟢 Excellent"
        elif uniqueness_ratio >= 0.7:
            return "🟡 Good"
        elif uniqueness_ratio >= 0.5:
            return "🟡 Moderate"
        else:
            return "🔴 Poor"

    # =============================================================================
    # EXISTING METHODS FROM ORIGINAL COMPARISON.PY
    # =============================================================================

    def _compare_column_details(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compare detailed column statistics"""
        column_comparisons = []

        # Get all columns from both datasets
        all_columns = set(orig["columns"].keys()) | set(synth["columns"].keys())

        for col_name in all_columns:
            if col_name in orig["columns"] and col_name in synth["columns"]:
                orig_col = orig["columns"][col_name]
                synth_col = synth["columns"][col_name]

                comparison = {
                    "column": col_name,
                    "type_match": orig_col["type"] == synth_col["type"],
                    "original_type": orig_col["type"],
                    "synthetic_type": synth_col["type"]
                }

                # Compare statistics based on type
                if orig_col["type"] in ["integer", "float"] and synth_col["type"] in ["integer", "float"]:
                    comparison.update(self._compare_numeric_stats(orig_col["stats"], synth_col["stats"]))
                elif orig_col["type"] == "string" and synth_col["type"] == "string":
                    comparison.update(self._compare_string_stats(orig_col["stats"], synth_col["stats"]))
                elif orig_col["type"] == "boolean" and synth_col["type"] == "boolean":
                    comparison.update(self._compare_boolean_stats(orig_col["stats"], synth_col["stats"]))
                elif orig_col["type"] == "datetime" and synth_col["type"] == "datetime":
                    comparison.update(self._compare_datetime_stats(orig_col["stats"], synth_col["stats"]))
                else:
                    # Type mismatch or unsupported type
                    comparison.update({
                        "type_mismatch": True,
                        "comparison_possible": False
                    })

                column_comparisons.append(comparison)

            elif col_name in orig["columns"]:
                # Column only in original
                column_comparisons.append({
                    "column": col_name,
                    "original_type": orig["columns"][col_name]["type"],
                    "synthetic_type": "MISSING",
                    "type_match": False,
                    "missing_in_synthetic": True
                })

            elif col_name in synth["columns"]:
                # Column only in synthetic
                column_comparisons.append({
                    "column": col_name,
                    "original_type": "MISSING",
                    "synthetic_type": synth["columns"][col_name]["type"],
                    "type_match": False,
                    "extra_in_synthetic": True
                })

        return column_comparisons

    def _compare_numeric_stats(self, orig_stats: Dict, synth_stats: Dict) -> Dict[str, Any]:
        """Compare numeric column statistics"""
        orig_mean = orig_stats.get("mean", 0)
        synth_mean = synth_stats.get("mean", 0)
        orig_std = orig_stats.get("std", 1)
        synth_std = synth_stats.get("std", 1)
        orig_min = orig_stats.get("min", 0)
        synth_min = synth_stats.get("min", 0)
        orig_max = orig_stats.get("max", 0)
        synth_max = synth_stats.get("max", 0)

        return {
            "mean_comparison": {
                "original": orig_mean,
                "synthetic": synth_mean,
                "difference": abs(orig_mean - synth_mean),
                "similarity_score": self._calculate_numeric_similarity_score(orig_mean, synth_mean, orig_std)
            },
            "std_comparison": {
                "original": orig_std,
                "synthetic": synth_std,
                "difference": abs(orig_std - synth_std),
                "similarity_score": self._calculate_numeric_similarity_score(orig_std, synth_std, max(orig_std, 1))
            },
            "range_comparison": {
                "original_range": f"{orig_min} - {orig_max}",
                "synthetic_range": f"{synth_min} - {synth_max}",
                "min_difference": abs(orig_min - synth_min),
                "max_difference": abs(orig_max - synth_max)
            },
            "overall_numeric_similarity": (
                                                  self._calculate_numeric_similarity_score(orig_mean, synth_mean,
                                                                                           orig_std) +
                                                  self._calculate_numeric_similarity_score(orig_std, synth_std,
                                                                                           max(orig_std, 1))
                                          ) / 2
        }

    def _compare_string_stats(self, orig_stats: Dict, synth_stats: Dict) -> Dict[str, Any]:
        """Compare string column statistics"""
        orig_unique = orig_stats.get("unique", 0)
        synth_unique = synth_stats.get("unique", 0)
        orig_unique_pct = orig_stats.get("unique_percent", 0)
        synth_unique_pct = synth_stats.get("unique_percent", 0)

        return {
            "unique_comparison": {
                "original": orig_unique,
                "synthetic": synth_unique,
                "difference": abs(orig_unique - synth_unique),
                "similarity_score": min(orig_unique, synth_unique) / max(orig_unique, synth_unique) if max(orig_unique,
                                                                                                           synth_unique) > 0 else 1.0
            },
            "unique_percent_comparison": {
                "original": orig_unique_pct,
                "synthetic": synth_unique_pct,
                "difference": abs(orig_unique_pct - synth_unique_pct)
            },
            "overall_string_similarity": min(orig_unique, synth_unique) / max(orig_unique, synth_unique) if max(
                orig_unique, synth_unique) > 0 else 1.0
        }

    def _compare_boolean_stats(self, orig_stats: Dict, synth_stats: Dict) -> Dict[str, Any]:
        """Compare boolean column statistics"""
        orig_true_pct = orig_stats.get("true_percentage", 0)
        synth_true_pct = synth_stats.get("true_percentage", 0)

        return {
            "true_percentage_comparison": {
                "original": orig_true_pct,
                "synthetic": synth_true_pct,
                "difference": abs(orig_true_pct - synth_true_pct)
            },
            "overall_boolean_similarity": max(0, 1 - abs(orig_true_pct - synth_true_pct) / 100)
        }

    def _compare_datetime_stats(self, orig_stats: Dict, synth_stats: Dict) -> Dict[str, Any]:
        """Compare datetime column statistics"""
        return {
            "date_range_comparison": {
                "original_range": f"{orig_stats.get('min', 'N/A')} - {orig_stats.get('max', 'N/A')}",
                "synthetic_range": f"{synth_stats.get('min', 'N/A')} - {synth_stats.get('max', 'N/A')}",
                "range_days_original": orig_stats.get("range_days", 0),
                "range_days_synthetic": synth_stats.get("range_days", 0)
            },
            "overall_datetime_similarity": 0.5  # Simplified for now
        }

    def _compare_relationships(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare relationship structures"""
        orig_rel = orig.get("relationships", {})
        synth_rel = synth.get("relationships", {})

        return {
            "one_to_many": self._compare_one_to_many(
                orig_rel.get("one_to_many", []),
                synth_rel.get("one_to_many", [])
            ),
            "many_to_many": self._compare_many_to_many(
                orig_rel.get("many_to_many", []),
                synth_rel.get("many_to_many", [])
            ),
            "value_mappings": self._compare_value_mappings(
                orig_rel.get("value_mappings", {}),
                synth_rel.get("value_mappings", {})
            )
        }

    def _compare_one_to_many(self, orig_relations: List, synth_relations: List) -> Dict[str, Any]:
        """Compare one-to-many relationships"""
        matched_relations = []
        unmatched_original = []
        unmatched_synthetic = []

        # Create copies to avoid modifying original lists
        remaining_synth = synth_relations.copy()

        # Find matching relationships
        for orig_rel in orig_relations:
            match_found = False
            for i, synth_rel in enumerate(remaining_synth):
                if (orig_rel["parent"] == synth_rel["parent"] and
                        orig_rel["child"] == synth_rel["child"]):
                    matched_relations.append({
                        "parent": orig_rel["parent"],
                        "child": orig_rel["child"],
                        "original_ratio": orig_rel["avg_children"],
                        "synthetic_ratio": synth_rel["avg_children"],
                        "ratio_difference": abs(orig_rel["avg_children"] - synth_rel["avg_children"]),
                        "similarity_score": min(orig_rel["avg_children"], synth_rel["avg_children"]) /
                                            max(orig_rel["avg_children"], synth_rel["avg_children"])
                    })
                    remaining_synth.pop(i)
                    match_found = True
                    break

            if not match_found:
                unmatched_original.append(orig_rel)

        # Remaining synthetic relationships are unmatched
        unmatched_synthetic = remaining_synth

        return {
            "matched_relationships": matched_relations,
            "unmatched_original": unmatched_original,
            "unmatched_synthetic": unmatched_synthetic,
            "relationship_preservation_score": len(matched_relations) / max(len(orig_relations),
                                                                            1) if orig_relations else 0.0
        }

    def _compare_many_to_many(self, orig_relations: List, synth_relations: List) -> Dict[str, Any]:
        """Compare many-to-many relationships"""
        matched_relations = []
        unmatched_original = orig_relations.copy()
        unmatched_synthetic = synth_relations.copy()

        # Simple matching for many-to-many (can be enhanced)
        for orig_rel in orig_relations:
            for synth_rel in synth_relations:
                if ((orig_rel["col1"] == synth_rel["col1"] and orig_rel["col2"] == synth_rel["col2"]) or
                        (orig_rel["col1"] == synth_rel["col2"] and orig_rel["col2"] == synth_rel["col1"])):
                    matched_relations.append({
                        "col1": orig_rel["col1"],
                        "col2": orig_rel["col2"],
                        "original_ratio": orig_rel["avg_ratio"],
                        "synthetic_ratio": synth_rel["avg_ratio"],
                        "description": f"Many-to-many between {orig_rel['col1']} and {orig_rel['col2']}"
                    })
                    if orig_rel in unmatched_original:
                        unmatched_original.remove(orig_rel)
                    if synth_rel in unmatched_synthetic:
                        unmatched_synthetic.remove(synth_rel)
                    break

        return {
            "matched_relationships": matched_relations,
            "unmatched_original": unmatched_original,
            "unmatched_synthetic": unmatched_synthetic,
            "relationship_preservation_score": len(matched_relations) / max(len(orig_relations),
                                                                            1) if orig_relations else 0.0
        }

    def _compare_value_mappings(self, orig_mappings: Dict, synth_mappings: Dict) -> Dict[str, Any]:
        """Compare value pattern mappings"""
        pattern_comparisons = {}

        for col in set(orig_mappings.keys()) | set(synth_mappings.keys()):
            if col in orig_mappings and col in synth_mappings:
                orig_patterns = orig_mappings[col]
                synth_patterns = synth_mappings[col]

                pattern_comparisons[col] = {}

                # Compare email domains
                if "email_domains" in orig_patterns and "email_domains" in synth_patterns:
                    orig_domains = set(orig_patterns["email_domains"]["top_domains"].keys())
                    synth_domains = set(synth_patterns["email_domains"]["top_domains"].keys())

                    overlap = len(orig_domains & synth_domains)
                    total = len(orig_domains | synth_domains)

                    pattern_comparisons[col]["email_domains"] = {
                        "domain_overlap": overlap / total if total > 0 else 0,
                        "original_domain_count": orig_patterns["email_domains"]["total_domains"],
                        "synthetic_domain_count": synth_patterns["email_domains"]["total_domains"]
                    }

                # Compare phone patterns
                if "phone_patterns" in orig_patterns and "phone_patterns" in synth_patterns:
                    orig_patterns_set = set(orig_patterns["phone_patterns"]["top_patterns"].keys())
                    synth_patterns_set = set(synth_patterns["phone_patterns"]["top_patterns"].keys())

                    overlap = len(orig_patterns_set & synth_patterns_set)
                    total = len(orig_patterns_set | synth_patterns_set)

                    pattern_comparisons[col]["phone_patterns"] = {
                        "pattern_overlap": overlap / total if total > 0 else 0,
                        "original_pattern_count": orig_patterns["phone_patterns"]["total_patterns"],
                        "synthetic_pattern_count": synth_patterns["phone_patterns"]["total_patterns"]
                    }

                # Compare common prefixes
                if "common_prefixes" in orig_patterns and "common_prefixes" in synth_patterns:
                    orig_prefixes = set(orig_patterns["common_prefixes"]["top_prefixes"].keys())
                    synth_prefixes = set(synth_patterns["common_prefixes"]["top_prefixes"].keys())

                    overlap = len(orig_prefixes & synth_prefixes)
                    total = len(orig_prefixes | synth_prefixes)

                    pattern_comparisons[col]["common_prefixes"] = {
                        "prefix_overlap": overlap / total if total > 0 else 0,
                        "original_prefix_count": len(orig_prefixes),
                        "synthetic_prefix_count": len(synth_prefixes)
                    }

        return pattern_comparisons

    def _compare_patterns(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare general data patterns"""
        orig_patterns = set(orig.get("patterns", []))
        synth_patterns = set(synth.get("patterns", []))

        common = orig_patterns & synth_patterns
        total = orig_patterns | synth_patterns

        return {
            "common_patterns": list(common),
            "original_only": list(orig_patterns - synth_patterns),
            "synthetic_only": list(synth_patterns - orig_patterns),
            "pattern_similarity": len(common) / len(total) if total else 1.0
        }

    def _compare_correlations(self, orig: Dict[str, Any], synth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare correlation structures"""
        orig_corr = orig.get("correlations", {}).get("high_correlations", [])
        synth_corr = synth.get("correlations", {}).get("high_correlations", [])

        matched_correlations = []

        for orig_c in orig_corr:
            for synth_c in synth_corr:
                if ((orig_c[0] == synth_c[0] and orig_c[1] == synth_c[1]) or
                        (orig_c[0] == synth_c[1] and orig_c[1] == synth_c[0])):
                    matched_correlations.append({
                        "columns": f"{orig_c[0]} - {orig_c[1]}",
                        "original_correlation": orig_c[2],
                        "synthetic_correlation": synth_c[2],
                        "difference": abs(orig_c[2] - synth_c[2])
                    })
                    break

        return {
            "matched_correlations": matched_correlations,
            "correlation_preservation_score": len(matched_correlations) / max(len(orig_corr), 1) if orig_corr else 0.0
        }

    def _calculate_numeric_similarity_score(self, val1: float, val2: float, std_dev: float) -> float:
        """Calculate similarity score for numeric values"""
        if std_dev == 0:
            return 1.0 if val1 == val2 else 0.0

        normalized_diff = abs(val1 - val2) / max(std_dev, 1e-8)
        return max(0, 1 - min(normalized_diff, 1))

    def _compare_columns(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, Any]:
        """Compare individual columns"""
        orig_clean = orig_series.dropna()
        synth_clean = synth_series.dropna()

        if np.issubdtype(orig_clean.dtype, np.number) and np.issubdtype(synth_clean.dtype, np.number):
            return self._compare_numeric_columns(orig_clean, synth_clean)
        else:
            return self._compare_categorical_columns(orig_clean, synth_clean)

    def _compare_numeric_columns(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, Any]:
        """Compare numeric columns"""
        try:
            ks_stat, ks_p_value = stats.ks_2samp(orig_series, synth_series)
        except:
            ks_stat, ks_p_value = np.nan, np.nan

        return {
            "type": "numeric",
            "mean_diff": abs(orig_series.mean() - synth_series.mean()),
            "std_diff": abs(orig_series.std() - synth_series.std()),
            "median_diff": abs(orig_series.median() - synth_series.median()),
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p_value,
            "distributions_similar": ks_p_value > 0.05 if not np.isnan(ks_p_value) else False,
            "similarity_score": self._calculate_numeric_similarity(orig_series, synth_series)
        }

    def _compare_categorical_columns(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, Any]:
        """Compare categorical columns"""
        orig_freq = orig_series.value_counts(normalize=True).sort_index()
        synth_freq = synth_series.value_counts(normalize=True).sort_index()

        all_values = sorted(set(orig_freq.index) | set(synth_freq.index))
        orig_aligned = orig_freq.reindex(all_values, fill_value=0)
        synth_aligned = synth_freq.reindex(all_values, fill_value=0)

        try:
            js_divergence = distance.jensenshannon(orig_aligned, synth_aligned)
        except:
            js_divergence = 1.0

        common_values = set(orig_freq.index) & set(synth_freq.index)
        union_values = set(orig_freq.index) | set(synth_freq.index)

        return {
            "type": "categorical",
            "js_divergence": js_divergence,
            "distributions_similar": js_divergence < 0.3,
            "value_overlap": len(common_values) / len(union_values) if union_values else 0,
            "unique_count_diff": abs(len(orig_freq) - len(synth_freq)),
            "similarity_score": 1 - js_divergence
        }

    def _calculate_numeric_similarity(self, orig_series: pd.Series, synth_series: pd.Series) -> float:
        """Calculate similarity score for numeric columns (0-1)"""
        mean_diff_norm = abs(orig_series.mean() - synth_series.mean()) / (orig_series.std() + 1e-8)
        std_diff_norm = abs(orig_series.std() - synth_series.std()) / (orig_series.std() + 1e-8)

        mean_sim = max(0, 1 - min(mean_diff_norm, 1))
        std_sim = max(0, 1 - min(std_diff_norm, 1))

        return (mean_sim + std_sim) / 2

    def _calculate_overall_similarity(self, comparison: Dict[str, Any]) -> float:
        """Calculate overall similarity score"""
        if not comparison:
            return 0.0

        scores = [comp.get("similarity_score", 0) for comp in comparison.values()]
        return sum(scores) / len(scores) if scores else 0.0