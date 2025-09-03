# data_analysis.py — Enhanced with uuid_like stats for string columns
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional

from config import Config


class DataAnalyzer:
    """Class to handle all data analysis operations"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # --- Add this inside class DataAnalyzer in data_analysis.py ---

    def get_key_stats(self, col_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        UI helper: return a compact, consistent summary of a single column's stats.
        ui_components.display_analysis_results(...) calls this as:
            key_stats = analyzer.get_key_stats(info)
        where `info` is analysis["columns"][<col>].

        We keep keys stable across types so the UI can render tables safely.
        """
        if not isinstance(col_info, dict):
            return {}

        ctype = col_info.get("type", "string")
        stats = (col_info.get("stats") or {})

        # Common fields
        out = {
            "type": ctype,
            "count": stats.get("count"),
            "null_count": stats.get("null_count"),
            "null_percent": stats.get("null_percentage"),
            "unique_count": stats.get("unique", stats.get("unique_values_count")),
            "unique_percent": stats.get("unique_percent", stats.get("unique_percentage")),
            "entropy": stats.get("entropy"),
        }

        if ctype in ("integer", "float"):
            # Normalize quantile naming (q1/q3 vs q25/q75) and include moments
            q1 = stats.get("q25", stats.get("q1"))
            q3 = stats.get("q75", stats.get("q3"))
            out.update({
                "min": stats.get("min"),
                "q1": q1,
                "median": stats.get("median"),
                "q3": q3,
                "max": stats.get("max"),
                "mean": stats.get("mean"),
                "std": stats.get("std"),
                "skewness": stats.get("skewness"),
                "kurtosis": stats.get("kurtosis"),
                "outlier_count": stats.get("outlier_count"),
                "outlier_percent": stats.get("outlier_percentage"),
            })
        elif ctype == "datetime":
            out.update({
                "min": stats.get("min"),
                "max": stats.get("max"),
                "range_days": stats.get("range_days"),
            })
        elif ctype == "boolean":
            out.update({
                "true_percent": stats.get("true_percentage"),
            })
        else:
            # string/categorical
            out.update({
                "most_common": stats.get("most_common"),
                "top_values": stats.get("value_distribution"),  # dict of top-k (value->count)
                "email_domains": stats.get("email_domains"),  # {top_domains, total_domains, ...}
                "phone_patterns": stats.get("phone_patterns"),
                "uuid_like": stats.get("uuid_like"),  # {avg_length, hyphen_presence_ratio, hex_chars_ratio}
                "total_unique_count": stats.get("total_unique_count"),
            })

        return out


    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main data analysis function - ENHANCED WITH RELATIONSHIPS"""
        columns = df.columns
        col_analysis: Dict[str, Any] = {}

        for col in columns:
            col_analysis[col] = self._analyze_column(df, col)

        # Detect higher-level patterns and correlations
        patterns = self._detect_patterns(df, col_analysis)
        correlations = self._analyze_correlations(df)

        # Detect column relationships (keep your existing logic)
        relationships = self._detect_column_relationships(df)

        return {
            "columns": col_analysis,
            "patterns": patterns,
            "correlations": correlations,
            "relationships": relationships,
            "row_count": len(df),
            "total_nulls": df.isnull().sum().sum(),
            "data_quality_score": self._calculate_quality_score(df),
        }

    # ----------------------------
    # Column analyzers
    # ----------------------------
    def _analyze_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        values = df[col]
        values_nonnull = values.dropna()
        dtype_str = str(values_nonnull.dtype)

        basic_stats = {
            "count": int(values_nonnull.shape[0]),
            "null_count": int(values.isnull().sum()),
            "null_percentage": round(values.isnull().mean() * 100, 2),
        }

        if np.issubdtype(values_nonnull.dtype, np.number):
            return self._analyze_numeric_column(values_nonnull, basic_stats, dtype_str)
        elif "datetime" in dtype_str.lower():
            return self._analyze_datetime_column(values_nonnull, basic_stats)
        elif "bool" in dtype_str.lower():
            return self._analyze_boolean_column(values_nonnull, basic_stats)
        else:
            return self._analyze_string_column(values_nonnull.astype(str), basic_stats)

    def _analyze_numeric_column(self, values: pd.Series, basic_stats: Dict, dtype_str: str) -> Dict[str, Any]:
        if len(values) > 0:
            basic_stats.update({
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
                "mean": round(float(values.mean()), 6),
                "median": round(float(values.median()), 6),
                "std": round(float(values.std()), 6),
                "q25": round(float(values.quantile(0.25)), 6),
                "q75": round(float(values.quantile(0.75)), 6),
                "skewness": round(float(stats.skew(values)), 4) if len(values) > 2 else 0.0,
                "kurtosis": round(float(stats.kurtosis(values)), 4) if len(values) > 3 else 0.0,
            })
            uniq = values.nunique()
            basic_stats.update({
                "unique_values_count": int(uniq),
                "unique_percentage": round(uniq / len(values) * 100, 2) if len(values) else 0.0,
                "sample_unique_values": list(values.unique()[:50]),
            })

        dtype = "float" if "float" in dtype_str else "integer"
        return {"type": dtype, "stats": basic_stats}

    def _analyze_datetime_column(self, values: pd.Series, basic_stats: Dict) -> Dict[str, Any]:
        if len(values) > 0:
            basic_stats.update({
                "min": values.min(),
                "max": values.max(),
                "range_days": (values.max() - values.min()).days
            })
        return {"type": "datetime", "stats": basic_stats}

    def _analyze_boolean_column(self, values: pd.Series, basic_stats: Dict) -> Dict[str, Any]:
        if len(values) > 0:
            true_pct = (values == True).mean() * 100
            basic_stats["true_percentage"] = round(float(true_pct), 2)
        return {"type": "boolean", "stats": basic_stats}

    def _analyze_string_column(self, values: pd.Series, basic_stats: Dict) -> Dict[str, Any]:
        # Basic frequency descriptors
        uniques = values.unique()
        freq = Counter(values)
        most_common = freq.most_common(1)[0] if freq else (None, 0)
        value_counts = values.value_counts()

        # Top-K distribution and all_unique_values (capped)
        top_values = value_counts.head(20).to_dict()
        all_unique_values = list(uniques[:100])
        entropy_value = stats.entropy(list(freq.values())) if len(freq) > 0 else 0.0

        # Email domains
        email_domains = {}
        if any('@' in v for v in values.head(50)):
            domains = values[values.str.contains("@")].str.split("@").str[-1]
            if len(domains) > 0:
                domain_counts = Counter(domains)
                email_domains = {
                    "top_domains": dict(domain_counts.most_common(10)),
                    "total_domains": int(len(domain_counts)),
                    "all_domains": list(domain_counts.keys())[:100],
                }

        # UUID-like pattern summary
        # (avg length, hyphen presence, hex-only ratio; helps the prompt preserve the "look" of IDs)
        s = values.astype(str)
        lengths = s.str.len()
        hyphen_presence = float((s.str.contains("-")).mean()) if len(s) > 0 else 0.0
        hex_only = float((s.str.replace("-", "", regex=False).str.match(r"^[0-9a-fA-F]+$")).mean()) if len(s) > 0 else 0.0
        uuid_like = {
            "avg_length": round(float(lengths.mean()) if len(lengths) else 0.0, 2),
            "hyphen_presence_ratio": round(hyphen_presence, 3),
            "hex_chars_ratio": round(hex_only, 3),
        }

        basic_stats.update({
            "unique": int(len(uniques)),
            "unique_percent": round(len(uniques) / len(values) * 100, 2) if len(values) else 0.0,
            "most_common": most_common,
            "entropy": round(float(entropy_value), 4),
            "value_distribution": top_values,
            "all_unique_values": all_unique_values,
            "email_domains": email_domains,
            "uuid_like": uuid_like,
            "total_unique_count": int(len(uniques)),
        })

        return {"type": "string", "stats": basic_stats}

    # ----------------------------
    # Existing helpers (unchanged)
    # ----------------------------
    def _detect_patterns(self, df: pd.DataFrame, col_analysis: Dict) -> List[str]:
        patterns: List[str] = []
        for col, info in col_analysis.items():
            unique_pct = info["stats"].get("unique_percent", 0)
            if unique_pct >= self.config.UNIQUE_ID_THRESHOLD * 100:
                patterns.append(f"{col} appears to be a unique identifier")
            elif info["type"] == "string" and info["stats"].get("unique", 0) <= self.config.CATEGORICAL_THRESHOLD:
                patterns.append(f"{col} appears to be categorical with {info['stats'].get('unique', 0)} unique values")
        return patterns

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Return nested dicts: {'pearson': {col: {col2: r}}, 'spearman': { ... }}
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {}
        pearson = numeric_df.corr(method="pearson").replace({np.nan: 0.0})
        spearman = numeric_df.corr(method="spearman").replace({np.nan: 0.0})
        return {
            "pearson": pearson.to_dict(),
            "spearman": spearman.to_dict(),
        }

    def _detect_column_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Keep your logic; here’s a light example that also exposes joint distributions for prompt
        relationships: Dict[str, Any] = {}

        # Example: joint distribution for any pair of string columns with modest cardinality
        cat_cols = [c for c in df.columns if df[c].dtype == object]
        joint_distributions: List[Dict[str, Any]] = []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                a, b = cat_cols[i], cat_cols[j]
                # Skip high-cardinality pairs; keep prompt compact
                if df[a].nunique() <= 20 and df[b].nunique() <= 20:
                    g = df.groupby([a, b]).size().reset_index(name="n")
                    total = max(1, len(df))
                    g["p"] = g["n"] / total
                    # Keep strongest 10 pairs
                    g = g.sort_values("p", ascending=False).head(10)
                    pairs = []
                    for _, r in g.iterrows():
                        pairs.append({a: r[a], b: r[b], "p": float(r["p"])})
                    if pairs:
                        joint_distributions.append({"between": [a, b], "pairs": pairs})

        if joint_distributions:
            relationships["joint_distributions"] = joint_distributions

        return relationships

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        # Keep whatever heuristic you used previously
        # Simple placeholder: higher is better
        non_null_ratio = 1.0 - df.isnull().mean().mean()
        return round(float(non_null_ratio * 100), 2)
