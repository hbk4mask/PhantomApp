# prompt_generator.py — Compact, precedence-aware prompt builder
import json
from typing import Dict, Any, List, Tuple

from config import Config  # keeps your model+limits config


class PromptGenerator:
    """
    Builds a compact, precedence-aware prompt that:
      - Makes row count the ONLY hard constraint
      - Uses tolerances for stats, distributions, correlations
      - Includes patterns (email domains, UUID-like) when present
      - Respects relationship filtering (expects filtered analysis)
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

    # ---------------------------
    # Public API
    # ---------------------------
    # def generate_from_analysis(self, analysis: Dict[str, Any], num_rows: int = 50) -> str:
    #     """
    #     Create a compact generation prompt from the code-based analysis.
    #     Assumes relationships were already filtered, if needed.
    #     """
    #     cols = analysis.get("columns", {})
    #     correlations = analysis.get("correlations", {}) or {}
    #     relationships = analysis.get("relationships", {}) or {}
    #
    #     # Schema types summary
    #     schema_types_str = self._format_schema_types(cols)
    #
    #     # Header & global rules
    #     lines: List[str] = []
    #     lines.append("You are generating SYNTHETIC TABULAR DATA.\n")
    #     lines.append(f"Output:\n- Pure CSV with header, EXACTLY {num_rows} rows. No prose.\n")
    #     lines.append("Global rules (take precedence in this order):")
    #     lines.append(f"1) Row count must be EXACTLY {num_rows}.")
    #     lines.append(f"2) Schema: use exactly these columns with these data types: {schema_types_str}.")
    #     lines.append("3) Statistical fidelity: match each numeric column’s mean, std, min, max, median, Q1, Q3 "
    #                  "within ±10% (or within original IQR for quantiles).")
    #     lines.append("4) Distribution fidelity (categoricals/strings): match value-count proportions within a small "
    #                  "divergence (target KL≤0.1); keep top-k categories and long-tail share.")
    #     lines.append("5) Uniqueness & entropy: match uniqueness% and entropy within ±5 percentage points where available.")
    #     lines.append("6) Correlations: preserve numeric pairwise Pearson/Spearman within ±0.10 absolute where computed.")
    #     lines.append("7) Patterns: preserve detected patterns (email domains, UUID-like length/hyphens) and their relative frequencies.\n")
    #
    #     # Schema & targets per column
    #     lines.append("Schema & targets:")
    #     for col_name, info in cols.items():
    #         lines.extend(self._format_column_targets(col_name, info))
    #
    #     # Correlation targets (numeric)
    #     corr_block = self._format_correlation_targets(correlations)
    #     if corr_block:
    #         lines.append("\nCorrelation targets (numeric only):")
    #         lines.extend(corr_block)
    #
    #     # Relationship / joint distribution targets (soft)
    #     rel_block = self._format_relationship_targets(relationships)
    #     if rel_block:
    #         lines.append("\nRelationships to preserve (soft targets):")
    #         lines.extend(rel_block)
    #
    #     lines.append("\nDo NOT preserve relationships that were excluded by integrity/relationship filtering.")
    #     lines.append("\nFormatting:\n- No markdown/code fences.\n- Escape commas inside fields if needed; otherwise no quotes.")
    #
    #     return "\n".join(lines)
    def generate_from_analysis(self, analysis: Dict[str, Any], num_rows: int = 50) -> str:
        """
        Create a compact, precedence-aware synthetic data prompt from the code-based analysis.
        - Row count is the ONLY hard constraint.
        - Per-column stats & patterns are expressed as targets with tolerances.
        - Numeric correlations and categorical joint distributions are soft targets.
        - Assumes relationships were already filtered upstream (if you use RelationshipFilter).
        """

        # ---------------------------
        # Small helpers (local to keep this method drop-in)
        # ---------------------------
        def _fmt(x):
            if x is None:
                return "N/A"
            try:
                return round(float(x), 6)
            except Exception:
                return x

        def _schema_types(cols: Dict[str, Any]) -> str:
            parts = []
            for col_name, info in (cols or {}).items():
                ctype = info.get("type", "string")
                parts.append(f"{col_name}:{ctype}")
            return ", ".join(parts)

        def _topk_proportions(value_dist: Dict[str, int], count_total: int):
            """Convert top-k counts to proportions and compute tail share."""
            if not value_dist:
                return {}, 0.0
            if count_total and count_total > 0:
                topk_pct = {k: round(v / count_total, 4) for k, v in value_dist.items()}
                tail_share = round(max(0.0, 1.0 - sum(topk_pct.values())), 4)
                return topk_pct, tail_share
            # Fallback if total missing: normalize by sum of provided counts
            s = float(sum(value_dist.values()))
            if s <= 0:
                return {}, 0.0
            topk_pct = {k: round(v / s, 4) for k, v in value_dist.items()}
            return topk_pct, 0.0

        def _proportionize(d: Dict[str, int]) -> Dict[str, float]:
            if not d:
                return {}
            total = float(sum(d.values()))
            if total <= 0:
                return {k: 0.0 for k in d}
            return {k: round(v / total, 4) for k, v in d.items()}

        def _format_column_targets(col_name: str, info: Dict[str, Any]) -> list[str]:
            stats = (info or {}).get("stats", {}) or {}
            ctype = (info or {}).get("type", "string")
            out = []

            null_pct = round(stats.get("null_percentage", 0.0), 2)
            uniq_pct = round(
                stats.get("unique_percent", stats.get("unique_percentage", 0.0)),
                2
            )
            entropy = stats.get("entropy", None)
            entropy_str = f"{entropy:.4f}" if isinstance(entropy, (int, float)) else "N/A"

            out.append(f"- {col_name}: type={ctype}, null%≈{null_pct}, uniqueness%≈{uniq_pct}, entropy≈{entropy_str}.")

            if ctype in ("integer", "float"):
                # Normalize q1/q3 naming
                q1 = stats.get("q25", stats.get("q1"))
                q3 = stats.get("q75", stats.get("q3"))
                out.append(
                    "  moments: mean={mean}, std={std}, q1={q1}, median={median}, q3={q3}, range=[{vmin},{vmax}]".format(
                        mean=_fmt(stats.get("mean")),
                        std=_fmt(stats.get("std")),
                        q1=_fmt(q1),
                        median=_fmt(stats.get("median")),
                        q3=_fmt(q3),
                        vmin=_fmt(stats.get("min")),
                        vmax=_fmt(stats.get("max")),
                    )
                )
            else:
                # Categorical / string
                value_dist = stats.get("value_distribution", {}) or {}
                count_total = int(stats.get("count") or 0)
                topk_pct, tail_share = _topk_proportions(value_dist, count_total)
                out.append(f"  top values (proportions): {json.dumps(topk_pct)} and tail_share≈{_fmt(tail_share)}")

                # Email domains (if analyzer provided them)
                email_domains = stats.get("email_domains") or {}
                if email_domains:
                    top_domains = email_domains.get("top_domains", {}) or {}
                    total_domains = email_domains.get("total_domains")
                    out.append(
                        "  email domain targets: " +
                        json.dumps({"top_domains": _proportionize(top_domains), "total_domains": total_domains})
                    )

                # UUID-like pattern summary (avg length, hyphen ratio, hex-only ratio)
                uuid_like = stats.get("uuid_like") or {}
                if uuid_like:
                    out.append(f"  uuid-like: {json.dumps(uuid_like)}")

            return out

        def _format_correlation_targets(correlations: Dict[str, Any]) -> list[str]:
            out = []
            if not isinstance(correlations, dict):
                return out
            pearson = correlations.get("pearson")
            spearman = correlations.get("spearman")
            if not isinstance(pearson, dict) or not isinstance(spearman, dict):
                return out

            cols = list(pearson.keys())
            for i in range(len(cols)):
                c1 = cols[i]
                for j in range(i + 1, len(cols)):
                    c2 = cols[j]
                    try:
                        r = float(pearson.get(c1, {}).get(c2, 0.0))
                        s = float(spearman.get(c1, {}).get(c2, 0.0))
                    except (TypeError, ValueError):
                        continue
                    if abs(r) >= 0.2 or abs(s) >= 0.2:
                        out.append(f"- {c1}↔{c2}: Pearson≈{round(r, 3)}, Spearman≈{round(s, 3)} (tolerance ±0.10)")
            return out

        def _format_relationship_targets(relationships: Dict[str, Any]) -> list[str]:
            out = []
            if not isinstance(relationships, dict):
                return out
            # Preferred structure from analyzer: {'joint_distributions': [{'between':[A,B], 'pairs':[{'A':..., 'B':..., 'p':...}]}]}
            joint_list = relationships.get("joint_distributions") or []
            for jd in joint_list:
                between = jd.get("between") or []
                pairs = jd.get("pairs") or []
                if len(between) == 2 and pairs:
                    a, b = between
                    out.append(
                        f"- Preserve joint distribution between {a} and {b} (no hard quotas). Notable pairs to approximate:")
                    for rec in pairs[:10]:  # limit to 10 to stay compact
                        v1 = rec.get(a) or rec.get("v1")
                        v2 = rec.get(b) or rec.get("v2")
                        p = rec.get("p")
                        if p is not None:
                            out.append(f"  • {v1} × {v2} ≈ {round(float(p) * 100, 2)}%")
            # Generic note if analyzer also flags many-to-many without explicit pairs
            if relationships.get("many_to_many"):
                out.append(
                    "- Maintain realistic overlap for detected many-to-many categorical relationships (soft target).")
            return out

        # ---------------------------
        # Build prompt from analysis
        # ---------------------------
        cols = analysis.get("columns", {}) or {}
        correlations = analysis.get("correlations", {}) or {}
        relationships = analysis.get("relationships", {}) or {}

        schema_types_str = _schema_types(cols)

        lines = []
        lines.append("You are generating SYNTHETIC TABULAR DATA.\n")
        lines.append(f"Output:\n- Pure CSV with header, EXACTLY {num_rows} rows. No prose.\n")
        lines.append("Global rules (take precedence in this order):")
        lines.append(f"1) Row count must be EXACTLY {num_rows}.")
        lines.append(f"2) Schema: use exactly these columns with these data types: {schema_types_str}.")
        lines.append(
            "3) Statistical fidelity: match each numeric column’s mean, std, min, max, median, Q1, Q3 within ±10% (or within original IQR for quantiles).")
        lines.append(
            "4) Distribution fidelity (categoricals/strings): match value-count proportions within a small divergence (target KL≤0.1); keep top-k categories and long-tail share.")
        lines.append(
            "5) Uniqueness & entropy: match uniqueness% and entropy within ±5 percentage points where available.")
        lines.append(
            "6) Correlations: preserve numeric pairwise Pearson/Spearman within ±0.10 absolute where computed.")
        lines.append(
            "7) Patterns: preserve detected patterns (email domains, UUID-like length/hyphens) and their relative frequencies.\n")

        # Per-column targets
        lines.append("Schema & targets:")
        for col_name, info in cols.items():
            lines.extend(_format_column_targets(col_name, info))

        # Correlation targets
        corr_block = _format_correlation_targets(correlations)
        if corr_block:
            lines.append("\nCorrelation targets (numeric only):")
            lines.extend(corr_block)

        # Relationship targets (soft)
        rel_block = _format_relationship_targets(relationships)
        if rel_block:
            lines.append("\nRelationships to preserve (soft targets):")
            lines.extend(rel_block)

        lines.append("\nDo NOT preserve relationships that were excluded by integrity/relationship filtering.")
        lines.append(
            "\nFormatting:\n- No markdown/code fences.\n- Escape commas inside fields if needed; otherwise no quotes.")

        return "\n".join(lines)

    # ---------------------------
    # Internals
    # ---------------------------
    def _format_schema_types(self, cols: Dict[str, Any]) -> str:
        parts: List[str] = []
        for col_name, info in cols.items():
            ctype = info.get("type", "string")
            parts.append(f"{col_name}:{ctype}")
        return ", ".join(parts)

    def _format_column_targets(self, col_name: str, info: Dict[str, Any]) -> List[str]:
        stats = info.get("stats", {}) or {}
        ctype = info.get("type", "string")
        out: List[str] = []

        null_pct = round(stats.get("null_percentage", 0.0), 2)
        uniq_pct = round(stats.get("unique_percent", 0.0), 2)
        entropy = stats.get("entropy", None)
        entropy_str = f"{entropy:.4f}" if isinstance(entropy, (int, float)) else "N/A"

        out.append(f"- {col_name}: type={ctype}, null%≈{null_pct}, uniqueness%≈{uniq_pct}, entropy≈{entropy_str}.")

        if ctype in ("integer", "float"):
            # Numeric moments
            num_keys = ("mean", "std", "q25", "median", "q75", "min", "max")
            # some analyses use q25/q75; fall back to q1/q3 if present
            q1 = stats.get("q25", stats.get("q1"))
            q3 = stats.get("q75", stats.get("q3"))
            mean = stats.get("mean")
            std = stats.get("std")
            median = stats.get("median")
            vmin = stats.get("min")
            vmax = stats.get("max")

            # Guard against None for safety
            out.append(
                "  moments: mean={mean}, std={std}, q1={q1}, median={median}, q3={q3}, range=[{vmin},{vmax}]".format(
                    mean=_fmt(mean), std=_fmt(std), q1=_fmt(q1), median=_fmt(median),
                    q3=_fmt(q3), vmin=_fmt(vmin), vmax=_fmt(vmax)
                )
            )
        else:
            # Categorical / string distributions
            value_dist = stats.get("value_distribution", {}) or {}
            count_total = stats.get("count", 0) or 0
            topk_pct, tail_share = self._topk_proportions(value_dist, count_total)
            out.append(f"  top values (proportions): {json.dumps(topk_pct)} and tail_share≈{_fmt(tail_share)}")

            # Email domains (if present)
            email_domains = stats.get("email_domains") or {}
            if email_domains:
                # normalize counts→proportions when possible
                top_domains = email_domains.get("top_domains", {})
                total_domains = email_domains.get("total_domains", None)
                out.append(f"  email domain targets: {json.dumps({'top_domains': _proportionize(top_domains), 'total_domains': total_domains})}")

            # UUID-like pattern (requires analyzer to include; see data_analysis.py update)
            uuid_like = stats.get("uuid_like") or {}
            if uuid_like:
                out.append(f"  uuid-like: {json.dumps(uuid_like)}")

        return out

    def _format_correlation_targets(self, correlations: Dict[str, Any]) -> List[str]:
        """
        Accepts analysis['correlations'] that may contain Pearson/Spearman matrices.
        Emits only pairs with notable absolute value (>=0.2) to keep prompt compact.
        """
        out: List[str] = []
        pearson = correlations.get("pearson")
        spearman = correlations.get("spearman")

        if not isinstance(pearson, dict) or not isinstance(spearman, dict):
            return out

        # pearson and spearman are expected as nested dicts: {col1: {col2: r}}
        cols = list(pearson.keys())
        for i in range(len(cols)):
            c1 = cols[i]
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                try:
                    r = float(pearson.get(c1, {}).get(c2, 0.0))
                    s = float(spearman.get(c1, {}).get(c2, 0.0))
                except (TypeError, ValueError):
                    continue
                if abs(r) >= 0.2 or abs(s) >= 0.2:
                    out.append(f"- {c1}↔{c2}: Pearson≈{round(r, 3)}, Spearman≈{round(s, 3)} (tolerance ±0.10)")
        return out

    def _format_relationship_targets(self, relationships: Dict[str, Any]) -> List[str]:
        """
        Emits soft joint-distribution tips for categorical-categorical links when present.
        Compatible with your relationship_filter: we just "preserve" what analysis kept.
        """
        out: List[str] = []

        # If your analyzer stores any joint distributions explicitly, include a compact summary.
        # We look for keys like 'joint_distributions' as a list of dicts with:
        #  {'between': ['DeviceType','DeviceOS'], 'pairs': [{'v1': 'Desktop', 'v2': 'Android', 'p': 0.06}, ...]}
        joint_list: List[Dict[str, Any]] = relationships.get("joint_distributions", []) or []
        for jd in joint_list:
            between = jd.get("between") or []
            pairs = jd.get("pairs") or []
            if len(between) == 2 and pairs:
                a, b = between
                out.append(f"- Preserve joint distribution between {a} and {b} (no hard quotas). Notable pairs to approximate:")
                # Limit to top 10 pairs to avoid verbosity
                for rec in pairs[:10]:
                    v1 = rec.get(a) or rec.get("v1")
                    v2 = rec.get(b) or rec.get("v2")
                    p = rec.get("p")
                    if p is not None:
                        out.append(f"  • {v1} × {v2} ≈ {round(float(p) * 100, 2)}%")

        # If your analyzer instead exposes high-level many-to-many notes, we emit a generic soft-preservation line
        if relationships.get("many_to_many"):
            out.append("- Maintain realistic overlap for detected many-to-many categorical relationships (soft target).")

        return out

    # ---------------------------
    # Utilities
    # ---------------------------
    def _topk_proportions(self, value_dist: Dict[str, int], total_count: int) -> Tuple[Dict[str, float], float]:
        """
        Convert top-k counts to proportions and compute tail share.
        If total_count is missing or zero, fall back to normalized counts sum.
        """
        if not value_dist:
            return {}, 0.0

        # Prefer the column's true total count
        if total_count and total_count > 0:
            topk_pct = {k: round(v / total_count, 4) for k, v in value_dist.items()}
            tail_share = round(max(0.0, 1.0 - sum(topk_pct.values())), 4)
            return topk_pct, tail_share

        # Fallback: normalize by the sum of provided top-k counts
        s = float(sum(value_dist.values()))
        if s <= 0:
            return {}, 0.0
        topk_pct = {k: round(v / s, 4) for k, v in value_dist.items()}
        tail_share = 0.0  # no idea of the true tail without total count
        return topk_pct, tail_share


def _fmt(x):
    if x is None:
        return "N/A"
    try:
        return round(float(x), 6)
    except Exception:
        return x


def _proportionize(d: Dict[str, int]) -> Dict[str, float]:
    if not d:
        return {}
    total = float(sum(d.values()))
    if total <= 0:
        return {k: 0.0 for k in d}
    return {k: round(v / total, 4) for k, v in d.items()}
