# =============================================================================
# File: relationship_filter.py
"""Relationship filtering system to exclude meaningless relationships"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class FieldType(Enum):
    """Field types for relationship filtering"""
    GEOGRAPHIC = "geographic"
    IDENTIFIER = "identifier"
    PERSONAL = "personal"
    TEMPORAL = "temporal"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    TEXT = "text"
    CONTACT = "contact"


@dataclass
class FieldClassification:
    """Classification of a field with its type and properties"""
    name: str
    field_type: FieldType
    confidence: float
    reasoning: str


class RelationshipFilter:
    """Class to filter out meaningless relationships"""

    def __init__(self):
        # Define which field type combinations should NOT have relationships
        self.excluded_combinations = {
            (FieldType.GEOGRAPHIC, FieldType.CONTACT),  # lat/lon with email/phone
            (FieldType.GEOGRAPHIC, FieldType.PERSONAL),  # lat/lon with names
            (FieldType.IDENTIFIER, FieldType.GEOGRAPHIC),  # IDs with coordinates
            (FieldType.IDENTIFIER, FieldType.CONTACT),  # IDs with email/phone (unless business logic)
            (FieldType.TEMPORAL, FieldType.GEOGRAPHIC),  # timestamps with coordinates
        }

        # Keywords for automatic field classification
        self.field_keywords = {
            FieldType.GEOGRAPHIC: {
                'latitude', 'lat', 'longitude', 'lon', 'lng', 'coord', 'x_coord',
                'y_coord', 'geo_lat', 'geo_lon', 'location_lat', 'location_lon'
            },
            FieldType.IDENTIFIER: {
                'id', 'uuid', 'key', 'identifier', 'code', 'ref', 'reference',
                'user_id', 'customer_id', 'order_id', 'transaction_id'
            },
            FieldType.PERSONAL: {
                'name', 'first_name', 'last_name', 'full_name', 'username',
                'display_name', 'nickname', 'fname', 'lname'
            },
            FieldType.TEMPORAL: {
                'date', 'time', 'timestamp', 'created_at', 'updated_at', 'modified',
                'birth_date', 'dob', 'start_date', 'end_date', 'datetime'
            },
            FieldType.CONTACT: {
                'email', 'phone', 'mobile', 'telephone', 'contact', 'mail',
                'phone_number', 'email_address', 'cell_phone'
            },
            FieldType.CATEGORICAL: {
                'category', 'type', 'status', 'state', 'country', 'region',
                'department', 'role', 'level', 'grade', 'class'
            }
        }

    def classify_fields(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, FieldClassification]:
        """Automatically classify fields based on name and content"""
        classifications = {}

        for col_name, col_info in analysis["columns"].items():
            classification = self._classify_single_field(col_name, col_info, df[col_name])
            classifications[col_name] = classification

        return classifications

    def _classify_single_field(self, col_name: str, col_info: Dict, series: pd.Series) -> FieldClassification:
        """Classify a single field"""
        col_name_lower = col_name.lower()
        col_type = col_info["type"]
        stats = col_info["stats"]

        # Check keywords first
        for field_type, keywords in self.field_keywords.items():
            for keyword in keywords:
                if keyword in col_name_lower:
                    confidence = 0.9 if keyword == col_name_lower else 0.7
                    return FieldClassification(
                        name=col_name,
                        field_type=field_type,
                        confidence=confidence,
                        reasoning=f"Contains keyword '{keyword}'"
                    )

        # Content-based classification
        if col_type in ["integer", "float"]:
            # Check for coordinates (typical ranges)
            if col_type == "float":
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)

                # Latitude range: -90 to 90
                if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                    if "lat" in col_name_lower or abs(max_val - min_val) < 180:
                        return FieldClassification(
                            name=col_name,
                            field_type=FieldType.GEOGRAPHIC,
                            confidence=0.8,
                            reasoning="Numeric range suggests latitude"
                        )

                # Longitude range: -180 to 180
                if -180 <= min_val <= 180 and -180 <= max_val <= 180:
                    if "lon" in col_name_lower or abs(max_val - min_val) > 180:
                        return FieldClassification(
                            name=col_name,
                            field_type=FieldType.GEOGRAPHIC,
                            confidence=0.8,
                            reasoning="Numeric range suggests longitude"
                        )

            # Check for ID-like behavior (high uniqueness)
            unique_pct = stats.get("unique_percent", 0)
            if unique_pct > 95:
                return FieldClassification(
                    name=col_name,
                    field_type=FieldType.IDENTIFIER,
                    confidence=0.6,
                    reasoning="High uniqueness suggests identifier"
                )

        elif col_type == "string":
            # Check string patterns
            unique_pct = stats.get("unique_percent", 0)

            # Email patterns
            if stats.get("email_domains"):
                return FieldClassification(
                    name=col_name,
                    field_type=FieldType.CONTACT,
                    confidence=0.95,
                    reasoning="Contains email patterns"
                )

            # Phone patterns
            if stats.get("phone_patterns"):
                return FieldClassification(
                    name=col_name,
                    field_type=FieldType.CONTACT,
                    confidence=0.95,
                    reasoning="Contains phone patterns"
                )

            # High uniqueness string could be identifier or personal
            if unique_pct > 90:
                if any(keyword in col_name_lower for keyword in ['name', 'user', 'person']):
                    return FieldClassification(
                        name=col_name,
                        field_type=FieldType.PERSONAL,
                        confidence=0.7,
                        reasoning="High uniqueness + name-related"
                    )
                else:
                    return FieldClassification(
                        name=col_name,
                        field_type=FieldType.IDENTIFIER,
                        confidence=0.6,
                        reasoning="High uniqueness suggests identifier"
                    )

            # Low uniqueness could be categorical
            elif unique_pct < 10:
                return FieldClassification(
                    name=col_name,
                    field_type=FieldType.CATEGORICAL,
                    confidence=0.7,
                    reasoning="Low uniqueness suggests categorical"
                )

        elif col_type == "datetime":
            return FieldClassification(
                name=col_name,
                field_type=FieldType.TEMPORAL,
                confidence=0.9,
                reasoning="Datetime type"
            )

        # Default classification
        if col_type in ["integer", "float"]:
            field_type = FieldType.NUMERIC
        else:
            field_type = FieldType.TEXT

        return FieldClassification(
            name=col_name,
            field_type=field_type,
            confidence=0.3,
            reasoning="Default classification"
        )

    def should_exclude_relationship(self, field1_type: FieldType, field2_type: FieldType) -> bool:
        """Check if relationship between two field types should be excluded"""
        combination = (field1_type, field2_type)
        reverse_combination = (field2_type, field1_type)

        return combination in self.excluded_combinations or reverse_combination in self.excluded_combinations

    def filter_relationships(self, analysis: Dict[str, Any],
                             classifications: Dict[str, FieldClassification],
                             user_exclusions: Set[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Filter relationships based on classifications and user exclusions"""
        if "relationships" not in analysis:
            return analysis

        filtered_analysis = analysis.copy()
        relationships = analysis["relationships"]

        # Filter one-to-many relationships
        if "one_to_many" in relationships:
            filtered_otm = []
            for rel in relationships["one_to_many"]:
                parent_col = rel["parent"]
                child_col = rel["child"]

                # Check automatic exclusions
                if self._should_exclude_auto(parent_col, child_col, classifications):
                    continue

                # Check user exclusions
                if user_exclusions and (parent_col, child_col) in user_exclusions:
                    continue
                if user_exclusions and (child_col, parent_col) in user_exclusions:
                    continue

                filtered_otm.append(rel)

            filtered_analysis["relationships"]["one_to_many"] = filtered_otm

        # Filter many-to-many relationships
        if "many_to_many" in relationships:
            filtered_mtm = []
            for rel in relationships["many_to_many"]:
                col1 = rel["col1"]
                col2 = rel["col2"]

                # Check automatic exclusions
                if self._should_exclude_auto(col1, col2, classifications):
                    continue

                # Check user exclusions
                if user_exclusions and (col1, col2) in user_exclusions:
                    continue
                if user_exclusions and (col2, col1) in user_exclusions:
                    continue

                filtered_mtm.append(rel)

            filtered_analysis["relationships"]["many_to_many"] = filtered_mtm

        return filtered_analysis

    def _should_exclude_auto(self, col1: str, col2: str,
                             classifications: Dict[str, FieldClassification]) -> bool:
        """Check if relationship should be automatically excluded"""
        if col1 not in classifications or col2 not in classifications:
            return False

        type1 = classifications[col1].field_type
        type2 = classifications[col2].field_type

        return self.should_exclude_relationship(type1, type2)


class RelationshipFilterUI:
    """UI components for relationship filtering"""

    def __init__(self, relationship_filter: RelationshipFilter):
        self.filter = relationship_filter

    def display_field_classification(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[
        str, FieldClassification]:
        """Display field classification interface"""
        st.markdown("### 🏷️ Field Classification")
        st.info(
            "Classify your fields to automatically filter out meaningless relationships (e.g., latitude/longitude with email addresses)")

        # Get automatic classifications
        auto_classifications = self.filter.classify_fields(df, analysis)

        # Allow user to modify classifications
        classifications = {}

        # Create columns for the interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Field Classifications:**")

            for field_name, auto_class in auto_classifications.items():
                with st.expander(
                        f"📋 {field_name} - {auto_class.field_type.value.title()} ({auto_class.confidence:.0%})"):
                    st.write(f"**Reasoning:** {auto_class.reasoning}")

                    # Allow user to override
                    field_types = [ft.value for ft in FieldType]
                    current_index = field_types.index(auto_class.field_type.value)

                    selected_type = st.selectbox(
                        "Override field type:",
                        field_types,
                        index=current_index,
                        key=f"field_type_{field_name}",
                        format_func=lambda x: x.replace('_', ' ').title()
                    )

                    classifications[field_name] = FieldClassification(
                        name=field_name,
                        field_type=FieldType(selected_type),
                        confidence=1.0 if selected_type != auto_class.field_type.value else auto_class.confidence,
                        reasoning="User override" if selected_type != auto_class.field_type.value else auto_class.reasoning
                    )

        with col2:
            st.markdown("**Auto-Exclusion Rules:**")
            st.markdown("""
            The following field type combinations will be automatically excluded from relationship analysis:

            - 📍 **Geographic** ↔ 📧 **Contact** (lat/lon with email/phone)
            - 📍 **Geographic** ↔ 👤 **Personal** (lat/lon with names)
            - 🆔 **Identifier** ↔ 📍 **Geographic** (IDs with coordinates)
            - 🆔 **Identifier** ↔ 📧 **Contact** (IDs with email/phone)
            - ⏰ **Temporal** ↔ 📍 **Geographic** (timestamps with coordinates)
            """)

        return classifications

    def display_relationship_exclusion_interface(self, analysis: Dict[str, Any],
                                                 classifications: Dict[str, FieldClassification]) -> Set[
        Tuple[str, str]]:
        """Display interface for manually excluding specific relationships"""
        st.markdown("### ⚙️ Manual Relationship Exclusions")

        if "relationships" not in analysis:
            st.info("No relationships detected to exclude.")
            return set()

        user_exclusions = set()
        relationships = analysis["relationships"]

        # Show one-to-many relationships
        if relationships.get("one_to_many"):
            st.markdown("**One-to-Many Relationships:**")

            for i, rel in enumerate(relationships["one_to_many"]):
                parent_col = rel["parent"]
                child_col = rel["child"]
                ratio = rel["avg_children"]

                # Check if this would be auto-excluded
                auto_excluded = self.filter._should_exclude_auto(parent_col, child_col, classifications)

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    relationship_desc = f"{parent_col} → {child_col} (avg: {ratio:.2f})"
                    if auto_excluded:
                        st.write(f"🚫 ~~{relationship_desc}~~ (auto-excluded)")
                    else:
                        st.write(f"✅ {relationship_desc}")

                with col2:
                    if not auto_excluded:
                        exclude = st.checkbox(
                            "Exclude",
                            key=f"exclude_otm_{i}",
                            help=f"Exclude {parent_col} → {child_col} relationship"
                        )
                        if exclude:
                            user_exclusions.add((parent_col, child_col))

                with col3:
                    # Show field types
                    parent_type = classifications.get(parent_col, "unknown")
                    child_type = classifications.get(child_col, "unknown")
                    if hasattr(parent_type, 'field_type') and hasattr(child_type, 'field_type'):
                        st.caption(f"{parent_type.field_type.value} → {child_type.field_type.value}")

        # Show many-to-many relationships
        if relationships.get("many_to_many"):
            st.markdown("**Many-to-Many Relationships:**")

            for i, rel in enumerate(relationships["many_to_many"]):
                col1_name = rel["col1"]
                col2_name = rel["col2"]
                ratio = rel["avg_ratio"]

                # Check if this would be auto-excluded
                auto_excluded = self.filter._should_exclude_auto(col1_name, col2_name, classifications)

                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    relationship_desc = f"{col1_name} ↔ {col2_name} (avg: {ratio:.2f})"
                    if auto_excluded:
                        st.write(f"🚫 ~~{relationship_desc}~~ (auto-excluded)")
                    else:
                        st.write(f"✅ {relationship_desc}")

                with col2:
                    if not auto_excluded:
                        exclude = st.checkbox(
                            "Exclude",
                            key=f"exclude_mtm_{i}",
                            help=f"Exclude {col1_name} ↔ {col2_name} relationship"
                        )
                        if exclude:
                            user_exclusions.add((col1_name, col2_name))

                with col3:
                    # Show field types
                    type1 = classifications.get(col1_name, "unknown")
                    type2 = classifications.get(col2_name, "unknown")
                    if hasattr(type1, 'field_type') and hasattr(type2, 'field_type'):
                        st.caption(f"{type1.field_type.value} ↔ {type2.field_type.value}")

        return user_exclusions

    def display_filtered_analysis_summary(self, original_analysis: Dict[str, Any],
                                          filtered_analysis: Dict[str, Any]):
        """Display summary of filtering results"""
        st.markdown("### 📊 Filtering Summary")

        orig_otm = len(original_analysis.get("relationships", {}).get("one_to_many", []))
        filtered_otm = len(filtered_analysis.get("relationships", {}).get("one_to_many", []))

        orig_mtm = len(original_analysis.get("relationships", {}).get("many_to_many", []))
        filtered_mtm = len(filtered_analysis.get("relationships", {}).get("many_to_many", []))

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("One-to-Many (Original)", orig_otm)
        with col2:
            st.metric("One-to-Many (Filtered)", filtered_otm, delta=filtered_otm - orig_otm)
        with col3:
            st.metric("Many-to-Many (Original)", orig_mtm)
        with col4:
            st.metric("Many-to-Many (Filtered)", filtered_mtm, delta=filtered_mtm - orig_mtm)

        excluded_otm = orig_otm - filtered_otm
        excluded_mtm = orig_mtm - filtered_mtm

        if excluded_otm > 0 or excluded_mtm > 0:
            st.success(f"✅ Filtered out {excluded_otm + excluded_mtm} meaningless relationships")
        else:
            st.info("ℹ️ No relationships were filtered out")


# Integration function for main app
def integrate_relationship_filtering(df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Main integration function to add to your existing app"""

    # Initialize the filtering system
    relationship_filter = RelationshipFilter()
    filter_ui = RelationshipFilterUI(relationship_filter)

    # Check if there are relationships to filter
    if not analysis.get("relationships") or not any([
        analysis["relationships"].get("one_to_many"),
        analysis["relationships"].get("many_to_many")
    ]):
        st.info("📝 No relationships detected in the data to filter.")
        return analysis

    st.markdown("---")
    st.header("🎯 Relationship Filtering")
    st.markdown("Filter out meaningless relationships before generating synthetic data.")

    # Step 1: Field Classification
    classifications = filter_ui.display_field_classification(df, analysis)

    # Step 2: Manual Exclusions
    user_exclusions = filter_ui.display_relationship_exclusion_interface(analysis, classifications)

    # Step 3: Apply Filtering
    if st.button("🔄 Apply Relationship Filtering", type="primary"):
        filtered_analysis = relationship_filter.filter_relationships(
            analysis, classifications, user_exclusions
        )

        # Display summary
        filter_ui.display_filtered_analysis_summary(analysis, filtered_analysis)

        # Store filtered analysis in session state
        st.session_state["filtered_analysis"] = filtered_analysis
        st.session_state["relationship_classifications"] = classifications

        st.success("✅ Relationship filtering applied successfully!")
        return filtered_analysis

    # Return filtered analysis if it exists in session state
    return st.session_state.get("filtered_analysis", analysis)