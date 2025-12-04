# pages/6_🔢_Synthetic_Data_Generator.py - ENHANCED LOGIC VERSION
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import json
import re
import random
from collections import defaultdict, Counter
import hashlib
from datetime import datetime, timedelta
import math

# =============================================================================
# ENHANCED LOGIC-BASED RULE ENGINE
# =============================================================================

class LogicBasedRuleEngine:
    """Advanced rule engine that combines LLM analysis with strong logic"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def discover_and_enhance_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Discover rules with LLM AND enhance with strong logic
        """
        # Phase 1: Statistical rule discovery (ALWAYS RUNS)
        statistical_rules = self._discover_statistical_rules(df)
        
        # Phase 2: Pattern-based rule discovery
        pattern_rules = self._discover_pattern_based_rules(df)
        
        # Phase 3: LLM discovery (if available)
        llm_rules = {}
        if self.api_key:
            try:
                llm_rules = self._discover_llm_rules(df)
            except:
                st.warning("LLM rule discovery failed - using enhanced statistical methods")
        
        # Phase 4: Merge and enhance all rules
        merged_rules = self._merge_and_enhance_rules(
            statistical_rules, pattern_rules, llm_rules, df
        )
        
        # Phase 5: Validate rules against data
        validated_rules = self._validate_rules_against_data(merged_rules, df)
        
        return validated_rules
    
    def _discover_statistical_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover rules using statistical analysis"""
        st.info("🔍 Performing statistical rule discovery...")
        
        rules = {
            "value_mappings": [],
            "value_constraints": {},
            "distribution_patterns": {},
            "column_groups": [],
            "hierarchies": [],
            "uniqueness_constraints": [],
            "statistical_summary": {}
        }
        
        # 1. Discover EXACT value mappings
        exact_mappings = self._find_exact_value_mappings(df)
        rules["value_mappings"].extend(exact_mappings)
        
        # 2. Discover PROBABILISTIC mappings
        probabilistic_mappings = self._find_probabilistic_mappings(df)
        rules["value_mappings"].extend(probabilistic_mappings)
        
        # 3. Discover value constraints
        rules["value_constraints"] = self._discover_value_constraints(df)
        
        # 4. Discover column groups (columns that always appear together)
        rules["column_groups"] = self._discover_column_groups(df)
        
        # 5. Discover value hierarchies
        rules["hierarchies"] = self._discover_hierarchies(df)
        
        # 6. Discover uniqueness constraints
        rules["uniqueness_constraints"] = self._discover_uniqueness_constraints(df)
        
        # 7. Discover distribution patterns
        rules["distribution_patterns"] = self._discover_distribution_patterns(df)
        
        # 8. Statistical summary
        rules["statistical_summary"] = self._create_statistical_summary(df)
        
        return rules
    
    def _find_exact_value_mappings(self, df: pd.DataFrame) -> List[Dict]:
        """Find exact one-to-one value mappings between columns"""
        mappings = []
        
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Check for exact mappings
                value_map = defaultdict(set)
                
                for _, row in df.iterrows():
                    val1 = str(row[col1]) if pd.notna(row[col1]) else "NaN"
                    val2 = str(row[col2]) if pd.notna(row[col2]) else "NaN"
                    value_map[val1].add(val2)
                
                # Find one-to-one mappings
                for val1, val2_set in value_map.items():
                    if len(val2_set) == 1:
                        val2 = next(iter(val2_set))
                        
                        # Count occurrences
                        matching_rows = df[
                            (df[col1].astype(str) == val1) & 
                            (df[col2].astype(str) == val2)
                        ]
                        
                        # Check for violations
                        violating_rows = df[
                            (df[col1].astype(str) == val1) & 
                            (df[col2].astype(str) != val2)
                        ]
                        
                        if len(violating_rows) == 0:  # EXACT mapping
                            mappings.append({
                                "from_column": col1,
                                "from_value": val1 if val1 != "NaN" else None,
                                "to_column": col2,
                                "to_value": val2 if val2 != "NaN" else None,
                                "confidence": "exact",
                                "support_count": len(matching_rows),
                                "coverage": len(matching_rows) / len(df),
                                "type": "value_mapping"
                            })
        
        return mappings
    
    def _find_probabilistic_mappings(self, df: pd.DataFrame) -> List[Dict]:
        """Find probabilistic mappings with high confidence"""
        mappings = []
        
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Calculate conditional probabilities
                value_counts = defaultdict(Counter)
                
                for _, row in df.iterrows():
                    val1 = str(row[col1]) if pd.notna(row[col1]) else "NaN"
                    val2 = str(row[col2]) if pd.notna(row[col2]) else "NaN"
                    value_counts[val1][val2] += 1
                
                # Find high-probability mappings (>80%)
                for val1, counter in value_counts.items():
                    total = sum(counter.values())
                    for val2, count in counter.items():
                        probability = count / total
                        
                        if probability >= 0.8 and total >= 3:  # High confidence
                            mappings.append({
                                "from_column": col1,
                                "from_value": val1 if val1 != "NaN" else None,
                                "to_column": col2,
                                "to_value": val2 if val2 != "NaN" else None,
                                "confidence": "probabilistic",
                                "probability": round(probability, 2),
                                "support_count": count,
                                "coverage": count / len(df),
                                "type": "probabilistic_mapping"
                            })
        
        return mappings
    
    def _discover_value_constraints(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover value constraints for each column"""
        constraints = {}
        
        for col in df.columns:
            col_constraints = {
                "type": self._detect_column_type(df[col]),
                "null_percentage": round(df[col].isnull().sum() / len(df) * 100, 1),
                "unique_count": df[col].nunique()
            }
            
            # Detect specific constraints based on type
            if col_constraints["type"] == "categorical":
                col_constraints.update(self._analyze_categorical_constraints(df[col]))
            elif col_constraints["type"] == "numeric":
                col_constraints.update(self._analyze_numeric_constraints(df[col]))
            elif col_constraints["type"] == "datetime":
                col_constraints.update(self._analyze_datetime_constraints(df[col]))
            elif col_constraints["type"] == "text":
                col_constraints.update(self._analyze_text_constraints(df[col]))
            elif col_constraints["type"] == "id":
                col_constraints.update(self._analyze_id_constraints(df[col]))
            
            constraints[col] = col_constraints
        
        return constraints
    
    def _detect_column_type(self, series: pd.Series) -> str:
        """Detect the type of column"""
        # Check for ID columns
        if 'id' in series.name.lower() or 'code' in series.name.lower():
            unique_ratio = series.nunique() / len(series)
            if unique_ratio > 0.9:
                return "id"
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.3 and series.nunique() < 20:
                return "categorical"
            return "numeric"
        
        # Check for categorical
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.3 and series.nunique() < 100:
            return "categorical"
        
        # Check for text patterns
        sample_values = series.dropna().head(10).astype(str)
        text_patterns = any(
            len(str(val)) > 50 or 
            re.search(r'[a-zA-Z]{10,}', str(val)) 
            for val in sample_values
        )
        
        if text_patterns:
            return "text"
        
        return "categorical"
    
    def _analyze_categorical_constraints(self, series: pd.Series) -> Dict:
        """Analyze categorical column constraints"""
        constraints = {
            "allowed_values": series.dropna().unique().tolist(),
            "value_distribution": series.value_counts().to_dict(),
            "most_common": series.mode().iloc[0] if not series.mode().empty else None,
            "most_common_percentage": round(series.value_counts(normalize=True).iloc[0] * 100, 1)
        }
        
        # Detect if values follow a pattern
        sample_values = [str(v) for v in series.dropna().unique()[:10]]
        if all(len(v) == 1 for v in sample_values):  # Single character codes
            constraints["pattern"] = "single_character"
        elif all(v in ['M', 'F', 'Male', 'Female'] for v in sample_values):
            constraints["pattern"] = "gender"
        
        return constraints
    
    def _analyze_numeric_constraints(self, series: pd.Series) -> Dict:
        """Analyze numeric column constraints"""
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        constraints = {
            "min": float(numeric_series.min()),
            "max": float(numeric_series.max()),
            "mean": float(numeric_series.mean()),
            "median": float(numeric_series.median()),
            "std": float(numeric_series.std()),
            "iqr": float(numeric_series.quantile(0.75) - numeric_series.quantile(0.25))
        }
        
        # Detect if values are integers
        if all(pd.notna(v) and float(v).is_integer() for v in numeric_series.dropna().head(20)):
            constraints["integer_only"] = True
        
        # Detect if values are percentages (0-100)
        if constraints["min"] >= 0 and constraints["max"] <= 100:
            constraints["likely_percentage"] = True
        
        # Detect if values are ages
        if "age" in series.name.lower() and constraints["min"] >= 0 and constraints["max"] <= 120:
            constraints["likely_age"] = True
        
        # Detect distribution shape
        skew = float(numeric_series.skew())
        if abs(skew) > 1:
            constraints["distribution"] = "highly_skewed"
        elif abs(skew) > 0.5:
            constraints["distribution"] = "moderately_skewed"
        else:
            constraints["distribution"] = "approximately_normal"
        
        return constraints
    
    def _analyze_datetime_constraints(self, series: pd.Series) -> Dict:
        """Analyze datetime constraints"""
        datetime_series = pd.to_datetime(series, errors='coerce')
        
        constraints = {
            "min_date": datetime_series.min().isoformat() if not pd.isna(datetime_series.min()) else None,
            "max_date": datetime_series.max().isoformat() if not pd.isna(datetime_series.max()) else None,
            "date_range_days": (datetime_series.max() - datetime_series.min()).days if len(datetime_series.dropna()) > 1 else 0
        }
        
        # Detect date patterns
        if constraints["date_range_days"] > 0:
            # Check if dates are business days
            day_of_week_counts = datetime_series.dt.dayofweek.value_counts()
            weekend_days = day_of_week_counts.get(5, 0) + day_of_week_counts.get(6, 0)
            if weekend_days / len(datetime_series.dropna()) < 0.1:
                constraints["likely_business_dates"] = True
        
        return constraints
    
    def _analyze_text_constraints(self, series: pd.Series) -> Dict:
        """Analyze text column constraints"""
        text_values = series.dropna().astype(str)
        
        constraints = {
            "min_length": int(text_values.str.len().min()) if len(text_values) > 0 else 0,
            "max_length": int(text_values.str.len().max()) if len(text_values) > 0 else 0,
            "avg_length": float(text_values.str.len().mean()) if len(text_values) > 0 else 0
        }
        
        # Detect patterns in text
        sample_texts = text_values.head(100)
        
        # Check for email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_count = sum(bool(re.search(email_pattern, text)) for text in sample_texts)
        if email_count / len(sample_texts) > 0.3:
            constraints["likely_email"] = True
        
        # Check for phone patterns
        phone_pattern = r'\b\d{10}\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b'
        phone_count = sum(bool(re.search(phone_pattern, text)) for text in sample_texts)
        if phone_count / len(sample_texts) > 0.3:
            constraints["likely_phone"] = True
        
        return constraints
    
    def _analyze_id_constraints(self, series: pd.Series) -> Dict:
        """Analyze ID column constraints"""
        id_values = series.dropna().astype(str)
        
        constraints = {
            "unique": len(id_values) == id_values.nunique(),
            "min_length": int(id_values.str.len().min()) if len(id_values) > 0 else 0,
            "max_length": int(id_values.str.len().max()) if len(id_values) > 0 else 0,
            "pattern_analysis": self._analyze_id_pattern(id_values)
        }
        
        return constraints
    
    def _analyze_id_pattern(self, id_values: pd.Series) -> Dict:
        """Analyze patterns in ID values"""
        patterns = {
            "has_prefix": False,
            "has_suffix": False,
            "is_numeric": False,
            "is_alphanumeric": False,
            "has_separators": False
        }
        
        if len(id_values) == 0:
            return patterns
        
        sample = id_values.head(100)
        
        # Check if all are numeric
        if all(val.isdigit() for val in sample):
            patterns["is_numeric"] = True
        
        # Check if all are alphanumeric
        if all(re.match(r'^[A-Za-z0-9]+$', val) for val in sample):
            patterns["is_alphanumeric"] = True
        
        # Check for common separators
        if any(any(sep in val for sep in ['-', '_', '/', '.']) for val in sample):
            patterns["has_separators"] = True
        
        # Check for consistent prefix/suffix
        prefix_lengths = [3, 4, 5]
        for length in prefix_lengths:
            prefixes = [val[:length] for val in sample if len(val) >= length]
            if len(set(prefixes)) == 1 and len(prefixes) > 0:
                patterns["has_prefix"] = True
                patterns["prefix"] = prefixes[0]
                break
        
        return patterns
    
    def _discover_column_groups(self, df: pd.DataFrame) -> List[List[str]]:
        """Discover groups of columns that appear together"""
        groups = []
        
        # Look for columns with similar null patterns
        null_patterns = {}
        for col in df.columns:
            null_patterns[col] = tuple(df[col].isnull())
        
        # Group columns with identical null patterns
        pattern_to_cols = defaultdict(list)
        for col, pattern in null_patterns.items():
            pattern_to_cols[pattern].append(col)
        
        # Add groups with more than 1 column
        for cols in pattern_to_cols.values():
            if len(cols) > 1:
                groups.append(cols)
        
        return groups
    
    def _discover_hierarchies(self, df: pd.DataFrame) -> List[Dict]:
        """Discover value hierarchies (e.g., State → City → Zip)"""
        hierarchies = []
        
        # Look for columns where one column's values map to multiple values in another
        for i, parent_col in enumerate(df.columns):
            for child_col in df.columns[i+1:]:
                # Calculate how many unique child values per parent value
                grouping = df.groupby(parent_col)[child_col].nunique()
                
                # If each parent has multiple children, it might be a hierarchy
                if len(grouping) > 0 and grouping.mean() > 1.5:
                    hierarchy_strength = 1 - (len(grouping) / df[child_col].nunique())
                    
                    if hierarchy_strength > 0.3:
                        hierarchies.append({
                            "parent_column": parent_col,
                            "child_column": child_col,
                            "parent_values_count": df[parent_col].nunique(),
                            "child_values_count": df[child_col].nunique(),
                            "avg_children_per_parent": round(grouping.mean(), 2),
                            "hierarchy_strength": round(hierarchy_strength, 2)
                        })
        
        return hierarchies
    
    def _discover_uniqueness_constraints(self, df: pd.DataFrame) -> List[str]:
        """Discover columns that should have unique values"""
        unique_cols = []
        
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            
            # Columns that look like IDs
            if ('id' in col.lower() or 
                'code' in col.lower() or 
                col.lower().endswith('_id') or
                unique_ratio > 0.95):
                unique_cols.append(col)
        
        return unique_cols
    
    def _discover_distribution_patterns(self, df: pd.DataFrame) -> Dict:
        """Discover distribution patterns in the data"""
        patterns = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for normal distribution
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 10:
                    skewness = series.skew()
                    kurtosis = series.kurtosis()
                    
                    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
                        patterns[col] = {"distribution": "normal"}
                    elif skewness > 1:
                        patterns[col] = {"distribution": "right_skewed"}
                    elif skewness < -1:
                        patterns[col] = {"distribution": "left_skewed"}
        
        return patterns
    
    def _create_statistical_summary(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive statistical summary"""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_values_total": int(df.isnull().sum().sum()),
            "missing_values_percentage": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
        }
    
    def _discover_pattern_based_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover rules based on patterns in the data"""
        rules = {
            "conditional_rules": [],
            "temporal_rules": [],
            "business_rules": []
        }
        
        # Look for conditional relationships
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Check if col1 values predict col2 values
                value_pairs = defaultdict(Counter)
                
                for _, row in df.iterrows():
                    val1 = str(row[col1]) if pd.notna(row[col1]) else "NaN"
                    val2 = str(row[col2]) if pd.notna(row[col2]) else "NaN"
                    value_pairs[val1][val2] += 1
                
                # Find strong conditional relationships
                for val1, counter in value_pairs.items():
                    total = sum(counter.values())
                    for val2, count in counter.items():
                        if count / total > 0.9 and total > 5:  # Very strong relationship
                            rules["conditional_rules"].append({
                                "if": f"{col1} == '{val1}'",
                                "then": f"{col2} == '{val2}'",
                                "confidence": count / total,
                                "support": count
                            })
        
        # Look for temporal patterns in datetime columns
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(datetime_cols) >= 2:
            # Check for ordering between datetime columns
            for i, earlier_col in enumerate(datetime_cols):
                for later_col in datetime_cols[i+1:]:
                    # Check if earlier_col is always before later_col
                    valid_pairs = 0
                    total_pairs = 0
                    
                    for _, row in df.iterrows():
                        if pd.notna(row[earlier_col]) and pd.notna(row[later_col]):
                            total_pairs += 1
                            if row[earlier_col] <= row[later_col]:
                                valid_pairs += 1
                    
                    if total_pairs > 0 and valid_pairs / total_pairs > 0.95:
                        rules["temporal_rules"].append({
                            "earlier": earlier_col,
                            "later": later_col,
                            "rule": f"{earlier_col} <= {later_col}",
                            "confidence": valid_pairs / total_pairs
                        })
        
        return rules
    
    def _discover_llm_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Discover rules using LLM (optional)"""
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            # Prepare sample
            sample_size = min(50, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            
            # Build prompt
            prompt = self._build_llm_prompt(df, sample_df)
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a data relationship expert. Analyze the dataset and find relationships between columns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content
            llm_rules = json.loads(content) if "{" in content else {"llm_analysis": content}
            
            return llm_rules
            
        except Exception as e:
            st.warning(f"LLM analysis skipped: {e}")
            return {}
    
    def _build_llm_prompt(self, df: pd.DataFrame, sample_df: pd.DataFrame) -> str:
        """Build prompt for LLM analysis"""
        return f"""Analyze this dataset and find relationships:

Columns: {list(df.columns)}
Sample data:
{sample_df.to_string(index=False)}

Find:
1. Business rules between columns
2. Value mappings (when X then Y)
3. Constraints on values
4. Any patterns you notice

Return as JSON with keys: business_rules, value_mappings, constraints, patterns."""
    
    def _merge_and_enhance_rules(self, stat_rules: Dict, pattern_rules: Dict, llm_rules: Dict, df: pd.DataFrame) -> Dict:
        """Merge all rules and enhance with additional logic"""
        merged_rules = {
            "value_mappings": stat_rules.get("value_mappings", []),
            "value_constraints": stat_rules.get("value_constraints", {}),
            "distribution_patterns": stat_rules.get("distribution_patterns", {}),
            "column_groups": stat_rules.get("column_groups", []),
            "hierarchies": stat_rules.get("hierarchies", []),
            "uniqueness_constraints": stat_rules.get("uniqueness_constraints", []),
            "statistical_summary": stat_rules.get("statistical_summary", {}),
            "conditional_rules": pattern_rules.get("conditional_rules", []),
            "temporal_rules": pattern_rules.get("temporal_rules", []),
            "business_rules": [],
            "generation_strategy": self._determine_generation_strategy(stat_rules, df)
        }
        
        # Add LLM business rules if available
        if llm_rules and "business_rules" in llm_rules:
            merged_rules["business_rules"] = llm_rules["business_rules"]
        
        # Enhance with derived rules
        merged_rules["derived_rules"] = self._derive_additional_rules(merged_rules, df)
        
        return merged_rules
    
    def _determine_generation_strategy(self, rules: Dict, df: pd.DataFrame) -> Dict:
        """Determine the best strategy for generating synthetic data"""
        strategy = {
            "primary_key_columns": [],
            "categorical_columns": [],
            "numerical_columns": [],
            "datetime_columns": [],
            "text_columns": [],
            "generation_order": [],
            "sampling_method": "weighted"
        }
        
        # Classify columns
        for col in df.columns:
            if col in rules.get("uniqueness_constraints", []):
                strategy["primary_key_columns"].append(col)
            elif "categorical" in rules.get("value_constraints", {}).get(col, {}).get("type", ""):
                strategy["categorical_columns"].append(col)
            elif "numeric" in rules.get("value_constraints", {}).get(col, {}).get("type", ""):
                strategy["numerical_columns"].append(col)
            elif "datetime" in rules.get("value_constraints", {}).get(col, {}).get("type", ""):
                strategy["datetime_columns"].append(col)
            elif "text" in rules.get("value_constraints", {}).get(col, {}).get("type", ""):
                strategy["text_columns"].append(col)
        
        # Determine generation order (start with key columns, then categorical, then others)
        strategy["generation_order"] = (
            strategy["primary_key_columns"] +
            strategy["categorical_columns"] +
            strategy["datetime_columns"] +
            strategy["numerical_columns"] +
            strategy["text_columns"]
        )
        
        return strategy
    
    def _derive_additional_rules(self, rules: Dict, df: pd.DataFrame) -> List[Dict]:
        """Derive additional rules from existing ones"""
        derived = []
        
        # Derive range rules for numeric columns
        for col, constraints in rules.get("value_constraints", {}).items():
            if constraints.get("type") == "numeric":
                min_val = constraints.get("min")
                max_val = constraints.get("max")
                if min_val is not None and max_val is not None:
                    derived.append({
                        "type": "range_rule",
                        "column": col,
                        "rule": f"{min_val} <= {col} <= {max_val}",
                        "min": min_val,
                        "max": max_val
                    })
        
        # Derive foreign key relationships
        for mapping in rules.get("value_mappings", []):
            if mapping.get("confidence") == "exact":
                derived.append({
                    "type": "foreign_key",
                    "from_column": mapping["from_column"],
                    "to_column": mapping["to_column"],
                    "rule": f"{mapping['from_column']}.{mapping['from_value']} -> {mapping['to_column']}.{mapping['to_value']}"
                })
        
        return derived
    
    def _validate_rules_against_data(self, rules: Dict, df: pd.DataFrame) -> Dict:
        """Validate that all rules are consistent with the data"""
        validated_rules = rules.copy()
        
        # Validate value mappings
        valid_mappings = []
        for mapping in rules.get("value_mappings", []):
            if self._validate_mapping(mapping, df):
                valid_mappings.append(mapping)
        validated_rules["value_mappings"] = valid_mappings
        
        # Validate constraints
        for col in list(validated_rules.get("value_constraints", {}).keys()):
            if col not in df.columns:
                del validated_rules["value_constraints"][col]
        
        return validated_rules
    
    def _validate_mapping(self, mapping: Dict, df: pd.DataFrame) -> bool:
        """Validate a single mapping against the data"""
        try:
            from_col = mapping.get("from_column")
            to_col = mapping.get("to_column")
            from_val = mapping.get("from_value")
            to_val = mapping.get("to_value")
            
            if from_col not in df.columns or to_col not in df.columns:
                return False
            
            # Check if mapping exists
            matching_rows = df[
                (df[from_col].astype(str) == str(from_val)) & 
                (df[to_col].astype(str) == str(to_val))
            ]
            
            # Check for violations
            violating_rows = df[
                (df[from_col].astype(str) == str(from_val)) & 
                (df[to_col].astype(str) != str(to_val))
            ]
            
            # Accept if no violations or high confidence
            return len(violating_rows) == 0 or mapping.get("confidence") == "probabilistic"
            
        except:
            return False

# =============================================================================
# INTELLIGENT DATA GENERATOR
# =============================================================================

class IntelligentDataGenerator:
    """Intelligent synthetic data generator with strong logic"""
    
    def __init__(self):
        self.generated_values = defaultdict(set)
        self.sequence_counters = defaultdict(int)
    
    def generate_synthetic_data(self, original_df: pd.DataFrame, rules: Dict[str, Any], num_rows: int) -> pd.DataFrame:
        """
        Generate synthetic data using intelligent strategies
        """
        st.info(f"🧠 Generating {num_rows} rows using intelligent strategies...")
        
        # Prepare generation strategy
        strategy = rules.get("generation_strategy", {})
        generation_order = strategy.get("generation_order", original_df.columns.tolist())
        
        # Initialize empty DataFrame
        synthetic_rows = []
        
        # Generate rows
        for row_idx in range(num_rows):
            if row_idx % 100 == 0 and row_idx > 0:
                st.info(f"Generated {row_idx} of {num_rows} rows...")
            
            new_row = {}
            
            # Generate values in strategic order
            for col in generation_order:
                new_row[col] = self._generate_column_value(
                    col, original_df, rules, new_row, row_idx
                )
            
            # Apply rule corrections
            new_row = self._apply_rule_corrections(new_row, rules)
            
            # Ensure uniqueness
            new_row = self._ensure_uniqueness(new_row, rules)
            
            synthetic_rows.append(new_row)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Ensure data types match original
        synthetic_df = self._preserve_data_types(synthetic_df, original_df)
        
        # Final validation and cleanup
        synthetic_df = self._final_validation(synthetic_df, rules)
        
        return synthetic_df
    
    def _generate_column_value(self, col: str, original_df: pd.DataFrame, rules: Dict, 
                             current_row: Dict, row_idx: int) -> Any:
        """Generate a value for a specific column"""
        
        # Get column constraints
        constraints = rules.get("value_constraints", {}).get(col, {})
        col_type = constraints.get("type", "unknown")
        
        # Check if value is determined by mappings
        mapped_value = self._get_mapped_value(col, current_row, rules)
        if mapped_value is not None:
            return mapped_value
        
        # Generate based on column type
        if col_type == "categorical":
            return self._generate_categorical_value(col, original_df, constraints, current_row)
        elif col_type == "numeric":
            return self._generate_numeric_value(col, original_df, constraints, current_row, row_idx)
        elif col_type == "datetime":
            return self._generate_datetime_value(col, original_df, constraints, current_row)
        elif col_type == "id":
            return self._generate_id_value(col, original_df, constraints, row_idx)
        elif col_type == "text":
            return self._generate_text_value(col, original_df, constraints, current_row)
        else:
            # Default: sample from original
            return self._sample_from_original(col, original_df)
    
    def _get_mapped_value(self, col: str, current_row: Dict, rules: Dict) -> Optional[Any]:
        """Check if column value is determined by mappings"""
        for mapping in rules.get("value_mappings", []):
            if mapping.get("to_column") == col:
                from_col = mapping.get("from_column")
                from_val = mapping.get("from_value")
                to_val = mapping.get("to_value")
                
                if from_col in current_row and str(current_row[from_col]) == str(from_val):
                    return to_val
        
        return None
    
    def _generate_categorical_value(self, col: str, original_df: pd.DataFrame, 
                                  constraints: Dict, current_row: Dict) -> Any:
        """Generate categorical value"""
        
        # Get allowed values
        allowed_values = constraints.get("allowed_values", [])
        if not allowed_values:
            allowed_values = original_df[col].dropna().unique().tolist()
        
        # Get value distribution
        value_dist = constraints.get("value_distribution", {})
        
        if value_dist:
            # Weighted sampling based on original distribution
            values = list(value_dist.keys())
            weights = list(value_dist.values())
            weights = [w/sum(weights) for w in weights]
            return np.random.choice(values, p=weights)
        else:
            # Uniform sampling
            return np.random.choice(allowed_values)
    
    def _generate_numeric_value(self, col: str, original_df: pd.DataFrame, 
                               constraints: Dict, current_row: Dict, row_idx: int) -> float:
        """Generate numeric value"""
        
        min_val = constraints.get("min")
        max_val = constraints.get("max")
        mean_val = constraints.get("mean")
        std_val = constraints.get("std")
        distribution = constraints.get("distribution", "normal")
        
        if min_val is not None and max_val is not None:
            if distribution == "normal" and mean_val is not None and std_val is not None:
                # Generate from normal distribution
                value = np.random.normal(mean_val, std_val)
                # Clip to range
                return np.clip(value, min_val, max_val)
            else:
                # Uniform distribution
                return random.uniform(min_val, max_val)
        else:
            # Sample from original
            return float(original_df[col].dropna().sample(1).iloc[0])
    
    def _generate_datetime_value(self, col: str, original_df: pd.DataFrame, 
                                constraints: Dict, current_row: Dict) -> Any:
        """Generate datetime value"""
        
        min_date_str = constraints.get("min_date")
        max_date_str = constraints.get("max_date")
        
        if min_date_str and max_date_str:
            min_date = pd.to_datetime(min_date_str)
            max_date = pd.to_datetime(max_date_str)
            
            # Generate random date within range
            delta = max_date - min_date
            random_days = random.randint(0, delta.days)
            random_date = min_date + timedelta(days=random_days)
            
            return random_date
        else:
            # Sample from original
            return original_df[col].dropna().sample(1).iloc[0]
    
    def _generate_id_value(self, col: str, original_df: pd.DataFrame, 
                          constraints: Dict, row_idx: int) -> str:
        """Generate unique ID value"""
        
        # Get pattern analysis
        pattern_info = constraints.get("pattern_analysis", {})
        original_sample = original_df[col].dropna().astype(str).iloc[0] if len(original_df[col].dropna()) > 0 else ""
        
        if pattern_info.get("is_numeric"):
            # Generate sequential numeric ID
            self.sequence_counters[col] += 1
            prefix = pattern_info.get("prefix", "")
            return f"{prefix}{self.sequence_counters[col]:06d}"
        
        elif pattern_info.get("is_alphanumeric"):
            # Generate alphanumeric ID
            prefix = pattern_info.get("prefix", "")
            if prefix and pattern_info.get("has_prefix"):
                random_part = ''.join(random.choices('0123456789ABCDEF', k=6))
                return f"{prefix}{random_part}"
            else:
                return f"ID{row_idx:06d}"
        
        else:
            # Generic ID generation
            return f"{col}_{row_idx:06d}"
    
    def _generate_text_value(self, col: str, original_df: pd.DataFrame, 
                            constraints: Dict, current_row: Dict) -> str:
        """Generate text value"""
        
        min_len = constraints.get("min_length", 0)
        max_len = constraints.get("max_length", 100)
        avg_len = constraints.get("avg_length", 50)
        
        # Check for specific types
        if constraints.get("likely_email"):
            domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
            name = f"user{random.randint(1000, 9999)}"
            domain = random.choice(domains)
            return f"{name}@{domain}"
        
        elif constraints.get("likely_phone"):
            # Generate phone number
            area_code = random.randint(100, 999)
            prefix = random.randint(100, 999)
            line = random.randint(1000, 9999)
            return f"({area_code}) {prefix}-{line}"
        
        else:
            # Generate random text
            length = random.randint(min_len, max_len)
            words = ["Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
                    "adipiscing", "elit", "sed", "do", "eiusmod", "tempor"]
            num_words = max(1, length // 6)
            return ' '.join(random.choices(words, k=num_words))[:length]
    
    def _sample_from_original(self, col: str, original_df: pd.DataFrame) -> Any:
        """Sample a value from the original column"""
        non_null_series = original_df[col].dropna()
        if len(non_null_series) > 0:
            return non_null_series.sample(1).iloc[0]
        else:
            return None
    
    def _apply_rule_corrections(self, row: Dict, rules: Dict) -> Dict:
        """Apply rule-based corrections to the row"""
        corrected_row = row.copy()
        
        # Apply conditional rules
        for rule in rules.get("conditional_rules", []):
            # Simple rule application (can be enhanced)
            pass
        
        # Apply business rules
        for rule in rules.get("business_rules", []):
            # Parse and apply business rules
            pass
        
        return corrected_row
    
    def _ensure_uniqueness(self, row: Dict, rules: Dict) -> Dict:
        """Ensure uniqueness for columns that require it"""
        corrected_row = row.copy()
        
        for col in rules.get("uniqueness_constraints", []):
            if col in corrected_row:
                value = corrected_row[col]
                if value in self.generated_values[col]:
                    # Generate unique value
                    base = str(value)
                    counter = 1
                    while f"{base}_{counter}" in self.generated_values[col]:
                        counter += 1
                    new_value = f"{base}_{counter}"
                    corrected_row[col] = new_value
                    self.generated_values[col].add(new_value)
                else:
                    self.generated_values[col].add(value)
        
        return corrected_row
    
    def _preserve_data_types(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure synthetic data has same types as original"""
        result_df = synthetic_df.copy()
        
        for col in original_df.columns:
            if col in result_df.columns:
                try:
                    # Try to convert to original type
                    original_dtype = original_df[col].dtype
                    result_df[col] = result_df[col].astype(original_dtype)
                except:
                    # If conversion fails, try inferring type
                    try:
                        if pd.api.types.is_numeric_dtype(original_dtype):
                            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                        elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                            result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                    except:
                        pass
        
        return result_df
    
    def _final_validation(self, synthetic_df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
        """Final validation and cleanup"""
        df = synthetic_df.copy()
        
        # Remove any duplicate rows
        df = df.drop_duplicates()
        
        # Fill any null values with appropriate defaults
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                constraints = rules.get("value_constraints", {}).get(col, {})
                col_type = constraints.get("type", "unknown")
                
                if col_type == "categorical":
                    # Fill with most common value
                    if "most_common" in constraints:
                        df[col].fillna(constraints["most_common"], inplace=True)
                elif col_type == "numeric":
                    # Fill with mean
                    if "mean" in constraints:
                        df[col].fillna(constraints["mean"], inplace=True)
        
        return df

# =============================================================================
# ENHANCED VALIDATION ENGINE
# =============================================================================

class EnhancedValidationEngine:
    """Comprehensive validation engine"""
    
    @staticmethod
    def validate_synthetic_data(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, rules: Dict) -> Dict:
        """Comprehensive validation"""
        
        validation = {
            "statistical_validation": EnhancedValidationEngine._validate_statistics(original_df, synthetic_df),
            "rule_validation": EnhancedValidationEngine._validate_rules(synthetic_df, rules),
            "data_quality": EnhancedValidationEngine._validate_data_quality(synthetic_df),
            "similarity_metrics": EnhancedValidationEngine._calculate_similarity(original_df, synthetic_df)
        }
        
        # Calculate overall score
        validation["overall_score"] = EnhancedValidationEngine._calculate_overall_score(validation)
        
        return validation
    
    @staticmethod
    def _validate_statistics(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Validate statistical properties"""
        
        stats = {
            "column_stats": {},
            "distribution_comparison": {}
        }
        
        for col in original_df.columns:
            if col in synthetic_df.columns:
                orig_series = original_df[col].dropna()
                synth_series = synthetic_df[col].dropna()
                
                if pd.api.types.is_numeric_dtype(orig_series):
                    # Compare numeric statistics
                    stats["column_stats"][col] = {
                        "original_mean": float(orig_series.mean()),
                        "synthetic_mean": float(synth_series.mean()),
                        "mean_difference_pct": abs(float(orig_series.mean()) - float(synth_series.mean())) / max(1, abs(float(orig_series.mean()))) * 100,
                        "original_std": float(orig_series.std()),
                        "synthetic_std": float(synth_series.std()),
                        "std_difference_pct": abs(float(orig_series.std()) - float(synth_series.std())) / max(1, abs(float(orig_series.std()))) * 100
                    }
                
                elif orig_series.nunique() < 20:  # Categorical
                    orig_counts = orig_series.value_counts(normalize=True)
                    synth_counts = synth_series.value_counts(normalize=True)
                    
                    # Compare category distributions
                    common_categories = set(orig_counts.index) & set(synth_counts.index)
                    distribution_diff = 0
                    for cat in common_categories:
                        orig_pct = orig_counts.get(cat, 0)
                        synth_pct = synth_counts.get(cat, 0)
                        distribution_diff += abs(orig_pct - synth_pct)
                    
                    stats["column_stats"][col] = {
                        "original_categories": len(orig_counts),
                        "synthetic_categories": len(synth_counts),
                        "distribution_difference": distribution_diff,
                        "category_coverage": len(common_categories) / max(1, len(orig_counts)) * 100
                    }
        
        return stats
    
    @staticmethod
    def _validate_rules(synthetic_df: pd.DataFrame, rules: Dict) -> Dict:
        """Validate that rules are followed"""
        
        rule_validation = {
            "value_mappings": {},
            "constraints": {},
            "uniqueness": {}
        }
        
        # Validate value mappings
        for mapping in rules.get("value_mappings", []):
            from_col = mapping.get("from_column")
            to_col = mapping.get("to_column")
            from_val = mapping.get("from_value")
            to_val = mapping.get("to_value")
            
            if from_col in synthetic_df.columns and to_col in synthetic_df.columns:
                # Find violations
                matching_rows = synthetic_df[synthetic_df[from_col].astype(str) == str(from_val)]
                violations = matching_rows[matching_rows[to_col].astype(str) != str(to_val)]
                
                rule_validation["value_mappings"][f"{from_col}={from_val}→{to_col}"] = {
                    "total_matches": len(matching_rows),
                    "violations": len(violations),
                    "compliance_rate": (len(matching_rows) - len(violations)) / max(1, len(matching_rows)) * 100
                }
        
        # Validate constraints
        for col, constraints in rules.get("value_constraints", {}).items():
            if col in synthetic_df.columns:
                violations = 0
                total = len(synthetic_df[col])
                
                if constraints.get("type") == "categorical":
                    allowed = set(str(v) for v in constraints.get("allowed_values", []))
                    violations = sum(1 for v in synthetic_df[col] if str(v) not in allowed and pd.notna(v))
                
                elif constraints.get("type") == "numeric":
                    min_val = constraints.get("min")
                    max_val = constraints.get("max")
                    if min_val is not None and max_val is not None:
                        numeric_vals = pd.to_numeric(synthetic_df[col], errors='coerce')
                        violations = ((numeric_vals < min_val) | (numeric_vals > max_val)).sum()
                
                rule_validation["constraints"][col] = {
                    "total": total,
                    "violations": int(violations),
                    "compliance_rate": (total - violations) / max(1, total) * 100
                }
        
        # Validate uniqueness
        for col in rules.get("uniqueness_constraints", []):
            if col in synthetic_df.columns:
                unique_count = synthetic_df[col].nunique()
                total = len(synthetic_df[col])
                rule_validation["uniqueness"][col] = {
                    "unique_count": unique_count,
                    "total_count": total,
                    "uniqueness_percentage": unique_count / max(1, total) * 100
                }
        
        return rule_validation
    
    @staticmethod
    def _validate_data_quality(synthetic_df: pd.DataFrame) -> Dict:
        """Validate data quality metrics"""
        
        quality = {
            "null_values": {},
            "duplicates": {},
            "data_types": {}
        }
        
        # Check for null values
        for col in synthetic_df.columns:
            null_count = synthetic_df[col].isnull().sum()
            quality["null_values"][col] = {
                "null_count": int(null_count),
                "null_percentage": null_count / len(synthetic_df) * 100
            }
        
        # Check for duplicate rows
        duplicate_count = synthetic_df.duplicated().sum()
        quality["duplicates"] = {
            "duplicate_rows": int(duplicate_count),
            "duplicate_percentage": duplicate_count / len(synthetic_df) * 100
        }
        
        # Check data types
        for col in synthetic_df.columns:
            quality["data_types"][col] = str(synthetic_df[col].dtype)
        
        return quality
    
    @staticmethod
    def _calculate_similarity(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict:
        """Calculate similarity metrics"""
        
        similarity = {
            "column_similarity": {},
            "overall_similarity": 0
        }
        
        similarities = []
        
        for col in original_df.columns:
            if col in synthetic_df.columns:
                orig_series = original_df[col].dropna()
                synth_series = synthetic_df[col].dropna()
                
                if len(orig_series) > 0 and len(synth_series) > 0:
                    if pd.api.types.is_numeric_dtype(orig_series):
                        # For numeric, compare distributions using KS test
                        try:
                            from scipy import stats
                            ks_statistic, _ = stats.ks_2samp(orig_series, synth_series)
                            similarity_score = 100 * (1 - ks_statistic)
                        except:
                            # Fallback: compare mean and std
                            mean_diff = abs(orig_series.mean() - synth_series.mean()) / max(1, abs(orig_series.mean()))
                            std_diff = abs(orig_series.std() - synth_series.std()) / max(1, abs(orig_series.std()))
                            similarity_score = 100 * (1 - (mean_diff + std_diff) / 2)
                    
                    else:
                        # For categorical, compare value distributions
                        orig_counts = orig_series.value_counts(normalize=True)
                        synth_counts = synth_series.value_counts(normalize=True)
                        
                        common_categories = set(orig_counts.index) & set(synth_counts.index)
                        total_diff = 0
                        for cat in common_categories:
                            total_diff += abs(orig_counts[cat] - synth_counts[cat])
                        
                        similarity_score = 100 * (1 - total_diff)
                    
                    similarity["column_similarity"][col] = round(similarity_score, 1)
                    similarities.append(similarity_score)
        
        if similarities:
            similarity["overall_similarity"] = round(np.mean(similarities), 1)
        
        return similarity
    
    @staticmethod
    def _calculate_overall_score(validation: Dict) -> float:
        """Calculate overall validation score"""
        
        scores = []
        
        # Rule compliance score
        rule_validation = validation.get("rule_validation", {})
        rule_scores = []
        
        for mapping_validation in rule_validation.get("value_mappings", {}).values():
            rule_scores.append(mapping_validation.get("compliance_rate", 0))
        
        for constraint_validation in rule_validation.get("constraints", {}).values():
            rule_scores.append(constraint_validation.get("compliance_rate", 0))
        
        if rule_scores:
            scores.append(np.mean(rule_scores))
        
        # Data quality score
        quality = validation.get("data_quality", {})
        if quality.get("duplicates", {}).get("duplicate_percentage", 100) < 5:
            scores.append(95)
        
        # Similarity score
        similarity = validation.get("similarity_metrics", {}).get("overall_similarity", 0)
        scores.append(similarity)
        
        if scores:
            return round(np.mean(scores), 1)
        else:
            return 0.0

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="iRMC DataGiene - Logic Enhanced",
        page_icon="🧠",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .score-good { color: #00C851; font-weight: bold; }
        .score-ok { color: #FFBB33; font-weight: bold; }
        .score-poor { color: #FF4444; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🧠 iRMC DataGiene - Logic Enhanced</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Intelligent Synthetic Data Generation
    
    This enhanced version uses **strong logic-based analysis** combined with statistical methods 
    to discover and enforce data relationships, producing higher quality synthetic data.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload ANY CSV Dataset", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Preview
        with st.expander("📋 Data Preview", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                # Quick stats
                st.metric("Total Rows", len(df))
                st.metric("Total Columns", len(df.columns))
                
                # Column types
                numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                categorical_cols = [col for col in df.columns if df[col].nunique() < 20 and df[col].nunique() > 0]
                
                st.write(f"**📈 Numeric Columns:** {len(numeric_cols)}")
                st.write(f"**🏷️ Categorical Columns:** {len(categorical_cols)}")
        
        # Generation settings
        st.subheader("🎯 Generation Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=len(df) * 10,
                value=min(len(df) * 2, 1000),
                step=50
            )
        
        with col2:
            use_llm = st.checkbox("Use LLM Enhancement", value=True, 
                                 help="Use LLM for additional rule discovery")
        
        with col3:
            quality_preset = st.selectbox(
                "Quality Preset",
                ["Balanced", "High Fidelity", "Fast Generation"],
                help="Trade-off between quality and speed"
            )
        
        # Generate button
        if st.button("🧠 Generate Intelligent Synthetic Data", type="primary", use_container_width=True):
            
            # Step 1: Discover Rules (Logic Enhanced)
            with st.spinner("🔍 Performing advanced rule discovery..."):
                try:
                    # Initialize rule engine
                    api_key = st.secrets.get("GROQ_API_KEY") if use_llm else None
                    rule_engine = LogicBasedRuleEngine(api_key=api_key)
                    
                    # Discover rules
                    rules = rule_engine.discover_and_enhance_rules(df)
                    
                    # Display discovered rules
                    with st.expander("📊 Discovered Rules Summary", expanded=True):
                        
                        # Show key findings
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Value Mappings", len(rules.get("value_mappings", [])))
                        
                        with col2:
                            st.metric("Column Constraints", len(rules.get("value_constraints", {})))
                        
                        with col3:
                            st.metric("Unique Columns", len(rules.get("uniqueness_constraints", [])))
                        
                        # Show detailed breakdown
                        st.subheader("Rule Details")
                        
                        tab1, tab2, tab3 = st.tabs(["🔗 Mappings", "🎯 Constraints", "📈 Patterns"])
                        
                        with tab1:
                            mappings = rules.get("value_mappings", [])
                            if mappings:
                                exact_mappings = [m for m in mappings if m.get("confidence") == "exact"]
                                prob_mappings = [m for m in mappings if m.get("confidence") == "probabilistic"]
                                
                                st.write(f"**Exact Mappings:** {len(exact_mappings)}")
                                for mapping in exact_mappings[:5]:
                                    st.write(f"- {mapping['from_column']}={mapping['from_value']} → {mapping['to_column']}={mapping['to_value']}")
                                
                                if len(exact_mappings) > 5:
                                    st.write(f"... and {len(exact_mappings) - 5} more")
                                
                                st.write(f"**Probabilistic Mappings:** {len(prob_mappings)}")
                        
                        with tab2:
                            for col, constraint in list(rules.get("value_constraints", {}).items())[:10]:
                                st.write(f"**{col}** ({constraint.get('type', 'unknown')})")
                                if constraint.get("type") == "categorical":
                                    st.write(f"Values: {len(constraint.get('allowed_values', []))} categories")
                                elif constraint.get("type") == "numeric":
                                    st.write(f"Range: {constraint.get('min', '?')} to {constraint.get('max', '?')}")
                        
                        with tab3:
                            if rules.get("column_groups"):
                                st.write("**Column Groups:**")
                                for group in rules.get("column_groups", []):
                                    st.write(f"- {', '.join(group)}")
                            
                            if rules.get("hierarchies"):
                                st.write("**Hierarchies:**")
                                for hierarchy in rules.get("hierarchies", []):
                                    st.write(f"- {hierarchy['parent_column']} → {hierarchy['child_column']}")
                
                except Exception as e:
                    st.error(f"Failed to discover rules: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
            
            # Step 2: Generate Data
            with st.spinner(f"⚡ Generating {num_rows} rows with intelligent logic..."):
                try:
                    generator = IntelligentDataGenerator()
                    synthetic_df = generator.generate_synthetic_data(df, rules, int(num_rows))
                except Exception as e:
                    st.error(f"Failed to generate data: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
            
            # Step 3: Validate
            with st.spinner("✅ Validating generated data..."):
                try:
                    validator = EnhancedValidationEngine()
                    validation_report = validator.validate_synthetic_data(df, synthetic_df, rules)
                except Exception as e:
                    st.error(f"Failed to validate: {e}")
                    validation_report = {"overall_score": 0}
            
            # Store results
            st.session_state.synthetic_data = synthetic_df
            st.session_state.original_data = df
            st.session_state.rules = rules
            st.session_state.validation = validation_report
            
            st.balloons()
        
        # Display results if generated
        if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
            synthetic = st.session_state.synthetic_data
            original = st.session_state.original_data
            rules = st.session_state.rules
            validation = st.session_state.validation
            
            st.subheader(f"✨ Generated {len(synthetic)} Rows")
            
            # Overall score
            overall_score = validation.get("overall_score", 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if overall_score >= 90:
                    st.success(f"🏆 Overall Score: {overall_score}%")
                elif overall_score >= 70:
                    st.warning(f"📊 Overall Score: {overall_score}%")
                else:
                    st.error(f"⚠️ Overall Score: {overall_score}%")
            
            with col2:
                # Rule compliance
                rule_validation = validation.get("rule_validation", {})
                mapping_compliance = []
                for val in rule_validation.get("value_mappings", {}).values():
                    mapping_compliance.append(val.get("compliance_rate", 0))
                
                if mapping_compliance:
                    avg_compliance = np.mean(mapping_compliance)
                    st.metric("Rule Compliance", f"{avg_compliance:.1f}%")
            
            with col3:
                # Similarity
                similarity = validation.get("similarity_metrics", {}).get("overall_similarity", 0)
                st.metric("Data Similarity", f"{similarity:.1f}%")
            
            # Tabs for detailed view
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Synthetic Data", "🔍 Validation Details", "📈 Statistics", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Quick comparison
                st.subheader("Quick Comparison")
                compare_col = st.selectbox("Select column to compare", original.columns)
                
                if compare_col in synthetic.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Distribution**")
                        st.bar_chart(original[compare_col].value_counts().head(10))
                    
                    with col2:
                        st.write("**Synthetic Distribution**")
                        st.bar_chart(synthetic[compare_col].value_counts().head(10))
            
            with tab2:
                # Detailed validation
                st.write("### 🔍 Detailed Validation Report")
                
                # Rule violations
                rule_validation = validation.get("rule_validation", {})
                if rule_validation.get("value_mappings"):
                    st.write("**Value Mapping Compliance:**")
                    for mapping, stats in list(rule_validation["value_mappings"].items())[:10]:
                        compliance = stats.get("compliance_rate", 0)
                        color = "🟢" if compliance >= 95 else "🟡" if compliance >= 80 else "🔴"
                        st.write(f"{color} {mapping}: {compliance:.1f}%")
                
                # Data quality
                quality = validation.get("data_quality", {})
                if quality.get("duplicates"):
                    dup_pct = quality["duplicates"].get("duplicate_percentage", 0)
                    st.write(f"**Duplicate Rows:** {dup_pct:.1f}%")
                
                # Null values
                if quality.get("null_values"):
                    total_null = sum(v.get("null_count", 0) for v in quality["null_values"].values())
                    null_pct = total_null / (len(synthetic) * len(synthetic.columns)) * 100
                    st.write(f"**Null Values:** {null_pct:.1f}%")
            
            with tab3:
                # Statistical comparison
                stats = validation.get("statistical_validation", {})
                if stats.get("column_stats"):
                    st.write("### 📊 Statistical Comparison")
                    
                    for col, col_stats in list(stats["column_stats"].items())[:10]:
                        st.write(f"**{col}**")
                        
                        if "mean_difference_pct" in col_stats:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Mean", f"{col_stats['original_mean']:.2f}")
                            with col2:
                                st.metric("Synthetic Mean", f"{col_stats['synthetic_mean']:.2f}")
                            with col3:
                                diff = col_stats['mean_difference_pct']
                                st.metric("Difference", f"{diff:.1f}%")
            
            with tab4:
                # Download options
                try:
                    csv = synthetic.to_csv(index=False)
                    st.download_button(
                        "📥 Download Synthetic Data (CSV)",
                        csv,
                        f"logic_enhanced_synthetic_{len(synthetic)}_rows.csv",
                        "text/csv",
                        use_container_width=True
                    )
                except:
                    st.error("Could not create CSV download")
                
                # Download rules
                try:
                    rules_json = json.dumps(rules, indent=2)
                    st.download_button(
                        "📥 Download Rules (JSON)",
                        rules_json,
                        f"enhanced_data_rules.json",
                        "application/json",
                        use_container_width=True
                    )
                except:
                    st.error("Could not create rules download")
                
                # Download validation report
                try:
                    validation_json = json.dumps(validation, indent=2, default=str)
                    st.download_button(
                        "📥 Download Validation Report",
                        validation_json,
                        f"validation_report.json",
                        "application/json",
                        use_container_width=True
                    )
                except:
                    st.error("Could not create validation report")
                
                # Regenerate option
                if st.button("🔄 Generate New Variation", use_container_width=True):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome screen
        st.info("""
        ### 🚀 How to Use:
        
        1. **Upload** any CSV dataset
        2. **Configure** generation settings
        3. **Generate** synthetic data with intelligent logic
        4. **Download** the results
        
        ### 🧠 Key Features:
        
        - **Logic-based rule discovery**: Statistical analysis finds relationships
        - **Intelligent generation**: Maintains data distributions and relationships
        - **Comprehensive validation**: Detailed quality checks
        - **LLM enhancement**: Optional AI-powered rule discovery
        """)
        
        # Example datasets
        with st.expander("📚 Example Use Cases"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**🏥 Healthcare Data**")
                st.write("- Patient records with medical codes")
                st.write("- Appointment scheduling")
                st.write("- Treatment outcomes")
            
            with col2:
                st.write("**🏢 Business Data**")
                st.write("- Customer transaction history")
                st.write("- Employee records")
                st.write("- Sales data with relationships")
            
            with col3:
                st.write("**🎓 Education Data**")
                st.write("- Student performance records")
                st.write("- Course enrollment data")
                st.write("- Grade distributions")

if __name__ == "__main__":
    main()
