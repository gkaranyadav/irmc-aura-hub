# pages/6_🔢_Synthetic_Data_Generator.py - GENERIC VERSION
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
# GENERIC PROMPT ENGINE - NO ASSUMPTIONS
# =============================================================================

class GenericPromptEngine:
    """Completely generic prompting - learns from ANY dataset"""
    
    @staticmethod
    def create_analysis_prompt(df: pd.DataFrame, sample_df: pd.DataFrame) -> str:
        """Create generic prompt that works for ANY dataset"""
        
        # Get column type analysis
        column_analysis = GenericPromptEngine._analyze_columns(df)
        
        prompt = f"""
# COMPREHENSIVE DATA PATTERN ANALYSIS

## DATASET INFORMATION
- Total Rows: {len(df)}
- Total Columns: {len(df.columns)}
- Column Names: {list(df.columns)}

## COLUMN TYPE ANALYSIS
{column_analysis}

## SAMPLE DATA FOR ANALYSIS
{sample_df.head(20).to_string(index=False)}

## YOUR TASK: DISCOVER DATA PATTERNS

Analyze this dataset COMPLETELY and find ALL relationships and patterns that exist in the data.

### 1. VALUE RELATIONSHIPS BETWEEN COLUMNS
For EVERY combination of columns, check if values in one column determine values in another:
- When value X appears in column A, what value appears in column B?
- Are these mappings consistent?
- Are they one-to-one or one-to-many?

**Search for:**
- Exact mappings (100% consistent)
- Strong correlations (>90% consistent)
- Common value combinations

### 2. COLUMN-SPECIFIC PATTERNS
For EACH column individually, analyze:
- What type of data does it contain?
- What values are allowed?
- Are there any patterns in the values?
- What are the statistical properties?

### 3. DATA INTEGRITY RULES
Look for rules that maintain data quality:
- Value ranges that must be respected
- Format patterns that must be followed
- Mandatory vs optional fields
- Uniqueness requirements

### 4. BUSINESS/DOMAIN LOGIC (IF ANY EXISTS IN DATA)
Only identify rules that are ACTUALLY PRESENT in the data:
- If the data shows certain value combinations always occur together
- If there are dependencies between columns
- If certain values imply others

## ANALYSIS METHODOLOGY

1. **Start column-by-column**: Understand each column individually
2. **Check column pairs**: Look for relationships between every column pair
3. **Validate across entire sample**: Ensure patterns hold consistently
4. **Look for edge cases**: Find exceptions to patterns
5. **Document everything**: Record ALL findings

## IMPORTANT RULES

1. **NO ASSUMPTIONS**: Only report what you actually see in the data
2. **VALIDATE**: Every claim must be backed by examples from the sample
3. **BE SPECIFIC**: Use exact column names and values from the data
4. **COMPREHENSIVE**: Check every column and relationship

## EXPECTED OUTPUT FORMAT

Return a JSON object with:

{{
  "dataset_description": "Brief description of what the dataset contains",
  "value_mappings": [
    {{
      "from_column": "exact_column_name_from_data",
      "from_value": "exact_value_from_data",
      "to_column": "exact_column_name_from_data", 
      "to_value": "exact_value_from_data",
      "confidence": "exact/strong/weak",
      "evidence_count": number_of_times_observed
    }}
  ],
  "column_constraints": {{
    "column_name": {{
      "data_type": "inferred_type",
      "observed_values": ["list", "of", "all", "observed", "values"],
      "value_range": {{"min": min_value, "max": max_value}},
      "patterns_detected": ["any_patterns_found"],
      "null_allowed": true/false
    }}
  }},
  "data_quality_rules": [
    "rules_like: 'phone_numbers_must_be_10_digits'",
    "rules_like: 'appointment_date_must_be_future'"
  ],
  "unique_constraints": ["columns_that_appear_unique"],
  "relationships_summary": "Overall description of column relationships"
}}

## CRITICAL: Your analysis will be used to generate NEW data that follows the SAME patterns.
## If you miss a relationship, the generated data will be WRONG.
## Be THOROUGH and ACCURATE.
"""
        return prompt
    
    @staticmethod
    def _analyze_columns(df: pd.DataFrame) -> str:
        """Analyze columns without assumptions"""
        analysis = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            analysis.append(f"**{col}**:")
            analysis.append(f"  - Data Type: {dtype}")
            analysis.append(f"  - Unique Values: {unique_count} ({unique_count/len(df)*100:.1f}% of rows)")
            analysis.append(f"  - Null Values: {null_count} ({null_percentage:.1f}%)")
            
            # Show sample values
            if unique_count < 10:
                analysis.append(f"  - All Values: {df[col].dropna().unique().tolist()}")
            else:
                sample_vals = df[col].dropna().sample(min(5, unique_count)).tolist()
                analysis.append(f"  - Sample Values: {sample_vals}")
            
            # Detect potential patterns
            if unique_count > 0:
                sample_val = str(df[col].dropna().iloc[0])
                patterns = GenericPromptEngine._detect_patterns(sample_val)
                if patterns:
                    analysis.append(f"  - Patterns Detected: {patterns}")
            
            analysis.append("")
        
        return "\n".join(analysis)
    
    @staticmethod
    def _detect_patterns(value: str) -> List[str]:
        """Detect common patterns in values"""
        patterns = []
        
        # Convert to string for pattern matching
        val_str = str(value)
        
        # ID patterns
        if re.match(r'^[A-Z0-9]{8,10}$', val_str):
            patterns.append("alphanumeric_id")
        elif re.match(r'^\d{8,10}$', val_str):
            patterns.append("numeric_id")
        
        # Phone patterns
        if re.match(r'^\d{10}$', val_str):
            patterns.append("10_digit_phone")
        elif re.match(r'^\d{3}[-.\s]?\d{3}[-.\s]?\d{4}$', val_str):
            patterns.append("formatted_phone")
        
        # Date patterns
        date_patterns = [
            r'\d{2}[-/]\d{2}[-/]\d{4}',
            r'\d{4}[-/]\d{2}[-/]\d{2}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'
        ]
        if any(re.match(p, val_str) for p in date_patterns):
            patterns.append("date_like")
        
        # Time patterns
        if re.match(r'\d{1,2}[:]\d{2}\s*(?:AM|PM|am|pm)?', val_str, re.IGNORECASE):
            patterns.append("time_like")
        
        # Name patterns
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', val_str):
            patterns.append("name_like")
        
        return patterns

# =============================================================================
# ENHANCED STATISTICAL RULE DISCOVERY
# =============================================================================

class EnhancedStatisticalDiscovery:
    """Discover rules using statistics only - no hardcoding"""
    
    @staticmethod
    def discover_rules(df: pd.DataFrame) -> Dict[str, Any]:
        """Discover all rules statistically"""
        
        rules = {
            "value_mappings": [],
            "column_constraints": {},
            "data_quality_rules": [],
            "unique_constraints": [],
            "statistical_patterns": {},
            "generation_strategy": {}
        }
        
        # 1. Discover exact value mappings
        rules["value_mappings"] = EnhancedStatisticalDiscovery._find_all_mappings(df)
        
        # 2. Analyze each column's constraints
        rules["column_constraints"] = EnhancedStatisticalDiscovery._analyze_column_constraints(df)
        
        # 3. Find unique columns
        rules["unique_constraints"] = EnhancedStatisticalDiscovery._find_unique_columns(df)
        
        # 4. Discover statistical patterns
        rules["statistical_patterns"] = EnhancedStatisticalDiscovery._find_statistical_patterns(df)
        
        # 5. Create generation strategy
        rules["generation_strategy"] = EnhancedStatisticalDiscovery._create_generation_strategy(df, rules)
        
        return rules
    
    @staticmethod
    def _find_all_mappings(df: pd.DataFrame) -> List[Dict]:
        """Find all value mappings between columns"""
        mappings = []
        
        # Check every pair of columns
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                col_mappings = EnhancedStatisticalDiscovery._analyze_column_pair(df, col1, col2)
                mappings.extend(col_mappings)
        
        return mappings
    
    @staticmethod
    def _analyze_column_pair(df: pd.DataFrame, col1: str, col2: str) -> List[Dict]:
        """Analyze relationship between two columns"""
        mappings = []
        
        # Group by col1 values and see what col2 values they map to
        grouped = df.groupby(col1)[col2].agg(['unique', 'count'])
        
        for val1, row in grouped.iterrows():
            unique_vals = row['unique']
            count = row['count']
            
            # If exactly one unique value, it's a mapping
            if len(unique_vals) == 1:
                val2 = unique_vals[0]
                
                # Verify this is consistent
                consistent = len(df[(df[col1] == val1) & (df[col2] == val2)]) == count
                
                if consistent:
                    mappings.append({
                        "from_column": col1,
                        "from_value": val1,
                        "to_column": col2,
                        "to_value": val2,
                        "confidence": "exact",
                        "support": int(count),
                        "coverage": count / len(df)
                    })
            
            # If mostly one value (strong correlation)
            elif len(unique_vals) > 1:
                # Find most common value
                value_counts = df[df[col1] == val1][col2].value_counts()
                most_common_val = value_counts.index[0]
                most_common_count = value_counts.iloc[0]
                
                # Check if strong correlation (>90%)
                if most_common_count / count > 0.9:
                    mappings.append({
                        "from_column": col1,
                        "from_value": val1,
                        "to_column": col2,
                        "to_value": most_common_val,
                        "confidence": "strong",
                        "support": int(most_common_count),
                        "coverage": most_common_count / len(df),
                        "probability": most_common_count / count
                    })
        
        # Also check reverse mapping
        grouped_reverse = df.groupby(col2)[col1].agg(['unique', 'count'])
        
        for val2, row in grouped_reverse.iterrows():
            unique_vals = row['unique']
            count = row['count']
            
            if len(unique_vals) == 1:
                val1 = unique_vals[0]
                
                # Check if we already have this mapping
                existing = any(
                    m['from_column'] == col2 and m['from_value'] == val2 and 
                    m['to_column'] == col1 and m['to_value'] == val1
                    for m in mappings
                )
                
                if not existing:
                    consistent = len(df[(df[col2] == val2) & (df[col1] == val1)]) == count
                    
                    if consistent:
                        mappings.append({
                            "from_column": col2,
                            "from_value": val2,
                            "to_column": col1,
                            "to_value": val1,
                            "confidence": "exact",
                            "support": int(count),
                            "coverage": count / len(df)
                        })
        
        return mappings
    
    @staticmethod
    def _analyze_column_constraints(df: pd.DataFrame) -> Dict:
        """Analyze constraints for each column"""
        constraints = {}
        
        for col in df.columns:
            col_constraints = {
                "data_type": str(df[col].dtype),
                "unique_count": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float(df[col].isnull().sum() / len(df)),
                "observed_values": EnhancedStatisticalDiscovery._get_observed_values(df[col])
            }
            
            # Add type-specific constraints
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                col_constraints.update({
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                    "mean": float(numeric_series.mean()),
                    "std": float(numeric_series.std()),
                    "type": "numeric"
                })
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_constraints.update({
                    "type": "datetime",
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max())
                })
            
            else:
                # Text or categorical
                sample_values = df[col].dropna().astype(str).tolist()
                patterns = EnhancedStatisticalDiscovery._analyze_text_patterns(sample_values)
                
                col_constraints.update({
                    "type": "text" if len(sample_values) > df[col].nunique() * 0.9 else "categorical",
                    "patterns": patterns,
                    "length_stats": {
                        "min": min(len(str(v)) for v in sample_values) if sample_values else 0,
                        "max": max(len(str(v)) for v in sample_values) if sample_values else 0,
                        "mean": np.mean([len(str(v)) for v in sample_values]) if sample_values else 0
                    }
                })
            
            constraints[col] = col_constraints
        
        return constraints
    
    @staticmethod
    def _get_observed_values(series: pd.Series) -> List:
        """Get observed values for a column"""
        if series.nunique() <= 20:
            return series.dropna().unique().tolist()
        else:
            return series.dropna().sample(min(20, len(series))).tolist()
    
    @staticmethod
    def _analyze_text_patterns(values: List[str]) -> List[str]:
        """Analyze patterns in text values"""
        if not values:
            return []
        
        patterns = []
        
        # Check for ID patterns
        sample_size = min(10, len(values))
        sample = values[:sample_size]
        
        # Check if all values match certain patterns
        all_match_id = all(re.match(r'^[A-Z0-9]{8,10}$', str(v)) for v in sample)
        if all_match_id:
            patterns.append("alphanumeric_id_8_10_chars")
        
        all_match_phone = all(re.match(r'^\d{10}$', str(v)) for v in sample)
        if all_match_phone:
            patterns.append("10_digit_phone")
        
        # Check for date patterns
        date_patterns = [
            r'\d{2}[-/]\d{2}[-/]\d{4}',
            r'\d{4}[-/]\d{2}[-/]\d{2}'
        ]
        for pattern in date_patterns:
            if all(re.search(pattern, str(v)) for v in sample if v):
                patterns.append(f"date_pattern_{pattern}")
        
        return patterns
    
    @staticmethod
    def _find_unique_columns(df: pd.DataFrame) -> List[str]:
        """Find columns that should be unique"""
        unique_cols = []
        
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            
            # High uniqueness ratio or ID-like column names
            if (unique_ratio > 0.95 or 
                'id' in col.lower() or 
                'code' in col.lower() or 
                col.lower().endswith('_id')):
                unique_cols.append(col)
        
        return unique_cols
    
    @staticmethod
    def _find_statistical_patterns(df: pd.DataFrame) -> Dict:
        """Find statistical patterns in the data"""
        patterns = {}
        
        # Distribution patterns for numeric columns
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            
            if len(series) > 10:
                skewness = series.skew()
                
                patterns[col] = {
                    "skewness": float(skewness),
                    "distribution": "normal" if abs(skewness) < 0.5 else 
                                   "right_skewed" if skewness > 0.5 else 
                                   "left_skewed",
                    "outlier_threshold": float(series.quantile(0.99))
                }
        
        # Frequency patterns for categorical columns
        categorical_cols = [col for col in df.columns if df[col].nunique() < 20]
        
        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True).to_dict()
            patterns[col] = {
                "value_distribution": value_counts,
                "most_common": list(value_counts.keys())[0] if value_counts else None,
                "most_common_percentage": list(value_counts.values())[0] if value_counts else 0
            }
        
        return patterns
    
    @staticmethod
    def _create_generation_strategy(df: pd.DataFrame, rules: Dict) -> Dict:
        """Create strategy for generating synthetic data"""
        
        # Classify columns by their role
        strategy = {
            "id_columns": [],
            "categorical_columns": [],
            "numeric_columns": [],
            "datetime_columns": [],
            "text_columns": [],
            "dependent_columns": [],  # Columns that depend on others
            "independent_columns": [],  # Columns that can be generated independently
            "generation_order": []
        }
        
        for col in df.columns:
            constraints = rules.get("column_constraints", {}).get(col, {})
            col_type = constraints.get("type", "unknown")
            
            if col_type == "numeric":
                strategy["numeric_columns"].append(col)
            elif col_type == "categorical":
                strategy["categorical_columns"].append(col)
            elif col_type == "datetime":
                strategy["datetime_columns"].append(col)
            elif "id" in str(constraints.get("patterns", [])):
                strategy["id_columns"].append(col)
            else:
                strategy["text_columns"].append(col)
        
        # Determine which columns are dependent on others
        for mapping in rules.get("value_mappings", []):
            to_col = mapping.get("to_column")
            if to_col not in strategy["dependent_columns"]:
                strategy["dependent_columns"].append(to_col)
        
        # Independent columns are those not dependent
        strategy["independent_columns"] = [
            col for col in df.columns 
            if col not in strategy["dependent_columns"]
        ]
        
        # Create generation order: independent first, then dependent
        strategy["generation_order"] = (
            strategy["independent_columns"] + 
            strategy["dependent_columns"]
        )
        
        return strategy

# =============================================================================
# SMART DATA GENERATOR
# =============================================================================

class SmartDataGenerator:
    """Generate synthetic data while preserving patterns"""
    
    def __init__(self):
        self.generated_values = defaultdict(set)
        self.sequence_counters = defaultdict(int)
        self.value_pools = {}
    
    def generate_data(self, original_df: pd.DataFrame, rules: Dict, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data"""
        
        synthetic_rows = []
        strategy = rules.get("generation_strategy", {})
        generation_order = strategy.get("generation_order", original_df.columns.tolist())
        
        # Prepare value pools for categorical columns
        self._prepare_value_pools(original_df, rules)
        
        for row_idx in range(num_rows):
            new_row = {}
            
            # Generate values in order
            for col in generation_order:
                new_row[col] = self._generate_smart_value(
                    col, original_df, rules, new_row, row_idx
                )
            
            # Apply cross-column consistency
            new_row = self._apply_cross_column_rules(new_row, rules)
            
            # Ensure uniqueness where required
            new_row = self._ensure_uniqueness(new_row, rules)
            
            synthetic_rows.append(new_row)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Preserve data types
        synthetic_df = self._preserve_types(synthetic_df, original_df)
        
        return synthetic_df
    
    def _prepare_value_pools(self, original_df: pd.DataFrame, rules: Dict):
        """Prepare pools of values for each column"""
        for col in original_df.columns:
            constraints = rules.get("column_constraints", {}).get(col, {})
            col_type = constraints.get("type", "unknown")
            
            if col_type in ["categorical", "text"]:
                # Use observed values as pool
                observed = constraints.get("observed_values", [])
                if observed:
                    self.value_pools[col] = observed
                else:
                    self.value_pools[col] = original_df[col].dropna().unique().tolist()
    
    def _generate_smart_value(self, col: str, original_df: pd.DataFrame, 
                            rules: Dict, current_row: Dict, row_idx: int) -> Any:
        """Generate a smart value for a column"""
        
        # Check if value is determined by mappings
        mapped_value = self._get_mapped_value(col, current_row, rules)
        if mapped_value is not None:
            return mapped_value
        
        constraints = rules.get("column_constraints", {}).get(col, {})
        col_type = constraints.get("type", "unknown")
        
        if col_type == "categorical":
            return self._generate_categorical(col, constraints)
        elif col_type == "numeric":
            return self._generate_numeric(col, constraints, row_idx)
        elif col_type == "datetime":
            return self._generate_datetime(col, constraints)
        elif "id" in str(constraints.get("patterns", [])):
            return self._generate_id(col, constraints, row_idx)
        else:
            # Default: sample from pool or original
            if col in self.value_pools:
                return random.choice(self.value_pools[col])
            else:
                return self._sample_from_original(original_df[col])
    
    def _get_mapped_value(self, col: str, current_row: Dict, rules: Dict) -> Optional[Any]:
        """Get value from mappings"""
        for mapping in rules.get("value_mappings", []):
            if mapping.get("to_column") == col:
                from_col = mapping.get("from_column")
                from_val = mapping.get("from_value")
                
                if from_col in current_row and str(current_row[from_col]) == str(from_val):
                    return mapping.get("to_value")
        
        return None
    
    def _generate_categorical(self, col: str, constraints: Dict) -> Any:
        """Generate categorical value"""
        if col in self.value_pools:
            return random.choice(self.value_pools[col])
        else:
            observed = constraints.get("observed_values", [])
            if observed:
                return random.choice(observed)
            return "Unknown"
    
    def _generate_numeric(self, col: str, constraints: Dict, row_idx: int) -> float:
        """Generate numeric value"""
        if "min" in constraints and "max" in constraints:
            min_val = constraints["min"]
            max_val = constraints["max"]
            
            # Add some variation
            if row_idx % 3 == 0:
                # Generate near mean
                if "mean" in constraints:
                    mean_val = constraints["mean"]
                    std_val = constraints.get("std", (max_val - min_val) / 6)
                    value = np.random.normal(mean_val, std_val)
                    return np.clip(value, min_val, max_val)
            
            # Random uniform
            return random.uniform(min_val, max_val)
        
        return random.randint(1, 100)  # Fallback
    
    def _generate_datetime(self, col: str, constraints: Dict) -> Any:
        """Generate datetime value"""
        if "min_date" in constraints and "max_date" in constraints:
            try:
                min_date = pd.to_datetime(constraints["min_date"])
                max_date = pd.to_datetime(constraints["max_date"])
                
                delta = max_date - min_date
                random_days = random.randint(0, delta.days)
                return min_date + timedelta(days=random_days)
            except:
                pass
        
        # Return current date as fallback
        return datetime.now()
    
    def _generate_id(self, col: str, constraints: Dict, row_idx: int) -> str:
        """Generate ID value"""
        patterns = constraints.get("patterns", [])
        
        if "alphanumeric_id_8_10_chars" in patterns:
            # Generate alphanumeric ID
            chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            length = random.randint(8, 10)
            return ''.join(random.choices(chars, k=length))
        
        elif "numeric_id" in str(patterns):
            # Generate numeric ID
            self.sequence_counters[col] += 1
            return f"{self.sequence_counters[col]:08d}"
        
        else:
            # Generic ID
            return f"{col}_{row_idx:06d}"
    
    def _sample_from_original(self, series: pd.Series) -> Any:
        """Sample from original series"""
        non_null = series.dropna()
        if len(non_null) > 0:
            return non_null.sample(1).iloc[0]
        return None
    
    def _apply_cross_column_rules(self, row: Dict, rules: Dict) -> Dict:
        """Apply cross-column consistency rules"""
        corrected_row = row.copy()
        
        # Apply exact mappings
        for mapping in rules.get("value_mappings", []):
            if mapping.get("confidence") == "exact":
                from_col = mapping["from_column"]
                from_val = mapping["from_value"]
                to_col = mapping["to_column"]
                to_val = mapping["to_value"]
                
                if from_col in corrected_row and str(corrected_row[from_col]) == str(from_val):
                    corrected_row[to_col] = to_val
        
        return corrected_row
    
    def _ensure_uniqueness(self, row: Dict, rules: Dict) -> Dict:
        """Ensure uniqueness for required columns"""
        corrected_row = row.copy()
        
        for col in rules.get("unique_constraints", []):
            if col in corrected_row:
                value = str(corrected_row[col])
                
                if value in self.generated_values[col]:
                    # Make unique
                    counter = 1
                    while f"{value}_{counter}" in self.generated_values[col]:
                        counter += 1
                    new_value = f"{value}_{counter}"
                    corrected_row[col] = new_value
                    self.generated_values[col].add(new_value)
                else:
                    self.generated_values[col].add(value)
        
        return corrected_row
    
    def _preserve_types(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Preserve original data types"""
        result_df = synthetic_df.copy()
        
        for col in original_df.columns:
            if col in result_df.columns:
                try:
                    result_df[col] = result_df[col].astype(original_df[col].dtype)
                except:
                    pass
        
        return result_df

# =============================================================================
# QUALITY VALIDATOR
# =============================================================================

class QualityValidator:
    """Validate quality of generated data"""
    
    @staticmethod
    def validate(original_df: pd.DataFrame, synthetic_df: pd.DataFrame, rules: Dict) -> Dict:
        """Comprehensive validation"""
        
        validation = {
            "basic_stats": QualityValidator._compare_basic_stats(original_df, synthetic_df),
            "rule_compliance": QualityValidator._check_rule_compliance(synthetic_df, rules),
            "data_quality": QualityValidator._check_data_quality(synthetic_df),
            "distribution_similarity": QualityValidator._compare_distributions(original_df, synthetic_df),
            "pattern_preservation": QualityValidator._check_patterns(synthetic_df, rules)
        }
        
        # Calculate overall score
        validation["overall_score"] = QualityValidator._calculate_overall_score(validation)
        
        return validation
    
    @staticmethod
    def _compare_basic_stats(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Compare basic statistics"""
        stats = {}
        
        for col in original.columns:
            if col in synthetic.columns:
                stats[col] = {
                    "original_rows": len(original),
                    "synthetic_rows": len(synthetic),
                    "original_nunique": original[col].nunique(),
                    "synthetic_nunique": synthetic[col].nunique(),
                    "original_null_pct": (original[col].isnull().sum() / len(original)) * 100,
                    "synthetic_null_pct": (synthetic[col].isnull().sum() / len(synthetic)) * 100
                }
        
        return stats
    
    @staticmethod
    def _check_rule_compliance(synthetic_df: pd.DataFrame, rules: Dict) -> Dict:
        """Check compliance with discovered rules"""
        compliance = {
            "value_mappings": {},
            "unique_constraints": {}
        }
        
        # Check value mappings
        for mapping in rules.get("value_mappings", []):
            if mapping.get("confidence") == "exact":
                from_col = mapping["from_column"]
                to_col = mapping["to_column"]
                from_val = mapping["from_value"]
                to_val = mapping["to_value"]
                
                if from_col in synthetic_df.columns and to_col in synthetic_df.columns:
                    matches = synthetic_df[synthetic_df[from_col] == from_val]
                    violations = matches[matches[to_col] != to_val]
                    
                    compliance["value_mappings"][f"{from_col}={from_val}"] = {
                        "expected": to_val,
                        "matches": len(matches),
                        "violations": len(violations),
                        "compliance_rate": (len(matches) - len(violations)) / max(1, len(matches)) * 100
                    }
        
        # Check uniqueness
        for col in rules.get("unique_constraints", []):
            if col in synthetic_df.columns:
                unique_count = synthetic_df[col].nunique()
                total = len(synthetic_df[col])
                
                compliance["unique_constraints"][col] = {
                    "unique_count": unique_count,
                    "total_count": total,
                    "uniqueness_percentage": (unique_count / total) * 100
                }
        
        return compliance
    
    @staticmethod
    def _check_data_quality(synthetic_df: pd.DataFrame) -> Dict:
        """Check data quality metrics"""
        quality = {
            "null_values": {},
            "duplicates": len(synthetic_df[synthetic_df.duplicated()]),
            "data_types": {}
        }
        
        for col in synthetic_df.columns:
            null_count = synthetic_df[col].isnull().sum()
            quality["null_values"][col] = {
                "count": int(null_count),
                "percentage": (null_count / len(synthetic_df)) * 100
            }
            quality["data_types"][col] = str(synthetic_df[col].dtype)
        
        return quality
    
    @staticmethod
    def _compare_distributions(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Compare value distributions"""
        comparisons = {}
        
        for col in original.columns:
            if col in synthetic.columns:
                if pd.api.types.is_numeric_dtype(original[col]):
                    # Compare numeric distributions
                    orig_mean = original[col].mean()
                    synth_mean = synthetic[col].mean()
                    mean_diff = abs(orig_mean - synth_mean) / max(1, abs(orig_mean)) * 100
                    
                    comparisons[col] = {
                        "type": "numeric",
                        "mean_difference_pct": mean_diff,
                        "original_mean": orig_mean,
                        "synthetic_mean": synth_mean
                    }
                
                elif original[col].nunique() < 20:
                    # Compare categorical distributions
                    orig_counts = original[col].value_counts(normalize=True)
                    synth_counts = synthetic[col].value_counts(normalize=True)
                    
                    # Calculate overlap
                    common_cats = set(orig_counts.index) & set(synth_counts.index)
                    total_diff = 0
                    for cat in common_cats:
                        total_diff += abs(orig_counts[cat] - synth_counts[cat])
                    
                    comparisons[col] = {
                        "type": "categorical",
                        "distribution_difference": total_diff,
                        "common_categories": len(common_cats)
                    }
        
        return comparisons
    
    @staticmethod
    def _check_patterns(synthetic_df: pd.DataFrame, rules: Dict) -> Dict:
        """Check if patterns are preserved"""
        patterns = {}
        
        for col, constraints in rules.get("column_constraints", {}).items():
            if col in synthetic_df.columns:
                col_patterns = constraints.get("patterns", [])
                if col_patterns:
                    patterns[col] = {
                        "expected_patterns": col_patterns,
                        "pattern_compliance": QualityValidator._check_pattern_compliance(
                            synthetic_df[col], col_patterns
                        )
                    }
        
        return patterns
    
    @staticmethod
    def _check_pattern_compliance(series: pd.Series, patterns: List[str]) -> Dict:
        """Check compliance with patterns"""
        compliance = {}
        
        for pattern in patterns:
            if "phone" in pattern:
                # Check phone pattern
                phone_count = series.astype(str).str.match(r'^\d{10}$').sum()
                compliance["10_digit_phone"] = {
                    "count": int(phone_count),
                    "percentage": (phone_count / len(series)) * 100
                }
            
            elif "id" in pattern:
                # Check ID pattern
                id_count = series.astype(str).str.match(r'^[A-Z0-9]{8,10}$').sum()
                compliance["alphanumeric_id"] = {
                    "count": int(id_count),
                    "percentage": (id_count / len(series)) * 100
                }
        
        return compliance
    
    @staticmethod
    def _calculate_overall_score(validation: Dict) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Rule compliance score
        rule_compliance = validation.get("rule_compliance", {})
        for mapping in rule_compliance.get("value_mappings", {}).values():
            scores.append(mapping.get("compliance_rate", 0))
        
        # Data quality score
        quality = validation.get("data_quality", {})
        duplicate_pct = quality.get("duplicates", 0) / max(1, len(quality.get("null_values", {}))) * 100
        if duplicate_pct < 5:
            scores.append(95)
        
        # Distribution similarity score
        distribution = validation.get("distribution_similarity", {})
        for col_stats in distribution.values():
            if col_stats.get("type") == "numeric":
                mean_diff = col_stats.get("mean_difference_pct", 100)
                scores.append(100 - min(mean_diff, 100))
        
        if scores:
            return round(np.mean(scores), 1)
        return 0.0

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Smart Data Generator",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Smart Synthetic Data Generator")
    st.markdown("Generate high-quality synthetic data while preserving ALL patterns from your original data.")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload your CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error: {e}")
            return
        
        # Preview
        with st.expander("📋 Preview Data"):
            st.dataframe(df.head(), use_container_width=True)
            
            # Show quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                numeric_cols = sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))
                st.metric("Numeric Columns", numeric_cols)
        
        # Generation settings
        st.subheader("⚙️ Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Number of rows to generate",
                min_value=len(df),
                max_value=len(df) * 10,
                value=len(df) * 2,
                step=50
            )
        
        with col2:
            use_llm = st.checkbox(
                "Use LLM for enhanced analysis",
                value=True,
                help="Uses LLM to discover additional patterns (requires API key)"
            )
        
        if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
            
            # Step 1: Discover Rules
            with st.spinner("🔍 Analyzing data patterns..."):
                try:
                    # Always use statistical discovery
                    stats_engine = EnhancedStatisticalDiscovery()
                    rules = stats_engine.discover_rules(df)
                    
                    # Optional LLM enhancement
                    if use_llm and "GROQ_API_KEY" in st.secrets:
                        st.info("Using LLM for enhanced pattern discovery...")
                        try:
                            from groq import Groq
                            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                            
                            prompt_engine = GenericPromptEngine()
                            prompt = prompt_engine.create_analysis_prompt(df, df.sample(min(30, len(df))))
                            
                            response = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=[
                                    {"role": "system", "content": "You are a data pattern expert. Analyze the dataset thoroughly."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.1,
                                max_tokens=2000
                            )
                            
                            llm_analysis = response.choices[0].message.content
                            
                            # Try to parse JSON if returned
                            if "{" in llm_analysis:
                                try:
                                    llm_rules = json.loads(llm_analysis)
                                    # Merge with statistical rules
                                    rules["llm_insights"] = llm_rules
                                except:
                                    rules["llm_analysis"] = llm_analysis
                            
                        except Exception as e:
                            st.warning(f"LLM analysis skipped: {e}")
                    
                except Exception as e:
                    st.error(f"Rule discovery failed: {e}")
                    return
            
            # Display discovered rules
            with st.expander("📊 Discovered Patterns", expanded=True):
                st.json(rules, expanded=False)
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Value Mappings", len(rules.get("value_mappings", [])))
                with col2:
                    st.metric("Unique Columns", len(rules.get("unique_constraints", [])))
                with col3:
                    exact_mappings = sum(1 for m in rules.get("value_mappings", []) if m.get("confidence") == "exact")
                    st.metric("Exact Mappings", exact_mappings)
            
            # Step 2: Generate Data
            with st.spinner(f"⚡ Generating {num_rows} rows..."):
                try:
                    generator = SmartDataGenerator()
                    synthetic_df = generator.generate_data(df, rules, int(num_rows))
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    return
            
            # Step 3: Validate
            with st.spinner("✅ Validating quality..."):
                try:
                    validator = QualityValidator()
                    validation = validator.validate(df, synthetic_df, rules)
                except Exception as e:
                    st.error(f"Validation failed: {e}")
                    validation = {"overall_score": 0}
            
            # Store results
            st.session_state.synthetic_data = synthetic_df
            st.session_state.rules = rules
            st.session_state.validation = validation
            
            st.balloons()
        
        # Display results if available
        if 'synthetic_data' in st.session_state:
            synthetic = st.session_state.synthetic_data
            validation = st.session_state.validation
            
            st.subheader(f"✨ Generated {len(synthetic)} Rows")
            
            # Quality score
            score = validation.get("overall_score", 0)
            if score >= 90:
                st.success(f"🏆 Overall Quality Score: {score}%")
            elif score >= 70:
                st.warning(f"📊 Overall Quality Score: {score}%")
            else:
                st.error(f"⚠️ Overall Quality Score: {score}%")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Validation", "🔍 Patterns", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Quick comparison
                st.subheader("Quick Comparison")
                compare_col = st.selectbox("Select column to compare", df.columns)
                
                if compare_col in synthetic.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Distribution**")
                        st.bar_chart(df[compare_col].value_counts().head(10))
                    with col2:
                        st.write("**Synthetic Distribution**")
                        st.bar_chart(synthetic[compare_col].value_counts().head(10))
            
            with tab2:
                # Validation details
                st.write("### Validation Details")
                
                # Rule compliance
                rule_compliance = validation.get("rule_compliance", {})
                if rule_compliance.get("value_mappings"):
                    st.write("**Rule Compliance:**")
                    for rule, stats in list(rule_compliance["value_mappings"].items())[:5]:
                        compliance = stats.get("compliance_rate", 0)
                        st.write(f"- {rule}: {compliance:.1f}% compliant")
                
                # Data quality
                quality = validation.get("data_quality", {})
                st.write(f"**Duplicate Rows:** {quality.get('duplicates', 0)}")
                
                # Distribution similarity
                distribution = validation.get("distribution_similarity", {})
                for col, stats in list(distribution.items())[:3]:
                    if stats.get("type") == "numeric":
                        st.write(f"**{col}** mean difference: {stats.get('mean_difference_pct', 0):.1f}%")
            
            with tab3:
                # Show preserved patterns
                patterns = validation.get("pattern_preservation", {})
                if patterns:
                    for col, pattern_info in patterns.items():
                        st.write(f"**{col}** patterns preserved:")
                        for pattern_name, stats in pattern_info.get("pattern_compliance", {}).items():
                            st.write(f"- {pattern_name}: {stats.get('percentage', 0):.1f}%")
            
            with tab4:
                # Download options
                try:
                    csv = synthetic.to_csv(index=False)
                    st.download_button(
                        "📥 Download Synthetic Data (CSV)",
                        csv,
                        f"synthetic_data_{len(synthetic)}_rows.csv",
                        "text/csv",
                        use_container_width=True
                    )
                except:
                    st.error("Could not create CSV")
                
                # Download rules
                try:
                    rules_json = json.dumps(st.session_state.rules, indent=2, default=str)
                    st.download_button(
                        "📥 Download Rules (JSON)",
                        rules_json,
                        f"data_rules.json",
                        "application/json",
                        use_container_width=True
                    )
                except:
                    st.error("Could not create rules download")
                
                # Regenerate
                if st.button("🔄 Generate New Variation"):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome message
        st.info("""
        ### 🎯 How it works:
        
        1. **Upload** any CSV file
        2. **Analyze** patterns automatically
        3. **Generate** synthetic data
        4. **Download** high-quality results
        
        ### ✨ Key Features:
        
        - **Pattern Preservation**: Maintains all relationships from original data
        - **No Hardcoding**: Works with ANY dataset structure
        - **Quality Validation**: Comprehensive quality checks
        - **Smart Generation**: Intelligent value generation
        """)

if __name__ == "__main__":
    main()
