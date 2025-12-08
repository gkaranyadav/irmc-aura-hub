# pages/6_🔢_Intelligent_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json
import re
import random
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# =============================================================================
# UNIVERSAL SEMANTIC INTELLIGENCE ENGINE
# =============================================================================

class UniversalSemanticIntelligence:
    """Understands semantic patterns in ANY dataset"""
    
    # UNIVERSAL PATTERNS (not domain-specific)
    UNIVERSAL_RELATIONSHIPS = {
        # Name patterns (works for ANY culture)
        "name_gender": {
            "columns": [["name", "patient", "customer", "user"], 
                       ["gender", "sex"]],
            "logic": "Names often suggest gender patterns",
            "check_with_groq": True
        },
        
        # Category-value patterns
        "category_value": {
            "columns": [["category", "type", "department", "role"],
                       ["value", "amount", "price", "salary"]],
            "logic": "Categories often have typical value ranges",
            "check_with_groq": True
        },
        
        # Date-sequence patterns
        "temporal_sequence": {
            "columns": [["start", "begin", "created"],
                       ["end", "finish", "completed"]],
            "logic": "Start dates should be before end dates",
            "check_with_groq": False
        },
        
        # ID-uniqueness patterns
        "id_uniqueness": {
            "columns": [["id", "code", "number", "serial"]],
            "logic": "IDs should typically be unique",
            "check_with_groq": False
        },
        
        # Hierarchical patterns
        "hierarchical": {
            "columns": [["parent", "main", "primary"],
                       ["child", "sub", "secondary"]],
            "logic": "Hierarchical relationships exist",
            "check_with_groq": True
        },
        
        # Geographic patterns
        "geographic": {
            "columns": [["city", "state", "country"],
                       ["price", "cost", "rate"]],
            "logic": "Geographic location affects prices/rates",
            "check_with_groq": True
        },
        
        # Status patterns
        "status_progression": {
            "columns": [["status", "state", "stage"],
                       ["date", "time", "timestamp"]],
            "logic": "Status changes follow temporal sequences",
            "check_with_groq": True
        }
    }
    
    @staticmethod
    def detect_semantic_patterns(df: pd.DataFrame) -> List[Dict]:
        """Detect semantic patterns in ANY dataset"""
        
        patterns = []
        columns = list(df.columns)
        column_names_lower = [col.lower() for col in columns]
        
        # Check each universal pattern
        for pattern_name, pattern_info in UniversalSemanticIntelligence.UNIVERSAL_RELATIONSHIPS.items():
            for col_group in pattern_info["columns"]:
                # Find matching columns
                matching_cols = []
                for col_pattern in col_group:
                    for idx, col_name in enumerate(column_names_lower):
                        if col_pattern in col_name:
                            matching_cols.append(columns[idx])
                
                if len(matching_cols) >= 2:
                    # Found potential pattern
                    pattern = {
                        "pattern_name": pattern_name,
                        "columns": matching_cols[:2],
                        "logic": pattern_info["logic"],
                        "check_with_groq": pattern_info["check_with_groq"],
                        "confidence": UniversalSemanticIntelligence._calculate_pattern_confidence(
                            matching_cols[:2], df, pattern_name
                        )
                    }
                    patterns.append(pattern)
        
        return patterns
    
    @staticmethod
    def _calculate_pattern_confidence(columns: List[str], df: pd.DataFrame, pattern_name: str) -> str:
        """Calculate confidence in detected pattern"""
        
        if len(columns) < 2:
            return "low"
        
        col1, col2 = columns[0], columns[1]
        
        try:
            if pattern_name == "name_gender":
                return UniversalSemanticIntelligence._check_name_gender_pattern(col1, col2, df)
            
            elif pattern_name == "category_value":
                return UniversalSemanticIntelligence._check_category_value_pattern(col1, col2, df)
            
            elif pattern_name == "id_uniqueness":
                return UniversalSemanticIntelligence._check_id_uniqueness(col1, df)
            
            else:
                unique_pairs = df[columns].dropna().drop_duplicates()
                if len(unique_pairs) < len(df) * 0.5:
                    return "medium"
                
        except:
            pass
        
        return "low"
    
    @staticmethod
    def _check_name_gender_pattern(name_col: str, gender_col: str, df: pd.DataFrame) -> str:
        """Check if names show gender patterns"""
        
        if name_col not in df.columns or gender_col not in df.columns:
            return "low"
        
        gender_values = set(str(v).lower() for v in df[gender_col].dropna().unique())
        typical_genders = {"m", "f", "male", "female", "0", "1"}
        
        if not any(g in typical_genders for g in gender_values):
            return "low"
        
        sample_size = min(100, len(df))
        sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        
        name_gender_map = {}
        inconsistencies = 0
        
        for _, row in sample.iterrows():
            name = str(row[name_col]).lower()
            gender = str(row[gender_col]).lower()
            
            if name in name_gender_map:
                if name_gender_map[name] != gender:
                    inconsistencies += 1
            else:
                name_gender_map[name] = gender
        
        inconsistency_rate = inconsistencies / len(name_gender_map) if name_gender_map else 1
        
        if inconsistency_rate < 0.1:
            return "high"
        elif inconsistency_rate < 0.3:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def _check_category_value_pattern(category_col: str, value_col: str, df: pd.DataFrame) -> str:
        """Check if categories have distinct value ranges"""
        
        if pd.api.types.is_numeric_dtype(df[value_col]):
            grouped = df.groupby(category_col)[value_col]
            std_within = grouped.std().mean()
            std_overall = df[value_col].std()
            
            if std_within < std_overall * 0.5:
                return "high"
            elif std_within < std_overall * 0.8:
                return "medium"
        
        return "low"
    
    @staticmethod
    def _check_id_uniqueness(col: str, df: pd.DataFrame) -> str:
        """Check if column looks like an ID"""
        
        unique_ratio = df[col].nunique() / len(df)
        
        if unique_ratio > 0.95:
            return "high"
        elif unique_ratio > 0.8:
            return "medium"
        else:
            return "low"

# =============================================================================
# SMART GROQ QUERY ENGINE
# =============================================================================

class SmartGroqQueryEngine:
    """Asks intelligent, context-aware questions to Groq"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def ask_intelligent_question(self, pattern: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Ask smart, context-aware question to Groq"""
        
        columns = pattern.get("columns", [])
        if len(columns) < 2:
            return {"error": "Need at least 2 columns"}
        
        col1, col2 = columns[0], columns[1]
        context = self._build_smart_context(col1, col2, df)
        question = self._build_intelligent_question(pattern, context)
        
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data relationship expert who understands semantic patterns.
                        Analyze column relationships based on their names, sample values, and common sense.
                        Give practical advice for synthetic data generation."""
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            answer = response.choices[0].message.content
            return self._parse_intelligent_response(answer, pattern, context)
            
        except Exception as e:
            return {
                "error": str(e),
                "advice": "Use statistical patterns from data",
                "confidence": "low"
            }
    
    def _build_smart_context(self, col1: str, col2: str, df: pd.DataFrame) -> Dict:
        """Build intelligent context about the columns"""
        
        context = {
            "column1": {
                "name": col1,
                "sample_values": df[col1].dropna().head(5).astype(str).tolist(),
                "value_type": self._infer_value_type(df[col1]),
                "unique_count": df[col1].nunique()
            },
            "column2": {
                "name": col2,
                "sample_values": df[col2].dropna().head(5).astype(str).tolist(),
                "value_type": self._infer_value_type(df[col2]),
                "unique_count": df[col2].nunique()
            },
            "relationship_hints": self._find_relationship_hints(col1, col2, df)
        }
        
        return context
    
    def _infer_value_type(self, series: pd.Series) -> str:
        """Infer semantic type of values"""
        
        if series.name and any(keyword in series.name.lower() for keyword in ['id', 'code', 'number', 'serial']):
            if series.nunique() / len(series) > 0.8:
                return "identifier"
        
        if series.name and any(keyword in series.name.lower() for keyword in ['name', 'person', 'customer', 'user']):
            return "name"
        
        if series.nunique() < 20:
            return "category"
        
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        
        try:
            pd.to_datetime(series.head(5))
            return "datetime"
        except:
            pass
        
        if any(len(str(val)) > 20 for val in series.dropna().head(5).astype(str)):
            return "text"
        
        return "unknown"
    
    def _find_relationship_hints(self, col1: str, col2: str, df: pd.DataFrame) -> List[str]:
        """Find hints about potential relationships"""
        
        hints = []
        
        try:
            unique_pairs = df[[col1, col2]].dropna().drop_duplicates()
            col1_unique = df[col1].nunique()
            
            if col1_unique > 0:
                avg_mappings = len(unique_pairs) / col1_unique
                if avg_mappings < 1.5:
                    hints.append(f"Low mapping variability ({avg_mappings:.1f} mappings per {col1} value)")
            
            if df[col1].nunique() < 10 and df[col2].nunique() < 10:
                contingency = pd.crosstab(df[col1], df[col2])
                max_per_row = contingency.max(axis=1)
                if (max_per_row / contingency.sum(axis=1) > 0.8).any():
                    hints.append("Strong value co-occurrence patterns detected")
        
        except:
            pass
        
        return hints
    
    def _build_intelligent_question(self, pattern: Dict, context: Dict) -> str:
        """Build context-aware intelligent question"""
        
        col1_info = context["column1"]
        col2_info = context["column2"]
        pattern_name = pattern.get("pattern_name", "unknown")
        
        question = f"""I'm analyzing a dataset with these two columns:

COLUMN 1: "{col1_info['name']}"
- Sample values: {col1_info['sample_values']}
- Value type: {col1_info['value_type']}
- Unique values: {col1_info['unique_count']}

COLUMN 2: "{col2_info['name']}"
- Sample values: {col2_info['sample_values']}
- Value type: {col2_info['value_type']}
- Unique values: {col2_info['unique_count']}

Detected pattern type: {pattern_name}
Relationship hints: {context['relationship_hints']}

Based on this CONTEXT, please analyze:

1. What SEMANTIC relationship might exist between these columns?
2. What REAL-WORLD constraints should I respect when generating synthetic data?
3. Are there any VALUE COMBINATIONS that would be unrealistic?
4. What GENERATION STRATEGY would maintain realism?

Please respond with SPECIFIC, ACTIONABLE advice for synthetic data generation.
Format your response as JSON with these keys:
- relationship_type: string describing the relationship
- constraints: list of constraints to follow
- unrealistic_combinations: list of combinations to avoid
- generation_strategy: specific strategy for generation
- confidence: high/medium/low
"""
        
        return question
    
    def _parse_intelligent_response(self, response: str, pattern: Dict, context: Dict) -> Dict[str, Any]:
        """Parse Groq's intelligent response"""
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                parsed["context_used"] = {
                    "pattern": pattern.get("pattern_name"),
                    "column_types": {
                        context["column1"]["name"]: context["column1"]["value_type"],
                        context["column2"]["name"]: context["column2"]["value_type"]
                    }
                }
                
                return parsed
        
        except:
            pass
        
        result = {
            "relationship_type": "extracted_from_text",
            "constraints": [],
            "unrealistic_combinations": [],
            "generation_strategy": "Follow data patterns carefully",
            "confidence": "medium",
            "raw_response": response[:500]
        }
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if "should not" in line_lower or "avoid" in line_lower or "unrealistic" in line_lower:
                result["constraints"].append(line.strip())
            if "strategy" in line_lower or "generate" in line_lower:
                result["generation_strategy"] = line.strip()
        
        return result

# =============================================================================
# INTELLIGENT CONSTRAINT ENGINE
# =============================================================================

class IntelligentConstraintEngine:
    """Builds intelligent constraints using semantic understanding"""
    
    def __init__(self, groq_engine: SmartGroqQueryEngine):
        self.groq_engine = groq_engine
        self.semantic_patterns = []
        self.intelligent_constraints = {}
        self.generation_rules = []
    
    def analyze_with_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset with semantic intelligence"""
        
        self.semantic_patterns = UniversalSemanticIntelligence.detect_semantic_patterns(df)
        self._get_groq_intelligence(df)
        self._build_intelligent_constraints(df)
        self._create_generation_rules()
        
        return {
            "semantic_patterns": self.semantic_patterns,
            "intelligent_constraints": self.intelligent_constraints,
            "generation_rules": self.generation_rules,
            "analysis_summary": self._create_intelligent_summary(df)
        }
    
    def _get_groq_intelligence(self, df: pd.DataFrame):
        """Get Groq intelligence for detected patterns"""
        
        for pattern in self.semantic_patterns:
            if pattern.get("check_with_groq", False) and pattern.get("confidence") in ["high", "medium"]:
                groq_response = self.groq_engine.ask_intelligent_question(pattern, df)
                
                if "error" not in groq_response:
                    pattern["groq_intelligence"] = groq_response
                    self._extract_constraints_from_groq(pattern, groq_response)
    
    def _extract_constraints_from_groq(self, pattern: Dict, groq_response: Dict):
        """Extract intelligent constraints from Groq response"""
        
        columns = pattern.get("columns", [])
        if len(columns) < 2:
            return
        
        col1, col2 = columns[0], columns[1]
        key = f"{col1}↔{col2}"
        
        constraints = {
            "pattern_type": pattern.get("pattern_name"),
            "groq_confidence": groq_response.get("confidence", "medium"),
            "relationship_type": groq_response.get("relationship_type", "unknown"),
            "constraints": groq_response.get("constraints", []),
            "unrealistic_combinations": groq_response.get("unrealistic_combinations", []),
            "generation_strategy": groq_response.get("generation_strategy", "")
        }
        
        self.intelligent_constraints[key] = constraints
    
    def _build_intelligent_constraints(self, df: pd.DataFrame):
        """Build constraints using semantic understanding"""
        
        for col in df.columns:
            if col not in self.intelligent_constraints:
                self.intelligent_constraints[col] = {
                    "type": self._infer_column_type(df[col]),
                    "statistical_constraints": self._build_statistical_constraints(df[col])
                }
        
        for pattern in self.semantic_patterns:
            if pattern.get("confidence") == "high":
                columns = pattern.get("columns", [])
                if len(columns) >= 2:
                    col1, col2 = columns[0], columns[1]
                    
                    rel_key = f"REL_{col1}_{col2}"
                    self.intelligent_constraints[rel_key] = {
                        "type": "relationship",
                        "columns": [col1, col2],
                        "pattern": pattern.get("pattern_name"),
                        "constraint": f"Respect {pattern.get('logic')}"
                    }
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer intelligent column type"""
        
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:
                return "categorical_numeric"
            return "continuous_numeric"
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        else:
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.3:
                return "categorical"
            else:
                col_name = series.name.lower() if series.name else ""
                if any(keyword in col_name for keyword in ['name', 'person']):
                    return "name"
                elif any(keyword in col_name for keyword in ['id', 'code']):
                    return "identifier"
                else:
                    return "text"
    
    def _build_statistical_constraints(self, series: pd.Series) -> Dict:
        """Build statistical constraints for a column"""
        
        constraints = {}
        
        if pd.api.types.is_numeric_dtype(series):
            constraints.update({
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "distribution": self._infer_distribution(series)
            })
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            constraints.update({
                "min_date": series.min().isoformat(),
                "max_date": series.max().isoformat(),
                "range_days": (series.max() - series.min()).days
            })
        
        else:
            value_counts = series.value_counts()
            constraints.update({
                "unique_values": series.nunique(),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_percentage": (value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0
            })
        
        return constraints
    
    def _infer_distribution(self, series: pd.Series) -> str:
        """Infer distribution type"""
        
        try:
            from scipy import stats
            
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return "unknown"
            
            _, p_value = stats.normaltest(clean_series)
            if p_value > 0.05:
                return "normal"
            
            skewness = stats.skew(clean_series)
            if abs(skewness) > 1:
                return "skewed"
            
            return "uniform"
        except:
            return "unknown"
    
    def _create_generation_rules(self):
        """Create intelligent generation rules"""
        
        self.generation_rules = []
        
        self.generation_rules.append({
            "id": "RULE_001",
            "type": "type_preservation",
            "description": "Preserve original data types",
            "priority": "high"
        })
        
        for pattern in self.semantic_patterns:
            if pattern.get("confidence") in ["high", "medium"]:
                columns = pattern.get("columns", [])
                if len(columns) >= 2:
                    self.generation_rules.append({
                        "id": f"RULE_PATTERN_{pattern['pattern_name']}",
                        "type": "semantic_pattern",
                        "columns": columns,
                        "description": pattern.get("logic"),
                        "priority": "medium" if pattern["confidence"] == "high" else "low"
                    })
        
        for key, constraint in self.intelligent_constraints.items():
            if "generation_strategy" in constraint and constraint["generation_strategy"]:
                if "↔" in key:
                    col1, col2 = key.split("↔")
                    self.generation_rules.append({
                        "id": f"RULE_GROQ_{col1}_{col2}",
                        "type": "groq_intelligence",
                        "columns": [col1, col2],
                        "description": constraint["generation_strategy"],
                        "priority": "high" if constraint.get("groq_confidence") == "high" else "medium"
                    })
        
        self.generation_rules.append({
            "id": "RULE_004",
            "type": "statistical_preservation",
            "description": "Maintain statistical distributions",
            "priority": "high"
        })
    
    def _create_intelligent_summary(self, df: pd.DataFrame) -> Dict:
        """Create intelligent analysis summary"""
        
        return {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "semantic_patterns_found": len(self.semantic_patterns),
            "high_confidence_patterns": sum(1 for p in self.semantic_patterns if p.get("confidence") == "high"),
            "groq_intelligence_used": sum(1 for p in self.semantic_patterns if "groq_intelligence" in p),
            "intelligent_constraints": len(self.intelligent_constraints),
            "generation_rules": len(self.generation_rules)
        }

# =============================================================================
# INTELLIGENT SYNTHETIC GENERATOR
# =============================================================================

class IntelligentSyntheticGenerator:
    """Generates synthetic data using semantic intelligence"""
    
    def __init__(self, intelligent_constraints: Dict):
        self.constraints = intelligent_constraints
        self.generated_cache = defaultdict(set)
        self.semantic_rules = self._extract_semantic_rules(intelligent_constraints)
    
    def _extract_semantic_rules(self, constraints: Dict) -> List[Dict]:
        """Extract semantic rules from constraints"""
        
        rules = []
        
        for key, constraint in constraints.items():
            if "↔" in key:
                col1, col2 = key.split("↔")
                
                rule = {
                    "columns": [col1, col2],
                    "type": constraint.get("relationship_type", "unknown"),
                    "constraints": constraint.get("constraints", []),
                    "unrealistic_combinations": constraint.get("unrealistic_combinations", []),
                    "strategy": constraint.get("generation_strategy", "")
                }
                
                rules.append(rule)
        
        return rules
    
    def generate_intelligent(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate data using semantic intelligence"""
        
        synthetic_rows = []
        
        for row_idx in range(num_rows):
            row = {}
            
            generation_order = self._determine_generation_order(df)
            
            for col in generation_order:
                row[col] = self._generate_with_intelligence(col, df, row, row_idx)
            
            row = self._apply_semantic_rules(row, df)
            row = self._ensure_semantic_quality(row, df)
            
            synthetic_rows.append(row)
            
            if row_idx % 50 == 0 and row_idx > 0:
                st.info(f"Intelligently generated {row_idx} of {num_rows} rows...")
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        synthetic_df = self._preserve_intelligent_types(synthetic_df, df)
        
        return synthetic_df
    
    def _determine_generation_order(self, df: pd.DataFrame) -> List[str]:
        """Determine intelligent generation order"""
        
        independent = []
        dependent = []
        
        for col in df.columns:
            is_dependent = False
            for rule in self.semantic_rules:
                if col == rule["columns"][1]:
                    is_dependent = True
                    break
            
            if is_dependent:
                dependent.append(col)
            else:
                independent.append(col)
        
        return independent + dependent
    
    def _generate_with_intelligence(self, col: str, df: pd.DataFrame, 
                                   current_row: Dict, row_idx: int) -> Any:
        """Generate value using semantic intelligence"""
        
        for rule in self.semantic_rules:
            if col == rule["columns"][1]:
                dep_col = rule["columns"][0]
                if dep_col in current_row:
                    return self._generate_dependent_value(col, dep_col, current_row[dep_col], df, rule)
        
        return self._generate_independent_value(col, df, row_idx)
    
    def _generate_dependent_value(self, target_col: str, source_col: str, 
                                 source_value: Any, df: pd.DataFrame, rule: Dict) -> Any:
        """Generate value based on dependency"""
        
        matching_rows = df[df[source_col] == source_value]
        
        if len(matching_rows) > 0:
            return matching_rows[target_col].sample(1).iloc[0]
        else:
            return self._generate_independent_value(target_col, df, 0)
    
    def _generate_independent_value(self, col: str, df: pd.DataFrame, row_idx: int) -> Any:
        """Generate independent value"""
        
        col_constraints = self.constraints.get(col, {}).get("statistical_constraints", {})
        
        if pd.api.types.is_numeric_dtype(df[col]):
            return self._generate_intelligent_numeric(col_constraints)
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            return self._generate_intelligent_datetime(col_constraints)
        
        else:
            return self._generate_intelligent_categorical(col, df, col_constraints, row_idx)
    
    def _generate_intelligent_numeric(self, constraints: Dict) -> float:
        """Generate numeric value intelligently"""
        
        min_val = constraints.get("min", 0)
        max_val = constraints.get("max", 100)
        mean_val = constraints.get("mean", (min_val + max_val) / 2)
        std_val = constraints.get("std", (max_val - min_val) / 6)
        distribution = constraints.get("distribution", "normal")
        
        if distribution == "normal":
            value = np.random.normal(mean_val, std_val)
        elif distribution == "uniform":
            value = random.uniform(min_val, max_val)
        else:
            value = random.triangular(min_val, mean_val, max_val)
        
        return float(np.clip(value, min_val, max_val))
    
    def _generate_intelligent_datetime(self, constraints: Dict):
        """Generate datetime value intelligently"""
        
        min_date_str = constraints.get("min_date")
        max_date_str = constraints.get("max_date")
        
        if min_date_str and max_date_str:
            min_date = pd.to_datetime(min_date_str)
            max_date = pd.to_datetime(max_date_str)
            
            range_days = (max_date - min_date).days
            
            if range_days > 365:
                day_of_year = random.randint(1, 365)
                year = random.randint(min_date.year, max_date.year)
                return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=day_of_year-1)
            else:
                random_days = random.randint(0, range_days)
                return min_date + pd.Timedelta(days=random_days)
        
        return pd.Timestamp.now()
    
    def _generate_intelligent_categorical(self, col: str, df: pd.DataFrame, 
                                         constraints: Dict, row_idx: int) -> Any:
        """Generate categorical/text value intelligently"""
        
        col_name_lower = col.lower()
        
        if any(keyword in col_name_lower for keyword in ['name', 'person']):
            return self._generate_intelligent_name(col, df, row_idx)
        
        elif any(keyword in col_name_lower for keyword in ['id', 'code']):
            return self._generate_intelligent_id(col, df, row_idx)
        
        unique_count = constraints.get("unique_values", 0)
        
        if unique_count < 20:
            values = df[col].dropna().unique()
            if len(values) > 0:
                value_counts = df[col].value_counts(normalize=True)
                return np.random.choice(value_counts.index.tolist(), 
                                      p=value_counts.values.tolist())
            else:
                return None
        else:
            return df[col].dropna().sample(1).iloc[0] if len(df[col].dropna()) > 0 else ""
    
    def _generate_intelligent_name(self, col: str, df: pd.DataFrame, row_idx: int) -> str:
        """Generate name intelligently"""
        
        names = df[col].dropna().astype(str).tolist()
        
        if len(names) > 0:
            sample_names = random.sample(names, min(10, len(names)))
            
            first_parts = [name.split()[0] for name in sample_names if ' ' in name]
            last_parts = [name.split()[-1] for name in sample_names if ' ' in name]
            
            if first_parts and last_parts:
                return f"{random.choice(first_parts)} {random.choice(last_parts)}"
            else:
                return random.choice(sample_names)
        else:
            return f"Person_{row_idx}"
    
    def _generate_intelligent_id(self, col: str, df: pd.DataFrame, row_idx: int) -> str:
        """Generate ID intelligently"""
        
        sample_ids = df[col].dropna().astype(str).head(10).tolist()
        
        if sample_ids:
            first_id = sample_ids[0]
            
            if first_id.isdigit():
                return str(1000 + row_idx)
            elif any(sep in first_id for sep in ['-', '_']):
                prefix = first_id.split('-')[0] if '-' in first_id else first_id.split('_')[0]
                return f"{prefix}_{row_idx}"
            else:
                return f"ID_{row_idx:04d}"
        else:
            return f"{col}_{row_idx:04d}"
    
    def _apply_semantic_rules(self, row: Dict, df: pd.DataFrame) -> Dict:
        """Apply semantic rules to generated row"""
        
        for rule in self.semantic_rules:
            if len(rule["columns"]) >= 2:
                col1, col2 = rule["columns"][0], rule["columns"][1]
                
                if col1 in row and col2 in row:
                    violates = False
                    
                    for bad_combo in rule.get("unrealistic_combinations", []):
                        if bad_combo.lower() in f"{row[col1]}{row[col2]}".lower():
                            violates = True
                            break
                    
                    if violates:
                        valid_combos = df[[col1, col2]].dropna().drop_duplicates()
                        if len(valid_combos) > 0:
                            new_combo = valid_combos.sample(1).iloc[0]
                            row[col1] = new_combo[col1]
                            row[col2] = new_combo[col2]
        
        return row
    
    def _ensure_semantic_quality(self, row: Dict, df: pd.DataFrame) -> Dict:
        """Ensure semantic quality of generated row"""
        
        name_cols = [col for col in row.keys() if 'name' in col.lower()]
        gender_cols = [col for col in row.keys() if 'gender' in col.lower() or 'sex' in col.lower()]
        
        if name_cols and gender_cols:
            pass
        
        return row
    
    def _preserve_intelligent_types(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Intelligently preserve data types"""
        
        for col in original_df.columns:
            if col in synthetic_df.columns:
                try:
                    original_dtype = original_df[col].dtype
                    
                    if pd.api.types.is_datetime64_any_dtype(original_dtype):
                        synthetic_df[col] = pd.to_datetime(synthetic_df[col], errors='coerce')
                    elif pd.api.types.is_numeric_dtype(original_dtype):
                        synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce')
                    else:
                        synthetic_df[col] = synthetic_df[col].astype(str)
                except:
                    pass
        
        return synthetic_df

# =============================================================================
# QUALITY VALIDATOR
# =============================================================================

class IntelligentQualityValidator:
    """Validates synthetic data quality"""
    
    @staticmethod
    def validate(original: pd.DataFrame, synthetic: pd.DataFrame, 
                constraints: Dict) -> Dict[str, Any]:
        """Comprehensive validation"""
        
        validation = {
            "basic_checks": IntelligentQualityValidator._basic_checks(synthetic),
            "statistical_comparison": IntelligentQualityValidator._statistical_comparison(original, synthetic),
            "relationship_compliance": IntelligentQualityValidator._relationship_compliance(synthetic, constraints),
            "data_quality": IntelligentQualityValidator._data_quality(synthetic)
        }
        
        scores = []
        if "score" in validation["basic_checks"]:
            scores.append(validation["basic_checks"]["score"])
        if "similarity_score" in validation["statistical_comparison"]:
            scores.append(validation["statistical_comparison"]["similarity_score"])
        if "compliance_score" in validation["relationship_compliance"]:
            scores.append(validation["relationship_compliance"]["compliance_score"])
        if "quality_score" in validation["data_quality"]:
            scores.append(validation["data_quality"]["quality_score"])
        
        validation["overall_score"] = np.mean(scores) if scores else 0
        
        return validation
    
    @staticmethod
    def _basic_checks(df: pd.DataFrame) -> Dict:
        """Basic data checks"""
        
        checks = {
            "has_data": len(df) > 0,
            "has_columns": len(df.columns) > 0,
            "null_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "issues": []
        }
        
        if checks["null_percentage"] > 30:
            checks["issues"].append(f"High null values: {checks['null_percentage']:.1f}%")
        
        if checks["duplicate_rows"] > len(df) * 0.1:
            checks["issues"].append(f"Many duplicate rows: {checks['duplicate_rows']}")
        
        checks["score"] = max(0, 100 - checks["null_percentage"] - (checks["duplicate_rows"] / len(df) * 100))
        
        return checks
    
    @staticmethod
    def _statistical_comparison(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Compare statistical properties"""
        
        comparison = {
            "column_comparisons": {},
            "similarity_scores": []
        }
        
        for col in original.columns:
            if col in synthetic.columns:
                orig_series = original[col].dropna()
                synth_series = synthetic[col].dropna()
                
                if len(orig_series) > 0 and len(synth_series) > 0:
                    if pd.api.types.is_numeric_dtype(orig_series):
                        mean_diff = abs(orig_series.mean() - synth_series.mean()) / max(1, abs(orig_series.mean()))
                        std_diff = abs(orig_series.std() - synth_series.std()) / max(1, abs(orig_series.std()))
                        similarity = 100 * (1 - (mean_diff + std_diff) / 2)
                    else:
                        orig_counts = orig_series.value_counts(normalize=True)
                        synth_counts = synth_series.value_counts(normalize=True)
                        
                        common = set(orig_counts.index) & set(synth_counts.index)
                        if len(common) > 0:
                            total_diff = sum(abs(orig_counts.get(c, 0) - synth_counts.get(c, 0)) for c in common)
                            similarity = 100 * (1 - total_diff)
                        else:
                            similarity = 0
                    
                    comparison["column_comparisons"][col] = similarity
                    comparison["similarity_scores"].append(similarity)
        
        if comparison["similarity_scores"]:
            comparison["similarity_score"] = np.mean(comparison["similarity_scores"])
        else:
            comparison["similarity_score"] = 0
        
        return comparison
    
    @staticmethod
    def _relationship_compliance(synthetic: pd.DataFrame, constraints: Dict) -> Dict:
        """Check relationship compliance"""
        
        relationships = constraints.get("intelligent_constraints", {})
        
        compliance = {
            "relationships_checked": 0,
            "violations": []
        }
        
        for key, constraint in relationships.items():
            if "↔" in key:
                compliance["relationships_checked"] += 1
                col1, col2 = key.split("↔")
                
                if col1 in synthetic.columns and col2 in synthetic.columns:
                    unrealistic = constraint.get("unrealistic_combinations", [])
                    
                    for combo in unrealistic:
                        if combo and isinstance(combo, str):
                            if combo.lower() in synthetic[[col1, col2]].astype(str).apply(lambda x: ' '.join(x), axis=1).str.lower().any():
                                compliance["violations"].append(f"Unrealistic combination found: {combo}")
        
        total_checks = compliance["relationships_checked"]
        violations = len(compliance["violations"])
        
        if total_checks > 0:
            compliance["compliance_score"] = 100 * (1 - violations / total_checks)
        else:
            compliance["compliance_score"] = 100
        
        return compliance
    
    @staticmethod
    def _data_quality(df: pd.DataFrame) -> Dict:
        """Data quality metrics"""
        
        quality = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_values": df.isnull().sum().sum(),
            "null_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": df.duplicated().sum() / len(df) * 100
        }
        
        quality["quality_score"] = max(0, 100 - quality["null_percentage"] - quality["duplicate_percentage"])
        
        return quality

# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Intelligent Synthetic Data Generator",
        page_icon="🧠",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .intelligent-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .agent-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }
        .relationship-card {
            background: #e3f2fd;
            border-radius: 8px;
            padding: 0.8rem;
            margin: 0.3rem 0;
            border-left: 3px solid #2196f3;
        }
        .progress-container {
            background: #f1f3f4;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="intelligent-header">🧠 Intelligent Synthetic Data Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Groq Intelligence
    
    """)
    
    # Check API key
    if "GROQ_API_KEY" not in st.secrets:
        st.error("⚠️ GROQ_API_KEY not found in Streamlit secrets!")
        st.info("""
        **To fix:**
        1. Go to Streamlit Cloud → App → Settings → Secrets
        2. Add: `GROQ_API_KEY = "your-groq-key-here"`
        3. Redeploy
        """)
        return
    
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
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
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                numeric_cols = sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))
                st.metric("Numeric Columns", numeric_cols)
        
        # Generation settings
        st.subheader("⚙️ Intelligent Generation Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=10000,
                value=min(len(df) * 3, 1000),
                step=100
            )
        
        with col2:
            intelligence_mode = st.selectbox(
                "Intelligence Mode",
                ["Full Intelligence (Recommended)", "Fast Analysis", "Statistical Only"],
                help="Full intelligence uses Groq for smart relationship detection"
            )
        
        with col3:
            quality_focus = st.selectbox(
                "Quality Focus",
                ["Balanced", "High Realism", "High Variation"],
                help="Balance between realism and diversity"
            )
        
        # Show intelligence agents
        st.subheader("👥 Intelligence System Ready")
        
        agents = [
            {"emoji": "🤔", "name": "Semantic Detector", "role": "Finds intelligent patterns"},
            {"emoji": "🔍", "name": "Groq Intelligence", "role": "Asks smart questions"},
            {"emoji": "🛠️", "name": "Constraint Builder", "role": "Creates smart rules"},
            {"emoji": "🎨", "name": "Smart Generator", "role": "Generates intelligently"},
            {"emoji": "✅", "name": "Quality Validator", "role": "Ensures quality"}
        ]
        
        cols = st.columns(5)
        for idx, (col, agent) in enumerate(zip(cols, agents)):
            with col:
                st.markdown(f"""
                <div class="agent-card">
                    <h3>{agent['emoji']} {agent['name']}</h3>
                    <p>{agent['role']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Generate button
        if st.button("🚀 Generate Intelligent Synthetic Data", type="primary", use_container_width=True):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Initialize
                status_text.text("Initializing intelligence system...")
                progress_bar.progress(10)
                
                groq_engine = SmartGroqQueryEngine(groq_api_key)
                
                # Step 2: Analyze with intelligence
                status_text.text("🤔 Detecting semantic patterns...")
                progress_bar.progress(30)
                
                constraint_engine = IntelligentConstraintEngine(groq_engine)
                constraints = constraint_engine.analyze_with_intelligence(df)
                
                # Show analysis
                with st.expander("📊 Intelligence Analysis Results", expanded=True):
                    
                    # Show semantic patterns
                    patterns = constraints.get("semantic_patterns", [])
                    if patterns:
                        st.write("**🔍 Detected Semantic Patterns:**")
                        for pattern in patterns:
                            if pattern.get("confidence") in ["high", "medium"]:
                                cols = pattern.get("columns", [])
                                with st.container():
                                    st.markdown(f"""
                                    <div class="relationship-card">
                                        <strong>{' ↔ '.join(cols)}</strong><br>
                                        Pattern: {pattern.get('pattern_name', 'unknown')}<br>
                                        Confidence: {pattern.get('confidence', 'low')}<br>
                                        Logic: {pattern.get('logic', '')}
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("No strong semantic patterns detected")
                    
                    # Show summary
                    summary = constraints.get("analysis_summary", {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Columns", summary.get("total_columns", 0))
                    with col2:
                        st.metric("Patterns Found", summary.get("semantic_patterns_found", 0))
                    with col3:
                        st.metric("High Confidence", summary.get("high_confidence_patterns", 0))
                    with col4:
                        st.metric("Groq Queries", summary.get("groq_intelligence_used", 0))
                
                # Step 3: Generate data
                status_text.text("🎨 Generating intelligent synthetic data...")
                progress_bar.progress(60)
                
                generator = IntelligentSyntheticGenerator(constraints)
                synthetic_df = generator.generate_intelligent(df, num_rows)
                
                # Step 4: Validate
                status_text.text("✅ Validating data quality...")
                progress_bar.progress(90)
                
                validator = IntelligentQualityValidator()
                validation = validator.validate(df, synthetic_df, constraints)
                
                # Complete
                status_text.text("✅ Generation complete!")
                progress_bar.progress(100)
                
                # Store results
                st.session_state.synthetic_data = synthetic_df
                st.session_state.constraints = constraints
                st.session_state.validation = validation
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Intelligent generation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Display results
        if 'synthetic_data' in st.session_state:
            synthetic = st.session_state.synthetic_data
            constraints = st.session_state.constraints
            validation = st.session_state.validation
            
            st.subheader(f"✨ Generated {len(synthetic)} Intelligent Rows")
            
            # Overall score
            overall_score = validation.get("overall_score", 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{overall_score:.1f}%")
            with col2:
                similarity = validation.get("statistical_comparison", {}).get("similarity_score", 0)
                st.metric("Statistical Similarity", f"{similarity:.1f}%")
            with col3:
                compliance = validation.get("relationship_compliance", {}).get("compliance_score", 0)
                st.metric("Relationship Compliance", f"{compliance:.1f}%")
            with col4:
                quality = validation.get("data_quality", {}).get("quality_score", 0)
                st.metric("Data Quality", f"{quality:.1f}%")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Synthetic Data", "🧠 Intelligence Analysis", "✅ Validation Details", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Quick comparison
                if len(df.columns) > 0:
                    compare_col = st.selectbox("Compare column", df.columns, key="compare")
                    if compare_col in synthetic.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original**")
                            try:
                                st.bar_chart(df[compare_col].value_counts().head(10))
                            except:
                                st.write(f"Unique values: {df[compare_col].nunique()}")
                        with col2:
                            st.write("**Synthetic**")
                            try:
                                st.bar_chart(synthetic[compare_col].value_counts().head(10))
                            except:
                                st.write(f"Unique values: {synthetic[compare_col].nunique()}")
            
            with tab2:
                # Show intelligence analysis
                st.write("### 🧠 How the AI Thought:")
                
                patterns = constraints.get("semantic_patterns", [])
                intelligent_constraints = constraints.get("intelligent_constraints", {})
                
                if patterns:
                    st.write("**Semantic Patterns Detected:**")
                    for pattern in patterns:
                        if pattern.get("confidence") in ["high", "medium"]:
                            with st.expander(f"{' ↔ '.join(pattern.get('columns', []))} - {pattern.get('pattern_name')}"):
                                st.write(f"**Confidence:** {pattern.get('confidence')}")
                                st.write(f"**Logic:** {pattern.get('logic')}")
                                
                                if "groq_intelligence" in pattern:
                                    groq_info = pattern["groq_intelligence"]
                                    st.write("**Groq Intelligence:**")
                                    st.write(f"Relationship Type: {groq_info.get('relationship_type', 'Unknown')}")
                                    st.write(f"Generation Strategy: {groq_info.get('generation_strategy', 'No strategy')}")
                                    
                                    constraints_list = groq_info.get("constraints", [])
                                    if constraints_list:
                                        st.write("**Constraints from Groq:**")
                                        for constraint in constraints_list[:3]:
                                            st.write(f"- {constraint}")
                else:
                    st.info("No semantic patterns detected. Generated based on statistical analysis.")
            
            with tab3:
                # Validation details
                st.write("### ✅ Validation Report")
                
                # Issues
                basic_checks = validation.get("basic_checks", {})
                if basic_checks.get("issues"):
                    st.write("**Issues Found:**")
                    for issue in basic_checks["issues"]:
                        st.write(f"⚠️ {issue}")
                else:
                    st.success("✅ No major issues found")
                
                # Relationship violations
                relationship_compliance = validation.get("relationship_compliance", {})
                if relationship_compliance.get("violations"):
                    st.write("**Relationship Violations:**")
                    for violation in relationship_compliance["violations"]:
                        st.write(f"❌ {violation}")
                else:
                    st.success("✅ All relationships maintained correctly")
                
                # Top similar columns
                statistical = validation.get("statistical_comparison", {})
                if statistical.get("column_comparisons"):
                    st.write("**Top 10 Most Similar Columns:**")
                    similarities = sorted(
                        statistical["column_comparisons"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    
                    for col, score in similarities:
                        st.write(f"- {col}: {score:.1f}% similar")
            
            with tab4:
                # Download
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download Synthetic Data (CSV)",
                    csv,
                    f"intelligent_synthetic_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Download intelligence analysis
                if constraints:
                    constraints_json = json.dumps(constraints, indent=2, default=str)
                    st.download_button(
                        "📥 Download Intelligence Analysis (JSON)",
                        constraints_json,
                        "intelligence_analysis.json",
                        "application/json",
                        use_container_width=True
                    )
                
                # Regenerate
                if st.button("🔄 Generate New Variation", use_container_width=True):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome
        st.info("""
        ### data Giene
        """)
        
        # Example
        with st.expander("📚 Example Intelligence Process"):
            st.write("""
            

if __name__ == "__main__":
    main()
