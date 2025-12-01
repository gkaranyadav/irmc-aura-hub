# pages/6_🔢_Synthetic_Data_Generator.py - ENHANCED PROMPT VERSION
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import json
import re
import random
from collections import defaultdict, Counter
import hashlib

# =============================================================================
# ENHANCED RULE DISCOVERY ENGINE
# =============================================================================

class EnhancedRuleDiscoveryEngine:
    """Enhanced rule discovery with better prompting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def discover_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced rule discovery with improved prompting
        """
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            # Prepare data sample
            sample_size = min(20, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            
            # Get column relationship analysis FIRST
            relationship_analysis = self._analyze_relationships_statistically(df)
            
            # Build ENHANCED prompt
            prompt = self._build_enhanced_prompt(df, sample_df, relationship_analysis)
            
            # Get analysis from Groq
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": """You are an EXPERT data pattern detective. 
                     Your ONLY job is to find EXACT relationships between columns.
                     Be METICULOUS - find EVERY relationship.
                     Return ONLY valid JSON."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            raw_rules = json.loads(response.choices[0].message.content)
            
            # Structure and ENHANCE rules with statistical validation
            structured_rules = self._structure_and_enhance_rules(raw_rules, df, relationship_analysis)
            
            return structured_rules
            
        except Exception as e:
            st.error(f"Rule discovery failed: {e}")
            return self._discover_rules_statistically_enhanced(df)
    
    def _analyze_relationships_statistically(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical analysis to help LLM"""
        relationships = {
            "strong_mappings": [],
            "value_constraints": {},
            "pattern_detection": {}
        }
        
        # Find strong column relationships
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Calculate relationship strength
                unique_pairs = df[[col1, col2]].drop_duplicates()
                pair_count = len(unique_pairs)
                total_possible = df[col1].nunique() * df[col2].nunique()
                
                if pair_count > 0 and pair_count < total_possible * 0.5:
                    # Strong relationship detected
                    relationship_strength = 1 - (pair_count / total_possible)
                    
                    if relationship_strength > 0.7:  # Very strong
                        relationships["strong_mappings"].append({
                            "columns": [col1, col2],
                            "unique_pairs": pair_count,
                            "strength": relationship_strength,
                            "examples": unique_pairs.head(3).to_dict('records')
                        })
        
        # Analyze value constraints
        for col in df.columns:
            unique_count = df[col].nunique()
            total = len(df)
            
            if unique_count < min(10, total * 0.3):
                # Likely categorical with constraints
                relationships["value_constraints"][col] = {
                    "type": "categorical",
                    "unique_values": df[col].dropna().unique().tolist(),
                    "value_counts": df[col].value_counts().to_dict()
                }
            elif pd.api.types.is_numeric_dtype(df[col]):
                relationships["value_constraints"][col] = {
                    "type": "numeric",
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
        
        return relationships
    
    def _build_enhanced_prompt(self, df: pd.DataFrame, sample_df: pd.DataFrame, stats: Dict) -> str:
        """Build ENHANCED prompt for better rule discovery"""
        
        # Create relationship hints from statistical analysis
        relationship_hints = ""
        if stats["strong_mappings"]:
            relationship_hints = "STATISTICAL HINTS (high confidence relationships):\n"
            for mapping in stats["strong_mappings"][:5]:  # Top 5
                cols = mapping["columns"]
                examples = mapping.get("examples", [])
                relationship_hints += f"- Columns {cols[0]} ↔ {cols[1]}: {len(examples)} example mappings\n"
                for ex in examples[:2]:
                    relationship_hints += f"  * {cols[0]}={ex.get(cols[0])} → {cols[1]}={ex.get(cols[1])}\n"
        
        return f"""CRITICAL DATA ANALYSIS TASK - FIND EVERY RELATIONSHIP

I need you to analyze this dataset METICULOUSLY and find EXACTLY how columns relate to each other.

DATASET OVERVIEW:
- Rows: {len(df)}
- Columns: {list(df.columns)}

{relationship_hints}

COLUMN DETAILS:
{self._get_detailed_column_analysis(df)}

DATA SAMPLE ({len(sample_df)} random rows for pattern analysis):
{sample_df.to_string(index=False)}

YOUR MISSION - FIND THESE RELATIONSHIPS:

1. **EXACT VALUE MAPPINGS** (MOST IMPORTANT):
   For EVERY column pair where values map exactly:
   - When column A has value X, column B ALWAYS has value Y
   - Example: When "Department" = "Cardiology", "Doctor" ALWAYS = "Dr. Sharma"
   
   Format each mapping as:
   {{
     "from_column": "Department",
     "from_value": "Cardiology", 
     "to_column": "Doctor",
     "to_value": "Dr. Sharma",
     "confidence": "exact"  # or "strong" or "partial"
   }}

2. **COMPLETE VALUE CONSTRAINTS**:
   For EACH column, list ALL possible values or valid ranges:
   - Categorical: ["M", "F"], ["Completed", "Cancelled", "No-Show"]
   - Numeric: min=0, max=120
   - Patterns: phone=10 digits, id=starts with letters
   
3. **LOGICAL RULES**:
   Business logic that MUST be followed:
   - "IF Symptom contains 'Chest' THEN Department must be 'Cardiology'"
   - "Age must be between 0 and 120"
   - "Phone numbers must be exactly 10 digits"

4. **UNIQUENESS RULES**:
   Which columns should have unique values?
   - "PatientID must be unique"
   - "AppointmentID must be unique"

CRITICAL INSTRUCTIONS:
1. Be EXHAUSTIVE - find EVERY relationship
2. Be PRECISE - exact values, not approximations
3. Check CONSISTENCY - if a mapping appears multiple times, it's a rule
4. Validate EVERYTHING - only include rules that are 100% consistent in the sample

RETURN COMPREHENSIVE JSON with this structure:
{{
  "dataset_summary": "brief description",
  "value_mappings": [
    // LIST EVERY EXACT MAPPING YOU FIND
    {{
      "from_column": "column_name",
      "from_value": "exact_value",
      "to_column": "other_column",
      "to_value": "exact_mapped_value",
      "confidence": "exact/strong/partial"
    }}
  ],
  "value_constraints": {{
    "column_name": {{
      "type": "categorical/numeric/date/text/id",
      "allowed_values": ["complete", "list", "of", "ALL", "values"],
      "min": 0,
      "max": 100,
      "pattern": "regex if applicable",
      "description": "constraint description"
    }}
  }},
  "logical_rules": [
    "clear business rules that must be followed"
  ],
  "uniqueness_constraints": ["columns that should be unique"],
  "analysis_notes": "what you focused on and why"
}}

IMPORTANT: Your analysis will be used to generate new data. If you miss a mapping, 
the generated data will be WRONG. Be THOROUGH."""
    
    def _get_detailed_column_analysis(self, df: pd.DataFrame) -> str:
        """Get detailed column analysis for prompt"""
        analysis = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            sample_values = df[col].dropna().unique()[:5].tolist()
            
            col_analysis = f"""
[{col}]
- Type: {dtype}
- Unique values: {unique_count} ({unique_count/len(df)*100:.1f}% of rows)
- Null values: {null_count}
- Sample: {sample_values}
"""
            
            # Add value distribution for categorical
            if unique_count < 10 and unique_count > 0:
                value_counts = df[col].value_counts().to_dict()
                col_analysis += f"- Value distribution: {value_counts}\n"
            
            # Detect potential ID columns
            if 'id' in col.lower() or unique_count == len(df):
                col_analysis += "- LIKELY IDENTIFIER (should be unique)\n"
            
            # Detect potential categories
            if unique_count < 20 and unique_count > 1:
                col_analysis += "- LIKELY CATEGORICAL\n"
            
            analysis.append(col_analysis)
        
        return "\n".join(analysis)
    
    def _structure_and_enhance_rules(self, raw_rules: Dict, df: pd.DataFrame, stats: Dict) -> Dict[str, Any]:
        """Structure rules and ENHANCE with statistical validation"""
        structured = {
            "value_mappings": [],
            "value_constraints": {},
            "logical_rules": [],
            "uniqueness_constraints": [],
            "dataset_summary": raw_rules.get("dataset_summary", ""),
            "analysis_notes": raw_rules.get("analysis_notes", "")
        }
        
        # Process value mappings from LLM
        llm_mappings = self._process_llm_mappings(raw_rules.get("value_mappings", []), df)
        
        # ENHANCE with statistical mappings
        statistical_mappings = self._extract_statistical_mappings(df, stats)
        
        # Combine and deduplicate
        all_mappings = self._combine_and_validate_mappings(llm_mappings, statistical_mappings, df)
        structured["value_mappings"] = all_mappings
        
        # Process constraints
        for col, constraints in raw_rules.get("value_constraints", {}).items():
            if col in df.columns:
                structured["value_constraints"][col] = constraints
        
        # ENHANCE constraints with statistical data
        self._enhance_constraints(structured, df)
        
        # Process logical rules
        for rule in raw_rules.get("logical_rules", []):
            if isinstance(rule, str) and len(rule) > 10:
                structured["logical_rules"].append(rule)
        
        # Process uniqueness
        for col in raw_rules.get("uniqueness_constraints", []):
            if col in df.columns:
                structured["uniqueness_constraints"].append(col)
        
        # Add auto-detected uniqueness (IDs, etc.)
        self._add_auto_uniqueness(structured, df)
        
        return structured
    
    def _process_llm_mappings(self, mappings: List, df: pd.DataFrame) -> List[Dict]:
        """Process and validate LLM mappings"""
        valid_mappings = []
        
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            
            # Check required fields
            required = ["from_column", "from_value", "to_column", "to_value"]
            if not all(k in mapping for k in required):
                continue
            
            from_col = mapping["from_column"]
            to_col = mapping["to_column"]
            
            # Validate columns exist
            if from_col not in df.columns or to_col not in df.columns:
                continue
            
            # Validate mapping exists in data
            from_val = str(mapping["from_value"])
            to_val = str(mapping["to_value"])
            
            # Check if this exact mapping exists in original data
            matching_rows = df[
                (df[from_col].astype(str) == from_val) & 
                (df[to_col].astype(str) == to_val)
            ]
            
            if len(matching_rows) > 0:
                # Also check if there are any violations
                violating_rows = df[
                    (df[from_col].astype(str) == from_val) & 
                    (df[to_col].astype(str) != to_val)
                ]
                
                confidence = "exact" if len(violating_rows) == 0 else "partial"
                
                valid_mappings.append({
                    "from_column": from_col,
                    "from_value": from_val,
                    "to_column": to_col,
                    "to_value": to_val,
                    "confidence": confidence,
                    "support_count": len(matching_rows),
                    "violation_count": len(violating_rows)
                })
        
        return valid_mappings
    
    def _extract_statistical_mappings(self, df: pd.DataFrame, stats: Dict) -> List[Dict]:
        """Extract mappings from statistical analysis"""
        mappings = []
        
        # Look for exact mappings
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Get unique value pairs
                value_pairs = defaultdict(set)
                
                for _, row in df.iterrows():
                    val1 = str(row[col1])
                    val2 = str(row[col2])
                    value_pairs[val1].add(val2)
                
                # Check for exact mappings (one-to-one)
                for val1, val2_set in value_pairs.items():
                    if len(val2_set) == 1:
                        val2 = next(iter(val2_set))
                        
                        # Count occurrences
                        count = len(df[
                            (df[col1].astype(str) == val1) & 
                            (df[col2].astype(str) == val2)
                        ])
                        
                        mappings.append({
                            "from_column": col1,
                            "from_value": val1,
                            "to_column": col2,
                            "to_value": val2,
                            "confidence": "exact",
                            "support_count": count,
                            "violation_count": 0
                        })
        
        return mappings
    
    def _combine_and_validate_mappings(self, llm_mappings: List, stat_mappings: List, df: pd.DataFrame) -> List[Dict]:
        """Combine and validate all mappings"""
        all_mappings = []
        seen = set()
        
        # Add statistical mappings first (more reliable)
        for mapping in stat_mappings:
            key = f"{mapping['from_column']}={mapping['from_value']}→{mapping['to_column']}"
            if key not in seen:
                all_mappings.append(mapping)
                seen.add(key)
        
        # Add LLM mappings that don't conflict
        for mapping in llm_mappings:
            key = f"{mapping['from_column']}={mapping['from_value']}→{mapping['to_column']}"
            
            if key not in seen:
                # Check if this conflicts with existing mapping
                conflicting = False
                for existing in all_mappings:
                    if (existing["from_column"] == mapping["from_column"] and
                        existing["from_value"] == mapping["from_value"] and
                        existing["to_column"] == mapping["to_column"] and
                        existing["to_value"] != mapping["to_value"]):
                        conflicting = True
                        break
                
                if not conflicting:
                    all_mappings.append(mapping)
                    seen.add(key)
        
        return all_mappings
    
    def _enhance_constraints(self, rules: Dict, df: pd.DataFrame):
        """Enhance constraints with statistical data"""
        for col in df.columns:
            if col not in rules["value_constraints"]:
                # Add auto-detected constraint
                if pd.api.types.is_numeric_dtype(df[col]):
                    rules["value_constraints"][col] = {
                        "type": "numeric",
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std())
                    }
                elif df[col].nunique() < min(20, len(df) * 0.3):
                    rules["value_constraints"][col] = {
                        "type": "categorical",
                        "allowed_values": df[col].dropna().unique().tolist(),
                        "value_counts": df[col].value_counts().to_dict()
                    }
    
    def _add_auto_uniqueness(self, rules: Dict, df: pd.DataFrame):
        """Add auto-detected uniqueness constraints"""
        for col in df.columns:
            # Columns that look like IDs
            if ('id' in col.lower() or 'code' in col.lower() or 
                df[col].nunique() == len(df) or
                df[col].nunique() > len(df) * 0.9):
                if col not in rules["uniqueness_constraints"]:
                    rules["uniqueness_constraints"].append(col)
    
    def _discover_rules_statistically_enhanced(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced statistical rule discovery"""
        st.warning("Using enhanced statistical rule discovery")
        
        rules = {
            "value_mappings": [],
            "value_constraints": {},
            "logical_rules": [],
            "uniqueness_constraints": [],
            "dataset_summary": "Statistically analyzed dataset",
            "analysis_notes": "Rules discovered through statistical analysis"
        }
        
        # Find ALL exact mappings
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Find exact one-to-one mappings
                value_pairs = defaultdict(set)
                
                for _, row in df.iterrows():
                    val1 = str(row[col1])
                    val2 = str(row[col2])
                    value_pairs[val1].add(val2)
                
                # Add exact mappings
                for val1, val2_set in value_pairs.items():
                    if len(val2_set) == 1:
                        val2 = next(iter(val2_set))
                        count = len(df[
                            (df[col1].astype(str) == val1) & 
                            (df[col2].astype(str) == val2)
                        ])
                        
                        rules["value_mappings"].append({
                            "from_column": col1,
                            "from_value": val1,
                            "to_column": col2,
                            "to_value": val2,
                            "confidence": "exact",
                            "support_count": count,
                            "violation_count": 0
                        })
        
        # Add constraints
        self._enhance_constraints(rules, df)
        
        # Add uniqueness
        self._add_auto_uniqueness(rules, df)
        
        return rules

# =============================================================================
# ENHANCED RULE APPLICATION ENGINE
# =============================================================================

class EnhancedRuleApplicationEngine:
    """Enhanced rule application with better mapping enforcement"""
    
    @staticmethod
    def generate_with_rules(
        original_df: pd.DataFrame, 
        rules: Dict[str, Any], 
        num_rows: int
    ) -> pd.DataFrame:
        """
        Generate synthetic data with ENHANCED rule enforcement
        """
        synthetic_rows = []
        
        # Build mapping lookup for fast access
        mapping_lookup = EnhancedRuleApplicationEngine._build_mapping_lookup(rules)
        
        # Track used values for uniqueness
        used_values = {col: set() for col in rules.get("uniqueness_constraints", [])}
        
        for i in range(num_rows):
            # Start with random template
            template_idx = random.randint(0, len(original_df) - 1)
            new_row = original_df.iloc[template_idx].copy()
            
            # Apply ALL mappings with priority
            new_row = EnhancedRuleApplicationEngine._apply_all_mappings(
                new_row, mapping_lookup, rules
            )
            
            # Apply constraints
            new_row = EnhancedRuleApplicationEngine._apply_constraints(
                new_row, rules, original_df, i
            )
            
            # Ensure uniqueness
            new_row = EnhancedRuleApplicationEngine._enforce_uniqueness(
                new_row, rules.get("uniqueness_constraints", []), used_values, i
            )
            
            synthetic_rows.append(new_row)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Final pass: Ensure all mappings are satisfied
        synthetic_df = EnhancedRuleApplicationEngine._final_validation_pass(
            synthetic_df, mapping_lookup, rules
        )
        
        # Ensure data types
        for col in original_df.columns:
            if col in synthetic_df.columns:
                try:
                    synthetic_df[col] = synthetic_df[col].astype(original_df[col].dtype)
                except:
                    pass
        
        return synthetic_df
    
    @staticmethod
    def _build_mapping_lookup(rules: Dict) -> Dict:
        """Build fast lookup for mappings"""
        lookup = {
            "by_from_column": defaultdict(list),
            "exact_mappings": set()
        }
        
        for mapping in rules.get("value_mappings", []):
            from_col = mapping["from_column"]
            to_col = mapping["to_column"]
            from_val = mapping["from_value"]
            to_val = mapping["to_value"]
            confidence = mapping.get("confidence", "partial")
            
            lookup["by_from_column"][from_col].append({
                "from_value": from_val,
                "to_column": to_col,
                "to_value": to_val,
                "confidence": confidence
            })
            
            if confidence == "exact":
                lookup["exact_mappings"].add(f"{from_col}={from_val}→{to_col}")
        
        return lookup
    
    @staticmethod
    def _apply_all_mappings(row: pd.Series, lookup: Dict, rules: Dict) -> pd.Series:
        """Apply ALL mappings to a row"""
        modified_row = row.copy()
        max_passes = 3  # Prevent infinite loops
        
        for _ in range(max_passes):
            changes_made = False
            
            for from_col in lookup["by_from_column"]:
                if from_col not in modified_row:
                    continue
                
                from_val = str(modified_row[from_col])
                
                for mapping in lookup["by_from_column"][from_col]:
                    if mapping["from_value"] == from_val:
                        to_col = mapping["to_column"]
                        to_val = mapping["to_value"]
                        
                        # Check if current value matches
                        if to_col in modified_row and str(modified_row[to_col]) != to_val:
                            # Apply mapping
                            modified_row[to_col] = to_val
                            changes_made = True
            
            if not changes_made:
                break
        
        return modified_row
    
    @staticmethod
    def _apply_constraints(row: pd.Series, rules: Dict, original_df: pd.DataFrame, row_index: int) -> pd.Series:
        """Apply value constraints"""
        modified_row = row.copy()
        
        for col, constraints in rules.get("value_constraints", {}).items():
            if col in modified_row:
                current_val = modified_row[col]
                constraint_type = constraints.get("type", "unknown")
                
                if constraint_type == "categorical":
                    allowed = constraints.get("allowed_values", [])
                    if allowed and str(current_val) not in [str(v) for v in allowed]:
                        # Pick from allowed, prefer values that maintain mappings
                        modified_row[col] = EnhancedRuleApplicationEngine._pick_smart_value(
                            col, current_val, allowed, modified_row, rules
                        )
                
                elif constraint_type == "numeric":
                    min_val = constraints.get("min")
                    max_val = constraints.get("max")
                    
                    if min_val is not None and max_val is not None:
                        try:
                            num_val = float(current_val)
                            if num_val < min_val or num_val > max_val:
                                # Generate within range
                                modified_row[col] = random.uniform(min_val, max_val)
                        except:
                            modified_row[col] = random.uniform(min_val, max_val)
        
        return modified_row
    
    @staticmethod
    def _pick_smart_value(col: str, current_val: Any, allowed: List, row: pd.Series, rules: Dict) -> Any:
        """Pick a value that maintains mappings"""
        # First, check if any mapping depends on this column
        for mapping in rules.get("value_mappings", []):
            if mapping["from_column"] == col and mapping["from_value"] == str(current_val):
                # This value is part of a mapping - keep it if possible
                if str(current_val) in [str(v) for v in allowed]:
                    return current_val
        
        # Otherwise, pick random allowed value
        return random.choice(allowed)
    
    @staticmethod
    def _enforce_uniqueness(row: pd.Series, unique_cols: List[str], used_values: Dict[str, Set], row_index: int) -> pd.Series:
        """Ensure uniqueness for columns"""
        modified_row = row.copy()
        
        for col in unique_cols:
            if col in modified_row:
                current_val = str(modified_row[col])
                
                # Check if value already used
                if current_val in used_values[col]:
                    # Generate unique value
                    base = current_val.split('_')[0] if '_' in current_val else current_val
                    counter = 1
                    
                    while True:
                        new_val = f"{base}_{counter}"
                        if new_val not in used_values[col]:
                            modified_row[col] = new_val
                            used_values[col].add(new_val)
                            break
                        counter += 1
                else:
                    used_values[col].add(current_val)
        
        return modified_row
    
    @staticmethod
    def _final_validation_pass(synthetic_df: pd.DataFrame, lookup: Dict, rules: Dict) -> pd.DataFrame:
        """Final pass to fix any remaining violations"""
        df = synthetic_df.copy()
        
        # Fix exact mapping violations
        for mapping in rules.get("value_mappings", []):
            if mapping.get("confidence") == "exact":
                from_col = mapping["from_column"]
                from_val = mapping["from_value"]
                to_col = mapping["to_column"]
                to_val = mapping["to_value"]
                
                if from_col in df.columns and to_col in df.columns:
                    # Find violations
                    mask = df[from_col].astype(str) == from_val
                    violations = df.loc[mask & (df[to_col].astype(str) != to_val)]
                    
                    # Fix violations
                    if len(violations) > 0:
                        df.loc[mask, to_col] = to_val
        
        return df

# [Keep the RuleValidationEngine and main function the same as before]
# =============================================================================
# RULE VALIDATION ENGINE (same as before)
# =============================================================================

class RuleValidationEngine:
    """Validates that generated data follows all rules"""
    
    @staticmethod
    def validate_rules(synthetic_df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all rules are followed"""
        validation_report = {
            "summary": {
                "total_rows": len(synthetic_df),
                "columns": len(synthetic_df.columns),
                "rules_checked": 0,
                "rules_violated": 0
            },
            "value_mappings_validation": [],
            "constraints_validation": {},
            "logical_rules_validation": [],
            "overall_score": 100.0
        }
        
        # Validate value mappings
        mapping_violations = 0
        for mapping in rules.get("value_mappings", []):
            from_col = mapping["from_column"]
            to_col = mapping["to_column"]
            from_val = mapping["from_value"]
            to_val = mapping["to_value"]
            
            if from_col in synthetic_df.columns and to_col in synthetic_df.columns:
                # Check all rows with from_val have to_val
                matching_rows = synthetic_df[synthetic_df[from_col].astype(str) == str(from_val)]
                violations = matching_rows[matching_rows[to_col].astype(str) != str(to_val)]
                
                validation_report["value_mappings_validation"].append({
                    "mapping": f"{from_col}={from_val} → {to_col}={to_val}",
                    "total_matches": len(matching_rows),
                    "violations": len(violations),
                    "violation_percentage": len(violations) / max(1, len(matching_rows)) * 100
                })
                
                if len(violations) > 0:
                    mapping_violations += 1
        
        # Validate constraints
        for col, constraints in rules.get("value_constraints", {}).items():
            if col in synthetic_df.columns:
                violations = 0
                total = len(synthetic_df)
                
                if constraints.get("type") == "categorical":
                    allowed = constraints.get("allowed_values", [])
                    if allowed:
                        allowed_set = set(str(v) for v in allowed)
                        violations = sum(1 for v in synthetic_df[col] if str(v) not in allowed_set)
                
                elif constraints.get("type") == "numeric":
                    min_val = constraints.get("min")
                    max_val = constraints.get("max")
                    if min_val is not None and max_val is not None:
                        numeric_vals = pd.to_numeric(synthetic_df[col], errors='coerce')
                        violations = ((numeric_vals < min_val) | (numeric_vals > max_val)).sum()
                
                validation_report["constraints_validation"][col] = {
                    "type": constraints.get("type", "unknown"),
                    "total_values": total,
                    "violations": int(violations),
                    "violation_percentage": violations / max(1, total) * 100
                }
        
        # Calculate overall score
        total_rules = (
            len(rules.get("value_mappings", [])) +
            len(rules.get("value_constraints", {})) +
            len(rules.get("logical_rules", []))
        )
        
        total_violations = mapping_violations
        for col_validation in validation_report["constraints_validation"].values():
            if col_validation["violations"] > 0:
                total_violations += 1
        
        validation_report["summary"]["rules_checked"] = total_rules
        validation_report["summary"]["rules_violated"] = total_violations
        
        if total_rules > 0:
            validation_report["overall_score"] = round(
                (1 - total_violations / total_rules) * 100, 1
            )
        
        return validation_report

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="iRMC DataGiene",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("iRMC DataGiene")
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
            
            # Show column stats
            col_info_data = []
            for col in df.columns:
                null_pct = (df[col].isnull().sum() / len(df) * 100).round(1) if len(df) > 0 else 0
                col_info_data.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Unique": df[col].nunique(),
                    "Null %": null_pct
                })
            
            col_info = pd.DataFrame(col_info_data)
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Check for API key
        if "GROQ_API_KEY" not in st.secrets:
            st.error("❌ GROQ_API_KEY not found in secrets.toml")
            st.info("Add this to `.streamlit/secrets.toml`:")
            st.code("GROQ_API_KEY = 'your-api-key-here'")
            st.stop()
        
        # Generation settings
        st.subheader("🎯 Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=len(df) * 10,
                value=len(df) * 2,
                step=50
            )
        
        with col2:
            st.metric("Multiplier", f"{num_rows/len(df):.1f}x")
        
        # Generate button
        if st.button("🚀 Generate with Enhanced Method", type="primary", use_container_width=True):
            if len(df) < 10:
                st.warning("⚠️ Very small dataset - rules may be limited")
            
            # Step 1: Discover Rules (ENHANCED)
            with st.spinner("🔍 LLM + Statistical analysis discovering ALL rules..."):
                try:
                    rule_discoverer = EnhancedRuleDiscoveryEngine(st.secrets["GROQ_API_KEY"])
                    rules = rule_discoverer.discover_rules(df)
                except Exception as e:
                    st.error(f"Failed to discover rules: {e}")
                    return
            
            # Display discovered rules
            st.subheader("📋 Discovered Rules (Enhanced)")
            with st.expander("View Rules Summary", expanded=True):
                # Show mapping summary
                if rules.get("value_mappings"):
                    st.write(f"**🔗 Found {len(rules['value_mappings'])} Value Mappings:**")
                    exact_count = sum(1 for m in rules["value_mappings"] if m.get("confidence") == "exact")
                    st.write(f"- Exact mappings: {exact_count}")
                    st.write(f"- Partial mappings: {len(rules['value_mappings']) - exact_count}")
                
                # Show constraints summary
                if rules.get("value_constraints"):
                    st.write(f"**🎯 Found {len(rules['value_constraints'])} Value Constraints:**")
                
                st.json(rules, expanded=False)
            
            # Step 2: Apply Rules (ENHANCED)
            with st.spinner(f"⚡ Generating {num_rows} rows with ENHANCED rule enforcement..."):
                try:
                    synthetic_df = EnhancedRuleApplicationEngine.generate_with_rules(df, rules, int(num_rows))
                except Exception as e:
                    st.error(f"Failed to generate data: {e}")
                    return
            
            # Step 3: Validate
            with st.spinner("✅ Validating rule compliance..."):
                try:
                    validator = RuleValidationEngine()
                    validation_report = validator.validate_rules(synthetic_df, rules)
                except Exception as e:
                    st.error(f"Failed to validate: {e}")
                    validation_report = {"overall_score": 0, "summary": {"rules_checked": 0}}
            
            # Store results
            st.session_state.synthetic_data = synthetic_df
            st.session_state.original_data = df
            st.session_state.rules = rules
            st.session_state.validation = validation_report
            
            st.balloons()
        
        # Show results if generated
        if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
            synthetic = st.session_state.synthetic_data
            original = st.session_state.original_data
            rules = st.session_state.rules
            validation = st.session_state.validation
            
            st.subheader(f"✨ Generated {len(synthetic)} Rows")
            
            # Quality score
            score = validation.get("overall_score", 0)
            score_color = "🟢" if score >= 95 else "🟡" if score >= 80 else "🔴"
            st.metric("Rule Compliance Score", f"{score_color} {score}%")
            
            # Show exact mapping violations
            if validation.get("value_mappings_validation"):
                violations = [v for v in validation["value_mappings_validation"] if v.get("violations", 0) > 0]
                if violations:
                    st.warning(f"⚠️ {len(violations)} mapping violations detected")
                    
                    with st.expander("View Violations", expanded=True):
                        for v in violations[:10]:  # Show first 10
                            st.error(f"❌ {v['mapping']}: {v['violations']} violations ({v.get('violation_percentage', 0):.1f}%)")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 Rules", "✅ Validation", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
            
            with tab2:
                st.write("**Applied Rules Summary**")
                
                # Show exact mappings that were enforced
                exact_mappings = [m for m in rules.get("value_mappings", []) if m.get("confidence") == "exact"]
                if exact_mappings:
                    st.write(f"**🔗 {len(exact_mappings)} EXACT Mappings Enforced:**")
                    for mapping in exact_mappings[:10]:
                        st.write(f"- {mapping['from_column']}={mapping['from_value']} → {mapping['to_column']}={mapping['to_value']}")
            
            with tab3:
                st.write("**Validation Report**")
                
                # Summary
                summary = validation.get("summary", {})
                st.write(f"- Total rules checked: {summary.get('rules_checked', 0)}")
                st.write(f"- Rules violated: {summary.get('rules_violated', 0)}")
                
                if summary.get('rules_checked', 0) > 0:
                    compliance_rate = 100 - summary.get('rules_violated', 0)/summary.get('rules_checked', 1)*100
                    st.write(f"- Compliance rate: {compliance_rate:.1f}%")
            
            with tab4:
                try:
                    csv = synthetic.to_csv(index=False)
                    st.download_button(
                        "📥 Download Synthetic Data (CSV)",
                        csv,
                        f"enhanced_synthetic_{len(synthetic)}_rows.csv",
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
                
                # Regenerate option
                if st.button("🔄 Generate New Variation"):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome screen
        st.info("""
        """)

if __name__ == "__main__":
    main()
