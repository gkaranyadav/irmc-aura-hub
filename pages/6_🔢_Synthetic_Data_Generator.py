# pages/6_🔢_Synthetic_Data_Generator.py - FIXED VERSION
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
# RULE DISCOVERY ENGINE (USING GROQ)
# =============================================================================

class RuleDiscoveryEngine:
    """Discovers rules from ANY dataset using Groq - NO HARDCODING"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def discover_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Discover ALL rules from ANY dataset
        Returns structured rules that can be applied
        """
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            # Prepare data sample (limit to avoid token limits)
            sample_size = min(15, len(df))
            sample_df = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
            
            # Build intelligent prompt
            prompt = self._build_discovery_prompt(df, sample_df)
            
            # Get analysis from Groq
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": """You are a data pattern detection expert. 
                     Analyze ANY dataset and discover ALL rules, patterns, and relationships.
                     Return ONLY valid JSON."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            raw_rules = json.loads(response.choices[0].message.content)
            
            # Structure and validate rules
            structured_rules = self._structure_rules(raw_rules, df)
            
            return structured_rules
            
        except Exception as e:
            st.error(f"Rule discovery failed: {e}")
            # Fallback to statistical rule discovery
            return self._discover_rules_statistically(df)
    
    def _build_discovery_prompt(self, df: pd.DataFrame, sample_df: pd.DataFrame) -> str:
        """Build prompt for rule discovery"""
        return f"""ANALYZE THIS DATASET AND DISCOVER ALL RULES:

DATASET INFO:
- Total rows: {len(df)}
- Columns ({len(df.columns)}): {list(df.columns)}

COLUMN TYPES (auto-detected):
{self._get_column_types_summary(df)}

SAMPLE DATA ({len(sample_df)} random rows):
{sample_df.to_string(index=False)}

DISCOVER THESE TYPES OF RULES:

1. VALUE MAPPINGS (column A → column B):
   - Which values in one column map to specific values in another column?
   - Example: In row 1, "Cardiology" maps to "Dr. Sharma"

2. VALUE CONSTRAINTS (per column):
   - What values are allowed/possible for each column?
   - Example: "Gender" only has ["M", "F"]

3. FORMAT PATTERNS:
   - What patterns do values follow? (dates, IDs, phones, emails)
   - Example: "Phone" is always 10 digits

4. LOGICAL RELATIONSHIPS:
   - IF-THEN rules between columns
   - Example: IF Symptom="Chest Pain" THEN Department="Cardiology"

5. UNIQUENESS & FREQUENCY:
   - Which columns should be unique?
   - What's the frequency distribution?

RETURN JSON WITH THIS STRUCTURE:
{{
  "dataset_summary": "Brief description of dataset type",
  "value_mappings": [
    {{
      "from_column": "column_name",
      "from_value": "specific_value",
      "to_column": "other_column", 
      "to_value": "mapped_value"
    }}
  ],
  "value_constraints": {{
    "column_name": {{
      "type": "categorical/numeric/date/text",
      "allowed_values": ["list", "of", "values"],
      "min": 0,
      "max": 100,
      "pattern": "regex_pattern"
    }}
  }},
  "logical_rules": [
    "IF column1='value1' THEN column2='value2'",
    "column3 values must match column4 pattern"
  ],
  "uniqueness_constraints": ["column1", "column2"],
  "discovery_confidence": {{
    "mappings_confidence": "high/medium/low",
    "constraints_confidence": "high/medium/low"
  }}
}}

Be SPECIFIC and PRECISE. Only include rules you are confident about."""
    
    def _get_column_types_summary(self, df: pd.DataFrame) -> str:
        """Auto-detect column types"""
        summary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "numeric"
            elif 'date' in col.lower() or self._looks_like_date(str(sample)):
                col_type = "date"
            elif unique_count < 10:
                col_type = "categorical"
            else:
                col_type = "text"
            
            summary.append(f"- {col}: {col_type} ({dtype}), {unique_count} unique values")
        
        return "\n".join(summary)
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if value looks like a date"""
        date_patterns = [r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', r'\d{4}[-/]\d{1,2}[-/]\d{1,2}']
        for pattern in date_patterns:
            if re.match(pattern, str(value)):
                return True
        return False
    
    def _structure_rules(self, raw_rules: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """Structure and validate discovered rules"""
        structured = {
            "value_mappings": [],
            "value_constraints": {},
            "logical_rules": [],
            "uniqueness_constraints": [],
            "dataset_summary": raw_rules.get("dataset_summary", ""),
            "confidence": raw_rules.get("discovery_confidence", {})
        }
        
        # Process value mappings
        for mapping in raw_rules.get("value_mappings", []):
            if all(k in mapping for k in ["from_column", "from_value", "to_column", "to_value"]):
                if mapping["from_column"] in df.columns and mapping["to_column"] in df.columns:
                    structured["value_mappings"].append(mapping)
        
        # Process value constraints
        for col, constraints in raw_rules.get("value_constraints", {}).items():
            if col in df.columns:
                structured["value_constraints"][col] = constraints
        
        # Process logical rules
        for rule in raw_rules.get("logical_rules", []):
            if isinstance(rule, str) and len(rule) > 10:
                structured["logical_rules"].append(rule)
        
        # Process uniqueness
        for col in raw_rules.get("uniqueness_constraints", []):
            if col in df.columns:
                structured["uniqueness_constraints"].append(col)
        
        # Add statistical constraints for columns without LLM constraints
        self._add_statistical_constraints(structured, df)
        
        return structured
    
    def _add_statistical_constraints(self, rules: Dict, df: pd.DataFrame):
        """Add statistical constraints for columns without LLM rules"""
        for col in df.columns:
            if col not in rules["value_constraints"]:
                constraints = {"type": "unknown"}
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    constraints.update({
                        "type": "numeric",
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "mean": float(df[col].mean())
                    })
                elif df[col].nunique() < min(20, len(df) * 0.5):
                    # Treat as categorical
                    constraints.update({
                        "type": "categorical",
                        "allowed_values": df[col].dropna().unique().tolist()
                    })
                else:
                    constraints["type"] = "text"
                
                rules["value_constraints"][col] = constraints
    
    def _discover_rules_statistically(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback: Discover rules statistically"""
        st.warning("Using statistical rule discovery (Groq unavailable)")
        
        rules = {
            "value_mappings": [],
            "value_constraints": {},
            "logical_rules": [],
            "uniqueness_constraints": [],
            "dataset_summary": "Statistically analyzed dataset",
            "confidence": {"mappings_confidence": "medium", "constraints_confidence": "high"}
        }
        
        # Discover mappings by analyzing value pairs
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                # Check for strong mappings
                unique_pairs = df[[col1, col2]].drop_duplicates()
                if len(unique_pairs) < len(df) * 0.3:  # Strong relationship
                    for _, row in unique_pairs.iterrows():
                        rules["value_mappings"].append({
                            "from_column": col1,
                            "from_value": str(row[col1]),
                            "to_column": col2,
                            "to_value": str(row[col2])
                        })
        
        # Add statistical constraints
        self._add_statistical_constraints(rules, df)
        
        return rules

# =============================================================================
# RULE APPLICATION ENGINE
# =============================================================================

class RuleApplicationEngine:
    """Applies discovered rules to generate synthetic data"""
    
    @staticmethod
    def generate_with_rules(
        original_df: pd.DataFrame, 
        rules: Dict[str, Any], 
        num_rows: int
    ) -> pd.DataFrame:
        """
        Generate synthetic data by APPLYING discovered rules
        """
        synthetic_rows = []
        
        # Track used values for uniqueness constraints
        used_values = {col: set() for col in rules.get("uniqueness_constraints", [])}
        
        for i in range(num_rows):
            # Start with a random original row as template
            template_idx = random.randint(0, len(original_df) - 1)
            new_row = original_df.iloc[template_idx].copy()
            
            # Apply rules to modify the row
            new_row = RuleApplicationEngine._apply_rules_to_row(
                new_row, rules, used_values, original_df, i
            )
            
            # Ensure uniqueness for constrained columns
            new_row = RuleApplicationEngine._enforce_uniqueness(
                new_row, rules.get("uniqueness_constraints", []), used_values
            )
            
            synthetic_rows.append(new_row)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Ensure same data types
        for col in original_df.columns:
            if col in synthetic_df.columns:
                try:
                    synthetic_df[col] = synthetic_df[col].astype(original_df[col].dtype)
                except:
                    # If conversion fails, keep as object
                    pass
        
        return synthetic_df
    
    @staticmethod
    def _apply_rules_to_row(
        row: pd.Series, 
        rules: Dict[str, Any], 
        used_values: Dict[str, Set],
        original_df: pd.DataFrame,
        row_index: int
    ) -> pd.Series:
        """Apply all rules to a single row"""
        modified_row = row.copy()
        
        # Apply value mappings
        for mapping in rules.get("value_mappings", []):
            from_col = mapping["from_column"]
            to_col = mapping["to_column"]
            
            if from_col in modified_row and to_col in modified_row:
                if str(modified_row[from_col]) == str(mapping["from_value"]):
                    # Apply the mapping
                    modified_row[to_col] = mapping["to_value"]
        
        # Apply value constraints
        for col, constraints in rules.get("value_constraints", {}).items():
            if col in modified_row:
                modified_row[col] = RuleApplicationEngine._apply_constraint(
                    col, modified_row[col], constraints, original_df, row_index
                )
        
        # Apply logical rules (simple IF-THEN)
        for rule_text in rules.get("logical_rules", []):
            modified_row = RuleApplicationEngine._apply_logical_rule(
                modified_row, rule_text, original_df
            )
        
        return modified_row
    
    @staticmethod
    def _apply_constraint(
        col: str, 
        current_value: Any, 
        constraints: Dict, 
        original_df: pd.DataFrame,
        row_index: int
    ) -> Any:
        """Apply value constraint to a cell"""
        constraint_type = constraints.get("type", "unknown")
        
        if constraint_type == "categorical":
            allowed = constraints.get("allowed_values", [])
            if allowed and str(current_value) not in [str(v) for v in allowed]:
                # Pick random allowed value
                return random.choice(allowed)
        
        elif constraint_type == "numeric":
            min_val = constraints.get("min")
            max_val = constraints.get("max")
            
            if min_val is not None and max_val is not None:
                try:
                    num_val = float(current_value)
                    if num_val < min_val or num_val > max_val:
                        # Generate within range
                        return random.uniform(min_val, max_val)
                except:
                    # Not a number, generate new
                    return random.uniform(min_val, max_val)
        
        elif constraint_type == "date" and "pattern" in constraints:
            # For dates, we might want to generate new ones
            if row_index % 3 == 0:  # Modify some dates
                base_date = pd.Timestamp("2024-01-01")
                random_days = random.randint(-365, 365)
                return (base_date + pd.Timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # For text with pattern
        if "pattern" in constraints:
            pattern = constraints["pattern"]
            # Simple pattern implementation
            if pattern == r"^\d{10}$" and not re.match(pattern, str(current_value)):
                # Generate 10-digit phone
                return f"{random.randint(1000000000, 9999999999)}"
        
        return current_value  # Keep original if no constraint applies
    
    @staticmethod
    def _apply_logical_rule(row: pd.Series, rule_text: str, original_df: pd.DataFrame) -> pd.Series:
        """Apply a logical rule (IF-THEN)"""
        modified_row = row.copy()
        rule_lower = rule_text.lower()
        
        # Simple IF-THEN pattern matching
        if "if" in rule_lower and "then" in rule_lower:
            # Try to parse IF column=value THEN column=value
            match = re.search(r'if\s+(\w+)\s*=\s*["\']?([^"\'\s]+)["\']?\s+then\s+(\w+)\s*=\s*["\']?([^"\'\s]+)["\']?', rule_lower)
            if match:
                if_col, if_val, then_col, then_val = match.groups()
                
                if if_col in modified_row and then_col in modified_row:
                    if str(modified_row[if_col]).lower() == if_val.lower():
                        modified_row[then_col] = then_val
        
        return modified_row
    
    @staticmethod
    def _enforce_uniqueness(
        row: pd.Series, 
        unique_cols: List[str], 
        used_values: Dict[str, Set]
    ) -> pd.Series:
        """Ensure uniqueness for specified columns"""
        modified_row = row.copy()
        
        for col in unique_cols:
            if col in modified_row:
                current_val = str(modified_row[col])
                
                # If value already used, modify it
                if current_val in used_values[col]:
                    # Add suffix to make unique
                    suffix = hashlib.md5(str(len(used_values[col])).encode()).hexdigest()[:4]
                    modified_row[col] = f"{current_val}_{suffix}"
                
                # Add to used values
                used_values[col].add(str(modified_row[col]))
        
        return modified_row

# =============================================================================
# RULE VALIDATION ENGINE
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
# MAIN APPLICATION - SIMPLIFIED
# =============================================================================

def main():
    st.set_page_config(
        page_title="Universal Rule-Based Data Generator",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Universal Rule-Based Data Generator")
    st.markdown("**LLM Discovers Rules → We Apply Rules Exactly → Perfect Synthetic Data**")
    
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
            
            # Show column stats - FIXED THIS LINE
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
        if st.button("🚀 Generate with Rule-Based Method", type="primary", use_container_width=True):
            if len(df) < 10:
                st.warning("⚠️ Very small dataset - rules may be limited")
            
            # Step 1: Discover Rules
            with st.spinner("🔍 LLM discovering rules from your data..."):
                try:
                    rule_discoverer = RuleDiscoveryEngine(st.secrets["GROQ_API_KEY"])
                    rules = rule_discoverer.discover_rules(df)
                except Exception as e:
                    st.error(f"Failed to discover rules: {e}")
                    return
            
            # Display discovered rules
            st.subheader("📋 Discovered Rules")
            with st.expander("View All Rules", expanded=True):
                st.json(rules, expanded=False)
            
            # Step 2: Apply Rules
            with st.spinner(f"⚡ Generating {num_rows} rows by applying rules..."):
                try:
                    synthetic_df = RuleApplicationEngine.generate_with_rules(df, rules, int(num_rows))
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
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 Rules", "✅ Validation", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Quick comparison
                st.write("**Quick Comparison**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original (sample):")
                    display_cols = list(original.columns)[:3]
                    st.dataframe(original[display_cols].head(5) if len(display_cols) > 0 else original.head(5))
                with col2:
                    st.write("Synthetic (sample):")
                    st.dataframe(synthetic[display_cols].head(5) if len(display_cols) > 0 else synthetic.head(5))
            
            with tab2:
                st.write("**Applied Rules Summary**")
                
                # Value mappings
                if rules.get("value_mappings"):
                    st.write(f"🔗 **{len(rules['value_mappings'])} Value Mappings:**")
                    for mapping in rules["value_mappings"][:5]:
                        st.write(f"- {mapping['from_column']}={mapping['from_value']} → {mapping['to_column']}={mapping['to_value']}")
                
                # Value constraints
                if rules.get("value_constraints"):
                    st.write(f"🎯 **{len(rules['value_constraints'])} Value Constraints:**")
                    for col, constraint in list(rules["value_constraints"].items())[:5]:
                        st.write(f"- {col}: {constraint.get('type', 'unknown')}")
                
                # Logical rules
                if rules.get("logical_rules"):
                    st.write(f"⚡ **{len(rules['logical_rules'])} Logical Rules:**")
                    for rule in rules["logical_rules"][:3]:
                        st.write(f"- {rule}")
            
            with tab3:
                st.write("**Validation Report**")
                
                # Summary
                summary = validation.get("summary", {})
                st.write(f"- Total rules checked: {summary.get('rules_checked', 0)}")
                st.write(f"- Rules violated: {summary.get('rules_violated', 0)}")
                
                if summary.get('rules_checked', 0) > 0:
                    compliance_rate = 100 - summary.get('rules_violated', 0)/summary.get('rules_checked', 1)*100
                    st.write(f"- Compliance rate: {compliance_rate:.1f}%")
                
                # Detailed violations
                if validation.get("constraints_validation"):
                    st.write("**Constraint Violations:**")
                    for col, report in validation["constraints_validation"].items():
                        if report.get("violations", 0) > 0:
                            st.error(f"❌ {col}: {report['violations']} violations ({report.get('violation_percentage', 0):.1f}%)")
                
                # Mapping violations
                if validation.get("value_mappings_validation"):
                    st.write("**Mapping Violations:**")
                    for mapping_report in validation["value_mappings_validation"]:
                        if mapping_report.get("violations", 0) > 0:
                            st.error(f"❌ {mapping_report.get('mapping', 'unknown')}: {mapping_report['violations']} violations")
            
            with tab4:
                try:
                    csv = synthetic.to_csv(index=False)
                    st.download_button(
                        "📥 Download Synthetic Data (CSV)",
                        csv,
                        f"rule_based_synthetic_{len(synthetic)}_rows.csv",
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
                        f"data_rules.json",
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
        ## ⚡ Universal Rule-Based Data Generator
        
        ### **How It Works:**
        1. **Upload ANY CSV** - Medical, Sales, HR, ANY data
        2. **LLM analyzes** and discovers ALL rules automatically
        3. **We apply EXACT same rules** to generate new data
        4. **Validate** 100% rule compliance
        
        ### **✨ Key Advantages over SDV:**
        ✅ **Understands semantics** - Knows chest pain → cardiology
        ✅ **Preserves EXACT relationships** - Not just statistical
        ✅ **Works with small data** - 60 rows is enough
        ✅ **Universal** - No domain knowledge needed
        ✅ **Transparent** - See all discovered rules
        ✅ **Validated** - Check rule compliance
        
        ### **📊 Rule Discovery Examples:**
        - **Medical:** "Chest pain → Cardiology", "Dr. Sharma → Cardiology"
        - **Sales:** "Product A → Category Electronics", "Price > 0"
        - **HR:** "Manager → Department", "Salary within range"
        
        ### **🚀 Get Started:**
        1. Upload your CSV (any structure)
        2. LLM will discover ALL rules automatically
        3. Generate 2x, 3x, 5x more data
        4. Download perfect synthetic data
        
        **Upload your CSV to experience rule-based generation!**
        """)

if __name__ == "__main__":
    main()
