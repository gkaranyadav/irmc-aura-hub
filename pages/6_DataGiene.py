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
# CORE INTELLIGENCE ENGINE - DOMAIN AGNOSTIC
# =============================================================================

class IntelligentRelationshipFinder:
    """Universal relationship detector - works for ANY dataset"""
    
    @staticmethod
    def find_potential_relationships(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """Find columns that MIGHT be related in ANY dataset"""
        
        relationships = []
        columns = list(df.columns)
        
        # Look for column pairs that semantically might be related
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                
                # Check based on column name semantics (generic)
                reason = IntelligentRelationshipFinder._get_relationship_reason(col1, col2, df)
                if reason:
                    relationships.append((col1, col2, reason))
        
        return relationships
    
    @staticmethod
    def _get_relationship_reason(col1: str, col2: str, df: pd.DataFrame) -> Optional[str]:
        """Generic reason why columns might be related"""
        
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Common relationship patterns (GENERIC - no domain specifics)
        relationship_patterns = [
            # Demographic patterns
            (['age', 'years', 'birth'], ['salary', 'income', 'revenue', 'price', 'amount'], 
             "demographic and financial often correlate"),
            
            (['gender', 'sex'], ['category', 'type', 'department', 'role'], 
             "gender often relates to categories/departments"),
            
            # Temporal patterns
            (['date', 'time', 'hour'], ['status', 'activity', 'event'], 
             "timing often relates to activities"),
            
            (['date', 'time'], ['amount', 'price', 'quantity'], 
             "temporal patterns in transactions"),
            
            # Hierarchical patterns
            (['category', 'type'], ['subcategory', 'subtype'], 
             "hierarchical relationship"),
            
            (['parent', 'main'], ['child', 'sub'], 
             "parent-child relationship"),
            
            # Business patterns
            (['product', 'item'], ['category', 'type'], 
             "products belong to categories"),
            
            (['customer', 'client'], ['segment', 'tier', 'type'], 
             "customers have segments"),
            
            # Location patterns
            (['city', 'state', 'country'], ['price', 'cost', 'rate'], 
             "geographic price variations"),
            
            # ID patterns
            (['id', 'code', 'number'], ['type', 'category', 'status'], 
             "IDs often encode information"),
        ]
        
        # Check each pattern
        for pattern1, pattern2, reason in relationship_patterns:
            col1_match = any(p in col1_lower for p in pattern1)
            col2_match = any(p in col2_lower for p in pattern2)
            
            if col1_match and col2_match:
                return reason
        
        # Check data-based relationships
        if IntelligentRelationshipFinder._data_suggests_relationship(col1, col2, df):
            return "data shows potential relationship"
        
        return None
    
    @staticmethod
    def _data_suggests_relationship(col1: str, col2: str, df: pd.DataFrame) -> bool:
        """Check if data itself suggests a relationship"""
        
        # Check for functional dependencies
        unique_pairs = df[[col1, col2]].dropna().drop_duplicates()
        col1_unique = df[col1].nunique()
        
        # If each col1 value maps to few col2 values, might be relationship
        if col1_unique > 0:
            avg_mappings = len(unique_pairs) / col1_unique
            if avg_mappings < 2.0:  # Low average mappings per value
                return True
        
        return False

# =============================================================================
# GROQ INTELLIGENCE CONNECTOR
# =============================================================================

class GroqIntelligenceConnector:
    """Connects to Groq for real-world knowledge"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def ask_column_relationship(self, col1: str, col2: str, 
                               sample1: List, sample2: List) -> Dict[str, Any]:
        """Ask Groq about real-world column relationships"""
        
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            prompt = self._build_relationship_prompt(col1, col2, sample1, sample2)
            
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data relationship expert. 
                        Analyze column relationships GENERICALLY without assuming specific domains.
                        Give insights about how such columns TYPICALLY relate in real-world datasets."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return self._parse_groq_response(answer)
            
        except Exception as e:
            return {
                "error": str(e),
                "relationship_strength": "unknown",
                "advice": "Assume independent columns"
            }
    
    def _build_relationship_prompt(self, col1: str, col2: str, 
                                  sample1: List, sample2: List) -> str:
        """Build intelligent prompt for ANY dataset"""
        
        return f"""I'm analyzing a dataset with these two columns:

Column 1: "{col1}"
Sample values: {sample1[:10]}

Column 2: "{col2}"
Sample values: {sample2[:10]}

Please analyze GENERICALLY (don't assume specific domain):

1. In typical real-world datasets, do columns with these NAMES usually have relationships?
2. What KIND of relationship might exist (if any)?
3. Should I be careful about certain value combinations when generating synthetic data?
4. Are there common patterns or constraints I should respect?

Please respond in this JSON format:
{{
    "likely_related": true/false,
    "relationship_type": "string describing relationship type",
    "typical_patterns": ["list of typical patterns"],
    "generation_advice": "advice for synthetic data generation",
    "confidence": "high/medium/low"
}}"""
    
    def _parse_groq_response(self, response: str) -> Dict[str, Any]:
        """Parse Groq response into structured data"""
        
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        result = {
            "likely_related": "unknown" in response.lower(),
            "relationship_type": "extracted from response",
            "generation_advice": response[:200],
            "confidence": "medium"
        }
        
        # Extract key insights
        if "usually related" in response.lower() or "typically related" in response.lower():
            result["likely_related"] = True
        if "independent" in response.lower() or "no relationship" in response.lower():
            result["likely_related"] = False
        
        return result

# =============================================================================
# INTELLIGENT CONSTRAINT BUILDER
# =============================================================================

class IntelligentConstraintBuilder:
    """Builds generation constraints using intelligence"""
    
    def __init__(self, groq_connector: GroqIntelligenceConnector):
        self.groq_connector = groq_connector
        self.constraints = {}
        self.learned_relationships = {}
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset and build intelligent constraints"""
        
        # Step 1: Find potential relationships
        relationships = IntelligentRelationshipFinder.find_potential_relationships(df)
        
        # Step 2: Analyze each relationship
        for col1, col2, reason in relationships[:5]:  # Limit to top 5 for speed
            self._analyze_relationship(col1, col2, df)
        
        # Step 3: Build statistical constraints
        self._build_statistical_constraints(df)
        
        # Step 4: Build generation rules
        generation_rules = self._build_generation_rules(df)
        
        return {
            "discovered_relationships": self.learned_relationships,
            "constraints": self.constraints,
            "generation_rules": generation_rules,
            "analysis_summary": self._create_summary(df)
        }
    
    def _analyze_relationship(self, col1: str, col2: str, df: pd.DataFrame):
        """Analyze a specific column relationship"""
        
        # Get sample values
        sample1 = df[col1].dropna().astype(str).unique().tolist()[:10]
        sample2 = df[col2].dropna().astype(str).unique().tolist()[:10]
        
        if len(sample1) == 0 or len(sample2) == 0:
            return
        
        # Check if data shows clear pattern
        data_pattern = self._analyze_data_pattern(col1, col2, df)
        
        # Ask Groq for real-world knowledge
        groq_insights = self.groq_connector.ask_column_relationship(
            col1, col2, sample1, sample2
        )
        
        # Store learned relationship
        self.learned_relationships[f"{col1}-{col2}"] = {
            "data_pattern": data_pattern,
            "groq_insights": groq_insights,
            "generation_rule": self._create_generation_rule(data_pattern, groq_insights)
        }
    
    def _analyze_data_pattern(self, col1: str, col2: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze pattern in the data itself"""
        
        pattern = {
            "strength": "weak",
            "pattern_type": "unknown",
            "confidence": "low"
        }
        
        try:
            # Check for categorical relationships
            if df[col1].nunique() < 20 and df[col2].nunique() < 20:
                # Create contingency table
                contingency = pd.crosstab(df[col1], df[col2])
                
                # Check if patterns exist
                if contingency.size > 0:
                    # Calculate pattern strength
                    max_per_row = contingency.max(axis=1)
                    total_per_row = contingency.sum(axis=1)
                    strength_scores = max_per_row / total_per_row
                    
                    avg_strength = strength_scores.mean()
                    if avg_strength > 0.8:
                        pattern["strength"] = "strong"
                        pattern["confidence"] = "high"
                    elif avg_strength > 0.6:
                        pattern["strength"] = "medium"
                        pattern["confidence"] = "medium"
                    
                    pattern["pattern_type"] = "categorical_mapping"
            
            # Check for numeric correlations
            elif pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                correlation = df[col1].corr(df[col2])
                if not pd.isna(correlation):
                    pattern["strength"] = "numeric_correlation"
                    pattern["correlation"] = round(correlation, 3)
                    if abs(correlation) > 0.7:
                        pattern["confidence"] = "high"
                    elif abs(correlation) > 0.4:
                        pattern["confidence"] = "medium"
        
        except:
            pass
        
        return pattern
    
    def _create_generation_rule(self, data_pattern: Dict, groq_insights: Dict) -> str:
        """Create generation rule from data and Groq insights"""
        
        # If Groq says columns are related
        if groq_insights.get("likely_related"):
            advice = groq_insights.get("generation_advice", "")
            
            if "strong" in data_pattern.get("strength", "").lower():
                return f"Maintain strong relationship between columns"
            elif "respect" in advice.lower() or "careful" in advice.lower():
                return f"Respect typical relationship patterns"
            else:
                return f"Consider relationship when generating"
        
        # If data shows strong pattern but Groq unsure
        elif data_pattern.get("confidence") == "high":
            return f"Maintain observed data pattern"
        
        # Default
        return "Generate independently"
    
    def _build_statistical_constraints(self, df: pd.DataFrame):
        """Build basic statistical constraints for each column"""
        
        for col in df.columns:
            col_constraints = {}
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric constraints
                col_constraints["type"] = "numeric"
                col_constraints["min"] = float(df[col].min())
                col_constraints["max"] = float(df[col].max())
                col_constraints["mean"] = float(df[col].mean())
                col_constraints["std"] = float(df[col].std())
            
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Datetime constraints
                col_constraints["type"] = "datetime"
                col_constraints["min_date"] = df[col].min().isoformat()
                col_constraints["max_date"] = df[col].max().isoformat()
            
            else:
                # Categorical constraints
                col_constraints["type"] = "categorical"
                col_constraints["unique_values"] = df[col].nunique()
                col_constraints["most_common"] = df[col].mode().iloc[0] if not df[col].mode().empty else None
            
            self.constraints[col] = col_constraints
    
    def _build_generation_rules(self, df: pd.DataFrame) -> List[Dict]:
        """Build intelligent generation rules"""
        
        rules = []
        
        # Rule 1: Maintain column types
        rules.append({
            "rule": "maintain_data_types",
            "description": "Keep same data types as original",
            "priority": "high"
        })
        
        # Rule 2: Respect discovered relationships
        for rel_key, rel_info in self.learned_relationships.items():
            if rel_info["generation_rule"] != "Generate independently":
                col1, col2 = rel_key.split("-")
                rules.append({
                    "rule": f"respect_relationship_{col1}_{col2}",
                    "columns": [col1, col2],
                    "description": rel_info["generation_rule"],
                    "priority": "medium",
                    "details": rel_info
                })
        
        # Rule 3: Maintain statistical distributions
        rules.append({
            "rule": "maintain_distributions",
            "description": "Keep similar value distributions",
            "priority": "high"
        })
        
        return rules
    
    def _create_summary(self, df: pd.DataFrame) -> Dict:
        """Create analysis summary"""
        
        return {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "numeric_columns": sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col])),
            "categorical_columns": sum(1 for col in df.columns if df[col].nunique() < 20),
            "relationships_found": len(self.learned_relationships),
            "strong_relationships": sum(1 for rel in self.learned_relationships.values() 
                                      if rel["data_pattern"].get("confidence") == "high")
        }

# =============================================================================
# INTELLIGENT SYNTHETIC GENERATOR
# =============================================================================

class IntelligentSyntheticGenerator:
    """Generates synthetic data using intelligence"""
    
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        self.generated_values = defaultdict(set)
    
    def generate(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate intelligent synthetic data"""
        
        synthetic_rows = []
        rules = self.constraints.get("generation_rules", [])
        relationships = self.constraints.get("discovered_relationships", {})
        
        for row_idx in range(num_rows):
            row = {}
            
            # Generate each column intelligently
            for col in df.columns:
                row[col] = self._generate_column_value(col, df, row, relationships)
            
            # Apply relationship rules
            row = self._apply_relationship_rules(row, relationships)
            
            # Ensure uniqueness where needed
            row = self._ensure_uniqueness(row, df)
            
            synthetic_rows.append(row)
            
            # Progress update
            if row_idx % 100 == 0 and row_idx > 0:
                st.info(f"Generated {row_idx} of {num_rows} rows...")
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        # Preserve data types
        synthetic_df = self._preserve_data_types(synthetic_df, df)
        
        return synthetic_df
    
    def _generate_column_value(self, col: str, df: pd.DataFrame, 
                              current_row: Dict, relationships: Dict) -> Any:
        """Generate value for a column considering relationships"""
        
        # Check if this column is related to already generated columns
        related_value = self._get_related_value(col, current_row, relationships)
        if related_value is not None:
            return related_value
        
        # Get column constraints
        col_constraints = self.constraints.get("constraints", {}).get(col, {})
        col_type = col_constraints.get("type", "unknown")
        
        # Generate based on type
        if col_type == "numeric":
            return self._generate_numeric(col, df, col_constraints)
        elif col_type == "datetime":
            return self._generate_datetime(col, df, col_constraints)
        else:
            return self._generate_categorical(col, df, col_constraints)
    
    def _get_related_value(self, col: str, current_row: Dict, relationships: Dict) -> Optional[Any]:
        """Get value based on relationships with already generated columns"""
        
        for rel_key, rel_info in relationships.items():
            col1, col2 = rel_key.split("-")
            
            # If col2 is our target and col1 is already generated
            if col2 == col and col1 in current_row:
                rule = rel_info.get("generation_rule", "")
                
                # If strong relationship, derive value
                if "strong" in rule.lower() or "maintain" in rule.lower():
                    # Simple logic: sample from original conditional distribution
                    original_val = current_row[col1]
                    matching_rows = df[df[col1] == original_val]
                    if len(matching_rows) > 0:
                        return matching_rows[col].sample(1).iloc[0]
        
        return None
    
    def _generate_numeric(self, col: str, df: pd.DataFrame, constraints: Dict) -> float:
        """Generate numeric value"""
        
        min_val = constraints.get("min", 0)
        max_val = constraints.get("max", 100)
        mean_val = constraints.get("mean", (min_val + max_val) / 2)
        std_val = constraints.get("std", (max_val - min_val) / 4)
        
        # Generate with normal distribution, clipped to range
        value = np.random.normal(mean_val, std_val)
        value = np.clip(value, min_val, max_val)
        
        return float(value)
    
    def _generate_datetime(self, col: str, df: pd.DataFrame, constraints: Dict):
        """Generate datetime value"""
        
        min_date_str = constraints.get("min_date")
        max_date_str = constraints.get("max_date")
        
        if min_date_str and max_date_str:
            min_date = pd.to_datetime(min_date_str)
            max_date = pd.to_datetime(max_date_str)
            
            # Random date within range
            delta = max_date - min_date
            random_days = random.randint(0, delta.days)
            return min_date + timedelta(days=random_days)
        else:
            return df[col].sample(1).iloc[0]
    
    def _generate_categorical(self, col: str, df: pd.DataFrame, constraints: Dict):
        """Generate categorical value"""
        
        # Get value distribution from original
        value_counts = df[col].value_counts(normalize=True)
        
        if len(value_counts) > 0:
            # Weighted sampling
            values = value_counts.index.tolist()
            weights = value_counts.values.tolist()
            return np.random.choice(values, p=weights)
        else:
            return df[col].sample(1).iloc[0]
    
    def _apply_relationship_rules(self, row: Dict, relationships: Dict) -> Dict:
        """Apply relationship rules to generated row"""
        
        for rel_key, rel_info in relationships.items():
            rule = rel_info.get("generation_rule", "")
            
            if "respect" in rule.lower() or "careful" in rule.lower():
                col1, col2 = rel_key.split("-")
                
                # If both columns generated, check if combination makes sense
                if col1 in row and col2 in row:
                    # Simple check: does this combination exist in original data?
                    combination_exists = not df[
                        (df[col1] == row[col1]) & 
                        (df[col2] == row[col2])
                    ].empty
                    
                    if not combination_exists:
                        # Replace col2 with compatible value
                        compatible_values = df[df[col1] == row[col1]][col2].unique()
                        if len(compatible_values) > 0:
                            row[col2] = random.choice(list(compatible_values))
        
        return row
    
    def _ensure_uniqueness(self, row: Dict, df: pd.DataFrame) -> Dict:
        """Ensure uniqueness for ID-like columns"""
        
        for col in df.columns:
            col_lower = col.lower()
            if ('id' in col_lower or 'code' in col_lower) and col in row:
                value = row[col]
                if value in self.generated_values[col]:
                    # Make unique
                    base = str(value)
                    suffix = 1
                    while f"{base}_{suffix}" in self.generated_values[col]:
                        suffix += 1
                    new_value = f"{base}_{suffix}"
                    row[col] = new_value
                    self.generated_values[col].add(new_value)
                else:
                    self.generated_values[col].add(value)
        
        return row
    
    def _preserve_data_types(self, synthetic_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Preserve original data types"""
        
        for col in original_df.columns:
            if col in synthetic_df.columns:
                try:
                    original_dtype = original_df[col].dtype
                    synthetic_df[col] = synthetic_df[col].astype(original_dtype)
                except:
                    # If conversion fails, try best effort
                    try:
                        if pd.api.types.is_numeric_dtype(original_dtype):
                            synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce')
                        elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                            synthetic_df[col] = pd.to_datetime(synthetic_df[col], errors='coerce')
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
        
        # Calculate overall score
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
        
        # Calculate score
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
                        # Compare means
                        mean_diff = abs(orig_series.mean() - synth_series.mean()) / max(1, abs(orig_series.mean()))
                        std_diff = abs(orig_series.std() - synth_series.std()) / max(1, abs(orig_series.std()))
                        similarity = 100 * (1 - (mean_diff + std_diff) / 2)
                    else:
                        # Compare distributions
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
        
        relationships = constraints.get("discovered_relationships", {})
        rules = constraints.get("generation_rules", [])
        
        compliance = {
            "relationships_checked": len(relationships),
            "rules_checked": len(rules),
            "violations": []
        }
        
        # Check each relationship
        for rel_key, rel_info in relationships.items():
            col1, col2 = rel_key.split("-")
            
            if col1 in synthetic.columns and col2 in synthetic.columns:
                rule = rel_info.get("generation_rule", "")
                
                if "maintain" in rule.lower() or "respect" in rule.lower():
                    # Check if unusual combinations exist
                    value_pairs = synthetic[[col1, col2]].dropna().drop_duplicates()
                    
                    # Simple check: count unique pairs vs unique values
                    if value_pairs.shape[0] > synthetic[col1].nunique() * 2:
                        compliance["violations"].append(
                            f"Many unique combinations for {col1}-{col2}, might violate relationship"
                        )
        
        # Calculate compliance score
        total_checks = compliance["relationships_checked"] + compliance["rules_checked"]
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
        
        # Quality score
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
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="intelligent-header">🧠 Intelligent Synthetic Data Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Works with **ANY** Dataset • Uses **Groq Intelligence** • No Hardcoded Rules
    
    **🤔 Relationship Detective → 🔍 Groq Intelligence → 🛠️ Constraint Builder → 🎨 Smart Generator → ✅ Quality Validator**
    """)
    
    # Check API key in secrets
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
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=10000,
                value=min(len(df) * 3, 1000),
                step=100
            )
        
        with col2:
            intelligence_level = st.selectbox(
                "Intelligence Level",
                ["High (Use Groq for relationships)", "Medium (Statistical only)", "Low (Basic generation)"],
                help="High level uses Groq to understand column relationships"
            )
        
        # Show agents
        st.subheader("👥 Intelligent Agents Ready")
        
        agents = [
            {"emoji": "🤔", "name": "Relationship Detective", "role": "Finds column relationships"},
            {"emoji": "🔍", "name": "Groq Intelligence", "role": "Asks for real-world knowledge"},
            {"emoji": "🛠️", "name": "Constraint Builder", "role": "Creates generation rules"},
            {"emoji": "🎨", "name": "Smart Generator", "role": "Generates intelligent data"},
            {"emoji": "✅", "name": "Quality Validator", "role": "Validates everything"}
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
            
            with st.spinner("🧠 Analyzing dataset with intelligence..."):
                try:
                    # Initialize components
                    groq_connector = GroqIntelligenceConnector(groq_api_key)
                    
                    # Build constraints
                    constraint_builder = IntelligentConstraintBuilder(groq_connector)
                    constraints = constraint_builder.analyze_dataset(df)
                    
                    # Show analysis
                    with st.expander("📊 Intelligence Analysis Results", expanded=True):
                        
                        # Show discovered relationships
                        relationships = constraints.get("discovered_relationships", {})
                        if relationships:
                            st.write("**🔗 Discovered Relationships:**")
                            for rel_key, rel_info in relationships.items():
                                col1, col2 = rel_key.split("-")
                                with st.container():
                                    st.markdown(f"""
                                    <div class="relationship-card">
                                        <strong>{col1} ↔ {col2}</strong><br>
                                        Data Pattern: {rel_info.get('data_pattern', {}).get('strength', 'unknown')}<br>
                                        Generation Rule: {rel_info.get('generation_rule', 'Unknown')}
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No strong relationships detected between columns")
                        
                        # Show summary
                        summary = constraints.get("analysis_summary", {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Columns", summary.get("total_columns", 0))
                        with col2:
                            st.metric("Relationships Found", summary.get("relationships_found", 0))
                        with col3:
                            st.metric("Strong Relationships", summary.get("strong_relationships", 0))
                    
                    # Generate data
                    st.info("🎨 Generating synthetic data with intelligence...")
                    generator = IntelligentSyntheticGenerator(constraints)
                    synthetic_df = generator.generate(df, num_rows)
                    
                    # Validate
                    st.info("✅ Validating synthetic data quality...")
                    validator = IntelligentQualityValidator()
                    validation = validator.validate(df, synthetic_df, constraints)
                    
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
                
                relationships = constraints.get("discovered_relationships", {})
                if relationships:
                    st.write("**Intelligent Relationship Detection:**")
                    for rel_key, rel_info in relationships.items():
                        col1, col2 = rel_key.split("-")
                        
                        with st.expander(f"{col1} ↔ {col2}"):
                            st.write("**Data Pattern Found:**")
                            st.json(rel_info.get("data_pattern", {}))
                            
                            st.write("**Groq Intelligence Insights:**")
                            groq_insights = rel_info.get("groq_insights", {})
                            if "error" not in groq_insights:
                                st.write(f"Likely Related: {groq_insights.get('likely_related', 'Unknown')}")
                                st.write(f"Advice: {groq_insights.get('generation_advice', 'No advice')}")
                            else:
                                st.warning("Could not get Groq insights")
                            
                            st.write("**Generation Rule Applied:**")
                            st.code(rel_info.get("generation_rule", "No rule"))
                else:
                    st.info("No strong relationships detected. AI generated data statistically.")
            
            with tab3:
                # Validation details
                st.write("### ✅ Validation Report")
                
                # Issues
                basic_checks = validation.get("basic_checks", {})
                if basic_checks.get("issues"):
                    st.write("**Issues Found:**")
                    for issue in basic_checks["issues"]:
                        st.write(f"⚠️ {issue}")
                
                # Relationship violations
                relationship_compliance = validation.get("relationship_compliance", {})
                if relationship_compliance.get("violations"):
                    st.write("**Relationship Violations:**")
                    for violation in relationship_compliance["violations"]:
                        st.write(f"❌ {violation}")
                
                # Column similarities
                statistical = validation.get("statistical_comparison", {})
                if statistical.get("column_comparisons"):
                    st.write("**Column Similarities (Top 10):**")
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
                
                # Download analysis
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
        ### 🎯 **How It Works (Intelligently):**
        
        1. **🤔 Relationship Detection** - AI analyzes column names and data to find potential relationships
        2. **🔍 Groq Intelligence** - When unsure, asks Groq about real-world column relationships
        3. **🛠️ Constraint Building** - Creates intelligent generation rules
        4. **🎨 Smart Generation** - Generates data that respects discovered relationships
        5. **✅ Quality Validation** - Validates everything makes sense
        
        ### 🌟 **Key Features:**
        - **Zero hardcoding** - No domain-specific rules
        - **Universal intelligence** - Works with ANY dataset
        - **Real-world knowledge** - Uses Groq when confused
        - **Transparent reasoning** - Shows you how it thinks
        - **Quality guaranteed** - Multiple validation layers
        """)
        
        # Example
        with st.expander("📚 Example Intelligence Process"):
            st.write("**Dataset:** Hospital records with columns: `Department`, `Gender`, `Age`")
            
            st.write("""
            1. **AI thinks:** "Department and Gender might be related..."
            2. **Checks data:** "Some departments have mixed genders"
            3. **Asks Groq:** "In real datasets, do Department and Gender columns typically have relationships?"
            4. **Groq responds:** "Yes, certain departments like Gynecology typically serve specific genders"
            5. **AI creates rule:** "Respect typical department-gender relationships when generating"
            6. **Generates:** Gynecology patients as mostly female, Urology as mostly male
            """)

if __name__ == "__main__":
    main()
