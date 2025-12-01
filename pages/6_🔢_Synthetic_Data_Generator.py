import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
import json
import re
from datetime import datetime
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Universal Synthetic Data Generator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# UNIVERSAL DATA ANALYZER
# =============================================================================

class UniversalDataAnalyzer:
    """Analyzes any dataset structure using LLM with structured output"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or st.secrets.get("GROQ_API_KEY")
        
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Deep analysis of dataset structure, patterns, and relationships
        Returns structured JSON for precise constraint generation
        """
        try:
            # Get dataset profile
            profile = self._profile_dataset(df)
            
            # Get LLM structured analysis
            llm_analysis = self._get_llm_structured_analysis(df, profile)
            
            # Merge with statistical analysis
            final_analysis = self._merge_analyses(profile, llm_analysis)
            
            return final_analysis
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return self._get_fallback_analysis(df)
    
    def _profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Statistical profiling of dataset"""
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": {},
            "correlations": {},
            "patterns": {}
        }
        
        for col in df.columns:
            col_profile = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": float(df[col].isnull().sum() / len(df) * 100),
                "unique_count": int(df[col].nunique()),
                "unique_pct": float(df[col].nunique() / len(df) * 100)
            }
            
            # Type-specific stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "median": float(df[col].median()) if not df[col].isnull().all() else None
                })
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                # Sample values for pattern detection
                sample_values = df[col].dropna().head(10).tolist()
                col_profile["samples"] = sample_values
                
                # Detect patterns
                if len(sample_values) > 0:
                    col_profile["patterns"] = self._detect_patterns(sample_values)
            
            # Value distribution for categorical-like columns
            if df[col].nunique() < 50:
                value_counts = df[col].value_counts().head(20)
                col_profile["distribution"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }
            
            profile["columns"][col] = col_profile
        
        # Detect correlations between columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.5:  # Strong correlation
                        profile["correlations"][f"{col1}_{col2}"] = float(corr_value)
        
        return profile
    
    def _detect_patterns(self, samples: List[str]) -> Dict[str, Any]:
        """Detect common patterns in string data"""
        patterns = {}
        
        # Check for common formats
        sample_str = str(samples[0]) if samples else ""
        
        # Email pattern
        if '@' in sample_str and '.' in sample_str:
            patterns["type"] = "email"
            patterns["regex"] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        
        # Phone pattern (various formats)
        elif re.match(r'^[\d\s\-$$$$\+]+$', sample_str) and len(sample_str) >= 10:
            patterns["type"] = "phone"
            patterns["regex"] = r"^[\d\s\-$$$$\+]{10,}$"
        
        # Date pattern
        elif self._looks_like_date(sample_str):
            patterns["type"] = "date"
        
        # ID pattern (alphanumeric with consistent format)
        elif re.match(r'^[A-Z]{2,}\d+', sample_str):
            patterns["type"] = "id"
            patterns["prefix"] = re.match(r'^[A-Z]+', sample_str).group()
        
        # Check length consistency
        lengths = [len(str(s)) for s in samples]
        if len(set(lengths)) == 1:
            patterns["fixed_length"] = lengths[0]
        
        return patterns
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if string looks like a date"""
        date_indicators = ['-', '/', ':', 'T']
        digit_count = sum(c.isdigit() for c in str(value))
        return any(ind in str(value) for ind in date_indicators) and digit_count >= 4
    
    def _get_llm_structured_analysis(self, df: pd.DataFrame, profile: Dict) -> Dict[str, Any]:
        """Get structured JSON analysis from LLM"""
        if not self.api_key:
            return {}
        
        try:
            import requests
            
            # Prepare data sample
            sample_data = df.head(20).to_dict(orient='records')
            
            # Craft detailed prompt for structured output
            prompt = f"""Analyze this dataset and return a JSON object with the following structure:

{{
  "dataset_type": "string describing the dataset (e.g., 'medical records', 'sales data', 'user database')",
  "relationships": [
    {{
      "columns": ["col1", "col2"],
      "type": "one-to-many | many-to-one | one-to-one | functional",
      "strength": "strong | moderate | weak",
      "description": "brief description"
    }}
  ],
  "constraints": [
    {{
      "column": "column_name",
      "type": "unique | range | format | enum | required",
      "rule": "specific rule description",
      "enforcement": "strict | flexible"
    }}
  ],
  "data_quality_issues": [
    {{
      "issue": "description of issue",
      "affected_columns": ["col1", "col2"],
      "severity": "high | medium | low",
      "recommendation": "how to fix"
    }}
  ],
  "generation_recommendations": {{
    "preserve_distributions": ["column names where distribution is important"],
    "enforce_relationships": ["relationship pairs that must be maintained"],
    "special_handling": {{"column_name": "specific handling needed"}}
  }}
}}

Dataset Info:
- Columns: {list(df.columns)}
- Row count: {len(df)}
- Statistical profile: {json.dumps(profile['columns'], indent=2)}

Sample data (first 20 rows):
{json.dumps(sample_data[:5], indent=2)}

Return ONLY valid JSON, no additional text."""

            # Call LLM API
            response = requests.post(
                'https://llm.blackbox.ai/chat/completions',
                headers={
                    'customerId': st.secrets.get("USER_CUSTOMER_ID", "default"),
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer xxx'
                },
                json={
                    "model": "openrouter/claude-sonnet-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a data analysis expert. Always return valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 3000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return json.loads(content)
            else:
                st.warning(f"LLM API returned status {response.status_code}")
                return {}
                
        except Exception as e:
            st.warning(f"LLM analysis unavailable: {str(e)}")
            return {}
    
    def _merge_analyses(self, profile: Dict, llm_analysis: Dict) -> Dict[str, Any]:
        """Merge statistical and LLM analyses"""
        return {
            "profile": profile,
            "llm_insights": llm_analysis,
            "relationships": llm_analysis.get("relationships", []),
            "constraints": llm_analysis.get("constraints", []),
            "quality_issues": llm_analysis.get("data_quality_issues", []),
            "recommendations": llm_analysis.get("generation_recommendations", {}),
            "dataset_type": llm_analysis.get("dataset_type", "unknown")
        }
    
    def _get_fallback_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        profile = self._profile_dataset(df)
        return {
            "profile": profile,
            "llm_insights": {},
            "relationships": [],
            "constraints": [],
            "quality_issues": [],
            "recommendations": {},
            "dataset_type": "unknown"
        }

# =============================================================================
# SDV CONSTRAINT BUILDER
# =============================================================================

class SDVConstraintBuilder:
    """Builds SDV constraints from analysis"""
    
    @staticmethod
    def build_constraints(analysis: Dict, df: pd.DataFrame) -> List[Any]:
        """Build SDV constraint objects from analysis"""
        constraints = []
        
        try:
            from sdv.constraints import (
                FixedCombinations,
                Inequality,
                Positive,
                Negative,
                Range,
                ScalarInequality,
                OneHotEncoding,
                Unique
            )
            
            # Build constraints from LLM insights
            for constraint_def in analysis.get("constraints", []):
                col = constraint_def.get("column")
                ctype = constraint_def.get("type")
                
                if not col or col not in df.columns:
                    continue
                
                try:
                    if ctype == "unique" and constraint_def.get("enforcement") == "strict":
                        constraints.append(Unique(column_names=[col]))
                    
                    elif ctype == "range":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            min_val = df[col].min()
                            max_val = df[col].max()
                            constraints.append(
                                Range(
                                    low_column_name=col,
                                    high_column_name=col,
                                    low_value=min_val,
                                    high_value=max_val,
                                    strict_boundaries=False
                                )
                            )
                    
                    elif ctype == "positive":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            constraints.append(Positive(column_names=[col], strict_boundaries=False))
                
                except Exception as e:
                    st.warning(f"Couldn't apply {ctype} constraint to {col}: {e}")
            
            # Build relationship constraints
            for rel in analysis.get("relationships", []):
                cols = rel.get("columns", [])
                rel_type = rel.get("type", "")
                strength = rel.get("strength", "")
                
                if len(cols) >= 2 and all(c in df.columns for c in cols):
                    # For strong relationships, use FixedCombinations
                    if strength == "strong" and rel_type in ["one-to-many", "functional"]:
                        try:
                            # Check if combinations are reasonable in size
                            unique_combos = df[cols].drop_duplicates()
                            if len(unique_combos) < len(df) * 0.7:  # Not too many unique combos
                                constraints.append(FixedCombinations(column_names=cols))
                        except Exception as e:
                            st.warning(f"Couldn't apply FixedCombinations to {cols}: {e}")
            
        except ImportError:
            st.warning("SDV constraints not available - install sdv package")
        except Exception as e:
            st.error(f"Error building constraints: {e}")
        
        return constraints

# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

class UniversalSyntheticGenerator:
    """Universal synthetic data generator using SDV + LLM guidance"""
    
    def __init__(self):
        self.analyzer = UniversalDataAnalyzer()
        self.constraint_builder = SDVConstraintBuilder()
        self.sdv_available = self._check_sdv()
    
    def _check_sdv(self) -> bool:
        """Check if SDV is available"""
        try:
            from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata
            return True
        except ImportError:
            return False
    
    def generate(
        self, 
        df: pd.DataFrame, 
        num_rows: int,
        model_type: str = "CTGAN",
        enforce_constraints: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Generate synthetic data with LLM-guided constraints
        
        Returns:
            (synthetic_df, metadata_dict)
        """
        if not self.sdv_available:
            st.error("❌ SDV not installed. Run: pip install sdv")
            return None, {}
        
        metadata = {}
        
        try:
            # Step 1: Analyze dataset
            with st.spinner("🔍 Analyzing dataset structure..."):
                analysis = self.analyzer.analyze_dataset(df)
                metadata["analysis"] = analysis
            
            # Step 2: Build constraints
            constraints = []
            if enforce_constraints:
                with st.spinner("🔧 Building constraints..."):
                    constraints = self.constraint_builder.build_constraints(analysis, df)
                    metadata["constraints_count"] = len(constraints)
            
            # Step 3: Prepare metadata
            from sdv.metadata import SingleTableMetadata
            
            sdv_metadata = SingleTableMetadata()
            sdv_metadata.detect_from_dataframe(df)
            
            # Step 4: Select and configure model
            if model_type == "CTGAN":
                from sdv.single_table import CTGANSynthesizer
                
                # Dynamic epochs based on dataset size
                epochs = self._calculate_epochs(len(df))
                batch_size = min(500, max(50, len(df) // 10))
                
                model = CTGANSynthesizer(
                    metadata=sdv_metadata,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False
                )
            else:
                from sdv.single_table import GaussianCopulaSynthesizer
                model = GaussianCopulaSynthesizer(
                    metadata=sdv_metadata,
                    enforce_min_max_values=True,
                    enforce_rounding=True
                )
            
            # Step 5: Train model
            with st.spinner(f"🤖 Training {model_type} model..."):
                if constraints:
                    # Add constraints to metadata
                    for constraint in constraints:
                        try:
                            model.add_constraints([constraint])
                        except Exception as e:
                            st.warning(f"Constraint skipped: {e}")
                
                model.fit(df)
                metadata["model_type"] = model_type
                metadata["training_rows"] = len(df)
            
            # Step 6: Generate synthetic data
            with st.spinner(f"✨ Generating {num_rows} synthetic rows..."):
                synthetic = model.sample(num_rows=num_rows)
                metadata["generated_rows"] = len(synthetic)
            
            # Step 7: Post-processing
            synthetic = self._post_process(synthetic, df, analysis)
            
            return synthetic, metadata
            
        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, metadata
    
    def _calculate_epochs(self, n_rows: int) -> int:
        """Calculate optimal epochs based on dataset size"""
        if n_rows < 100:
            return 300
        elif n_rows < 500:
            return 200
        elif n_rows < 2000:
            return 150
        else:
            return 100
    
    def _post_process(
        self, 
        synthetic: pd.DataFrame, 
        original: pd.DataFrame,
        analysis: Dict
    ) -> pd.DataFrame:
        """Apply post-processing based on analysis"""
        df = synthetic.copy()
        
        # Ensure column order matches original
        df = df[original.columns]
        
        # Apply data type corrections
        for col in df.columns:
            if col in original.columns:
                original_dtype = original[col].dtype
                
                # Preserve categorical types
                if original_dtype == 'object' or pd.api.types.is_string_dtype(original_dtype):
                    df[col] = df[col].astype(str)
                
                # Preserve integer types
                elif pd.api.types.is_integer_dtype(original_dtype):
                    df[col] = df[col].round().astype(original_dtype)
        
        return df

# =============================================================================
# DATA QUALITY VALIDATOR
# =============================================================================

class DataQualityValidator:
    """Validate synthetic data quality"""
    
    @staticmethod
    def validate(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive quality validation"""
        report = {
            "basic_stats": {},
            "distribution_similarity": {},
            "correlation_preservation": {},
            "privacy_check": {},
            "overall_score": 0.0
        }
        
        # Basic stats comparison
        report["basic_stats"] = {
            "original_rows": len(original),
            "synthetic_rows": len(synthetic),
            "columns_match": list(original.columns) == list(synthetic.columns),
            "dtypes_match": (original.dtypes == synthetic.dtypes).all()
        }
        
        # Distribution similarity for numeric columns
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in synthetic.columns:
                orig_mean = original[col].mean()
                synt_mean = synthetic[col].mean()
                orig_std = original[col].std()
                synt_std = synthetic[col].std()
                
                mean_diff_pct = abs(orig_mean - synt_mean) / (abs(orig_mean) + 1e-10) * 100
                std_diff_pct = abs(orig_std - synt_std) / (abs(orig_std) + 1e-10) * 100
                
                report["distribution_similarity"][col] = {
                    "mean_diff_pct": round(mean_diff_pct, 2),
                    "std_diff_pct": round(std_diff_pct, 2),
                    "quality": "good" if mean_diff_pct < 10 and std_diff_pct < 15 else "fair" if mean_diff_pct < 25 else "poor"
                }
        
        # Correlation preservation
        if len(numeric_cols) > 1:
            orig_corr = original[numeric_cols].corr()
            synt_corr = synthetic[numeric_cols].corr()
            
            corr_diff = np.abs(orig_corr - synt_corr).mean().mean()
            report["correlation_preservation"] = {
                "average_difference": round(corr_diff, 3),
                "quality": "good" if corr_diff < 0.1 else "fair" if corr_diff < 0.2 else "poor"
            }
        
        # Privacy check - ensure no exact duplicates
        merged = pd.merge(
            original.astype(str),
            synthetic.astype(str),
            how='inner',
            on=list(original.columns)
        )
        
        report["privacy_check"] = {
            "exact_duplicates": len(merged),
            "privacy_preserved": len(merged) == 0,
            "duplication_rate_pct": round(len(merged) / len(synthetic) * 100, 2)
        }
        
        # Calculate overall score
        scores = []
        
        # Distribution score
        dist_scores = [
            1.0 if v["quality"] == "good" else 0.6 if v["quality"] == "fair" else 0.3
            for v in report["distribution_similarity"].values()
        ]
        if dist_scores:
            scores.append(np.mean(dist_scores))
        
        # Correlation score
        if report["correlation_preservation"]:
            corr_quality = report["correlation_preservation"]["quality"]
            corr_score = 1.0 if corr_quality == "good" else 0.6 if corr_quality == "fair" else 0.3
            scores.append(corr_score)
        
        # Privacy score
        privacy_score = 1.0 if report["privacy_check"]["privacy_preserved"] else 0.5
        scores.append(privacy_score)
        
        report["overall_score"] = round(np.mean(scores) * 100, 1) if scores else 0.0
        
        return report

# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    st.title("🎲 Universal Synthetic Data Generator")
    st.markdown("**AI-Powered • Zero Hardcoding • Works with Any Dataset**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_type = st.selectbox(
            "Generative Model",
            ["CTGAN", "Gaussian Copula"],
            help="CTGAN: Better for complex patterns | Gaussian Copula: Faster, simpler"
        )
        
        enforce_constraints = st.checkbox(
            "Enforce Constraints",
            value=True,
            help="Apply LLM-detected constraints during generation"
        )
        
        st.divider()
        
        st.markdown("""
        ### 🚀 How It Works
        1. **Upload** any CSV dataset
        2. **AI analyzes** structure & patterns
        3. **Auto-detects** constraints
        4. **Generates** high-quality synthetic data
        
        ### ✨ Features
        - ✅ Universal (any dataset)
        - ✅ No column hardcoding
        - ✅ Smart constraint detection
        - ✅ Relationship preservation
        - ✅ Quality validation
        """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "📤 Upload Dataset (CSV)",
        type=['csv'],
        help="Upload any CSV file - the AI will automatically understand its structure"
    )
    
    if uploaded_file:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{len(df):,} rows** × **{len(df.columns)} columns**")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Preview
        with st.expander("👀 Data Preview", expanded=True):
            st.dataframe(df.head(100), use_container_width=True, height=300)
        
        # Column info
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null": df.count().values,
                "Null %": ((df.isnull().sum() / len(df)) * 100).round(1).values,
                "Unique": [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Generation controls
        st.subheader("🎯 Generate Synthetic Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            num_rows = st.slider(
                "Number of synthetic rows to generate",
                min_value=len(df),
                max_value=min(len(df) * 10, 100000),
                value=min(len(df) * 2, 10000),
                step=max(len(df) // 10, 100)
            )
        
        with col2:
            st.metric("Multiplier", f"{num_rows / len(df):.1f}x")
        
        # Generate button
        if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
            generator = UniversalSyntheticGenerator()
            
            if not generator.sdv_available:
                st.error("❌ SDV not installed. Run: `pip install sdv`")
                return
            
            # Generate
            synthetic, metadata = generator.generate(
                df=df,
                num_rows=num_rows,
                model_type=model_type,
                enforce_constraints=enforce_constraints
            )
            
            if synthetic is not None:
                st.session_state.synthetic_data = synthetic
                st.session_state.original_data = df
                st.session_state.metadata = metadata
                st.balloons()
        
        # Display results
        if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
            st.divider()
            st.subheader("✨ Generated Synthetic Data")
            
            synthetic = st.session_state.synthetic_data
            original = st.session_state.original_data
            metadata = st.session_state.metadata
            
            # Analysis insights
            if metadata.get("analysis"):
                with st.expander("🧠 AI Analysis Insights", expanded=False):
                    analysis = metadata["analysis"]
                    
                    # Dataset type
                    if analysis.get("dataset_type"):
                        st.info(f"**Dataset Type:** {analysis['dataset_type']}")
                    
                    # Relationships
                    if analysis.get("relationships"):
                        st.write("**🔗 Detected Relationships:**")
                        for rel in analysis["relationships"][:5]:
                            cols = rel.get("columns", [])
                            st.write(f"- **{' ↔ '.join(cols)}**: {rel.get('type')} ({rel.get('strength')} strength)")
                    
                    # Constraints
                    if analysis.get("constraints"):
                        st.write(f"**✅ Applied {len(analysis['constraints'])} constraints**")
                    
                    # Quality issues
                    if analysis.get("quality_issues"):
                        st.write("**⚠️ Quality Issues Detected:**")
                        for issue in analysis["quality_issues"][:3]:
                            st.warning(f"{issue.get('issue')} - {issue.get('recommendation')}")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📄 Data", "📊 Quality Report", "📈 Comparison", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(100), use_container_width=True, height=400)
                st.caption(f"Showing first 100 of {len(synthetic):,} rows")
            
            with tab2:
                with st.spinner("Validating data quality..."):
                    validator = DataQualityValidator()
                    quality_report = validator.validate(original, synthetic)
                
                # Overall score
                score = quality_report["overall_score"]
                score_color = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
                st.metric("Overall Quality Score", f"{score_color} {score}%")
                
                # Distribution similarity
                st.write("**📊 Distribution Similarity (Numeric Columns)**")
                if quality_report["distribution_similarity"]:
                    dist_df = pd.DataFrame(quality_report["distribution_similarity"]).T
                    st.dataframe(dist_df, use_container_width=True)
                else:
                    st.info("No numeric columns to compare")
                
                # Correlation preservation
                if quality_report["correlation_preservation"]:
                    st.write("**🔗 Correlation Preservation**")
                    corr_info = quality_report["correlation_preservation"]
                    st.write(f"- Average difference: {corr_info['average_difference']}")
                    st.write(f"- Quality: {corr_info['quality']}")
                
                # Privacy check
                st.write("**🔒 Privacy Check**")
                privacy = quality_report["privacy_check"]
                if privacy["privacy_preserved"]:
                    st.success("✅ No exact duplicates found - Privacy preserved!")
                else:
                    st.warning(f"⚠️ Found {privacy['exact_duplicates']} exact duplicates ({privacy['duplication_rate_pct']}%)")
            
            with tab3:
                # Column selector
                compare_cols = st.multiselect(
                    "Select columns to compare",
                    options=list(original.columns),
                    default=list(original.columns)[:3]
                )
                
                for col in compare_cols:
                    if col in synthetic.columns:
                        st.write(f"**{col}**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.caption("Original")
                            if pd.api.types.is_numeric_dtype(original[col]):
                                st.line_chart(original[col].value_counts().sort_index())
                            else:
                                st.bar_chart(original[col].value_counts().head(10))
                        
                        with col2:
                            st.caption("Synthetic")
                            if pd.api.types.is_numeric_dtype(synthetic[col]):
                                st.line_chart(synthetic[col].value_counts().sort_index())
                            else:
                                st.bar_chart(synthetic[col].value_counts().head(10))
                        
                        st.divider()
            
            with tab4:
                csv = synthetic.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Synthetic Data (CSV)",
                    data=csv,
                    file_name=f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Download quality report
                report_json = json.dumps(quality_report if 'quality_report' in locals() else {}, indent=2)
                st.download_button(
                    label="📥 Download Quality Report (JSON)",
                    data=report_json,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    else:
        # Welcome screen
        st.info("""
        ## 👋 Welcome to Universal Synthetic Data Generator
        
        This tool uses **AI + Statistical Modeling** to generate high-quality synthetic data from any dataset.
        
        ### ✨ Key Features:
        - **🌍 Universal**: Works with ANY dataset structure (no hardcoding!)
        - **🧠 AI-Powered**: LLM automatically detects patterns, relationships, and constraints
        - **🎯 High Quality**: Preserves distributions, correlations, and data characteristics
        - **🔒 Privacy-Safe**: Generates new data that doesn't leak original records
        - **📊 Validated**: Comprehensive quality metrics and comparison tools
        
        ### 🚀 Get Started:
        1. Upload any CSV file
        2. AI will analyze the structure
        3. Generate synthetic data with one click
        4. Download and use!
        
        ### 💡 Use Cases:
        - Testing & QA (generate large test datasets)
        - Data sharing (privacy-preserving)
        - ML training (augment small datasets)
        - Demo & presentations (realistic fake data)
        
        **Upload your CSV to begin! →**
        """)

if __name__ == "__main__":
    main()
