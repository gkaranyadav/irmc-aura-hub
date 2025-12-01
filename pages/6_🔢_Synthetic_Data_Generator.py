# pages/6_🔢_Synthetic_Data_Generator.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import json
import re
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Universal Synthetic Data Generator",
    page_icon="🎲",
    layout="wide"
)

# =============================================================================
# SIMPLE LLM ANALYZER
# =============================================================================

class SimpleLLMAnalyzer:
    """Simple analyzer that works reliably"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or st.secrets.get("GROQ_API_KEY")
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simple but effective analysis"""
        try:
            # Try LLM analysis if API key exists
            if self.api_key:
                try:
                    from groq import Groq
                    client = Groq(api_key=self.api_key)
                    
                    # Create simple prompt that works reliably
                    sample_data = df.head(10).to_string(index=False)
                    
                    prompt = f"""Analyze this dataset and return JSON with:
                    1. dataset_type: what type of data is this?
                    2. main_patterns: main patterns you observe
                    3. key_relationships: key relationships between columns
                    4. quality_issues: any data quality issues
                    
                    Columns: {list(df.columns)}
                    
                    Sample data:
                    {sample_data}
                    
                    Return JSON only."""
                    
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",  # Use reliable model
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Return JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    content = response.choices[0].message.content
                    
                    # Extract JSON
                    try:
                        # Try to parse directly
                        llm_analysis = json.loads(content)
                    except:
                        # Try to extract JSON from text
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            llm_analysis = json.loads(json_match.group())
                        else:
                            llm_analysis = {"error": "Could not parse LLM response"}
                    
                except Exception as e:
                    st.warning(f"LLM analysis failed: {e}")
                    llm_analysis = {}
            else:
                llm_analysis = {}
            
            # Always include statistical analysis
            stats_analysis = self._statistical_analysis(df)
            
            # Merge analyses
            return {
                "statistics": stats_analysis,
                "llm_insights": llm_analysis,
                "dataset_type": llm_analysis.get("dataset_type", "unknown"),
                "patterns": llm_analysis.get("main_patterns", []),
                "relationships": llm_analysis.get("key_relationships", []),
                "quality_issues": llm_analysis.get("quality_issues", [])
            }
            
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return self._statistical_analysis(df)
    
    def _statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic statistical analysis"""
        analysis = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": {},
            "dtypes": {}
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col])
            }
            
            if col_info["is_numeric"]:
                col_info.update({
                    "mean": float(df[col].mean()) if df[col].notna().any() else None,
                    "min": float(df[col].min()) if df[col].notna().any() else None,
                    "max": float(df[col].max()) if df[col].notna().any() else None
                })
            
            analysis["columns"][col] = col_info
        
        return analysis

# =============================================================================
# SIMPLE SDV GENERATOR
# =============================================================================

class SimpleSDVGenerator:
    """Simple SDV generator without complex constraints"""
    
    def __init__(self):
        self.sdv_available = self._check_sdv()
        self.analyzer = SimpleLLMAnalyzer()
    
    def _check_sdv(self):
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            return True
        except ImportError:
            st.error("SDV not installed. Please add 'sdv' to requirements.txt")
            return False
    
    def generate(self, df: pd.DataFrame, num_rows: int) -> Optional[pd.DataFrame]:
        """Simple generation with SDV"""
        if not self.sdv_available:
            return None
        
        try:
            # Show analysis
            with st.spinner("🔍 Analyzing data..."):
                analysis = self.analyzer.analyze_dataset(df)
            
            self._show_analysis(analysis)
            
            # Train SDV
            with st.spinner("🤖 Training model..."):
                from sdv.metadata import SingleTableMetadata
                from sdv.single_table import CTGANSynthesizer
                
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data=df)
                
                # Simple model configuration
                epochs = 200 if len(df) < 100 else 100
                batch_size = min(50, len(df))
                
                model = CTGANSynthesizer(
                    metadata=metadata,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False
                )
                
                model.fit(df)
            
            # Generate
            with st.spinner(f"🎯 Generating {num_rows} rows..."):
                synthetic = model.sample(num_rows=num_rows)
            
            # Basic post-processing
            synthetic = self._post_process(synthetic, df)
            
            return synthetic
            
        except Exception as e:
            st.error(f"Generation failed: {e}")
            return None
    
    def _show_analysis(self, analysis: Dict):
        """Show analysis results"""
        st.subheader("📊 Analysis Results")
        
        # Statistics
        with st.expander("📈 Statistical Summary", expanded=True):
            stats = analysis.get("statistics", {})
            shape = stats.get("shape", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", shape.get("rows", 0))
            with col2:
                st.metric("Columns", shape.get("columns", 0))
            with col3:
                null_total = sum(col_info.get("null_count", 0) 
                               for col_info in stats.get("columns", {}).values())
                st.metric("Total Nulls", null_total)
        
        # LLM Insights
        if analysis.get("llm_insights"):
            with st.expander("🧠 LLM Insights", expanded=False):
                insights = analysis["llm_insights"]
                
                if "dataset_type" in insights:
                    st.info(f"**Dataset Type:** {insights['dataset_type']}")
                
                if "main_patterns" in insights:
                    st.write("**Main Patterns:**")
                    for pattern in insights["main_patterns"][:3]:
                        st.write(f"- {pattern}")
        
        # Quality Issues
        issues = analysis.get("quality_issues", [])
        if issues:
            with st.expander("⚠️ Quality Issues", expanded=True):
                for issue in issues[:3]:
                    st.warning(f"❌ {issue}")
    
    def _post_process(self, synthetic: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
        """Basic post-processing"""
        df = synthetic.copy()
        
        # Ensure same columns
        missing_cols = set(original.columns) - set(df.columns)
        extra_cols = set(df.columns) - set(original.columns)
        
        if missing_cols:
            for col in missing_cols:
                df[col] = np.nan
        
        if extra_cols:
            df = df.drop(columns=list(extra_cols))
        
        # Ensure column order
        df = df[original.columns]
        
        return df

# =============================================================================
# SIMPLE VALIDATOR
# =============================================================================

class SimpleValidator:
    """Simple data validation"""
    
    @staticmethod
    def validate(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Simple validation"""
        report = {
            "basic": {},
            "distribution_check": {},
            "quality_score": 0.0
        }
        
        # Basic checks
        report["basic"] = {
            "original_rows": len(original),
            "synthetic_rows": len(synthetic),
            "columns_match": list(original.columns) == list(synthetic.columns),
            "has_duplicates": synthetic.duplicated().any()
        }
        
        # Distribution check for numeric columns
        numeric_cols = original.select_dtypes(include=[np.number]).columns
        dist_scores = []
        
        for col in numeric_cols:
            if col in synthetic.columns:
                orig_mean = original[col].mean()
                synth_mean = synthetic[col].mean()
                
                if orig_mean != 0:
                    diff_pct = abs(orig_mean - synth_mean) / abs(orig_mean) * 100
                    report["distribution_check"][col] = {
                        "mean_diff_pct": round(diff_pct, 1),
                        "quality": "good" if diff_pct < 20 else "fair" if diff_pct < 50 else "poor"
                    }
                    dist_scores.append(1.0 if diff_pct < 20 else 0.5 if diff_pct < 50 else 0.0)
        
        # Calculate quality score
        if dist_scores:
            report["quality_score"] = round(np.mean(dist_scores) * 100, 1)
        else:
            report["quality_score"] = 75.0  # Default score if no numeric columns
        
        return report

# =============================================================================
# MAIN APP - SIMPLIFIED
# =============================================================================

def main():
    st.title("🎲 Universal Synthetic Data Generator")
    st.markdown("**Simple • Effective • No Hardcoding**")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload CSV File", type=['csv'])
    
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
        
        # Show column info
        st.write("**📊 Column Information**")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count().values,
            "Unique": [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Generation settings
        st.subheader("🎯 Generate Synthetic Data")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=len(df) * 10,
                value=len(df) * 2,
                step=100
            )
        
        with col2:
            st.metric("Multiplier", f"{num_rows/len(df):.1f}x")
        
        # Generate button
        if st.button("🚀 Generate Data", type="primary", use_container_width=True):
            generator = SimpleSDVGenerator()
            
            synthetic = generator.generate(df, int(num_rows))
            
            if synthetic is not None:
                st.session_state.synthetic_data = synthetic
                st.session_state.original_data = df
                st.balloons()
        
        # Show results
        if 'synthetic_data' in st.session_state and st.session_state.synthetic_data is not None:
            synthetic = st.session_state.synthetic_data
            original = st.session_state.original_data
            
            st.subheader(f"✨ Generated {len(synthetic)} Rows")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📄 Data", "📊 Validation", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Compare samples
                st.write("**🔍 Comparison**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original (first 5):")
                    st.dataframe(original.head())
                with col2:
                    st.write("Synthetic (first 5):")
                    st.dataframe(synthetic.head())
            
            with tab2:
                validator = SimpleValidator()
                report = validator.validate(original, synthetic)
                
                # Show score
                score = report["quality_score"]
                st.metric("Quality Score", f"{score}%")
                
                # Basic checks
                basic = report["basic"]
                st.write("**Basic Checks:**")
                st.write(f"- ✅ Columns match: {basic['columns_match']}")
                st.write(f"- ✅ No duplicates: {not basic['has_duplicates']}")
                
                # Distribution checks
                if report["distribution_check"]:
                    st.write("**Distribution Similarity:**")
                    for col, info in report["distribution_check"].items():
                        quality_icon = "🟢" if info["quality"] == "good" else "🟡" if info["quality"] == "fair" else "🔴"
                        st.write(f"- {col}: {quality_icon} {info['mean_diff_pct']}% difference")
            
            with tab3:
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"synthetic_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Option to regenerate
                if st.button("🔄 Generate New Variation"):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome screen
        st.info("""
        ## 🎲 Universal Synthetic Data Generator
        
        ### **How It Works:**
        1. **Upload** any CSV file
        2. **AI analyzes** patterns automatically
        3. **SDV generates** synthetic data
        4. **Download** and use
        
        ### **Key Features:**
        ✅ **Universal** - Works with any dataset
        ✅ **No Hardcoding** - No predefined rules
        ✅ **Simple** - Easy to use
        ✅ **Reliable** - Uses proven SDV technology
        
        ### **Best Practices:**
        - Upload **30+ rows** for good results
        - Use **descriptive column names**
        - Ensure **consistent formatting**
        
        **Upload a CSV file to get started!**
        """)

if __name__ == "__main__":
    main()
