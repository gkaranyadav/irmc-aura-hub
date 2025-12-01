# pages/6_🔢_Synthetic_Data_Generator.py - TRULY DYNAMIC
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import json
import re

# =============================================================================
# DYNAMIC LLM ANALYZER - NO PREDEFINED RULES
# =============================================================================

class DynamicLLMAnalyzer:
    """Dynamic analyzer - LLM discovers everything"""
    
    @staticmethod
    def analyze_dynamically(df: pd.DataFrame) -> Dict:
        """
        Let LLM analyze data dynamically without any guidance
        """
        try:
            if "GROQ_API_KEY" not in st.secrets:
                return {"error": "No API key"}
            
            from groq import Groq
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            # Get random sample
            sample = df.sample(min(10, len(df))).reset_index(drop=True)
            
            # TOTALLY OPEN-ENDED PROMPT - NO GUIDANCE AT ALL
            prompt = f"""
            Here is some data. Analyze it and tell me what you find.
            
            Data has {len(df)} rows and these columns: {list(df.columns)}
            
            First 10 rows:
            {sample.to_string(index=False)}
            
            What patterns do you see? What issues exist? What should be preserved in synthetic data?
            """
            
            messages = [
                {"role": "system", "content": "You analyze data patterns."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            raw_analysis = response.choices[0].message.content
            
            # Extract insights from raw text
            return DynamicLLMAnalyzer._extract_insights(raw_analysis, df)
            
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return {"analysis": "Could not analyze", "issues": []}
    
    @staticmethod
    def _extract_insights(raw_text: str, df: pd.DataFrame) -> Dict:
        """Extract insights from LLM's free-text response"""
        insights = {
            "raw_analysis": raw_text,
            "issues_found": [],
            "patterns_discovered": [],
            "recommendations": []
        }
        
        # Look for issue indicators in the text
        lines = raw_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Look for problem indicators
            if any(word in line_lower for word in ['wrong', 'error', 'incorrect', 'mismatch', 'issue', 'problem', 'inconsistent']):
                insights["issues_found"].append(line.strip())
            
            # Look for pattern indicators
            if any(word in line_lower for word in ['pattern', 'relationship', 'mapping', 'correlation', 'association']):
                insights["patterns_discovered"].append(line.strip())
            
            # Look for recommendations
            if any(word in line_lower for word in ['should', 'must', 'need to', 'recommend', 'suggest']):
                insights["recommendations"].append(line.strip())
        
        # Auto-detect some patterns from data
        insights["auto_detected"] = DynamicLLMAnalyzer._auto_detect_patterns(df)
        
        return insights
    
    @staticmethod
    def _auto_detect_patterns(df: pd.DataFrame) -> Dict:
        """Auto-detect patterns purely from data"""
        patterns = {
            "column_stats": {},
            "potential_issues": [],
            "data_quality": {}
        }
        
        # Analyze each column
        for col in df.columns:
            col_data = {
                "type": str(df[col].dtype),
                "unique_values": int(df[col].nunique()),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": df[col].dropna().unique()[:3].tolist()
            }
            patterns["column_stats"][col] = col_data
        
        # Look for potential data issues (no medical logic!)
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    # Check if columns might be related by looking at value pairs
                    unique_pairs = df[[col1, col2]].drop_duplicates().shape[0]
                    total_rows = len(df)
                    
                    # If many rows have same column1 but different column2, might be issue
                    if df[col1].nunique() < total_rows * 0.5:  # Column1 has repeats
                        pair_stats = df.groupby(col1)[col2].nunique()
                        inconsistent = pair_stats[pair_stats > 1]
                        if len(inconsistent) > 0:
                            patterns["potential_issues"].append(
                                f"Column '{col1}' has inconsistent mappings to '{col2}'"
                            )
        
        return patterns

# =============================================================================
# SMART SDV GENERATOR WITH DYNAMIC RULES
# =============================================================================

class DynamicGenerator:
    """Generate data with dynamic rules from LLM"""
    
    def __init__(self):
        self.sdv_available = self._check_sdv()
        self.analyzer = DynamicLLMAnalyzer()
    
    def _check_sdv(self):
        try:
            from sdv.single_table import CTGANSynthesizer
            return True
        except:
            return False
    
    def generate_dynamically(self, df: pd.DataFrame, num_rows: int) -> Optional[pd.DataFrame]:
        """Generate data using dynamic analysis"""
        if not self.sdv_available:
            st.error("SDV not available")
            return None
        
        # Step 1: Dynamic LLM Analysis
        with st.spinner("🧠 LLM analyzing data patterns..."):
            insights = self.analyzer.analyze_dynamically(df)
        
        # Display insights
        self._display_insights(insights)
        
        # Step 2: Train SDV
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            
            with st.spinner("🤖 Training model on your data..."):
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data=df)
                
                # Train with optimal settings for data size
                epochs = min(500, max(100, len(df) * 5))  # Dynamic epochs
                batch_size = min(32, max(8, len(df) // 10))  # Dynamic batch
                
                model = CTGANSynthesizer(
                    metadata=metadata,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False
                )
                
                model.fit(df)
            
            # Step 3: Generate
            with st.spinner(f"🎯 Generating {num_rows} rows..."):
                synthetic = model.sample(num_rows=num_rows)
            
            # Step 4: Apply dynamic fixes based on LLM insights
            synthetic = self._apply_dynamic_fixes(synthetic, insights, df)
            
            # Step 5: Show results
            self._show_results(synthetic, df)
            
            return synthetic
            
        except Exception as e:
            st.error(f"Generation failed: {e}")
            return None
    
    def _display_insights(self, insights: Dict):
        """Display LLM insights"""
        st.subheader("🔍 LLM Analysis Results")
        
        # Show raw analysis
        with st.expander("📝 LLM's Analysis", expanded=True):
            st.write(insights.get("raw_analysis", "No analysis"))
        
        # Show discovered patterns
        patterns = insights.get("patterns_discovered", [])
        if patterns:
            with st.expander("🔗 Discovered Patterns", expanded=False):
                for pattern in patterns[:5]:  # Show first 5
                    st.write(f"• {pattern}")
        
        # Show issues found
        issues = insights.get("issues_found", [])
        if issues:
            with st.expander("⚠️ Potential Issues", expanded=True):
                for issue in issues[:5]:  # Show first 5
                    st.error(f"❌ {issue}")
        
        # Show recommendations
        recommendations = insights.get("recommendations", [])
        if recommendations:
            with st.expander("🎯 LLM Recommendations", expanded=False):
                for rec in recommendations[:5]:
                    st.info(f"💡 {rec}")
    
    def _apply_dynamic_fixes(self, synthetic: pd.DataFrame, insights: Dict, original: pd.DataFrame) -> pd.DataFrame:
        """Apply fixes based on LLM insights"""
        df = synthetic.copy()
        
        # Extract column names from insights text
        raw_text = insights.get("raw_analysis", "").lower()
        
        # Look for column mentions in the analysis
        mentioned_cols = []
        for col in original.columns:
            if col.lower() in raw_text:
                mentioned_cols.append(col)
        
        # If LLM mentioned specific columns having issues, check them
        fixes_applied = []
        
        for col in mentioned_cols:
            if col in df.columns:
                # Check for basic consistency with original
                orig_unique = set(str(v).lower() for v in original[col].dropna().unique()[:20])
                synth_unique = set(str(v).lower() for v in df[col].dropna().unique()[:20])
                
                # If synthetic has values not in original, might be issue
                new_values = synth_unique - orig_unique
                if len(new_values) > 0 and len(orig_unique) > 0:
                    # Replace some new values with original ones
                    sample_size = min(10, len(df))
                    replace_indices = df.sample(sample_size).index
                    original_sample = np.random.choice(list(orig_unique), size=sample_size)
                    df.loc[replace_indices, col] = original_sample
                    fixes_applied.append(f"Adjusted '{col}' values")
        
        if fixes_applied:
            st.info(f"Applied {len(fixes_applied)} adjustments based on LLM insights")
        
        return df
    
    def _show_results(self, synthetic: pd.DataFrame, original: pd.DataFrame):
        """Show generation results"""
        st.subheader(f"✅ Generated {len(synthetic)} Rows")
        
        # Show comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data Sample**")
            st.dataframe(original.head(5), use_container_width=True)
        
        with col2:
            st.write("**Generated Data Sample**")
            st.dataframe(synthetic.head(5), use_container_width=True)
        
        # Check basic statistics
        st.write("**Basic Statistics Comparison**")
        
        stats_cols = [col for col in original.columns if pd.api.types.is_numeric_dtype(original[col])]
        if len(stats_cols) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original means:")
                for col in stats_cols[:3]:
                    st.write(f"{col}: {original[col].mean():.2f}")
            with col2:
                st.write("Generated means:")
                for col in stats_cols[:3]:
                    if col in synthetic.columns:
                        st.write(f"{col}: {synthetic[col].mean():.2f}")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Dynamic Data Generator",
        page_icon="🌀",
        layout="wide"
    )
    
    st.title("🌀 Dynamic Synthetic Data Generator")
    st.markdown("**Zero rules - LLM analyzes patterns, SDV generates, we validate**")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Your Data (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Quick preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column info
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(df[col].dtype) for col in df.columns],
                'Unique': [df[col].nunique() for col in df.columns],
                'Nulls': [df[col].isnull().sum() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Generation settings
        st.subheader("⚙️ Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=len(df) * 20,
                value=len(df) * 3,
                help="Based on your data size"
            )
        
        with col2:
            st.write("**Generation Strategy**")
            strategy = st.selectbox(
                "",
                ["Preserve Patterns", "Add Variation", "Balance"],
                help="How to generate data"
            )
        
        # Generate button
        if st.button("🚀 Generate Dynamic Synthetic Data", type="primary", use_container_width=True):
            if len(df) < 10:
                st.warning("⚠️ Very small dataset - patterns may not be clear")
            
            generator = DynamicGenerator()
            
            if not generator.sdv_available:
                st.error("SDV not installed")
            else:
                with st.spinner("Generating with dynamic analysis..."):
                    synthetic = generator.generate_dynamically(df, int(num_rows))
                
                if synthetic is not None:
                    st.session_state.generated_data = synthetic
                    st.balloons()
        
        # Show results
        if 'generated_data' in st.session_state and st.session_state.generated_data is not None:
            synthetic = st.session_state.generated_data
            
            st.subheader(f"📊 Generated Data ({len(synthetic)} rows)")
            
            tab1, tab2, tab3 = st.tabs(["Data", "Analysis", "Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
            
            with tab2:
                # Compare distributions for categorical columns
                cat_cols = [col for col in df.columns if df[col].nunique() < 20]
                
                if len(cat_cols) > 0:
                    for col in cat_cols[:2]:  # Show first 2
                        if col in synthetic.columns:
                            st.write(f"**{col} Distribution**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Original:")
                                st.bar_chart(df[col].value_counts().head(10))
                            with col2:
                                st.write("Generated:")
                                st.bar_chart(synthetic[col].value_counts().head(10))
            
            with tab3:
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"dynamic_synthetic_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Option to regenerate
                if st.button("🔄 Generate New Variation"):
                    st.session_state.generated_data = None
                    st.rerun()
    
    else:
        st.info("""
        ## 🌀 Dynamic Synthetic Data Generator
        
        
        """)

if __name__ == "__main__":
    main()
