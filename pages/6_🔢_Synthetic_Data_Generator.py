# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import random
from groq import Groq
from auth import check_session

# =============================================================================
# TRULY UNIVERSAL GENERATOR (NO CODE ASSUMPTIONS)
# =============================================================================
class UniversalPatternAnalyzer:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.llm_available = True
        except:
            self.llm_available = False
    
    def analyze_patterns(self, df_sample):
        """
        Let LLM analyze - code doesn't predefine anything
        """
        if not self.llm_available or df_sample.empty:
            return self._fallback_analysis(df_sample)
        
        # Prepare data for LLM
        data_info = {
            "columns": {},
            "sample_rows": []
        }
        
        # Add column info
        for col in df_sample.columns[:15]:  # Limit columns
            col_data = df_sample[col].dropna().head(10)
            if len(col_data) > 0:
                samples = [str(x)[:100] for x in col_data.tolist()]
                data_info["columns"][col] = {
                    "samples": samples,
                    "unique_count": col_data.nunique(),
                    "sample_count": len(col_data)
                }
        
        # Add sample rows
        for idx, row in df_sample.head(5).iterrows():
            data_info["sample_rows"].append(row.to_dict())
        
        # Universal prompt - NO constraints on LLM
        prompt = f"""
        Analyze this dataset and describe how to generate similar synthetic data.
        
        Data Info:
        {json.dumps(data_info, indent=2)}
        
        Analyze each column and provide generation rules.
        Focus on patterns, formats, and distributions.
        
        Return JSON with generation instructions.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a data pattern analyst."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON
            try:
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return self._enhance_with_stats(analysis, df_sample)
            except:
                pass
            
            return self._fallback_analysis(df_sample)
            
        except Exception as e:
            st.warning(f"LLM analysis failed: {str(e)}")
            return self._fallback_analysis(df_sample)
    
    def _enhance_with_stats(self, analysis, df):
        """Add statistical facts to LLM analysis"""
        if 'columns' not in analysis:
            analysis['columns'] = {}
        
        for col in df.columns:
            if col not in analysis['columns']:
                analysis['columns'][col] = {}
            
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info = analysis['columns'][col]
                
                # Add actual stats
                try:
                    # Check if can be numeric
                    numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_vals) > 0:
                        if 'numeric_stats' not in col_info:
                            col_info['numeric_stats'] = {}
                        col_info['numeric_stats']['min'] = float(numeric_vals.min())
                        col_info['numeric_stats']['max'] = float(numeric_vals.max())
                        col_info['numeric_stats']['mean'] = float(numeric_vals.mean())
                        col_info['numeric_stats']['std'] = float(numeric_vals.std()) if len(numeric_vals) > 1 else 0
                        
                        # Check decimal places
                        if not numeric_vals.apply(lambda x: float(x).is_integer()).all():
                            decimals = numeric_vals.apply(
                                lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0
                            )
                            col_info['numeric_stats']['max_decimals'] = int(decimals.max())
                            col_info['numeric_stats']['common_decimals'] = int(decimals.mode().iloc[0]) if len(decimals.mode()) > 0 else 2
                
                except:
                    pass
                
                # Always add these
                col_info['actual_unique_ratio'] = float(col_data.nunique() / len(col_data))
                col_info['actual_sample_count'] = len(col_data)
                
                # Character analysis
                if len(col_data) > 0:
                    samples = col_data.head(5).astype(str).tolist()
                    col_info['sample_patterns'] = samples
        
        return analysis
    
    def _fallback_analysis(self, df):
        """Statistical analysis without assumptions"""
        analysis = {'columns': {}}
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info = {
                    'generation_type': 'auto_detect',
                    'actual_stats': {}
                }
                
                # Check patterns statistically
                sample = str(col_data.iloc[0]) if len(col_data) > 0 else ""
                
                # Check character composition
                chars = set(''.join(col_data.head(10).astype(str).tolist()))
                digit_count = sum(1 for c in chars if c.isdigit())
                alpha_count = sum(1 for c in chars if c.isalpha())
                
                col_info['character_analysis'] = {
                    'has_digits': digit_count > 0,
                    'has_letters': alpha_count > 0,
                    'has_symbols': len(chars) > (digit_count + alpha_count),
                    'common_symbols': [c for c in ['-', '_', '@', '.', '/', ' '] if any(c in str(x) for x in col_data.head(5))]
                }
                
                # Check if numeric-like
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > len(col_data) * 0.7:  # Mostly numeric
                        col_info['generation_type'] = 'numeric'
                        col_info['actual_stats']['min'] = float(numeric_data.min())
                        col_info['actual_stats']['max'] = float(numeric_data.max())
                        col_info['actual_stats']['mean'] = float(numeric_data.mean())
                        
                        if numeric_data.apply(lambda x: float(x).is_integer()).all():
                            col_info['actual_stats']['is_integer'] = True
                        else:
                            col_info['actual_stats']['is_float'] = True
                except:
                    pass
                
                # Check length patterns
                lengths = col_data.astype(str).str.len()
                col_info['length_stats'] = {
                    'min': int(lengths.min()),
                    'max': int(lengths.max()),
                    'avg': float(lengths.mean())
                }
                
                # Check uniqueness
                col_info['uniqueness'] = float(col_data.nunique() / len(col_data))
                
                analysis['columns'][col] = col_info
        
        return analysis

# =============================================================================
# UNIVERSAL DATA GENERATOR
# =============================================================================
class UniversalDataGenerator:
    def __init__(self):
        self.analyzer = UniversalPatternAnalyzer()
    
    def generate_data(self, original_df, num_rows):
        """
        Generate synthetic data - NO assumptions about what data is
        """
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Analyze patterns
        with st.spinner("🔍 Analyzing patterns..."):
            analysis = self.analyzer.analyze_patterns(original_df.head(50))
        
        # Step 2: Generate each column
        generated_data = {}
        
        for col in original_df.columns:
            if col in analysis.get('columns', {}):
                generated_data[col] = self._generate_column(
                    col_name=col,
                    col_info=analysis['columns'][col],
                    original_data=original_df[col] if col in original_df.columns else None,
                    num_rows=num_rows
                )
            else:
                # Simple fallback
                generated_data[col] = self._simple_generate(original_df[col], num_rows)
        
        # Create DataFrame
        return pd.DataFrame(generated_data)
    
    def _generate_column(self, col_name, col_info, original_data, num_rows):
        """Generate based on pattern analysis"""
        
        # Check what type of generation
        gen_type = col_info.get('generation_type', 'auto_detect')
        
        # If we have actual samples, sometimes reuse patterns
        if original_data is not None and len(original_data) > 0:
            unique_vals = original_data.dropna().unique()
            if len(unique_vals) <= 20:  # Limited unique values
                return np.random.choice(unique_vals, num_rows)
        
        # Check character patterns
        char_analysis = col_info.get('character_analysis', {})
        
        # If numeric-like
        if gen_type == 'numeric' or ('actual_stats' in col_info and 'min' in col_info['actual_stats']):
            stats = col_info.get('actual_stats', {})
            min_val = stats.get('min', 0)
            max_val = stats.get('max', 100)
            
            if stats.get('is_integer', False):
                # Generate integers
                data = np.random.randint(int(min_val), int(max_val) + 1, num_rows)
            else:
                # Generate floats
                mean_val = stats.get('mean', (min_val + max_val) / 2)
                std_val = stats.get('std', (max_val - min_val) / 4)
                
                data = np.random.normal(mean_val, std_val, num_rows)
                data = np.clip(data, min_val, max_val)
                
                # Apply decimal places if specified
                if stats.get('is_float', False):
                    decimals = stats.get('common_decimals', 2)
                    data = np.round(data, decimals)
            
            return data
        
        # Text-like generation
        else:
            # Get length patterns
            length_stats = col_info.get('length_stats', {'min': 5, 'max': 20, 'avg': 10})
            
            # Check character sets to use
            use_digits = char_analysis.get('has_digits', True)
            use_letters = char_analysis.get('has_letters', True)
            
            # Get common symbols
            common_symbols = char_analysis.get('common_symbols', [])
            
            # Generate text
            generated = []
            for _ in range(num_rows):
                # Determine length
                length = int(np.random.normal(length_stats['avg'], 2))
                length = max(length_stats['min'], min(length_stats['max'], length))
                
                # Build characters
                chars = []
                for _ in range(length):
                    # Choose character type
                    if common_symbols and np.random.random() < 0.1:  # 10% chance symbol
                        chars.append(random.choice(common_symbols))
                    elif use_digits and use_letters:
                        # Mix digits and letters
                        if np.random.random() < 0.5:
                            chars.append(random.choice('0123456789'))
                        else:
                            chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
                    elif use_digits:
                        chars.append(random.choice('0123456789'))
                    else:
                        chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
                
                generated.append(''.join(chars))
            
            return generated
    
    def _simple_generate(self, original_series, num_rows):
        """Simple fallback generation"""
        if original_series is not None and len(original_series) > 0:
            unique_vals = original_series.dropna().unique()
            if len(unique_vals) > 0:
                return np.random.choice(unique_vals, num_rows)
        
        # Generate random strings as last resort
        return [f"VAL_{i:06d}" for i in range(num_rows)]

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Authentication
    if not check_session():
        st.warning("Please login first")
        st.stop()
    
    # Page config
    st.set_page_config(
        page_title="Universal Data Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Header
    st.title("🔢 Universal Synthetic Data Generator")
    st.markdown("Generate synthetic data for ANY dataset - **No assumptions about data meaning**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize
    if 'generator' not in st.session_state:
        st.session_state.generator = UniversalDataGenerator()
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    
    # Main interface
    uploaded_file = st.file_uploader("📤 Upload ANY CSV Dataset", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.error("Empty file")
            else:
                st.session_state.original_data = df
                
                st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                
                # Preview
                with st.expander("📋 Original Data Preview", expanded=True):
                    st.dataframe(df.head(10))
                
                # Analysis section
                st.subheader("🔍 Pattern Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Analyze Data Patterns", type="primary"):
                        with st.spinner("Analyzing patterns..."):
                            analysis = st.session_state.generator.analyzer.analyze_patterns(df.head(50))
                            st.session_state.analysis = analysis
                            
                            st.success("✅ Analysis complete!")
                
                with col2:
                    num_rows = st.number_input("Rows to generate", min_value=10, max_value=10000, value=1000, step=100)
                
                # Show analysis if available
                if st.session_state.analysis:
                    with st.expander("📊 Pattern Analysis Results", expanded=False):
                        for col, info in st.session_state.analysis.get('columns', {}).items():
                            st.write(f"**{col}**:")
                            if 'generation_type' in info:
                                st.write(f"Generation type: `{info['generation_type']}`")
                            if 'character_analysis' in info:
                                st.write(f"Characters: {info['character_analysis']}")
                            if 'actual_stats' in info and info['actual_stats']:
                                st.write(f"Stats: {info['actual_stats']}")
                            st.write("---")
                
                # Generation
                if st.session_state.analysis:
                    if st.button("🚀 Generate Synthetic Data", type="primary"):
                        with st.spinner("Generating synthetic data..."):
                            generated = st.session_state.generator.generate_data(
                                original_df=df,
                                num_rows=int(num_rows)
                            )
                            
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} synthetic rows!")
                            st.balloons()
                
                # Results
                if st.session_state.generated_data is not None:
                    st.subheader("📋 Generated Data")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Preview
                    st.dataframe(df_gen.head(20))
                    
                    # Stats comparison
                    with st.expander("📊 Statistics Comparison", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Data**")
                            st.write(f"Rows: {len(df)}")
                            st.write(f"Columns: {len(df.columns)}")
                        
                        with col2:
                            st.write("**Generated Data**")
                            st.write(f"Rows: {len(df_gen)}")
                            st.write(f"Columns: {len(df_gen.columns)}")
                    
                    # Download
                    st.subheader("💾 Download")
                    
                    csv = df_gen.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "synthetic_data.csv",
                        "text/csv"
                    )
                    
                    if st.button("🔄 Generate New Set"):
                        st.session_state.generated_data = None
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
