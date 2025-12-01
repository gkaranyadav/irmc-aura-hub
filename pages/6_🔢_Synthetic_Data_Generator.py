# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
import io
import random
from groq import Groq
from auth import check_session

# =============================================================================
# UNIVERSAL PATTERN ANALYZER (Works for ANY dataset)
# =============================================================================
class UniversalPatternAnalyzer:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.llm_available = True
        except:
            self.llm_available = False
    
    def analyze_universal_patterns(self, df_sample):
        """
        Analyze ANY dataset and create generation rules
        WITHOUT predefining what data represents
        """
        if not self.llm_available or df_sample.empty:
            return self._analyze_patterns_statistically(df_sample)
        
        # Create COMPACT representation (save tokens)
        compact_data = self._create_compact_representation(df_sample)
        
        # UNIVERSAL PROMPT (no assumptions about data)
        prompt = f"""
        Analyze this dataset and create rules for generating similar synthetic data.
        
        DATA SUMMARY:
        {json.dumps(compact_data, indent=2)}
        
        IMPORTANT: DO NOT assume what the data represents. 
        DO NOT use pre-defined categories like "age", "email", "price".
        
        Your task: Analyze the PATTERNS and FORMATS only.
        
        For EACH column, provide:
        1. DATA PATTERN ANALYSIS: What patterns do you see?
        2. FORMAT RULES: What format should generated data follow?
        3. VALUE CONSTRAINTS: Any constraints on values?
        4. GENERATION STRATEGY: How to generate similar data?
        
        Think CRITICALLY about what makes sense:
        - If values look like IDs, they should be integers/sequential
        - If values have decimal points, how many decimals make sense?
        - If values are text, what patterns exist?
        - If values are dates/times, what format?
        
        Return ONLY JSON:
        {{
            "columns": {{
                "column_name": {{
                    "pattern_analysis": "description of patterns found",
                    "detected_format": "integer|float|string|date|id_like|code_like|mixed",
                    "format_details": {{
                        "is_numeric": true/false,
                        "is_integer_like": true/false,
                        "has_decimal_points": true/false,
                        "decimal_places": 0|2|...,
                        "is_sequential": true/false,
                        "is_unique": true/false,
                        "text_pattern": "regex_pattern_if_applicable",
                        "length_range": [min, max],
                        "character_set": "description of allowed characters"
                    }},
                    "value_constraints": {{
                        "min_value": number or null,
                        "max_value": number or null,
                        "must_be_positive": true/false,
                        "common_endings": [".99", ".00"] or null,
                        "allowed_values": ["list"] or null
                    }},
                    "generation_strategy": "reuse_values|pattern_based|sequential|random_within_range"
                }}
            }},
            "overall_patterns": ["patterns across dataset"],
            "generation_advice": "advice for realistic generation"
        }}
        
        Example response for column with values ["001", "002", "003"]:
        {{
            "column_name": {{
                "pattern_analysis": "Sequential numeric codes with leading zeros",
                "detected_format": "id_like",
                "format_details": {{
                    "is_numeric": true,
                    "is_integer_like": true,
                    "has_decimal_points": false,
                    "decimal_places": 0,
                    "is_sequential": true,
                    "is_unique": true,
                    "text_pattern": "^\d{{3}}$",
                    "length_range": [3, 3],
                    "character_set": "digits only"
                }},
                "value_constraints": {{
                    "min_value": 1,
                    "max_value": 999,
                    "must_be_positive": true,
                    "common_endings": null,
                    "allowed_values": null
                }},
                "generation_strategy": "sequential"
            }}
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a pattern analysis expert. Analyze data patterns without assuming meaning."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=2500
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON
            try:
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Validate and enhance analysis
                    analysis = self._validate_and_enhance_analysis(analysis, df_sample)
                    
                    return analysis
            except json.JSONDecodeError as e:
                st.warning(f"JSON parse error: {e}")
            
            return self._analyze_patterns_statistically(df_sample)
            
        except Exception as e:
            st.warning(f"LLM analysis failed: {str(e)}")
            return self._analyze_patterns_statistically(df_sample)
    
    def _create_compact_representation(self, df):
        """Create compact data representation to save tokens"""
        compact = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": []
        }
        
        for col in df.columns[:20]:  # Limit to 20 columns
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # Take 3 samples, trimmed
                samples = [str(x)[:50] for x in col_data.head(3).tolist()]
                
                # Basic stats
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        stats = {
                            "min": float(numeric_data.min()),
                            "max": float(numeric_data.max()),
                            "mean": float(numeric_data.mean())
                        }
                    else:
                        stats = None
                except:
                    stats = None
                
                compact["columns"].append({
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "samples": samples,
                    "unique_count": col_data.nunique(),
                    "stats": stats
                })
        
        return compact
    
    def _validate_and_enhance_analysis(self, analysis, df):
        """Add statistical validation to LLM analysis"""
        if 'columns' not in analysis:
            return analysis
        
        for col_name, col_info in analysis['columns'].items():
            if col_name in df.columns:
                col_data = df[col_name].dropna()  # Fixed: changed col to col_name
                if len(col_data) > 0:
                    # Add actual statistical data
                    try:
                        # Check if numeric
                        numeric_series = pd.to_numeric(col_data, errors='coerce')
                        numeric_data = numeric_series.dropna()
                        
                        if 'format_details' not in col_info:
                            col_info['format_details'] = {}
                        
                        if len(numeric_data) > 0:
                            col_info['format_details']['is_numeric'] = True
                            col_info['format_details']['actual_min'] = float(numeric_data.min())
                            col_info['format_details']['actual_max'] = float(numeric_data.max())
                            
                            # Check if all are integers
                            if numeric_data.apply(lambda x: float(x).is_integer()).all():
                                col_info['format_details']['is_integer_like'] = True
                            else:
                                # Check decimal places
                                decimal_counts = numeric_data.apply(
                                    lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0
                                )
                                if decimal_counts.max() > 0:
                                    col_info['format_details']['has_decimal_points'] = True
                                    col_info['format_details']['typical_decimal_places'] = int(decimal_counts.mode().iloc[0] if len(decimal_counts.mode()) > 0 else 0)
                        
                        # Check for sequential pattern
                        if len(numeric_data) > 2:
                            sorted_vals = sorted(numeric_data.unique()[:10])
                            diffs = np.diff(sorted_vals)
                            if len(diffs) > 0 and np.allclose(diffs, diffs[0], rtol=0.1):
                                col_info['format_details']['is_sequential'] = True
                        
                        # Check uniqueness
                        unique_ratio = col_data.nunique() / len(col_data)
                        col_info['format_details']['uniqueness_ratio'] = float(unique_ratio)
                        if unique_ratio > 0.95:
                            col_info['format_details']['is_unique'] = True
                        
                    except Exception as e:
                        pass
        
        return analysis
    
    def _analyze_patterns_statistically(self, df):
        """Statistical pattern analysis without LLM"""
        analysis = {
            'columns': {},
            'overall_patterns': ['Statistical pattern analysis'],
            'generation_advice': 'Generate based on statistical distributions'
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info = {
                    'pattern_analysis': 'Statistical pattern detected',
                    'detected_format': 'auto_detect',
                    'format_details': {},
                    'value_constraints': {},
                    'generation_strategy': 'statistical'
                }
                
                # Try numeric analysis
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        col_info['detected_format'] = 'numeric'
                        col_info['format_details']['is_numeric'] = True
                        col_info['format_details']['actual_min'] = float(numeric_data.min())
                        col_info['format_details']['actual_max'] = float(numeric_data.max())
                        col_info['format_details']['actual_mean'] = float(numeric_data.mean())
                        
                        # Check if integers
                        if numeric_data.apply(lambda x: float(x).is_integer()).all():
                            col_info['format_details']['is_integer_like'] = True
                        else:
                            # Check decimal places
                            decimal_counts = numeric_data.apply(
                                lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0
                            )
                            if decimal_counts.max() > 0:
                                col_info['format_details']['has_decimal_points'] = True
                                col_info['format_details']['typical_decimal_places'] = int(decimal_counts.mode().iloc[0] if len(decimal_counts.mode()) > 0 else 2)
                except:
                    col_info['detected_format'] = 'text'
                    col_info['format_details']['is_numeric'] = False
                
                # Check unique values
                unique_ratio = col_data.nunique() / len(col_data)
                col_info['format_details']['uniqueness_ratio'] = float(unique_ratio)
                
                analysis['columns'][col] = col_info
        
        return analysis

# =============================================================================
# SMART DATA GENERATOR (Works for ANY dataset)
# =============================================================================
class SmartDataGenerator:  # Changed from UniversalDataGenerator to SmartDataGenerator to match error
    def __init__(self):
        self.analyzer = UniversalPatternAnalyzer()  # FIXED: Added analyzer attribute
    
    def generate_universal_data(self, original_df, num_rows, noise_level=0.1):
        """
        Generate synthetic data for ANY dataset
        """
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Analyze patterns (ONE LLM CALL)
        with st.spinner("🔍 Analyzing data patterns..."):
            analysis = self.analyzer.analyze_universal_patterns(original_df.head(50))
        
        # Step 2: Generate each column based on patterns
        generated_data = {}
        
        for col in original_df.columns:
            if col in analysis.get('columns', {}):
                col_info = analysis['columns'][col]
                generated_data[col] = self._generate_column_by_pattern(
                    col_name=col,
                    col_info=col_info,
                    original_series=original_df[col] if col in original_df.columns else None,
                    num_rows=num_rows
                )
            else:
                # Fallback: Use original samples
                generated_data[col] = self._generate_fallback(original_df[col], num_rows)
        
        # Step 3: Create DataFrame
        df_generated = pd.DataFrame(generated_data)
        
        # Step 4: Apply post-processing for realism
        df_generated = self._apply_smart_post_processing(df_generated, analysis)
        
        # Step 5: Add controlled noise
        if noise_level > 0:
            df_generated = self._add_intelligent_noise(df_generated, noise_level, analysis)
        
        return df_generated
    
    def _generate_column_by_pattern(self, col_name, col_info, original_series, num_rows):
        """Generate data based on detected patterns"""
        format_details = col_info.get('format_details', {})
        value_constraints = col_info.get('value_constraints', {})
        strategy = col_info.get('generation_strategy', 'statistical')
        
        # Strategy 1: Reuse existing values
        if strategy == 'reuse_values' and original_series is not None:
            unique_vals = original_series.dropna().unique()
            if len(unique_vals) > 0:
                return np.random.choice(unique_vals, num_rows)
        
        # Strategy 2: Generate based on format
        if format_details.get('is_numeric', False):
            return self._generate_numeric_by_pattern(format_details, value_constraints, num_rows, original_series)
        else:
            return self._generate_text_by_pattern(format_details, original_series, num_rows)
    
    def _generate_numeric_by_pattern(self, format_details, constraints, num_rows, original_series=None):
        """Generate numeric data with intelligence"""
        
        # Determine range
        min_val = constraints.get('min_value') or format_details.get('actual_min', 0)
        max_val = constraints.get('max_value') or format_details.get('actual_max', 100)
        
        # Adjust based on original if available
        if original_series is not None:
            try:
                numeric_data = pd.to_numeric(original_series, errors='coerce').dropna()
                if len(numeric_data) > 0:
                    mean_val = float(numeric_data.mean())
                    std_val = float(numeric_data.std()) if len(numeric_data) > 1 else (max_val - min_val) / 4
                else:
                    mean_val = (min_val + max_val) / 2
                    std_val = (max_val - min_val) / 4
            except:
                mean_val = (min_val + max_val) / 2
                std_val = (max_val - min_val) / 4
        else:
            mean_val = (min_val + max_val) / 2
            std_val = (max_val - min_val) / 4
        
        # Generate data
        if format_details.get('is_sequential', False):
            # Sequential numbers
            start = int(min_val) if pd.notnull(min_val) else 1000
            data = list(range(start, start + num_rows))
        
        elif format_details.get('is_integer_like', False):
            # Integer data
            data = np.random.normal(mean_val, std_val, num_rows)
            data = np.clip(data, min_val, max_val).astype(int)
            
            # Ensure positive if required
            if constraints.get('must_be_positive', False):
                data = np.abs(data)
        
        else:
            # Float data with decimal places
            decimal_places = format_details.get('typical_decimal_places', 2)
            
            # Generate with normal distribution
            data = np.random.normal(mean_val, std_val, num_rows)
            data = np.clip(data, min_val, max_val)
            
            # Apply decimal places
            data = np.round(data, decimal_places)
            
            # Apply common endings if specified
            common_endings = constraints.get('common_endings')
            if common_endings and decimal_places == 2:
                for i in range(num_rows):
                    if np.random.random() < 0.3:  # 30% chance
                        integer_part = int(data[i])
                        ending = np.random.choice(common_endings)
                        data[i] = integer_part + float(ending)
        
        return data
    
    def _generate_text_by_pattern(self, format_details, original_series, num_rows):
        """Generate text data based on patterns"""
        if original_series is not None:
            samples = original_series.dropna().astype(str).head(50).tolist()
            
            if len(samples) > 0:
                # Strategy 1: If few unique values, reuse them
                unique_vals = original_series.dropna().unique()
                if len(unique_vals) <= 10:
                    return np.random.choice(unique_vals, num_rows)
                
                # Strategy 2: Detect and replicate patterns
                pattern = self._detect_text_pattern(samples)
                if pattern:
                    return self._generate_by_pattern(pattern, num_rows)
                
                # Strategy 3: Mix and match parts
                return self._mix_and_match_generate(samples, num_rows)
        
        # Fallback: Generate placeholder text
        return [f"Sample_{i}" for i in range(num_rows)]
    
    def _detect_text_pattern(self, samples):
        """Detect patterns in text samples"""
        if not samples:
            return None
        
        # Check for alphanumeric codes
        first = samples[0]
        
        # Pattern 1: Codes with separators (ABC-123)
        if any(sep in first for sep in ['-', '_', '.']):
            parts = re.split(r'[-_.]+', first)
            pattern_parts = []
            for part in parts:
                if part.isdigit():
                    pattern_parts.append('#' * len(part))
                elif part.isalpha():
                    pattern_parts.append('@' * len(part))
                else:
                    pattern_parts.append(part)
            separator = first[re.search(r'[-_.]+', first).start()]
            return separator.join(pattern_parts)
        
        # Pattern 2: Email-like
        if '@' in first and '.' in first:
            return 'email'
        
        # Pattern 3: Date-like
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}']
        for pattern in date_patterns:
            if re.match(pattern, first):
                return 'date'
        
        return None
    
    def _generate_by_pattern(self, pattern, num_rows):
        """Generate data based on pattern"""
        if pattern == 'email':
            domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com']
            return [f"user{i}@{random.choice(domains)}" for i in range(num_rows)]
        
        elif pattern == 'date':
            start = datetime.now() - timedelta(days=365)
            return [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_rows)]
        
        else:
            # Generic pattern with separators
            generated = []
            for _ in range(num_rows):
                item = ''
                for char in pattern:
                    if char == '#':
                        item += str(random.randint(0, 9))
                    elif char == '@':
                        item += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    else:
                        item += char
                generated.append(item)
            return generated
    
    def _mix_and_match_generate(self, samples, num_rows):
        """Generate by mixing parts of existing samples"""
        if not samples:
            return [f"Item_{i}" for i in range(num_rows)]
        
        # Split samples into parts
        parts_list = []
        for sample in samples[:20]:  # Use first 20
            # Split by common separators
            parts = re.split(r'[\s\-_\.]+', sample)
            parts_list.append(parts)
        
        # Generate by combining random parts
        generated = []
        for _ in range(num_rows):
            parts = []
            for i in range(len(parts_list[0])):
                available = [p[i] for p in parts_list if i < len(p)]
                if available:
                    parts.append(random.choice(available))
            generated.append('_'.join(parts))
        
        return generated
    
    def _generate_fallback(self, original_series, num_rows):
        """Fallback generation"""
        if original_series is not None and len(original_series) > 0:
            unique_vals = original_series.dropna().unique()
            if len(unique_vals) > 0:
                return np.random.choice(unique_vals, num_rows)
        
        return [f"Data_{i}" for i in range(num_rows)]
    
    def _apply_smart_post_processing(self, df, analysis):
        """Apply intelligent post-processing"""
        
        for col in df.columns:
            if col in analysis.get('columns', {}):
                col_info = analysis['columns'][col]
                format_details = col_info.get('format_details', {})
                
                # Fix integer columns
                if format_details.get('is_integer_like', False):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].round().astype('Int64')
                    except:
                        pass
                
                # Ensure positive values if required
                constraints = col_info.get('value_constraints', {})
                if constraints.get('must_be_positive', False):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].abs()
                    except:
                        pass
                
                # Apply decimal places
                decimal_places = format_details.get('typical_decimal_places')
                if decimal_places is not None:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].round(decimal_places)
                    except:
                        pass
        
        return df
    
    def _add_intelligent_noise(self, df, noise_level, analysis):
        """Add intelligent noise"""
        for col in df.columns:
            if col in analysis.get('columns', {}):
                col_info = analysis['columns'][col]
                format_details = col_info.get('format_details', {})
                
                if format_details.get('is_numeric', False) and noise_level > 0:
                    try:
                        col_data = pd.to_numeric(df[col], errors='coerce')
                        
                        # Calculate noise based on range
                        if 'actual_min' in format_details and 'actual_max' in format_details:
                            data_range = format_details['actual_max'] - format_details['actual_min']
                            if data_range > 0:
                                noise = np.random.uniform(-noise_level, noise_level, len(df)) * data_range
                                df[col] = col_data + noise
                                
                                # Re-apply constraints
                                if format_details.get('is_integer_like', False):
                                    df[col] = df[col].round().astype('Int64')
                                else:
                                    decimal_places = format_details.get('typical_decimal_places', 2)
                                    df[col] = df[col].round(decimal_places)
                    except:
                        pass
        
        return df

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
    st.markdown("Generate realistic synthetic data for **ANY dataset**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize - FIXED: Use SmartDataGenerator instead of UniversalDataGenerator
    if 'generator' not in st.session_state:
        st.session_state.generator = SmartDataGenerator()  # FIXED
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
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10))
                
                # Analyze button
                if st.button("🔍 Analyze Patterns (One-time LLM)", type="primary"):
                    with st.spinner("Analyzing patterns with LLM..."):
                        analysis = st.session_state.generator.analyzer.analyze_universal_patterns(df.head(50))
                        st.session_state.analysis = analysis
                        
                        st.success("✅ Pattern analysis complete!")
                        
                        # Show analysis
                        with st.expander("📊 Pattern Analysis Results", expanded=True):
                            for col, info in analysis.get('columns', {}).items():
                                st.write(f"**{col}**:")
                                st.json(info, expanded=False)
                
                # Generate section
                if st.session_state.analysis:
                    st.subheader("⚙️ Generate Synthetic Data")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        num_rows = st.select_slider(
                            "Rows to generate",
                            options=[100, 500, 1000, 5000, 10000],
                            value=1000
                        )
                    
                    with col2:
                        noise = st.slider("Variation", 0.0, 0.5, 0.1, 0.05,
                                        help="Adds realistic variation")
                    
                    if st.button("🚀 Generate Realistic Data", type="primary"):
                        with st.spinner("Generating..."):
                            generated = st.session_state.generator.generate_universal_data(
                                original_df=df,
                                num_rows=num_rows,
                                noise_level=noise
                            )
                            
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} rows!")
                            st.balloons()
                
                # Download section
                if st.session_state.generated_data is not None:
                    st.subheader("💾 Download Results")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Preview
                    st.dataframe(df_gen.head(10))
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "synthetic_data.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        if st.button("🔄 Generate More"):
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
