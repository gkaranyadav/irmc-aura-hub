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
# UNIVERSAL PATTERN ANALYZER WITH GENERATION INSTRUCTIONS
# =============================================================================
class UniversalPatternAnalyzer:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.llm_available = True
        except:
            self.llm_available = False
    
    def analyze_with_generation(self, df_sample):
        """
        Ask LLM to analyze AND provide specific generation code
        """
        if not self.llm_available or df_sample.empty:
            return self._basic_analysis(df_sample)
        
        # Get data samples
        data_samples = {}
        for col in df_sample.columns:
            samples = df_sample[col].dropna().head(10).tolist()
            python_samples = []
            for sample in samples:
                if isinstance(sample, (np.integer, np.int64)):
                    python_samples.append(int(sample))
                elif isinstance(sample, (np.floating, np.float64)):
                    python_samples.append(float(sample))
                elif pd.isna(sample):
                    python_samples.append(None)
                else:
                    python_samples.append(str(sample))
            data_samples[col] = python_samples
        
        # CRITICAL PROMPT: Ask LLM for generation logic
        prompt = f"""
        I need to generate synthetic data that matches this pattern.
        
        ORIGINAL DATA SAMPLES:
        {json.dumps(data_samples, indent=2, default=str)}
        
        IMPORTANT: DO NOT predefine any categories in your code. 
        Instead, analyze the PATTERNS and tell me HOW to generate similar data.
        
        For EACH column, provide:
        1. pattern_analysis: What patterns do you see in the values?
        2. generation_logic: EXACT Python code to generate similar values
        3. parameters: What parameters control the generation?
        
        The generation_logic should be a Python function that takes:
        - n: number of values to generate
        - params: dictionary of parameters
        
        EXAMPLE RESPONSE for column with values ["user1@email.com", "john.doe@company.org"]:
        {{
            "Email": {{
                "pattern_analysis": "Email addresses with username@domain format",
                "generation_logic": "def generate_email(n, params):\\n    domains = ['gmail.com', 'yahoo.com', 'company.com']\\n    return [f'user{{i}}@{{random.choice(domains)}}' for i in range(n)]",
                "parameters": {{"domains": ["list", "of", "common", "domains"]}}
            }}
        }}
        
        Return ONLY JSON:
        {{
            "columns": {{
                "column_name": {{
                    "pattern_analysis": "description",
                    "generation_logic": "python_function_code",
                    "parameters": {{"param1": "value1", "param2": ["list", "of", "values"]}}
                }}
            }}
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a data generation expert. Provide Python code to generate synthetic data based on patterns."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON
            try:
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    
                    # Add statistical data
                    for col in df_sample.columns:
                        if col in analysis.get('columns', {}):
                            col_data = df_sample[col].dropna()
                            if len(col_data) > 0:
                                analysis['columns'][col]['actual_stats'] = {
                                    'unique_count': int(col_data.nunique()),
                                    'sample_count': len(col_data)
                                }
                    
                    return analysis
            except json.JSONDecodeError as e:
                st.warning(f"JSON parse error: {e}")
                st.text("LLM Response (first 1000 chars):")
                st.text(analysis_text[:1000])
            
            return self._basic_analysis(df_sample)
            
        except Exception as e:
            st.warning(f"LLM analysis failed: {str(e)}")
            return self._basic_analysis(df_sample)
    
    def _basic_analysis(self, df):
        """Basic analysis without LLM"""
        analysis = {'columns': {}}
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                analysis['columns'][col] = {
                    'pattern_analysis': 'Basic pattern detection',
                    'generation_logic': f"def generate_{col}(n, params):\n    return [f'{col}_{{i}}' for i in range(n)]",
                    'parameters': {}
                }
        return analysis

# =============================================================================
# UNIVERSAL DATA GENERATOR (NO PREDEFINED LOGIC)
# =============================================================================
class UniversalDataGenerator:
    def __init__(self):
        self.analyzer = UniversalPatternAnalyzer()
    
    def generate_data(self, original_df, num_rows):
        """
        Generate data using LLM's generation logic
        """
        if original_df.empty:
            return pd.DataFrame()
        
        # Get generation instructions from LLM
        with st.spinner("🤖 Getting generation logic from AI..."):
            analysis = self.analyzer.analyze_with_generation(original_df.head(50))
            st.session_state.llm_analysis = analysis
        
        # Execute LLM's generation logic
        generated_data = {}
        
        for col in original_df.columns:
            if col in analysis.get('columns', {}):
                col_info = analysis['columns'][col]
                generation_logic = col_info.get('generation_logic', '')
                parameters = col_info.get('parameters', {})
                
                # Try to execute LLM's generation code
                try:
                    generated_values = self._execute_llm_generation_logic(
                        col_name=col,
                        generation_logic=generation_logic,
                        parameters=parameters,
                        num_rows=num_rows,
                        original_values=original_df[col].dropna().tolist() if col in original_df.columns else []
                    )
                    generated_data[col] = generated_values
                    
                except Exception as e:
                    st.warning(f"Failed to generate {col}: {str(e)}")
                    # Fallback: use original patterns
                    generated_data[col] = self._generate_from_patterns(original_df[col], num_rows)
            else:
                generated_data[col] = self._generate_from_patterns(original_df[col], num_rows)
        
        return pd.DataFrame(generated_data)
    
    def _execute_llm_generation_logic(self, col_name, generation_logic, parameters, num_rows, original_values):
        """Execute LLM's generation code"""
        
        # Extract function from generation_logic
        function_match = re.search(r'def (\w+)\(.*?\):(.*?)(?=\ndef|\Z)', generation_logic, re.DOTALL)
        
        if function_match:
            func_name = function_match.group(1)
            func_body = function_match.group(2)
            
            # Create the function dynamically
            function_code = f"""
import random
import string
import numpy as np

{generation_logic}

# Generate the data
result = {func_name}({num_rows}, {parameters})
"""
            
            # Execute in a safe context
            safe_globals = {
                'random': random,
                'string': string,
                'np': np,
                '__builtins__': {}
            }
            
            try:
                exec(function_code, safe_globals)
                result = safe_globals.get('result', [])
                
                if result and len(result) == num_rows:
                    return result
            
            except Exception as e:
                st.warning(f"Could not execute LLM code for {col_name}: {e}")
        
        # If LLM code fails, use smart pattern generation
        return self._smart_generate_from_values(col_name, original_values, num_rows)
    
    def _smart_generate_from_values(self, col_name, original_values, num_rows):
        """Smart generation based on actual values"""
        if not original_values:
            return [f"{col_name}_{i}" for i in range(num_rows)]
        
        # Analyze the values
        samples = []
        for val in original_values[:20]:  # Use first 20 samples
            if isinstance(val, (np.integer, np.int64)):
                samples.append(int(val))
            elif isinstance(val, (np.floating, np.float64)):
                samples.append(float(val))
            elif pd.isna(val):
                samples.append(None)
            else:
                samples.append(str(val))
        
        # Check if numeric
        numeric_values = []
        for val in samples:
            if val is not None:
                try:
                    numeric_values.append(float(val))
                except:
                    pass
        
        if len(numeric_values) > len(samples) * 0.7:  # Mostly numeric
            # Generate numeric data
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            mean_val = np.mean(numeric_values)
            
            if all(v.is_integer() for v in numeric_values if isinstance(v, float)):
                # Integers
                data = np.random.normal(mean_val, (max_val - min_val) / 4, num_rows)
                data = np.clip(data, min_val, max_val)
                return [int(round(x)) for x in data]
            else:
                # Floats
                data = np.random.normal(mean_val, (max_val - min_val) / 4, num_rows)
                data = np.clip(data, min_val, max_val)
                return [float(x) for x in data]
        
        else:
            # Text data - analyze patterns
            text_samples = [str(s) for s in samples if s is not None]
            
            # Check for common patterns
            if len(text_samples) > 0:
                # Check for email pattern
                if any('@' in s for s in text_samples[:5]):
                    return self._generate_email_like(text_samples, num_rows)
                
                # Check for date pattern
                date_patterns = [r'\d{4}[-/]\d{2}[-/]\d{2}', r'\d{2}[-/]\d{2}[-/]\d{4}']
                if any(any(re.match(p, s) for p in date_patterns) for s in text_samples[:3]):
                    return self._generate_date_like(num_rows)
                
                # Check for name pattern
                if all(len(s.split()) >= 2 and s[0].isupper() for s in text_samples[:3]):
                    return self._generate_name_like(text_samples, num_rows)
            
            # Generic text generation based on patterns
            return self._generate_text_from_patterns(text_samples, num_rows)
    
    def _generate_email_like(self, samples, num_rows):
        """Generate email-like patterns"""
        # Extract patterns from samples
        domains = set()
        for sample in samples:
            if '@' in sample:
                parts = sample.split('@')
                if len(parts) == 2:
                    domains.add(parts[1])
        
        domain_list = list(domains) if domains else ['example.com', 'test.com', 'demo.org']
        
        emails = []
        for i in range(num_rows):
            username = f"user{random.randint(1, 9999)}"
            if random.random() < 0.5:
                username = f"{random.choice(['john', 'jane', 'alex', 'sam'])}.{random.choice(['doe', 'smith', 'jones', 'brown'])}{random.randint(1, 99)}"
            
            emails.append(f"{username}@{random.choice(domain_list)}")
        
        return emails
    
    def _generate_date_like(self, num_rows):
        """Generate date-like patterns"""
        import datetime
        
        dates = []
        for i in range(num_rows):
            days_ago = random.randint(1, 365*2)
            date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
            
            if random.random() < 0.5:
                dates.append(date.strftime('%Y-%m-%d'))
            else:
                dates.append(date.strftime('%d/%m/%Y'))
        
        return dates
    
    def _generate_name_like(self, samples, num_rows):
        """Generate name-like patterns"""
        # Extract name parts
        first_parts = []
        last_parts = []
        
        for sample in samples:
            parts = sample.split()
            if len(parts) >= 2:
                first_parts.append(parts[0])
                last_parts.append(' '.join(parts[1:]))
        
        if not first_parts:
            first_parts = ['John', 'Jane', 'Alex', 'Sam', 'Mike', 'Sarah']
        if not last_parts:
            last_parts = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']
        
        names = []
        for i in range(num_rows):
            first = random.choice(first_parts)
            last = random.choice(last_parts)
            names.append(f"{first} {last}")
        
        return names
    
    def _generate_text_from_patterns(self, samples, num_rows):
        """Generate text based on observed patterns"""
        if not samples:
            return [f"val_{i}" for i in range(num_rows)]
        
        # Analyze character patterns
        all_chars = ''.join(samples)
        has_letters = any(c.isalpha() for c in all_chars)
        has_digits = any(c.isdigit() for c in all_chars)
        has_upper = any(c.isupper() for c in all_chars)
        has_lower = any(c.islower() for c in all_chars)
        has_symbols = any(not c.isalnum() and not c.isspace() for c in all_chars)
        
        # Get length range
        lengths = [len(s) for s in samples]
        avg_len = np.mean(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        
        # Generate similar patterns
        results = []
        for i in range(num_rows):
            # Determine length
            length = int(np.random.normal(avg_len, avg_len/3))
            length = max(min_len, min(max_len, length))
            
            # Build string
            chars = []
            for _ in range(length):
                if has_digits and has_letters:
                    # Mix of digits and letters
                    if random.random() < 0.5:
                        chars.append(random.choice('0123456789'))
                    else:
                        if has_upper and has_lower:
                            if random.random() < 0.5:
                                chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                            else:
                                chars.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
                        elif has_upper:
                            chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                        else:
                            chars.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
                elif has_digits:
                    chars.append(random.choice('0123456789'))
                else:
                    if has_upper and has_lower:
                        if random.random() < 0.5:
                            chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                        else:
                            chars.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
                    elif has_upper:
                        chars.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                    else:
                        chars.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
            
            results.append(''.join(chars))
        
        return results
    
    def _generate_from_patterns(self, original_series, num_rows):
        """Generate from original data patterns"""
        if original_series is not None:
            values = original_series.dropna().tolist()
            if values:
                return self._smart_generate_from_values(original_series.name if hasattr(original_series, 'name') else 'col', 
                                                       values, num_rows)
        
        return [f"data_{i}" for i in range(num_rows)]

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
    st.markdown("Generate realistic data for **ANY dataset** - No predefined logic")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize
    if 'generator' not in st.session_state:
        st.session_state.generator = UniversalDataGenerator()
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload ANY CSV Dataset", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.error("Empty file")
            else:
                st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                
                # Preview
                with st.expander("📋 Original Data Preview", expanded=True):
                    st.dataframe(df.head(10))
                
                # Generation controls
                st.subheader("⚙️ Generation Settings")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_rows = st.number_input("Number of rows to generate", 
                                             min_value=10, 
                                             max_value=10000, 
                                             value=1000)
                
                with col2:
                    if st.button("🚀 Generate with AI Logic", type="primary"):
                        with st.spinner("AI is analyzing patterns and generating..."):
                            generated = st.session_state.generator.generate_data(df, int(num_rows))
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} rows using AI logic!")
                            st.balloons()
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📋 Generated Data")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Preview
                    st.dataframe(df_gen.head(20))
                    
                    # Show LLM analysis if available
                    if 'llm_analysis' in st.session_state:
                        with st.expander("🤖 AI Generation Logic", expanded=False):
                            analysis = st.session_state.llm_analysis
                            for col, info in analysis.get('columns', {}).items():
                                st.write(f"**{col}**")
                                st.write(f"Pattern analysis: {info.get('pattern_analysis', 'N/A')}")
                                
                                logic = info.get('generation_logic', '')
                                if logic:
                                    st.code(logic, language='python')
                                st.write("---")
                    
                    # Download
                    st.subheader("💾 Download")
                    
                    csv = df_gen.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "ai_generated_data.csv",
                        "text/csv"
                    )
                    
                    if st.button("🔄 Generate New"):
                        st.session_state.generated_data = None
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
