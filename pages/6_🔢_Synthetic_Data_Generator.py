# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import random
import math
from datetime import datetime, timedelta
from groq import Groq
from auth import check_session
from typing import Dict, List, Any

# =============================================================================
# UNIVERSAL LLM DATA GENERATOR
# =============================================================================

class UniversalDataGenerator:
    """Universal generator that adapts to ANY dataset"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
        except:
            self.available = False
            st.warning("LLM not available")
    
    def analyze_dataset(self, df):
        """Analyze dataset to understand structure and patterns"""
        analysis = {
            "columns": {},
            "types": {},
            "patterns": {},
            "sample_data": {}
        }
        
        for col in df.columns:
            # Skip empty columns
            if df[col].isnull().all():
                analysis["columns"][col] = "empty"
                continue
            
            # Get sample values
            non_null_vals = df[col].dropna()
            if len(non_null_vals) > 0:
                sample_vals = non_null_vals.head(5).tolist()
                analysis["sample_data"][col] = sample_vals
            
            # Detect data type
            col_str = df[col].astype(str)
            
            # Check for categorical
            unique_ratio = len(df[col].unique()) / len(df[col]) if len(df[col]) > 0 else 0
            if unique_ratio < 0.3 and len(df[col].unique()) <= 20:
                analysis["types"][col] = "categorical"
                analysis["patterns"][col] = {
                    "type": "categorical",
                    "values": df[col].unique().tolist()
                }
            
            # Check for numeric
            elif df[col].dtype in ['int64', 'float64'] or all(re.match(r'^-?\d+\.?\d*$', str(x)) for x in non_null_vals.head(10) if pd.notnull(x)):
                analysis["types"][col] = "numeric"
                if len(non_null_vals) > 0:
                    analysis["patterns"][col] = {
                        "type": "numeric",
                        "min": float(non_null_vals.min()),
                        "max": float(non_null_vals.max()),
                        "mean": float(non_null_vals.mean())
                    }
            
            # Check for dates
            elif any('date' in col.lower() or 'time' in col.lower() for x in [col]):
                date_patterns = [r'\d{2}[-/]\d{2}[-/]\d{4}', r'\d{4}[-/]\d{2}[-/]\d{2}']
                if any(any(re.search(pattern, str(x)) for pattern in date_patterns) for x in non_null_vals.head(10) if pd.notnull(x)):
                    analysis["types"][col] = "date"
            
            # Check for IDs
            elif any(x in col.lower() for x in ['id', 'code', 'num', 'no', 'number']) and all(re.match(r'^[A-Za-z0-9_-]+$', str(x)) for x in non_null_vals.head(10) if pd.notnull(x)):
                analysis["types"][col] = "id"
            
            # Check for names
            elif any(x in col.lower() for x in ['name', 'person', 'customer', 'user', 'client']):
                analysis["types"][col] = "name"
            
            # Check for emails
            elif any(x in col.lower() for x in ['email', 'mail']):
                analysis["types"][col] = "email"
            
            # Default to text
            else:
                analysis["types"][col] = "text"
        
        return analysis
    
    def generate_perfect_data(self, original_df, num_rows):
        """Generate PERFECT data that matches ANY dataset structure"""
        if not self.available or original_df.empty:
            return self._smart_fallback(original_df, num_rows)
        
        # Analyze the dataset first
        analysis = self.analyze_dataset(original_df)
        
        # Get LLM to generate data
        with st.spinner("🤖 LLM is analyzing your data and generating perfect synthetic data..."):
            llm_data = self._get_llm_generated_data(original_df, num_rows, analysis)
        
        if llm_data is not None:
            return llm_data
        
        return self._smart_fallback(original_df, num_rows)
    
    def _get_llm_generated_data(self, df, num_rows, analysis):
        """Get LLM to generate data for ANY dataset"""
        
        # Prepare samples
        samples = []
        for i, (idx, row) in enumerate(df.head(5).iterrows()):
            sample = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    sample[col] = None
                elif isinstance(val, (int, np.integer)):
                    sample[col] = int(val)
                elif isinstance(val, (float, np.floating)):
                    sample[col] = float(val)
                else:
                    sample[col] = str(val)
            samples.append(sample)
        
        # Build dynamic prompt based on analysis
        prompt = self._build_dynamic_prompt(df, num_rows, samples, analysis)
        
        try:
            messages = [
                {"role": "system", "content": """You are a data generation expert. You generate PERFECTLY realistic synthetic data that matches ANY dataset structure.
                
                CRITICAL INSTRUCTIONS:
                1. Analyze the given data structure and patterns
                2. Generate data that follows the SAME patterns and distributions
                3. All values MUST be realistic and make sense
                4. NO placeholder values (no dummy_1, temp_user, test@email.com)
                5. Maintain data types and formats
                6. Everything MUST be logically consistent
                
                You MUST adapt to whatever dataset is provided."""},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=8000
            )
            
            result = response.choices[0].message.content
            
            # Parse the response
            return self._parse_llm_response(result, df.columns, num_rows, analysis)
            
        except Exception as e:
            st.error(f"LLM generation failed: {str(e)}")
            return None
    
    def _build_dynamic_prompt(self, df, num_rows, samples, analysis):
        """Build dynamic prompt based on actual dataset"""
        
        column_descriptions = []
        for col in df.columns:
            col_type = analysis["types"].get(col, "unknown")
            col_samples = analysis["sample_data"].get(col, [])
            
            desc = f"- {col}: "
            if col_type == "id":
                desc += f"ID field. Samples: {col_samples[:3] if col_samples else 'N/A'}"
            elif col_type == "name":
                desc += f"Name field. Samples: {col_samples[:3] if col_samples else 'N/A'}"
            elif col_type == "email":
                desc += f"Email field. Samples: {col_samples[:3] if col_samples else 'N/A'}"
            elif col_type == "numeric":
                pattern = analysis["patterns"].get(col, {})
                desc += f"Numeric field. Range: {pattern.get('min', 'N/A')} to {pattern.get('max', 'N/A')}"
            elif col_type == "categorical":
                pattern = analysis["patterns"].get(col, {})
                values = pattern.get("values", [])
                desc += f"Categorical field. Values: {values[:5] if len(values) > 0 else 'N/A'}"
            elif col_type == "date":
                desc += f"Date field. Samples: {col_samples[:3] if col_samples else 'N/A'}"
            else:
                desc += f"Text/Other field. Samples: {col_samples[:3] if col_samples else 'N/A'}"
            column_descriptions.append(desc)
        
        prompt = f"""
        CRITICAL TASK: Generate {num_rows} rows of SYNTHETIC DATA that perfectly matches the structure and patterns of the given dataset.

        ORIGINAL DATASET INFO:
        - Number of columns: {len(df.columns)}
        - Columns: {', '.join(df.columns.tolist())}
        - Sample rows (first 5):
        {json.dumps(samples, indent=2, default=str)}

        COLUMN ANALYSIS:
        {chr(10).join(column_descriptions)}

        IMPORTANT INSTRUCTIONS:
        1. Generate data that follows the EXACT SAME patterns as the original
        2. Use the SAME data types and formats for each column
        3. If a column has specific values (like statuses), use similar realistic values
        4. If a column has numeric ranges, stay within similar ranges
        5. If a column has names/emails, generate REALISTIC ones
        6. Maintain logical consistency between related columns

        GENERATION RULES:
        - IDs: Continue sequence if sequential, otherwise generate similar format IDs
        - Names: Use realistic human names if name-like, otherwise similar text patterns
        - Emails: Generate proper email formats if email field
        - Numbers: Stay within observed ranges with realistic distributions
        - Categories: Use observed categories with similar distributions
        - Dates: Generate dates in same format and time range
        - Text: Generate realistic text that matches the pattern

        RETURN FORMAT: JSON array ONLY:
        [
            {{
                "{df.columns[0] if len(df.columns) > 0 else 'col1'}": "value1",
                "{df.columns[1] if len(df.columns) > 1 else 'col2'}": "value2",
                ...
            }},
            ...
        ]

        Generate EXACTLY {num_rows} rows. Make EVERY value realistic and match the original patterns!
        """
        
        return prompt
    
    def _parse_llm_response(self, response, expected_columns, num_rows, analysis):
        """Parse LLM's response"""
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Convert to DataFrame
                df = pd.DataFrame(data[:num_rows])
                
                # Apply validation and fixing
                df = self._validate_and_fix_data(df, expected_columns, analysis)
                
                return df
            
            # Alternative parsing
            lines = response.strip().split('\n')
            data = []
            current_obj = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                    except:
                        pass
                elif ':' in line and not line.startswith('```'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().replace('"', '').replace("'", "")
                        value = parts[1].strip().replace('"', '').replace("'", "")
                        current_obj[key] = value
                
                if line.endswith('},') or line.endswith('}'):
                    if current_obj:
                        data.append(current_obj.copy())
                        current_obj = {}
            
            if data:
                df = pd.DataFrame(data[:num_rows])
                df = self._validate_and_fix_data(df, expected_columns, analysis)
                return df
            
        except Exception as e:
            st.warning(f"Failed to parse LLM response: {str(e)}")
            # Show raw response for debugging
            with st.expander("Debug: LLM Raw Response"):
                st.code(response[:2000])
        
        return None
    
    def _validate_and_fix_data(self, df, expected_columns, analysis):
        """Validate and fix data quality based on analysis"""
        # Ensure all columns exist
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        
        # Fix each column based on analysis
        for col in expected_columns:
            if col in df.columns and not df[col].isnull().all():
                col_type = analysis["types"].get(col, "unknown")
                pattern = analysis["patterns"].get(col, {})
                
                # Fix based on type
                for idx in range(len(df)):
                    val = df.at[idx, col]
                    
                    if col_type == "id":
                        if pd.isna(val) or str(val).strip() == "":
                            df.at[idx, col] = f"{col}_{idx+1000}"
                    
                    elif col_type == "name":
                        if pd.isna(val) or not any(c.isalpha() for c in str(val)):
                            df.at[idx, col] = self._generate_real_name()
                    
                    elif col_type == "email":
                        if pd.isna(val) or '@' not in str(val):
                            df.at[idx, col] = self._generate_email_from_name("User", idx)
                    
                    elif col_type == "numeric":
                        try:
                            num_val = float(str(val).replace(',', ''))
                            if pattern:
                                min_val = pattern.get("min", 0)
                                max_val = pattern.get("max", 1000)
                                if num_val < min_val * 0.5 or num_val > max_val * 2:
                                    df.at[idx, col] = random.uniform(min_val, max_val)
                        except:
                            if pattern:
                                df.at[idx, col] = random.uniform(pattern.get("min", 0), pattern.get("max", 100))
                    
                    elif col_type == "categorical":
                        if pattern and "values" in pattern:
                            if pd.isna(val) or str(val) not in pattern["values"]:
                                df.at[idx, col] = random.choice(pattern["values"])
        
        # Reorder columns
        df = df[expected_columns]
        
        return df
    
    def _generate_real_name(self):
        """Generate realistic human name"""
        western_first = ['John', 'James', 'Michael', 'David', 'Robert', 'William', 'Mary', 'Patricia', 
                        'Jennifer', 'Linda', 'Elizabeth', 'Susan', 'Jessica', 'Sarah']
        indian_first = ['Rahul', 'Amit', 'Raj', 'Sanjay', 'Vikram', 'Priya', 'Neha', 'Anjali', 'Sneha']
        
        western_last = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis']
        indian_last = ['Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma']
        
        if random.random() < 0.6:
            first = random.choice(western_first)
            last = random.choice(western_last)
        else:
            first = random.choice(indian_first)
            last = random.choice(indian_last)
        
        return f"{first} {last}"
    
    def _generate_email_from_name(self, name, idx):
        """Generate realistic email from name"""
        parts = str(name).split()
        if len(parts) >= 2:
            first = parts[0].lower()
            last = parts[-1].lower()
            patterns = [
                f"{first}.{last}{random.randint(1,99)}@gmail.com",
                f"{first[0]}{last}{random.randint(10,99)}@yahoo.com",
                f"{first}.{last}@company.com"
            ]
        else:
            patterns = [
                f"user{idx+1000}@gmail.com",
                f"client{idx+500}@company.com"
            ]
        
        return random.choice(patterns)
    
    def _smart_fallback(self, df, num_rows):
        """Smart fallback generation that adapts to any dataset"""
        generated = {}
        
        for col in df.columns:
            col_lower = col.lower()
            original_vals = df[col].dropna().tolist()
            
            # Skip if all null
            if not original_vals:
                generated[col] = [None] * num_rows
                continue
            
            # Analyze column content
            sample_str = str(original_vals[0]) if original_vals else ""
            
            # Detect column type
            is_id = any(x in col_lower for x in ['id', 'code', 'num', 'no', 'number'])
            is_name = any(x in col_lower for x in ['name', 'person', 'customer', 'user'])
            is_email = any(x in col_lower for x in ['email', 'mail'])
            is_date = any(x in col_lower for x in ['date', 'time'])
            is_numeric = all(isinstance(x, (int, float, np.number)) or (isinstance(x, str) and re.match(r'^-?\d+\.?\d*$', x)) 
                           for x in original_vals[:10] if x is not None)
            
            # Generate based on type
            if is_id:
                # ID field
                if len(set(original_vals)) > len(original_vals) * 0.8:  # Mostly unique
                    start = 1000
                    if original_vals and any(str(x).isdigit() for x in original_vals[:5]):
                        nums = [int(re.sub(r'\D', '', str(x))) for x in original_vals[:5] if str(x).isdigit()]
                        if nums:
                            start = max(nums)
                    generated[col] = [str(start + i) for i in range(1, num_rows + 1)]
                else:
                    generated[col] = random.choices(original_vals, k=num_rows)
            
            elif is_name:
                # Name field
                names = []
                for i in range(num_rows):
                    if random.random() < 0.6:
                        first = random.choice(['John', 'James', 'Michael', 'David', 'Mary', 'Patricia', 'Linda'])
                        last = random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])
                    else:
                        first = random.choice(['Rahul', 'Amit', 'Raj', 'Priya', 'Neha', 'Anjali'])
                        last = random.choice(['Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta'])
                    names.append(f"{first} {last}")
                generated[col] = names
            
            elif is_email:
                # Email field
                emails = []
                for i in range(num_rows):
                    first = random.choice(['john', 'jane', 'alex', 'sam', 'mike'])
                    last = random.choice(['smith', 'johnson', 'williams', 'brown'])
                    num = random.randint(1, 99)
                    domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.com'])
                    emails.append(f"{first}.{last}{num}@{domain}")
                generated[col] = emails
            
            elif is_date:
                # Date field
                dates = []
                for i in range(num_rows):
                    days_ago = random.randint(1, 365)
                    date = datetime.now() - timedelta(days=days_ago)
                    dates.append(date.strftime('%d-%m-%Y'))
                generated[col] = dates
            
            elif is_numeric:
                # Numeric field
                if original_vals:
                    nums = [float(x) for x in original_vals if x is not None and str(x).strip() != '']
                    if nums:
                        min_val = min(nums)
                        max_val = max(nums)
                        values = []
                        for i in range(num_rows):
                            val = random.uniform(min_val, max_val)
                            if random.random() < 0.3:
                                val = round(val, 2)
                            values.append(val)
                        generated[col] = values
                    else:
                        generated[col] = [random.uniform(0, 100) for _ in range(num_rows)]
                else:
                    generated[col] = [random.uniform(0, 100) for _ in range(num_rows)]
            
            else:
                # Text or categorical
                if len(set(original_vals)) <= 10:  # Categorical
                    generated[col] = random.choices(original_vals, k=num_rows)
                else:
                    # Generate similar text patterns
                    values = []
                    for i in range(num_rows):
                        if original_vals:
                            base = random.choice(original_vals)
                            if isinstance(base, str):
                                values.append(f"{base}_{i}")
                            else:
                                values.append(f"Value_{i}")
                        else:
                            values.append(f"Data_{i}")
                    generated[col] = values
        
        return pd.DataFrame(generated)


# =============================================================================
# STREAMLIT APP
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
    st.title("✨ Universal Data Generator")
    st.markdown("**LLM-Powered • Works with ANY Dataset • No Predefined Schemas**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize generator
    if 'universal_generator' not in st.session_state:
        st.session_state.universal_generator = UniversalDataGenerator()
    
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload ANY Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if df.empty:
                st.error("Empty file")
            else:
                st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                
                # Analyze dataset
                if st.session_state.data_analysis is None:
                    with st.spinner("Analyzing dataset structure..."):
                        st.session_state.data_analysis = st.session_state.universal_generator.analyze_dataset(df)
                
                # Preview with analysis
                with st.expander("📋 Dataset Preview & Analysis", expanded=True):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(df.head(10))
                    
                    with col2:
                        st.write("**Dataset Analysis:**")
                        analysis = st.session_state.data_analysis
                        for col in df.columns[:10]:  # Show first 10 columns
                            col_type = analysis["types"].get(col, "unknown")
                            st.write(f"• **{col}**: {col_type}")
                        if len(df.columns) > 10:
                            st.write(f"... and {len(df.columns) - 10} more columns")
                
                # Generation controls
                st.subheader("⚙️ Generate Synthetic Data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=500,
                                             value=100)
                
                with col2:
                    use_llm = st.checkbox("Use AI for better quality", value=True)
                
                with col3:
                    if st.button("🚀 Generate Data", type="primary"):
                        if use_llm and not st.session_state.universal_generator.available:
                            st.warning("LLM not available. Using smart fallback.")
                        
                        with st.spinner("Generating synthetic data..."):
                            generator = st.session_state.universal_generator
                            if use_llm and generator.available:
                                generated = generator.generate_perfect_data(df, int(num_rows))
                            else:
                                generated = generator._smart_fallback(df, int(num_rows))
                            
                            st.session_state.generated_data = generated
                            if generated is not None:
                                st.success(f"✅ Generated {len(generated)} rows!")
                                st.balloons()
                            else:
                                st.error("Failed to generate data")
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📊 Generated Synthetic Data")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Comparison", "Quality Check", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows Generated", len(df_gen))
                        with col2:
                            st.metric("Columns", len(df_gen.columns))
                        with col3:
                            null_percent = (df_gen.isnull().sum().sum() / (len(df_gen) * len(df_gen.columns))) * 100
                            st.metric("Null Values", f"{null_percent:.1f}%")
                    
                    with tab2:
                        st.write("## Original vs Generated (First 5 rows)")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original Data**")
                            st.dataframe(df.head(5))
                        with col2:
                            st.write("**Generated Data**")
                            st.dataframe(df_gen.head(5))
                    
                    with tab3:
                        st.write("## Data Quality Report")
                        
                        quality_issues = []
                        good_columns = []
                        
                        for col in df_gen.columns:
                            issues = []
                            
                            # Check for nulls
                            null_count = df_gen[col].isnull().sum()
                            if null_count > 0:
                                issues.append(f"{null_count} null values")
                            
                            # Check for unique values
                            unique_count = df_gen[col].nunique()
                            if unique_count == 1 and len(df_gen) > 1:
                                issues.append("Only 1 unique value")
                            
                            # Check for empty strings
                            if df_gen[col].dtype == 'object':
                                empty_count = (df_gen[col].astype(str).str.strip() == '').sum()
                                if empty_count > 0:
                                    issues.append(f"{empty_count} empty strings")
                            
                            if issues:
                                quality_issues.append((col, issues))
                            else:
                                good_columns.append(col)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"✅ Good Columns: {len(good_columns)}")
                            for col in good_columns[:10]:
                                st.write(f"• {col}")
                        
                        with col2:
                            if quality_issues:
                                st.warning(f"⚠️ Columns with issues: {len(quality_issues)}")
                                for col, issues in quality_issues[:5]:
                                    st.write(f"• **{col}**: {', '.join(issues)}")
                            else:
                                st.success("All columns look good!")
                    
                    with tab4:
                        # Download
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "synthetic_data.csv",
                            "text/csv"
                        )
                        
                        # Regenerate options
                        st.write("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Generate New Dataset"):
                                st.session_state.generated_data = None
                                st.rerun()
                        with col2:
                            if st.button("📊 Analyze New File"):
                                st.session_state.generated_data = None
                                st.session_state.data_analysis = None
                                st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__": 
    main()
