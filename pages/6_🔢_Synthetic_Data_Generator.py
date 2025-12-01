# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
import json
import io
from groq import Groq
from faker import Faker
import hashlib
import time
from auth import check_session

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    MAX_SAMPLE_ROWS = 100  # Max rows to analyze with LLM
    MAX_GENERATE_ROWS = 10000  # Maximum rows to generate

# Initialize Faker for realistic data generation
faker = Faker()

# =============================================================================
# DATA TYPE DETECTOR
# =============================================================================
class DataTypeDetector:
    def __init__(self):
        pass
    
    def detect_column_type(self, series):
        """Detect the type of data in a column"""
        # Check if series is empty
        if len(series) == 0:
            return "unknown"
        
        # Convert to string for pattern matching
        sample_str = series.dropna().astype(str).iloc[0] if not series.dropna().empty else ""
        
        # Check for date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or M/D/YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, str(sample_str)):
                return "date"
        
        # Check for email
        if '@' in sample_str and '.' in sample_str:
            return "email"
        
        # Check for phone number
        phone_patterns = [r'\d{10}', r'\d{3}-\d{3}-\d{4}', r'\(\d{3}\) \d{3}-\d{4}']
        for pattern in phone_patterns:
            if re.match(pattern, sample_str.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')):
                return "phone"
        
        # Check for boolean
        if sample_str.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
            return "boolean"
        
        # Check for numeric
        try:
            float(series.dropna().iloc[0])
            # Check if all values are numeric
            numeric_count = sum(pd.to_numeric(series, errors='coerce').notna())
            if numeric_count / len(series) > 0.8:  # 80% numeric
                return "numeric"
        except:
            pass
        
        # Check for categorical (limited unique values)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.3 and series.nunique() < 20:
            return "categorical"
        
        # Default to text
        return "text"

# =============================================================================
# LLM PATTERN ANALYZER
# =============================================================================
class LLMPatternAnalyzer:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None
    
    def analyze_data_patterns(self, df_sample):
        """Use LLM to analyze patterns in the data"""
        if not self.client:
            return {}
        
        try:
            # Limit to first 50 rows for token efficiency
            sample_df = df_sample.head(50)
            
            # Convert to JSON for LLM
            sample_json = sample_df.head(10).to_dict(orient='records')
            
            prompt = f"""
            Analyze this dataset sample and identify patterns for synthetic data generation:
            
            SAMPLE DATA (first 10 rows):
            {json.dumps(sample_json, indent=2)}
            
            COLUMNS: {list(df_sample.columns)}
            TOTAL ROWS IN SAMPLE: {len(sample_df)}
            
            Analyze and return JSON with:
            1. For each column: data type (name, email, phone, date, number, category, text, id)
            2. Patterns detected (e.g., "age between 20-60", "dates in 2023", "names follow First Last pattern")
            3. Relationships between columns (e.g., "city and state go together", "price depends on category")
            4. Data distributions (uniform, normal, skewed, etc.)
            5. Unique constraints or patterns
            
            Return ONLY valid JSON:
            {{
                "columns": {{
                    "column_name": {{
                        "type": "detected_type",
                        "pattern": "pattern_description",
                        "unique_values": ["value1", "value2"],
                        "range": [min, max] if numeric,
                        "date_range": ["start_date", "end_date"] if date
                    }}
                }},
                "relationships": ["colA → colB dependency"],
                "overall_patterns": ["overall pattern descriptions"]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a data analysis expert. Analyze datasets for patterns to generate synthetic data."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    patterns = json.loads(json_match.group())
                    return patterns
            except:
                pass
            
            return {}
            
        except Exception as e:
            st.warning(f"⚠️ LLM analysis failed: {str(e)}")
            return {}

# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================
class SyntheticDataGenerator:
    def __init__(self):
        self.detector = DataTypeDetector()
        self.llm_analyzer = LLMPatternAnalyzer()
    
    def analyze_dataset(self, df):
        """Analyze dataset and create generation model"""
        if df.empty:
            return None
        
        # Basic analysis
        analysis = {
            'original_rows': len(df),
            'original_columns': list(df.columns),
            'column_types': {},
            'column_stats': {},
            'patterns': {},
            'relationships': []
        }
        
        # Detect column types
        for col in df.columns:
            col_type = self.detector.detect_column_type(df[col])
            analysis['column_types'][col] = col_type
            
            # Calculate basic statistics
            if col_type == 'numeric':
                analysis['column_stats'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std())
                }
            elif col_type == 'date':
                try:
                    dates = pd.to_datetime(df[col], errors='coerce')
                    analysis['column_stats'][col] = {
                        'min': dates.min(),
                        'max': dates.max()
                    }
                except:
                    pass
            elif col_type in ['categorical', 'text']:
                analysis['column_stats'][col] = {
                    'unique_values': df[col].unique().tolist(),
                    'value_counts': df[col].value_counts().to_dict()
                }
        
        # Use LLM for pattern analysis (on sample)
        sample_df = df.head(min(100, len(df)))
        llm_patterns = self.llm_analyzer.analyze_data_patterns(sample_df)
        analysis['llm_patterns'] = llm_patterns
        
        # Detect simple correlations
        numeric_cols = [col for col, t in analysis['column_types'].items() if t == 'numeric']
        if len(numeric_cols) > 1:
            numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            correlation = numeric_df.corr().to_dict()
            analysis['correlations'] = correlation
        
        return analysis
    
    def generate_data(self, analysis, num_rows, noise_level=0.1):
        """Generate synthetic data based on analysis"""
        if not analysis:
            return pd.DataFrame()
        
        generated_data = {}
        columns = analysis['original_columns']
        
        # Track relationships for dependent columns
        relationships = analysis.get('llm_patterns', {}).get('relationships', [])
        
        for col in columns:
            col_type = analysis['column_types'].get(col, 'text')
            col_stats = analysis['column_stats'].get(col, {})
            
            if col_type == 'numeric':
                min_val = col_stats.get('min', 0)
                max_val = col_stats.get('max', 100)
                mean_val = col_stats.get('mean', (min_val + max_val) / 2)
                std_val = col_stats.get('std', (max_val - min_val) / 4)
                
                # Generate normal distribution with noise
                base_data = np.random.normal(mean_val, std_val, num_rows)
                base_data = np.clip(base_data, min_val, max_val)
                
                # Add noise
                noise = np.random.uniform(-noise_level, noise_level, num_rows) * (max_val - min_val)
                generated_data[col] = base_data + noise
                
            elif col_type == 'date':
                min_date = col_stats.get('min', datetime(2020, 1, 1))
                max_date = col_stats.get('max', datetime(2023, 12, 31))
                
                if isinstance(min_date, str):
                    min_date = pd.to_datetime(min_date)
                if isinstance(max_date, str):
                    max_date = pd.to_datetime(max_date)
                
                date_range = (max_date - min_date).days
                random_days = np.random.randint(0, date_range, num_rows)
                
                generated_dates = [min_date + timedelta(days=int(days)) for days in random_days]
                
                # Add noise by shifting some dates
                if noise_level > 0:
                    noise_days = np.random.randint(-int(noise_level * 365), int(noise_level * 365), num_rows)
                    generated_dates = [date + timedelta(days=int(nd)) for date, nd in zip(generated_dates, noise_days)]
                
                generated_data[col] = generated_dates
                
            elif col_type == 'email':
                domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com']
                if 'unique_values' in col_stats:
                    existing_emails = col_stats['unique_values']
                    # Extract patterns from existing emails
                    local_parts = [email.split('@')[0] for email in existing_emails if '@' in email]
                    domains = list(set([email.split('@')[1] for email in existing_emails if '@' in email]))
                else:
                    local_parts = [faker.first_name().lower() + faker.last_name().lower() for _ in range(100)]
                
                emails = []
                for i in range(num_rows):
                    if local_parts and i < len(local_parts):
                        local = local_parts[i % len(local_parts)]
                    else:
                        local = faker.first_name().lower() + faker.last_name().lower()
                    
                    domain = random.choice(domains) if domains else 'example.com'
                    
                    # Add noise/variation
                    if noise_level > 0 and random.random() < noise_level:
                        local += str(random.randint(1, 999))
                    
                    emails.append(f"{local}@{domain}")
                
                generated_data[col] = emails
                
            elif col_type == 'phone':
                patterns = ['###-###-####', '(###) ###-####', '##########']
                phones = []
                for _ in range(num_rows):
                    pattern = random.choice(patterns)
                    phone = ''
                    for char in pattern:
                        if char == '#':
                            phone += str(random.randint(0, 9))
                        else:
                            phone += char
                    phones.append(phone)
                generated_data[col] = phones
                
            elif col_type == 'categorical':
                unique_values = col_stats.get('unique_values', ['A', 'B', 'C', 'D'])
                value_counts = col_stats.get('value_counts', {})
                
                if value_counts:
                    # Use original distribution
                    values, probs = zip(*value_counts.items())
                    total = sum(probs)
                    probabilities = [p/total for p in probs]
                    generated_data[col] = np.random.choice(values, num_rows, p=probabilities)
                else:
                    # Uniform distribution
                    generated_data[col] = np.random.choice(unique_values, num_rows)
                
            elif col_type == 'text':
                # For names
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['name', 'first', 'last', 'full']):
                    if 'first' in col_lower:
                        generated_data[col] = [faker.first_name() for _ in range(num_rows)]
                    elif 'last' in col_lower:
                        generated_data[col] = [faker.last_name() for _ in range(num_rows)]
                    else:
                        generated_data[col] = [faker.name() for _ in range(num_rows)]
                elif 'address' in col_lower:
                    generated_data[col] = [faker.address().replace('\n', ', ') for _ in range(num_rows)]
                elif 'city' in col_lower:
                    generated_data[col] = [faker.city() for _ in range(num_rows)]
                elif 'country' in col_lower:
                    generated_data[col] = [faker.country() for _ in range(num_rows)]
                else:
                    # Generic text
                    words = ['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit']
                    generated_data[col] = [' '.join(random.choices(words, k=random.randint(3, 10))) for _ in range(num_rows)]
                
            elif col_type == 'boolean':
                generated_data[col] = np.random.choice([True, False], num_rows)
                
            else:
                # Unknown type, generate random strings
                generated_data[col] = [faker.word() for _ in range(num_rows)]
        
        # Apply relationships if detected
        if relationships and len(generated_data) > 0:
            generated_data = self._apply_relationships(generated_data, relationships, analysis)
        
        # Create DataFrame
        df_generated = pd.DataFrame(generated_data)
        
        return df_generated
    
    def _apply_relationships(self, data_dict, relationships, analysis):
        """Apply simple relationships between columns"""
        # Simple relationship implementation
        # For example: if "city" and "state" should match
        for rel in relationships:
            if 'city' in rel.lower() and 'state' in rel.lower():
                # Simple city-state mapping
                city_state_map = {
                    'New York': 'NY',
                    'Los Angeles': 'CA',
                    'Chicago': 'IL',
                    'Houston': 'TX',
                    'Phoenix': 'AZ',
                    'Philadelphia': 'PA',
                    'San Antonio': 'TX',
                    'San Diego': 'CA',
                    'Dallas': 'TX',
                    'San Jose': 'CA'
                }
                
                # Find city and state columns
                city_col = None
                state_col = None
                for col in data_dict.keys():
                    if 'city' in col.lower():
                        city_col = col
                    elif 'state' in col.lower():
                        state_col = col
                
                if city_col and state_col:
                    # Update states based on cities
                    for i in range(len(data_dict[city_col])):
                        city = data_dict[city_col][i]
                        if city in city_state_map:
                            data_dict[state_col][i] = city_state_map[city]
        
        return data_dict

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Check authentication
    if not check_session():
        st.warning("Please login first to access Synthetic Data Generator")
        st.stop()
    
    # Page configuration
    st.set_page_config(
        page_title="Synthetic Data Generator - irmc Aura",
        page_icon="🔢",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #175CFF, #00A3FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .step-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #175CFF;
            box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1);
            margin-bottom: 1.5rem;
        }
        .data-preview {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .generated-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">🔢 Synthetic Data Generator</div>', unsafe_allow_html=True)
        st.markdown("### Generate realistic synthetic data from your sample data")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize session state
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = SyntheticDataGenerator()
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "⚙️ Configure Generation", "📊 Results & Download"])
    
    with tab1:
        st.markdown('<div class="step-card"><h3>Step 1: Upload Your Sample Data</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your sample data (5-1000 rows recommended)"
            )
            
            if uploaded_file is not None:
                try:
                    # Read CSV file
                    df = pd.read_csv(uploaded_file)
                    
                    if df.empty:
                        st.error("❌ The uploaded file is empty")
                    elif len(df) < 3:
                        st.warning("⚠️ Very small dataset. For better results, provide at least 5-10 rows.")
                    else:
                        st.session_state.uploaded_data = df
                        
                        # Display file info
                        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                        
                        # Preview data
                        with st.expander("📋 Preview Data (First 10 rows)", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Analyze button
                        if st.button("🔍 Analyze Data Patterns", type="primary", use_container_width=True):
                            with st.spinner("Analyzing data patterns..."):
                                analysis = st.session_state.data_generator.analyze_dataset(df)
                                st.session_state.data_analysis = analysis
                                st.success("✅ Data analysis complete!")
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
        
        with col2:
            st.subheader("📝 Quick Tips")
            st.markdown("""
            **Best Practices:**
            - Upload 5-1000 rows for best results
            - Include all column types you want to generate
            - Clean data works better
            - Remove sensitive information
            
            **Supported Data Types:**
            - Names, addresses, emails
            - Phone numbers, IDs
            - Dates, times
            - Numbers (prices, ages, quantities)
            - Categories, statuses
            - Boolean values
            
            **Limitations:**
            - Max 10,000 generated rows
            - Complex relationships may not be captured
            """)
    
    with tab2:
        if st.session_state.uploaded_data is None:
            st.info("📤 Please upload data in Step 1 first")
        else:
            st.markdown('<div class="step-card"><h3>Step 2: Configure Generation</h3></div>', unsafe_allow_html=True)
            
            # Show analysis results if available
            if st.session_state.data_analysis:
                st.subheader("📊 Detected Data Patterns")
                
                analysis = st.session_state.data_analysis
                
                # Show column types
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Column Types Detected:**")
                    for col, col_type in analysis['column_types'].items():
                        st.write(f"• **{col}**: `{col_type}`")
                
                with col2:
                    if 'llm_patterns' in analysis and analysis['llm_patterns']:
                        st.write("**LLM-Detected Patterns:**")
                        patterns = analysis['llm_patterns'].get('overall_patterns', [])
                        for pattern in patterns[:3]:  # Show first 3
                            st.write(f"• {pattern}")
            
            # Generation settings
            st.subheader("⚙️ Generation Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_rows = st.selectbox(
                    "Number of rows to generate",
                    options=[100, 500, 1000, 5000, 10000],
                    index=2,  # Default 1000
                    help="Select how many synthetic rows to generate"
                )
            
            with col2:
                noise_level = st.slider(
                    "Noise Level",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="Add randomness/variation to the data (0 = exact patterns, 0.5 = more random)"
                )
                st.caption(f"Noise: {int(noise_level*100)}%")
            
            with col3:
                include_original = st.checkbox(
                    "Include original data",
                    value=False,
                    help="Include original rows in generated data"
                )
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    preserve_distributions = st.checkbox(
                        "Preserve value distributions",
                        value=True,
                        help="Maintain original value frequencies"
                    )
                
                with col2:
                    maintain_relationships = st.checkbox(
                        "Maintain column relationships",
                        value=True,
                        help="Try to preserve relationships between columns"
                    )
            
            # Generate button
            if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
                if st.session_state.data_analysis:
                    with st.spinner(f"Generating {num_rows} synthetic rows..."):
                        # Generate data
                        generated_df = st.session_state.data_generator.generate_data(
                            analysis=st.session_state.data_analysis,
                            num_rows=num_rows,
                            noise_level=noise_level
                        )
                        
                        # Add original data if requested
                        if include_original and st.session_state.uploaded_data is not None:
                            original_df = st.session_state.uploaded_data
                            generated_df = pd.concat([original_df, generated_df], ignore_index=True)
                        
                        st.session_state.generated_data = generated_df
                        st.success(f"✅ Successfully generated {len(generated_df)} rows!")
                        st.rerun()
                else:
                    st.error("Please analyze data first")
    
    with tab3:
        if st.session_state.generated_data is None:
            st.info("⚙️ Please generate data in Step 2 first")
        else:
            st.markdown('<div class="step-card"><h3>Step 3: Download Your Synthetic Data</h3></div>', unsafe_allow_html=True)
            
            generated_df = st.session_state.generated_data
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Generated Rows", f"{len(generated_df):,}")
            with col2:
                st.metric("Columns", len(generated_df.columns))
            with col3:
                st.metric("Memory Usage", f"{generated_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            with col4:
                # Data quality indicator (simplified)
                if st.session_state.uploaded_data is not None:
                    orig_cols = set(st.session_state.uploaded_data.columns)
                    gen_cols = set(generated_df.columns)
                    match = len(orig_cols.intersection(gen_cols)) / len(orig_cols) * 100
                    st.metric("Column Match", f"{match:.0f}%")
            
            # Data preview
            st.subheader("📋 Preview Generated Data")
            
            preview_tab1, preview_tab2 = st.tabs(["First 10 Rows", "Last 10 Rows"])
            
            with preview_tab1:
                st.dataframe(generated_df.head(10), use_container_width=True)
            
            with preview_tab2:
                st.dataframe(generated_df.tail(10), use_container_width=True)
            
            # Download options
            st.subheader("💾 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Convert to CSV
                csv = generated_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    generated_df.to_excel(writer, index=False, sheet_name='SyntheticData')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📗 Download as Excel",
                    data=excel_data,
                    file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                if st.button("🔄 Generate More Data", use_container_width=True):
                    st.session_state.generated_data = None
                    st.rerun()
            
            # Data quality check
            st.subheader("📊 Data Quality Check")
            
            if st.session_state.uploaded_data is not None:
                original_df = st.session_state.uploaded_data
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Data Stats:**")
                    st.write(f"- Rows: {len(original_df):,}")
                    st.write(f"- Columns: {len(original_df.columns)}")
                    st.write(f"- Missing values: {original_df.isna().sum().sum()}")
                
                with col2:
                    st.write("**Generated Data Stats:**")
                    st.write(f"- Rows: {len(generated_df):,}")
                    st.write(f"- Columns: {len(generated_df.columns)}")
                    st.write(f"- Missing values: {generated_df.isna().sum().sum()}")
                
                # Column type comparison
                st.write("**Column Type Comparison:**")
                for col in original_df.columns:
                    if col in generated_df.columns:
                        orig_type = str(original_df[col].dtype)
                        gen_type = str(generated_df[col].dtype)
                        st.write(f"• **{col}**: Original=`{orig_type}`, Generated=`{gen_type}`")
            
            # Reset button
            st.markdown("---")
            if st.button("🔄 Start Over with New Data", use_container_width=True):
                st.session_state.uploaded_data = None
                st.session_state.data_analysis = None
                st.session_state.generated_data = None
                st.rerun()

# Run the app
if __name__ == "__main__":
    main()
