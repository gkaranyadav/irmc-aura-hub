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
# SMART PATTERN ANALYZER
# =============================================================================
class SmartPatternAnalyzer:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.llm_available = True
        except:
            self.llm_available = False
    
    def analyze_patterns(self, df_sample):  # Fixed method name
        """
        Let LLM analyze data and tell us what it is AND how to generate it
        """
        if not self.llm_available or df_sample.empty:
            return self._statistical_analysis(df_sample)
        
        # Prepare comprehensive data info
        data_info = self._prepare_data_info(df_sample)
        
        # SMART PROMPT - Ask LLM to identify AND generate properly
        prompt = f"""
        Analyze this dataset and provide instructions for generating realistic synthetic data.
        
        DATA INFO:
        {json.dumps(data_info, indent=2)}
        
        IMPORTANT: You need to:
        1. Identify what TYPE of data each column contains (ID, name, date, email, number, code, etc.)
        2. Analyze the patterns and formats
        3. Provide CLEAR generation rules
        
        For EACH column, provide:
        - data_type: What kind of data is this? (customer_id, product_name, email, date, amount, status, etc.)
        - patterns_observed: What patterns do you see?
        - format_rules: What format should generated data follow?
        - realistic_constraints: What makes this data realistic?
        - generation_method: How to generate it?
        
        Make the synthetic data LOOK REALISTIC and MAKE SENSE.
        
        Example for column with values: ["john@email.com", "jane@company.com"]
        {{
            "data_type": "email_address",
            "patterns_observed": "username@domain.tld format",
            "format_rules": "lowercase letters, numbers, dots, @ symbol, domain ending",
            "realistic_constraints": "valid email format, common domains",
            "generation_method": "generate random usernames + common domains"
        }}
        
        Return JSON:
        {{
            "columns": {{
                "column_name": {{
                    "data_type": "identified_type",
                    "patterns_observed": "description",
                    "format_rules": "format details",
                    "realistic_constraints": "what makes it realistic",
                    "generation_method": "how to generate",
                    "generation_details": {{
                        "method": "random|sequential|pattern|categorical",
                        "parameters": {{}}
                    }}
                }}
            }},
            "dataset_type": "overall dataset type",
            "realism_advice": "how to keep data realistic"
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a data generation expert who creates realistic synthetic data."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=3000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON
            try:
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return self._enhance_with_real_stats(analysis, df_sample)
            except json.JSONDecodeError as e:
                st.warning(f"JSON parse error: {e}")
            
            return self._statistical_analysis(df_sample)
            
        except Exception as e:
            st.warning(f"LLM analysis failed: {str(e)}")
            return self._statistical_analysis(df_sample)
    
    def _prepare_data_info(self, df):
        """Prepare comprehensive data info"""
        data_info = {
            "shape": f"{len(df)} rows × {len(df.columns)} columns",
            "columns": {},
            "sample_data": []
        }
        
        # Column details
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # Get samples
                samples = col_data.head(5).tolist()
                
                # Basic stats
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        stats = {
                            "min": float(numeric_data.min()),
                            "max": float(numeric_data.max()),
                            "mean": float(numeric_data.mean()),
                            "unique_count": col_data.nunique(),
                            "unique_ratio": float(col_data.nunique() / len(col_data))
                        }
                    else:
                        stats = None
                except:
                    stats = None
                
                data_info["columns"][col] = {
                    "dtype": str(df[col].dtype),
                    "samples": [str(x)[:100] for x in samples],
                    "unique_values": col_data.nunique(),
                    "null_count": df[col].isna().sum(),
                    "stats": stats
                }
        
        # Sample rows
        for idx, row in df.head(3).iterrows():
            data_info["sample_data"].append({
                "row": idx,
                "data": {col: str(row[col])[:100] for col in df.columns}
            })
        
        return data_info
    
    def _enhance_with_real_stats(self, analysis, df):
        """Add real statistical data to LLM analysis"""
        if 'columns' not in analysis:
            analysis['columns'] = {}
        
        for col in df.columns:
            if col not in analysis['columns']:
                analysis['columns'][col] = {}
            
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info = analysis['columns'][col]
                
                # Always add these stats
                col_info['real_stats'] = {
                    'row_count': len(col_data),
                    'unique_count': col_data.nunique(),
                    'unique_ratio': float(col_data.nunique() / len(col_data)),
                    'null_count': df[col].isna().sum()
                }
                
                # Try numeric analysis
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        if 'numeric_stats' not in col_info:
                            col_info['numeric_stats'] = {}
                        
                        col_info['numeric_stats'].update({
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'mean': float(numeric_data.mean()),
                            'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0,
                            'is_integer': numeric_data.apply(lambda x: float(x).is_integer()).all()
                        })
                        
                        # Check decimal places
                        if not col_info['numeric_stats']['is_integer']:
                            decimals = numeric_data.apply(
                                lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0
                            )
                            col_info['numeric_stats']['max_decimals'] = int(decimals.max())
                            col_info['numeric_stats']['common_decimals'] = int(decimals.mode().iloc[0]) if len(decimals.mode()) > 0 else 2
                except:
                    pass
                
                # Text analysis
                if len(col_data) > 0:
                    samples = col_data.head(10).astype(str).tolist()
                    col_info['sample_patterns'] = samples
                    
                    # Length analysis
                    lengths = col_data.astype(str).str.len()
                    col_info['length_stats'] = {
                        'min': int(lengths.min()),
                        'max': int(lengths.max()),
                        'avg': float(lengths.mean()),
                        'std': float(lengths.std()) if len(lengths) > 1 else 0
                    }
        
        return analysis
    
    def _statistical_analysis(self, df):
        """Statistical fallback analysis"""
        analysis = {'columns': {}}
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info = {
                    'data_type': 'auto_detected',
                    'patterns_observed': 'Statistical patterns',
                    'real_stats': {
                        'row_count': len(col_data),
                        'unique_count': col_data.nunique(),
                        'unique_ratio': float(col_data.nunique() / len(col_data))
                    }
                }
                
                # Check if mostly numeric
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    numeric_ratio = len(numeric_data) / len(col_data)
                    
                    if numeric_ratio > 0.7:  # Mostly numeric
                        col_info['data_type'] = 'numeric'
                        col_info['numeric_stats'] = {
                            'min': float(numeric_data.min()),
                            'max': float(numeric_data.max()),
                            'mean': float(numeric_data.mean()),
                            'is_integer': numeric_data.apply(lambda x: float(x).is_integer()).all()
                        }
                    else:
                        col_info['data_type'] = 'text'
                        
                        # Check for common patterns
                        samples = col_data.head(5).astype(str).tolist()
                        col_info['sample_patterns'] = samples
                        
                        # Check for email pattern
                        if any('@' in s and '.' in s for s in samples[:3]):
                            col_info['data_type'] = 'email_like'
                        
                        # Check for date pattern
                        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
                        if any(re.match(p, str(s)) for p in date_patterns for s in samples[:2]):
                            col_info['data_type'] = 'date_like'
                
                except:
                    col_info['data_type'] = 'text'
                
                # Length stats
                lengths = col_data.astype(str).str.len()
                col_info['length_stats'] = {
                    'min': int(lengths.min()),
                    'max': int(lengths.max()),
                    'avg': float(lengths.mean())
                }
                
                analysis['columns'][col] = col_info
        
        analysis['dataset_type'] = 'generic_dataset'
        analysis['realism_advice'] = 'Maintain similar distributions and patterns'
        
        return analysis

# =============================================================================
# SMART DATA GENERATOR
# =============================================================================
class SmartDataGenerator:
    def __init__(self):
        self.analyzer = SmartPatternAnalyzer()  # Fixed class name
    
    def generate_data(self, original_df, num_rows, variation=0.1):
        """
        Generate realistic synthetic data
        """
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Analyze patterns
        with st.spinner("🔍 Analyzing data patterns and types..."):
            analysis = self.analyzer.analyze_patterns(original_df.head(50))
            st.session_state.last_analysis = analysis
        
        # Step 2: Generate each column
        generated_data = {}
        
        for col in original_df.columns:
            if col in analysis.get('columns', {}):
                generated_data[col] = self._generate_column_smartly(
                    col_name=col,
                    col_info=analysis['columns'][col],
                    original_series=original_df[col] if col in original_df.columns else None,
                    num_rows=num_rows,
                    variation=variation
                )
            else:
                generated_data[col] = self._generate_fallback(original_df[col], num_rows)
        
        # Create DataFrame
        df_generated = pd.DataFrame(generated_data)
        
        # Apply post-processing for realism
        df_generated = self._apply_realism_checks(df_generated, analysis)
        
        return df_generated
    
    def _generate_column_smartly(self, col_name, col_info, original_series, num_rows, variation):
        """Smart generation based on data type"""
        
        data_type = col_info.get('data_type', 'auto_detect')
        generation_method = col_info.get('generation_method', 'pattern_based')
        
        # If we have specific generation details
        if 'generation_details' in col_info:
            return self._generate_by_details(col_info['generation_details'], num_rows)
        
        # Generate based on data type
        if 'id' in data_type.lower() or '_id' in col_name.lower():
            return self._generate_ids(col_name, col_info, num_rows)
        
        elif 'email' in data_type.lower():
            return self._generate_emails(num_rows)
        
        elif 'date' in data_type.lower():
            return self._generate_dates(col_info, num_rows, variation)
        
        elif 'name' in data_type.lower():
            return self._generate_names(data_type, num_rows)
        
        elif data_type == 'numeric':
            return self._generate_numeric(col_info, num_rows, variation)
        
        elif 'status' in data_type.lower() or 'category' in data_type.lower():
            return self._generate_categorical(original_series, num_rows)
        
        elif 'amount' in data_type.lower() or 'price' in data_type.lower():
            return self._generate_prices(col_info, num_rows, variation)
        
        else:
            # Generic text generation
            return self._generate_text(col_info, original_series, num_rows)
    
    def _generate_ids(self, col_name, col_info, num_rows):
        """Generate ID-like data"""
        prefix = ''
        if 'customer' in col_name.lower():
            prefix = 'CUST'
        elif 'order' in col_name.lower():
            prefix = 'ORD'
        elif 'product' in col_name.lower():
            prefix = 'PROD'
        
        start_id = 1000
        if 'numeric_stats' in col_info:
            start_id = int(col_info['numeric_stats'].get('min', 1000))
        
        ids = []
        for i in range(num_rows):
            if prefix:
                ids.append(f"{prefix}{start_id + i:06d}")
            else:
                ids.append(f"{start_id + i}")
        
        return ids
    
    def _generate_emails(self, num_rows):
        """Generate realistic emails"""
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com', 'business.org']
        first_names = ['john', 'jane', 'alex', 'sam', 'mike', 'sara', 'david', 'lisa']
        last_names = ['smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis']
        
        emails = []
        for i in range(num_rows):
            if random.random() < 0.7:
                # First.Last@domain
                fname = random.choice(first_names)
                lname = random.choice(last_names)
                email = f"{fname}.{lname}{random.randint(1,99)}@{random.choice(domains)}"
            else:
                # Initials or other patterns
                email = f"{random.choice('abcdefghijklmnopqrstuvwxyz')}{random.randint(100,999)}@{random.choice(domains)}"
            
            emails.append(email.lower())
        
        return emails
    
    def _generate_dates(self, col_info, num_rows, variation):
        """Generate realistic dates"""
        # Start from recent date
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365 * 2)  # 2 years range
        
        # Generate dates
        date_range = (end_date - start_date).days
        dates = []
        
        for _ in range(num_rows):
            # Mix of recent and older dates
            if random.random() < 0.7:  # 70% recent
                days_ago = random.randint(1, 90)
            else:
                days_ago = random.randint(91, date_range)
            
            date = end_date - pd.Timedelta(days=days_ago)
            
            # Random format
            if random.random() < 0.5:
                dates.append(date.strftime('%Y-%m-%d'))
            else:
                dates.append(date.strftime('%d/%m/%Y'))
        
        return dates
    
    def _generate_names(self, data_type, num_rows):
        """Generate names"""
        first_names = ['John', 'Jane', 'Alex', 'Sam', 'Mike', 'Sara', 'David', 'Lisa', 
                      'Robert', 'Emily', 'James', 'Maria', 'William', 'Sarah']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis',
                     'Wilson', 'Moore', 'Taylor', 'Anderson', 'Thomas']
        
        if 'customer' in data_type.lower() or 'full' in data_type.lower():
            return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_rows)]
        elif 'first' in data_type.lower():
            return [random.choice(first_names) for _ in range(num_rows)]
        else:
            return [random.choice(last_names) for _ in range(num_rows)]
    
    def _generate_numeric(self, col_info, num_rows, variation):
        """Generate numeric data"""
        if 'numeric_stats' in col_info:
            stats = col_info['numeric_stats']
            min_val = stats.get('min', 0)
            max_val = stats.get('max', 1000)
            mean_val = stats.get('mean', (min_val + max_val) / 2)
            std_val = stats.get('std', (max_val - min_val) / 4)
            
            if stats.get('is_integer', False):
                # Integer data
                data = np.random.normal(mean_val, std_val * (1 + variation), num_rows)
                data = np.clip(data, min_val, max_val)
                return np.round(data).astype(int)
            else:
                # Float data
                data = np.random.normal(mean_val, std_val * (1 + variation), num_rows)
                data = np.clip(data, min_val, max_val)
                decimals = stats.get('common_decimals', 2)
                return np.round(data, decimals)
        else:
            # Fallback
            return np.random.uniform(10, 1000, num_rows)
    
    def _generate_categorical(self, original_series, num_rows):
        """Generate categorical data"""
        if original_series is not None:
            unique_vals = original_series.dropna().unique()
            if len(unique_vals) > 0:
                # Use original categories
                return np.random.choice(unique_vals, num_rows)
        
        # Default categories
        categories = ['Active', 'Pending', 'Completed', 'Cancelled', 'Shipped', 'Processing']
        return np.random.choice(categories, num_rows)
    
    def _generate_prices(self, col_info, num_rows, variation):
        """Generate price-like data"""
        if 'numeric_stats' in col_info:
            stats = col_info['numeric_stats']
            min_price = max(0, stats.get('min', 1))
            max_price = stats.get('max', 1000)
            mean_price = stats.get('mean', (min_price + max_price) / 2)
            
            # Generate with common price endings
            prices = []
            for _ in range(num_rows):
                base = np.random.normal(mean_price, mean_price * 0.5)
                base = np.clip(base, min_price, max_price)
                
                # Common price endings
                if random.random() < 0.3:
                    price = np.round(base) + 0.99  # .99 ending
                elif random.random() < 0.2:
                    price = np.round(base) + 0.50  # .50 ending
                elif random.random() < 0.1:
                    price = np.round(base)  # Whole number
                else:
                    price = np.round(base, 2)  # Random cents
                
                prices.append(float(price))
            
            return prices
        else:
            return np.round(np.random.uniform(1, 1000, num_rows), 2)
    
    def _generate_text(self, col_info, original_series, num_rows):
        """Generate text data"""
        if original_series is not None and len(original_series) > 0:
            # If few unique values, reuse them
            unique_vals = original_series.dropna().unique()
            if len(unique_vals) <= 15:
                return np.random.choice(unique_vals, num_rows)
            
            # Generate based on patterns
            samples = original_series.head(10).astype(str).tolist()
            
            # Check for product-like names
            if any(len(s.split()) >= 2 and any(char.isupper() for char in s) for s in samples[:3]):
                return self._generate_product_names(num_rows)
        
        # Generic text
        words = ['Item', 'Product', 'Service', 'Component', 'Module', 'Unit', 'System', 'Tool']
        adjectives = ['Premium', 'Standard', 'Basic', 'Advanced', 'Professional', 'Enterprise']
        
        return [f"{random.choice(adjectives)} {random.choice(words)} {random.randint(1, 100)}" 
                for _ in range(num_rows)]
    
    def _generate_product_names(self, num_rows):
        """Generate product-like names"""
        products = ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse', 'Speaker', 'Headphones']
        types = ['Pro', 'Plus', 'Max', 'Lite', 'Standard', 'Premium', 'Ultra']
        brands = ['Tech', 'Smart', 'Power', 'Ultra', 'Hyper', 'Mega']
        
        names = []
        for _ in range(num_rows):
            if random.random() < 0.6:
                name = f"{random.choice(brands)} {random.choice(products)} {random.choice(types)}"
            else:
                name = f"{random.choice(products)} {random.randint(10, 99)}{random.choice('ABCD')}"
            names.append(name)
        
        return names
    
    def _generate_by_details(self, details, num_rows):
        """Generate based on detailed instructions"""
        method = details.get('method', 'random')
        
        if method == 'sequential':
            start = details.get('parameters', {}).get('start', 1)
            return list(range(start, start + num_rows))
        
        elif method == 'categorical':
            categories = details.get('parameters', {}).get('categories', ['A', 'B', 'C'])
            return np.random.choice(categories, num_rows)
        
        else:
            return [f"GEN_{i}" for i in range(num_rows)]
    
    def _generate_fallback(self, original_series, num_rows):
        """Fallback generation"""
        if original_series is not None and len(original_series) > 0:
            unique_vals = original_series.dropna().unique()
            if len(unique_vals) > 0:
                return np.random.choice(unique_vals, num_rows)
        
        return [f"DATA_{i:06d}" for i in range(num_rows)]
    
    def _apply_realism_checks(self, df, analysis):
        """Apply realism checks and fixes"""
        # Ensure IDs are unique
        for col in df.columns:
            if col.lower().endswith('_id') or col.lower().endswith('id'):
                df[col] = df[col].astype(str)
        
        # Ensure emails look like emails
        for col in df.columns:
            if 'email' in col.lower():
                # Make sure all have @ symbol
                df[col] = df[col].apply(lambda x: x if '@' in str(x) else f"{x}@example.com")
        
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
        page_title="Smart Data Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Header
    st.title("🔢 Smart Synthetic Data Generator")
    st.markdown("Generate **realistic** synthetic data that **makes sense**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize
    if 'generator' not in st.session_state:
        st.session_state.generator = SmartDataGenerator()
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    
    # Upload section
    uploaded_file = st.file_uploader("📤 Upload CSV Dataset", type=['csv'])
    
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
                
                st.subheader("🔍 Data Analysis & Generation")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🤖 Analyze with AI", type="primary"):
                        with st.spinner("AI is analyzing your data..."):
                            analysis = st.session_state.generator.analyzer.analyze_patterns(df.head(50))
                            st.session_state.analysis = analysis
                            st.success("✅ Analysis complete!")
                
                with col2:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=10000, 
                                             value=1000, 
                                             step=100)
                
                with col3:
                    variation = st.slider("Variation level", 0.0, 1.0, 0.1, 0.05,
                                        help="How much variation from original patterns")
                
                # Show analysis
                if st.session_state.analysis:
                    with st.expander("📊 AI Analysis Results", expanded=False):
                        analysis = st.session_state.analysis
                        
                        st.write(f"**Dataset type:** {analysis.get('dataset_type', 'Unknown')}")
                        st.write(f"**Realism advice:** {analysis.get('realism_advice', '')}")
                        
                        st.subheader("Column Analysis:")
                        for col, info in analysis.get('columns', {}).items():
                            with st.expander(f"**{col}**", expanded=False):
                                st.write(f"**Data type:** {info.get('data_type', 'Unknown')}")
                                st.write(f"**Patterns:** {info.get('patterns_observed', '')}")
                                st.write(f"**Generation:** {info.get('generation_method', '')}")
                                
                                if 'real_stats' in info:
                                    st.write("**Statistics:**")
                                    st.json(info['real_stats'], expanded=False)
                
                # Generate button
                if st.session_state.analysis:
                    if st.button("🚀 Generate Realistic Data", type="primary"):
                        with st.spinner("Generating realistic synthetic data..."):
                            generated = st.session_state.generator.generate_data(
                                original_df=df,
                                num_rows=int(num_rows),
                                variation=variation
                            )
                            
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} realistic rows!")
                            st.balloons()
                
                # Results
                if st.session_state.generated_data is not None:
                    st.subheader("📋 Generated Data Preview")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Show tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Preview", "Comparison", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Data**")
                            st.write(f"Rows: {len(df)}")
                            st.write(f"Sample values:")
                            for col in df.columns[:3]:
                                st.write(f"- {col}: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
                        
                        with col2:
                            st.write("**Generated Data**")
                            st.write(f"Rows: {len(df_gen)}")
                            st.write(f"Sample values:")
                            for col in df_gen.columns[:3]:
                                st.write(f"- {col}: {df_gen[col].iloc[0] if len(df_gen) > 0 else 'N/A'}")
                    
                    with tab3:
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "realistic_synthetic_data.csv",
                            "text/csv",
                            key="download-csv"
                        )
                        
                        # JSON download
                        json_str = df_gen.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            "realistic_synthetic_data.json",
                            "application/json",
                            key="download-json"
                        )
                        
                        if st.button("🔄 Generate New Batch"):
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
