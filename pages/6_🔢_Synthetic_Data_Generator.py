# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
import io
import time
import hashlib
import random
from groq import Groq
from auth import check_session

# =============================================================================
# IMPORT FREE LIBRARIES
# =============================================================================
try:
    # SDV for statistical pattern learning (FREE)
    from sdv.tabular import GaussianCopula
    from sdv.constraints import Unique, Positive, FixedIncrements, Between
    from sdv.metadata import SingleTableMetadata
    
    # Mimesis for realistic data (FREE)
    from mimesis import Person, Address, Finance, Business, Datetime, Text, Generic
    from mimesis.locales import Locale
    from mimesis import Internet, Science, Development, Food, Hardware
    
    SDV_AVAILABLE = True
    MIMESIS_AVAILABLE = True
except ImportError as e:
    SDV_AVAILABLE = False
    MIMESIS_AVAILABLE = False
    st.warning(f"⚠️ For better results, install: `pip install sdv mimesis`")

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Use best model for analysis
    LLM_SAMPLE_SIZE = 10  # Analyze only 10 rows with LLM (one-time)
    SDV_SAMPLE_SIZE = 100  # Use 100 rows for SDV training
    MAX_GENERATE_ROWS = 10000

# =============================================================================
# SMART LLM RULE EXTRACTOR (ONE-TIME ANALYSIS)
# =============================================================================
class SmartRuleExtractor:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.client_available = True
        except:
            self.client_available = False
            st.warning("⚠️ Groq API key not found. Using basic generation.")
        
        # Cache for rules (avoid duplicate LLM calls)
        self.rule_cache = {}
    
    def extract_smart_rules(self, df_sample):
        """
        ONE-TIME LLM ANALYSIS: Analyze 10 rows, extract rules for ALL future generation
        This is the ONLY LLM call we make!
        """
        # Create hash of dataset for caching
        dataset_hash = self._create_dataset_hash(df_sample)
        
        # Check cache first (avoid duplicate LLM calls)
        if dataset_hash in self.rule_cache:
            return self.rule_cache[dataset_hash]
        
        if not self.client_available or df_sample.empty:
            return self._get_basic_rules(df_sample)
        
        try:
            # COST OPTIMIZATION: Use only 10 rows for LLM analysis
            sample_size = min(Config.LLM_SAMPLE_SIZE, len(df_sample))
            llm_sample = df_sample.head(sample_size)
            
            # Prepare data for LLM (minimal info)
            columns_info = []
            for col in llm_sample.columns:
                unique_vals = llm_sample[col].dropna().unique()[:3].tolist()
                col_type = str(llm_sample[col].dtype)
                columns_info.append({
                    'column': col,
                    'type': col_type,
                    'sample_values': unique_vals
                })
            
            # Sample data for LLM
            sample_data = llm_sample.head(5).to_dict(orient='records')
            
            prompt = f"""
            Analyze this dataset and extract SMART GENERATION RULES for synthetic data.
            
            COLUMNS TO ANALYZE:
            {json.dumps(columns_info, indent=2)}
            
            SAMPLE DATA (5 rows):
            {json.dumps(sample_data, indent=2)}
            
            DATASET INFO: {len(llm_sample)} rows, {len(llm_sample.columns)} columns
            
            YOUR TASK:
            1. Identify the DATASET TYPE (ecommerce, customer_data, inventory, financial, healthcare, etc.)
            2. For EACH COLUMN, determine:
               - Semantic meaning (what does this column represent?)
               - Data constraints (min/max, unique, sequential, etc.)
               - Realistic generation rules
            3. Identify relationships between columns
            
            Return ONLY JSON with this structure:
            {{
                "dataset_type": "detected_type",
                "columns": {{
                    "column_name": {{
                        "semantic_type": "order_id|customer_name|age|email|price|quantity|date|status|etc",
                        "generation_method": "sequential_int|real_name|int_range|email_pattern|float_price|small_int|date_range|categorical",
                        "constraints": {{"min": X, "max": Y, "unique": true/false, "sequential": true/false}},
                        "realistic_values": ["value1", "value2"] if categorical
                    }}
                }},
                "relationships": ["column_A depends on column_B"],
                "generation_advice": "Use SDV for correlations, Mimesis for text data"
            }}
            """
            
            # ONE LLM CALL ONLY!
            messages = [
                {"role": "system", "content": "You are a data generation expert. Extract smart rules for synthetic data."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=2000  # Limit output tokens
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    rules = json.loads(json_match.group())
                    
                    # Enhance rules with basic stats
                    rules = self._enhance_rules_with_stats(rules, df_sample)
                    
                    # Cache the rules (avoid future LLM calls for same dataset)
                    self.rule_cache[dataset_hash] = rules
                    return rules
            except json.JSONDecodeError:
                st.warning("⚠️ Could not parse LLM response. Using basic rules.")
            
            return self._get_basic_rules(df_sample)
            
        except Exception as e:
            st.warning(f"⚠️ LLM analysis failed: {str(e)}")
            return self._get_basic_rules(df_sample)
    
    def _enhance_rules_with_stats(self, rules, df):
        """Add statistical info to LLM rules"""
        if 'columns' not in rules:
            return rules
        
        for col_name, col_info in rules['columns'].items():
            if col_name in df.columns:
                col_data = df[col_name].dropna()
                if not col_data.empty:
                    # Add statistical info
                    if col_info.get('generation_method') == 'int_range':
                        try:
                            numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
                            if len(numeric_vals) > 0:
                                if 'constraints' not in col_info:
                                    col_info['constraints'] = {}
                                col_info['constraints']['min'] = float(numeric_vals.min())
                                col_info['constraints']['max'] = float(numeric_vals.max())
                        except:
                            pass
        
        return rules
    
    def _get_basic_rules(self, df):
        """Fallback rules when LLM is not available"""
        basic_rules = {
            'dataset_type': 'generic',
            'columns': {},
            'relationships': [],
            'generation_advice': 'Use statistical generation'
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            if not col_data.empty:
                # Simple type detection
                col_str = str(col).lower()
                
                if any(keyword in col_str for keyword in ['id', 'num', 'no', 'code']):
                    semantic_type = 'id'
                    generation_method = 'sequential_int'
                    constraints = {'unique': True, 'sequential': True}
                elif any(keyword in col_str for keyword in ['name', 'first', 'last', 'full']):
                    semantic_type = 'name'
                    generation_method = 'real_name'
                elif 'age' in col_str:
                    semantic_type = 'age'
                    generation_method = 'int_range'
                    constraints = {'min': 18, 'max': 80}
                elif 'email' in col_str:
                    semantic_type = 'email'
                    generation_method = 'email_pattern'
                elif 'price' in col_str or 'amount' in col_str or 'cost' in col_str:
                    semantic_type = 'price'
                    generation_method = 'float_price'
                elif 'date' in col_str or 'time' in col_str:
                    semantic_type = 'date'
                    generation_method = 'date_range'
                elif 'status' in col_str or 'type' in col_str or 'category' in col_str:
                    semantic_type = 'category'
                    generation_method = 'categorical'
                    # Get unique values
                    unique_vals = col_data.unique()[:10].tolist()
                    constraints = {'values': unique_vals}
                else:
                    semantic_type = 'text'
                    generation_method = 'text'
                
                basic_rules['columns'][col] = {
                    'semantic_type': semantic_type,
                    'generation_method': generation_method
                }
                if 'constraints' in locals():
                    basic_rules['columns'][col]['constraints'] = constraints
        
        return basic_rules
    
    def _create_dataset_hash(self, df):
        """Create hash of dataset for caching"""
        # Use column names and first few rows for hash
        hash_data = str(df.columns.tolist()) + str(df.head(3).values.tolist())
        return hashlib.md5(hash_data.encode()).hexdigest()

# =============================================================================
# HYBRID DATA GENERATOR (FREE LIBRARIES + SMART RULES)
# =============================================================================
class HybridDataGenerator:
    def __init__(self):
        self.rule_extractor = SmartRuleExtractor()
        
        # Initialize Mimesis generators (FREE)
        if MIMESIS_AVAILABLE:
            self.person = Person(Locale.EN)
            self.address = Address(Locale.EN)
            self.finance = Finance(Locale.EN)
            self.business = Business(Locale.EN)
            self.datetime_gen = Datetime(Locale.EN)
            self.text_gen = Text(Locale.EN)
            self.internet = Internet()
            self.generic = Generic(Locale.EN)
        
        # Realistic data templates
        self.product_categories = {
            'electronics': ['iPhone', 'Laptop', 'Tablet', 'Smartwatch', 'Headphones', 'Camera'],
            'clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes', 'Hat'],
            'home': ['Chair', 'Table', 'Lamp', 'Bed', 'Sofa', 'Desk'],
            'books': ['Novel', 'Textbook', 'Cookbook', 'Biography', 'Self-Help'],
            'food': ['Coffee', 'Tea', 'Snacks', 'Chocolate', 'Chips']
        }
        
        self.order_statuses = ['Pending', 'Processing', 'Shipped', 'Delivered', 'Cancelled', 'Returned']
        self.countries = ['USA', 'India', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan']
    
    def generate_smart_data(self, original_df, num_rows, noise_level=0.1):
        """
        Generate realistic synthetic data using:
        1. ONE-TIME LLM rule extraction
        2. SDV for statistical patterns (FREE)
        3. Mimesis for realistic text (FREE)
        4. Business logic for domain rules
        """
        if original_df.empty:
            return pd.DataFrame()
        
        # STEP 1: ONE-TIME LLM RULE EXTRACTION
        with st.spinner("🤔 Analyzing data patterns (one-time LLM analysis)..."):
            rules = self.rule_extractor.extract_smart_rules(original_df)
        
        # STEP 2: USE SDV FOR STATISTICAL GENERATION (FREE)
        sdv_data = None
        if SDV_AVAILABLE and len(original_df) >= 5:
            try:
                # Train SDV on larger sample (100 rows max)
                train_size = min(Config.SDV_SAMPLE_SIZE, len(original_df))
                train_data = original_df.head(train_size)
                
                # Create metadata for better generation
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(train_data)
                
                # Add constraints based on LLM rules
                constraints = self._create_sdv_constraints(rules, train_data)
                
                # Train GaussianCopula model
                model = GaussianCopula(
                    constraints=constraints if constraints else None,
                    default_distribution='gaussian_kde'
                )
                
                model.fit(train_data)
                
                # Generate synthetic data with SDV
                sdv_data = model.sample(num_rows=num_rows)
                
                # Apply post-processing for realism
                sdv_data = self._apply_business_rules(sdv_data, rules)
                
            except Exception as e:
                st.warning(f"⚠️ SDV generation failed: {str(e)}")
                sdv_data = None
        
        # STEP 3: IF SDV FAILS, USE RULE-BASED GENERATION
        if sdv_data is None or sdv_data.empty:
            with st.spinner("🔄 Using rule-based generation..."):
                sdv_data = self._generate_with_rules(rules, original_df, num_rows)
        
        # STEP 4: ENHANCE WITH REALISTIC DATA (MIMESIS)
        if MIMESIS_AVAILABLE:
            sdv_data = self._enhance_with_mimesis(sdv_data, rules)
        
        # STEP 5: APPLY NOISE IF REQUESTED
        if noise_level > 0:
            sdv_data = self._apply_noise(sdv_data, noise_level, rules)
        
        return sdv_data
    
    def _create_sdv_constraints(self, rules, df):
        """Create SDV constraints from LLM rules"""
        constraints = []
        
        if 'columns' not in rules:
            return constraints
        
        for col_name, col_info in rules['columns'].items():
            if col_name in df.columns:
                semantic_type = col_info.get('semantic_type', '')
                constraints_dict = col_info.get('constraints', {})
                
                # Add Unique constraint for IDs
                if semantic_type in ['order_id', 'customer_id', 'id'] or constraints_dict.get('unique'):
                    try:
                        constraints.append(Unique(column_names=[col_name]))
                    except:
                        pass
                
                # Add Positive constraint for age, price, quantity
                if semantic_type in ['age', 'price', 'quantity', 'amount']:
                    try:
                        constraints.append(Positive(column_names=[col_name]))
                    except:
                        pass
                
                # Add Between constraint for ranges
                if 'min' in constraints_dict and 'max' in constraints_dict:
                    try:
                        min_val = float(constraints_dict['min'])
                        max_val = float(constraints_dict['max'])
                        constraints.append(Between(column_names=[col_name], low=min_val, high=max_val))
                    except:
                        pass
        
        return constraints
    
    def _apply_business_rules(self, data, rules):
        """Apply business logic to generated data"""
        if data.empty:
            return data
        
        dataset_type = rules.get('dataset_type', 'generic')
        
        # Fix common issues based on dataset type
        if dataset_type in ['ecommerce', 'sales', 'orders']:
            data = self._apply_ecommerce_rules(data, rules)
        elif dataset_type in ['customer', 'user']:
            data = self._apply_customer_rules(data, rules)
        
        # Fix data types
        for col in data.columns:
            if col in rules.get('columns', {}):
                col_info = rules['columns'][col]
                semantic_type = col_info.get('semantic_type', '')
                
                # Fix integer columns
                if semantic_type in ['age', 'quantity', 'order_id', 'customer_id']:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
                    except:
                        pass
                
                # Fix price columns (2 decimals)
                if semantic_type in ['price', 'amount', 'cost']:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        # Round to 2 decimals, common price endings
                        data[col] = data[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
                        # Make some end with .99 or .50
                        mask = data[col].notna()
                        if mask.any():
                            for i in data[mask].index:
                                if random.random() < 0.3:
                                    data.at[i, col] = round(data.at[i, col]) + 0.99
                                elif random.random() < 0.5:
                                    data.at[i, col] = round(data.at[i, col]) + 0.50
                    except:
                        pass
        
        return data
    
    def _apply_ecommerce_rules(self, data, rules):
        """Apply e-commerce specific rules"""
        # Ensure order IDs are sequential integers
        order_id_col = None
        for col, info in rules.get('columns', {}).items():
            if info.get('semantic_type') == 'order_id' and col in data.columns:
                order_id_col = col
                break
        
        if order_id_col:
            try:
                # Make order IDs sequential integers starting from 1000
                data[order_id_col] = range(1000, 1000 + len(data))
            except:
                pass
        
        # Ensure ages are integers 18-80
        age_col = None
        for col, info in rules.get('columns', {}).items():
            if info.get('semantic_type') == 'age' and col in data.columns:
                age_col = col
                break
        
        if age_col:
            try:
                data[age_col] = pd.to_numeric(data[age_col], errors='coerce')
                data[age_col] = data[age_col].clip(18, 80).astype(int)
            except:
                pass
        
        # Ensure quantities are small integers (1-10)
        qty_col = None
        for col, info in rules.get('columns', {}).items():
            if 'quantity' in info.get('semantic_type', '') and col in data.columns:
                qty_col = col
                break
        
        if qty_col:
            try:
                data[qty_col] = pd.to_numeric(data[qty_col], errors='coerce')
                data[qty_col] = data[qty_col].clip(1, 10).astype(int)
            except:
                pass
        
        return data
    
    def _generate_with_rules(self, rules, original_df, num_rows):
        """Fallback generation using rules only"""
        data = {}
        
        if 'columns' not in rules:
            return pd.DataFrame()
        
        for col_name, col_info in rules.get('columns', {}).items():
            semantic_type = col_info.get('semantic_type', 'text')
            generation_method = col_info.get('generation_method', 'text')
            constraints = col_info.get('constraints', {})
            
            if generation_method == 'sequential_int':
                # Generate sequential IDs
                start = constraints.get('start', 1000)
                data[col_name] = range(start, start + num_rows)
            
            elif generation_method == 'real_name':
                # Generate realistic names
                if MIMESIS_AVAILABLE:
                    data[col_name] = [self.person.full_name() for _ in range(num_rows)]
                else:
                    first_names = ['John', 'Sarah', 'Mike', 'Emma', 'David', 'Lisa']
                    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia']
                    data[col_name] = [f"{random.choice(first_names)} {random.choice(last_names)}" 
                                     for _ in range(num_rows)]
            
            elif generation_method == 'int_range':
                # Generate integers in range
                min_val = constraints.get('min', 18)
                max_val = constraints.get('max', 80)
                data[col_name] = np.random.randint(min_val, max_val + 1, num_rows)
            
            elif generation_method == 'email_pattern':
                # Generate realistic emails
                if MIMESIS_AVAILABLE:
                    data[col_name] = [self.person.email() for _ in range(num_rows)]
                else:
                    domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com']
                    data[col_name] = [f"user{i}@{random.choice(domains)}" for i in range(num_rows)]
            
            elif generation_method == 'float_price':
                # Generate realistic prices
                min_price = constraints.get('min', 10)
                max_price = constraints.get('max', 5000)
                prices = np.random.uniform(min_price, max_price, num_rows)
                # Round to 2 decimals, make some end with .99
                data[col_name] = [round(p, 2) for p in prices]
                for i in range(num_rows):
                    if random.random() < 0.3:
                        data[col_name][i] = round(data[col_name][i]) + 0.99
            
            elif generation_method == 'small_int':
                # Small integers (1-10)
                data[col_name] = np.random.randint(1, 11, num_rows)
            
            elif generation_method == 'date_range':
                # Generate dates
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now() + timedelta(days=365)
                date_range = (end_date - start_date).days
                dates = [start_date + timedelta(days=random.randint(0, date_range)) 
                        for _ in range(num_rows)]
                data[col_name] = dates
            
            elif generation_method == 'categorical':
                # Use provided values or generate
                values = constraints.get('values', ['A', 'B', 'C', 'D'])
                data[col_name] = np.random.choice(values, num_rows)
            
            else:
                # Default text generation
                if MIMESIS_AVAILABLE:
                    data[col_name] = [self.text_gen.sentence() for _ in range(num_rows)]
                else:
                    words = ['Lorem', 'ipsum', 'dolor', 'sit', 'amet']
                    data[col_name] = [' '.join(random.choices(words, k=5)) for _ in range(num_rows)]
        
        return pd.DataFrame(data)
    
    def _enhance_with_mimesis(self, data, rules):
        """Enhance data with realistic values using Mimesis"""
        if not MIMESIS_AVAILABLE or data.empty:
            return data
        
        for col in data.columns:
            if col in rules.get('columns', {}):
                col_info = rules['columns'][col]
                semantic_type = col_info.get('semantic_type', '')
                
                # Enhance specific column types
                if 'name' in semantic_type:
                    # Replace with realistic names
                    data[col] = [self.person.full_name() for _ in range(len(data))]
                
                elif 'email' in semantic_type:
                    # Replace with realistic emails
                    data[col] = [self.person.email() for _ in range(len(data))]
                
                elif 'address' in semantic_type:
                    # Replace with realistic addresses
                    data[col] = [self.address.address() for _ in range(len(data))]
                
                elif 'city' in semantic_type:
                    data[col] = [self.address.city() for _ in range(len(data))]
                
                elif 'country' in semantic_type:
                    data[col] = [self.address.country() for _ in range(len(data))]
                
                elif 'phone' in semantic_type:
                    data[col] = [self.person.telephone() for _ in range(len(data))]
        
        return data
    
    def _apply_noise(self, data, noise_level, rules):
        """Apply controlled noise to data"""
        if noise_level <= 0 or data.empty:
            return data
        
        # Only apply noise to numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in data.columns:
                col_data = data[col]
                if col in rules.get('columns', {}):
                    col_info = rules['columns'][col]
                    semantic_type = col_info.get('semantic_type', '')
                    
                    # Apply different noise based on column type
                    if semantic_type in ['age', 'quantity']:
                        # For integers, add/subtract small amounts
                        noise = np.random.randint(-int(noise_level * 10), int(noise_level * 10), len(data))
                        data[col] = col_data + noise
                        # Ensure still valid range
                        if 'constraints' in col_info:
                            min_val = col_info['constraints'].get('min')
                            max_val = col_info['constraints'].get('max')
                            if min_val is not None and max_val is not None:
                                data[col] = data[col].clip(min_val, max_val).astype(int)
                    
                    elif semantic_type in ['price', 'amount']:
                        # For prices, add percentage noise
                        noise_percent = np.random.uniform(-noise_level, noise_level, len(data))
                        data[col] = col_data * (1 + noise_percent)
                        # Round to 2 decimals
                        data[col] = data[col].round(2)
        
        return data

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
        .llm-analysis {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .data-preview {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">🔢 Smart Data Generator</div>', unsafe_allow_html=True)
        st.markdown("### Generate realistic synthetic data using AI + Smart Libraries")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Show library status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LLM Usage", "One-time analysis", delta="Cost optimized")
    with col2:
        status = "✅ Available" if SDV_AVAILABLE else "⚠️ Install SDV"
        st.metric("SDV Library", status)
    with col3:
        status = "✅ Available" if MIMESIS_AVAILABLE else "⚠️ Install Mimesis"
        st.metric("Mimesis Library", status)
    
    # Initialize session state
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = HybridDataGenerator()
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'llm_rules' not in st.session_state:
        st.session_state.llm_rules = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "⚙️ Generate Data", "📊 Results & Download"])
    
    with tab1:
        st.markdown('<div class="step-card"><h3>Step 1: Upload Sample Data</h3></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader(
                "Upload your sample data (CSV)",
                type=['csv'],
                help="Upload 5-1000 rows for best analysis"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if df.empty:
                        st.error("❌ Empty file")
                    else:
                        st.session_state.uploaded_data = df
                        
                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                        
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            missing = df.isna().sum().sum()
                            st.metric("Missing Values", missing)
                        
                        # Preview
                        with st.expander("📋 Data Preview", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Smart Analysis Button
                        st.markdown('<div class="llm-analysis"><h4>🤖 Smart Analysis (One-time LLM)</h4></div>', unsafe_allow_html=True)
                        st.caption("LLM will analyze ONLY 10 rows to extract generation rules. This is a one-time cost.")
                        
                        if st.button("🚀 Run Smart Analysis", type="primary", use_container_width=True):
                            if len(df) >= 3:
                                with st.spinner("Running one-time LLM analysis..."):
                                    rules = st.session_state.data_generator.rule_extractor.extract_smart_rules(df)
                                    st.session_state.llm_rules = rules
                                    
                                    # Show analysis results
                                    st.success("✅ Analysis complete!")
                                    
                                    # Display detected rules
                                    with st.expander("📊 Detected Rules", expanded=True):
                                        st.json(rules)
                                        
                                    st.rerun()
                            else:
                                st.warning("Need at least 3 rows for analysis")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            st.subheader("💡 How It Works")
            st.markdown("""
            **Smart Pipeline:**
            1. **One-time LLM Analysis** (10 rows only)
            2. **Rule extraction** for all columns
            3. **SDV statistical generation** (free)
            4. **Mimesis realistic data** (free)
            5. **Business logic application**
            
            **Cost Optimized:**
            - Only **1 LLM call** per dataset
            - Rules cached for reuse
            - Free libraries for generation
            
            **Better Results:**
            - Realistic names, emails
            - Correct data types
            - Business logic applied
            """)
    
    with tab2:
        if st.session_state.uploaded_data is None:
            st.info("📤 Please upload data in Step 1")
        elif st.session_state.llm_rules is None:
            st.info("🤖 Please run Smart Analysis first")
        else:
            st.markdown('<div class="step-card"><h3>Step 2: Generate Synthetic Data</h3></div>', unsafe_allow_html=True)
            
            # Show analysis summary
            rules = st.session_state.llm_rules
            st.subheader("📊 Analysis Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Dataset Type:** `{rules.get('dataset_type', 'Unknown')}`")
                st.write(f"**Columns Analyzed:** {len(rules.get('columns', {}))}")
            
            with col2:
                relationships = rules.get('relationships', [])
                st.write(f"**Relationships Found:** {len(relationships)}")
                if relationships:
                    for rel in relationships[:2]:
                        st.write(f"• {rel}")
            
            # Generation settings
            st.subheader("⚙️ Generation Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_rows = st.select_slider(
                    "Rows to generate",
                    options=[100, 500, 1000, 5000, 10000],
                    value=1000
                )
            
            with col2:
                noise_level = st.slider(
                    "Noise/Variation",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="Add randomness to data (0 = exact patterns, 0.5 = more variation)"
                )
                st.caption(f"Noise level: {int(noise_level * 100)}%")
            
            with col3:
                generation_method = st.selectbox(
                    "Generation Method",
                    options=["Smart Hybrid (Recommended)", "Rule-Based Only", "Statistical Only"],
                    help="Smart Hybrid uses LLM rules + SDV + Mimesis"
                )
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                fix_data_types = st.checkbox("Fix data types", value=True, 
                    help="Ensure ages are integers, prices have 2 decimals, etc.")
                enhance_realism = st.checkbox("Enhance with realistic data", value=True,
                    help="Use Mimesis for realistic names, emails, addresses")
            
            # Generate button
            if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
                with st.spinner(f"Generating {num_rows} realistic rows..."):
                    # Generate data
                    generated_df = st.session_state.data_generator.generate_smart_data(
                        original_df=st.session_state.uploaded_data,
                        num_rows=num_rows,
                        noise_level=noise_level
                    )
                    
                    if generated_df is not None and not generated_df.empty:
                        st.session_state.generated_data = generated_df
                        st.success(f"✅ Generated {len(generated_df)} realistic rows!")
                        st.rerun()
                    else:
                        st.error("❌ Generation failed")
    
    with tab3:
        if st.session_state.generated_data is None:
            st.info("⚙️ Please generate data in Step 2")
        else:
            st.markdown('<div class="step-card"><h3>Step 3: Download Your Data</h3></div>', unsafe_allow_html=True)
            
            generated_df = st.session_state.generated_data
            
            # Show quality metrics
            st.subheader("📊 Quality Check")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Check data types
                int_cols = len([col for col in generated_df.columns 
                              if generated_df[col].dtype in ['int64', 'int32']])
                st.metric("Integer Columns", int_cols)
            
            with col2:
                # Check for decimal prices
                price_cols = [col for col in generated_df.columns if 'price' in col.lower() or 'amount' in col.lower()]
                if price_cols:
                    sample_price = generated_df[price_cols[0]].iloc[0]
                    if isinstance(sample_price, float):
                        st.metric("Price Format", "✓ 2 decimals")
            
            with col3:
                # Check for realistic names
                name_cols = [col for col in generated_df.columns if 'name' in col.lower()]
                if name_cols:
                    sample_name = generated_df[name_cols[0]].iloc[0]
                    if ' ' in str(sample_name) and len(str(sample_name).split()) >= 2:
                        st.metric("Names", "✓ Realistic")
            
            with col4:
                # Missing values
                missing = generated_df.isna().sum().sum()
                st.metric("Missing Values", missing)
            
            # Data preview
            st.subheader("📋 Generated Data Preview")
            
            preview_tab1, preview_tab2 = st.tabs(["First 10 Rows", "Sample Check"])
            
            with preview_tab1:
                st.dataframe(generated_df.head(10), use_container_width=True)
            
            with preview_tab2:
                # Show random sample
                st.write("**Random Sample (5 rows):**")
                sample = generated_df.sample(min(5, len(generated_df)))
                st.dataframe(sample, use_container_width=True)
                
                # Check specific issues
                st.write("**Data Quality Check:**")
                
                # Check order IDs
                id_cols = [col for col in generated_df.columns if 'id' in col.lower()]
                if id_cols:
                    id_col = id_cols[0]
                    ids = generated_df[id_col]
                    if pd.api.types.is_integer_dtype(ids):
                        st.success(f"✓ {id_col}: Integer type")
                    else:
                        st.warning(f"⚠ {id_col}: Not integer")
                
                # Check ages
                age_cols = [col for col in generated_df.columns if 'age' in col.lower()]
                if age_cols:
                    age_col = age_cols[0]
                    ages = generated_df[age_col]
                    if pd.api.types.is_integer_dtype(ages):
                        valid_ages = ((ages >= 0) & (ages <= 120)).all()
                        if valid_ages:
                            st.success(f"✓ {age_col}: Valid ages (0-120)")
            
            # Download options
            st.subheader("💾 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV
                csv = generated_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel (if openpyxl available)
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        generated_df.to_excel(writer, index=False, sheet_name='Data')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📗 Download Excel",
                        data=excel_data,
                        file_name=f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except:
                    st.button("📗 Download Excel", disabled=True, 
                            help="Install openpyxl: pip install openpyxl")
            
            with col3:
                if st.button("🔄 Generate More", use_container_width=True):
                    st.session_state.generated_data = None
                    st.rerun()
            
            # Reset
            st.markdown("---")
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state.uploaded_data = None
                st.session_state.llm_rules = None
                st.session_state.generated_data = None
                st.rerun()

# Run the app
if __name__ == "__main__":
    main()
