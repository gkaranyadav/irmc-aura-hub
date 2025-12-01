# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
import io
import random
import hashlib
from groq import Groq
from auth import check_session

# =============================================================================
# SMART PATTERN DETECTOR (Understands just enough)
# =============================================================================
class SmartPatternDetector:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.llm_available = True
        except:
            self.llm_available = False
    
    def detect_semantic_types(self, df_sample):
        """
        SMART ANALYSIS: Understands WHAT data represents
        but only at HIGH LEVEL (cheap on tokens)
        """
        if not self.llm_available or df_sample.empty:
            return self._fallback_detection(df_sample)
        
        # OPTIMIZATION: Use only essential info
        column_info = []
        for col in df_sample.columns[:15]:  # Limit columns
            # Get 2 sample values (enough to understand)
            samples = []
            for i in range(min(2, len(df_sample))):
                val = str(df_sample[col].iloc[i])[:30]  # Trim to save tokens
                samples.append(val)
            
            column_info.append({
                'column': col,
                'samples': samples
            })
        
        # SMALL PROMPT for semantic understanding
        prompt = f"""
        Look at these data samples and tell me what KIND of data each column contains.
        Be SPECIFIC but BRIEF.
        
        COLUMNS:
        {json.dumps(column_info, indent=2)}
        
        For EACH column, identify the SEMANTIC TYPE from these options:
        
        1. PERSON_DATA: "customer_name", "employee_name", "first_name", "last_name", "full_name"
        2. CONTACT_INFO: "email_address", "phone_number", "mobile_number", "contact_email"
        3. IDENTIFIERS: "order_id", "customer_id", "product_id", "transaction_id", "invoice_number"
        4. NUMERIC_VALUES: "age", "price", "amount", "quantity", "weight", "height", "score"
        5. DATES_TIMES: "order_date", "birth_date", "timestamp", "created_at", "updated_at"
        6. CATEGORIES: "product_category", "status", "type", "gender", "country", "city"
        7. DESCRIPTIONS: "product_description", "notes", "comments", "address", "title"
        8. BOOLEAN: "is_active", "has_account", "verified", "completed"
        9. OTHER: "unknown_data", "codes", "references", "links"
        
        Return ONLY JSON:
        {{
            "columns": {{
                "column_name": "semantic_type",
                "specific_example": "customer_name" (more specific)
            }},
            "dataset_context": "ecommerce|customer|inventory|financial|other",
            "generation_advice": "specific advice for realistic generation"
        }}
        
        Example:
        {{
            "columns": {{
                "OrderID": "IDENTIFIERS",
                "specific_example": "order_id"
            }},
            "dataset_context": "ecommerce",
            "generation_advice": "Order IDs should be sequential integers starting from 1000"
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a data semantics expert. Identify what data represents."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b",  # Smaller model for cost
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Enhance with statistical analysis
            result = self._add_statistical_insights(result, df_sample)
            
            return result
            
        except Exception as e:
            st.warning(f"LLM analysis failed: {e}")
            return self._fallback_detection(df_sample)
    
    def _add_statistical_insights(self, analysis, df):
        """Add concrete numbers to LLM analysis"""
        if 'columns' not in analysis:
            return analysis
        
        for col, col_type in analysis['columns'].items():
            if col in df.columns:
                col_data = df[col].dropna()
                if not col_data.empty:
                    # Add concrete stats based on type
                    if 'IDENTIFIERS' in col_type:
                        # Check if sequential
                        if pd.api.types.is_numeric_dtype(col_data):
                            analysis[col] = analysis.get(col, {})
                            analysis[col]['is_sequential'] = self._check_sequential(col_data)
                            analysis[col]['start_value'] = int(col_data.min())
                    
                    elif 'NUMERIC_VALUES' in col_type:
                        if pd.api.types.is_numeric_dtype(col_data):
                            analysis[col] = analysis.get(col, {})
                            analysis[col]['min'] = float(col_data.min())
                            analysis[col]['max'] = float(col_data.max())
                            analysis[col]['mean'] = float(col_data.mean())
                            
                            # Detect if integer
                            if col_data.apply(lambda x: float(x).is_integer()).all():
                                analysis[col]['is_integer'] = True
                    
                    elif 'DATES_TIMES' in col_type:
                        # Try to parse as date
                        try:
                            dates = pd.to_datetime(col_data, errors='coerce')
                            if dates.notna().any():
                                analysis[col] = analysis.get(col, {})
                                analysis[col]['min_date'] = str(dates.min())
                                analysis[col]['max_date'] = str(dates.max())
                        except:
                            pass
        
        return analysis
    
    def _check_sequential(self, series):
        """Check if values are sequential"""
        if len(series) < 2:
            return False
        
        sorted_vals = sorted(series.dropna().unique())
        diffs = np.diff(sorted_vals[:10])  # Check first 10
        
        # If all diffs are 1 (or consistent), it's sequential
        if len(diffs) > 0 and np.allclose(diffs, diffs[0], rtol=0.1):
            return True
        return False
    
    def _fallback_detection(self, df):
        """Rule-based fallback"""
        analysis = {
            'columns': {},
            'dataset_context': 'unknown',
            'generation_advice': 'Use statistical patterns'
        }
        
        for col in df.columns:
            col_lower = col.lower()
            col_data = df[col].dropna()
            
            # Heuristic detection
            if any(x in col_lower for x in ['id', 'no', 'num', 'code']):
                analysis['columns'][col] = 'IDENTIFIERS'
            elif any(x in col_lower for x in ['name', 'first', 'last', 'full']):
                analysis['columns'][col] = 'PERSON_DATA'
            elif 'email' in col_lower:
                analysis['columns'][col] = 'CONTACT_INFO'
            elif 'phone' in col_lower:
                analysis['columns'][col] = 'CONTACT_INFO'
            elif 'age' in col_lower:
                analysis['columns'][col] = 'NUMERIC_VALUES'
            elif any(x in col_lower for x in ['price', 'amount', 'cost', 'total']):
                analysis['columns'][col] = 'NUMERIC_VALUES'
            elif any(x in col_lower for x in ['date', 'time', 'created', 'updated']):
                analysis['columns'][col] = 'DATES_TIMES'
            elif any(x in col_lower for x in ['category', 'type', 'status', 'gender']):
                analysis['columns'][col] = 'CATEGORIES'
            else:
                analysis['columns'][col] = 'OTHER'
        
        return analysis

# =============================================================================
# REALISTIC DATA GENERATORS (Domain-specific)
# =============================================================================
class RealisticGenerator:
    def __init__(self):
        # Realistic data pools
        self.first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 
                          'Michael', 'Linda', 'William', 'Elizabeth', 'David', 'Susan']
        self.last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                          'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez']
        
        self.product_categories = {
            'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smartwatch'],
            'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes'],
            'Home': ['Chair', 'Table', 'Lamp', 'Bed', 'Sofa'],
            'Books': ['Novel', 'Textbook', 'Cookbook', 'Biography', 'Self-Help'],
            'Food': ['Coffee', 'Tea', 'Snacks', 'Chocolate', 'Chips']
        }
        
        self.countries = ['USA', 'India', 'UK', 'Canada', 'Australia', 'Germany', 
                         'France', 'Japan', 'China', 'Brazil', 'Mexico', 'Spain']
        
        self.cities_by_country = {
            'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
            'India': ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai'],
            'UK': ['London', 'Manchester', 'Birmingham', 'Liverpool', 'Glasgow']
        }
    
    def generate_column(self, col_name, semantic_type, analysis, num_rows, original_data=None):
        """Generate realistic data based on semantic understanding"""
        
        if semantic_type == 'PERSON_DATA':
            if 'first' in col_name.lower():
                return [random.choice(self.first_names) for _ in range(num_rows)]
            elif 'last' in col_name.lower():
                return [random.choice(self.last_names) for _ in range(num_rows)]
            else:
                return [f"{random.choice(self.first_names)} {random.choice(self.last_names)}" 
                       for _ in range(num_rows)]
        
        elif semantic_type == 'CONTACT_INFO':
            if 'email' in col_name.lower():
                # Generate realistic emails
                domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com', 'business.com']
                emails = []
                for i in range(num_rows):
                    first = random.choice(self.first_names).lower()
                    last = random.choice(self.last_names).lower()
                    domain = random.choice(domains)
                    # Multiple email patterns
                    pattern = random.choice([1, 2, 3])
                    if pattern == 1:
                        email = f"{first}.{last}@{domain}"
                    elif pattern == 2:
                        email = f"{first[0]}{last}@{domain}"
                    else:
                        email = f"{first}{last}{random.randint(1, 99)}@{domain}"
                    emails.append(email)
                return emails
            
            elif 'phone' in col_name.lower():
                # Realistic phone numbers
                formats = ['(###) ###-####', '###-###-####', '+1 ### ### ####']
                phones = []
                for _ in range(num_rows):
                    fmt = random.choice(formats)
                    phone = ''
                    for char in fmt:
                        if char == '#':
                            phone += str(random.randint(0, 9))
                        else:
                            phone += char
                    phones.append(phone)
                return phones
        
        elif semantic_type == 'IDENTIFIERS':
            # Check if sequential
            if original_data is not None and analysis.get(col_name, {}).get('is_sequential'):
                start = analysis[col_name].get('start_value', 1000)
                return list(range(start, start + num_rows))
            else:
                # Generate realistic IDs
                patterns = ['ID-#####', 'REF-####', 'ORD-###', 'CUST-#####']
                ids = []
                for i in range(num_rows):
                    pattern = random.choice(patterns)
                    id_str = ''
                    for char in pattern:
                        if char == '#':
                            id_str += str(random.randint(0, 9))
                        else:
                            id_str += char
                    ids.append(id_str)
                return ids
        
        elif semantic_type == 'NUMERIC_VALUES':
            # Get realistic ranges from analysis
            min_val = analysis.get(col_name, {}).get('min', 0)
            max_val = analysis.get(col_name, {}).get('max', 100)
            mean_val = analysis.get(col_name, {}).get('mean', (min_val + max_val) / 2)
            
            if analysis.get(col_name, {}).get('is_integer', False):
                # Generate integers
                if 'age' in col_name.lower():
                    # Age-specific: mostly 18-65, some older
                    ages = []
                    for _ in range(num_rows):
                        if random.random() < 0.8:
                            ages.append(random.randint(18, 65))
                        else:
                            ages.append(random.randint(66, 90))
                    return ages
                elif 'quantity' in col_name.lower():
                    # Quantity: mostly small numbers
                    return np.random.poisson(lam=3, size=num_rows).clip(1, 20)
                else:
                    # General integers
                    return np.random.randint(min_val, max_val + 1, num_rows)
            else:
                # Generate floats (prices, amounts)
                if any(x in col_name.lower() for x in ['price', 'amount', 'cost', 'total']):
                    # Prices: realistic with .99/.50 endings
                    prices = np.random.uniform(min_val, max_val, num_rows)
                    # Apply realistic price patterns
                    realistic_prices = []
                    for price in prices:
                        if random.random() < 0.3:
                            # End with .99
                            realistic_prices.append(int(price) + 0.99)
                        elif random.random() < 0.2:
                            # End with .50
                            realistic_prices.append(int(price) + 0.50)
                        elif random.random() < 0.2:
                            # Round number
                            realistic_prices.append(round(price))
                        else:
                            # Keep as is, round to 2 decimals
                            realistic_prices.append(round(price, 2))
                    return realistic_prices
                else:
                    # General floats
                    return np.random.uniform(min_val, max_val, num_rows).round(2)
        
        elif semantic_type == 'DATES_TIMES':
            # Generate realistic dates
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now() + timedelta(days=30)
            
            if 'birth' in col_name.lower():
                # Birth dates: mostly 1950-2005
                start_date = datetime(1950, 1, 1)
                end_date = datetime(2005, 12, 31)
            
            date_range = (end_date - start_date).days
            dates = []
            for _ in range(num_rows):
                random_days = random.randint(0, date_range)
                date = start_date + timedelta(days=random_days)
                
                # Format based on original
                if original_data is not None and len(original_data) > 0:
                    sample = str(original_data.iloc[0])
                    if re.match(r'\d{4}-\d{2}-\d{2}', sample):
                        dates.append(date.strftime('%Y-%m-%d'))
                    elif re.match(r'\d{2}/\d{2}/\d{4}', sample):
                        dates.append(date.strftime('%m/%d/%Y'))
                    else:
                        dates.append(date.strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    dates.append(date.strftime('%Y-%m-%d'))
            
            return dates
        
        elif semantic_type == 'CATEGORIES':
            # Generate realistic categories
            if 'country' in col_name.lower():
                return [random.choice(self.countries) for _ in range(num_rows)]
            elif 'city' in col_name.lower():
                cities = []
                for _ in range(num_rows):
                    country = random.choice(self.countries)
                    if country in self.cities_by_country:
                        cities.append(random.choice(self.cities_by_country[country]))
                    else:
                        cities.append(f"City_{random.randint(1, 100)}")
                return cities
            elif 'gender' in col_name.lower():
                return random.choices(['Male', 'Female', 'Other'], 
                                    weights=[48, 48, 4], k=num_rows)
            elif any(x in col_name.lower() for x in ['status', 'state']):
                return random.choices(['Active', 'Inactive', 'Pending', 'Completed'], 
                                    k=num_rows)
            else:
                # Generic categories
                if original_data is not None and len(original_data) > 0:
                    # Use original categories
                    unique_vals = original_data.dropna().unique()
                    if len(unique_vals) <= 20:
                        return random.choices(list(unique_vals), k=num_rows)
                
                # Fallback
                letters = ['A', 'B', 'C', 'D', 'E', 'F']
                return [f"Category_{random.choice(letters)}" for _ in range(num_rows)]
        
        elif semantic_type == 'DESCRIPTIONS':
            # Generate realistic descriptions
            templates = [
                "High quality {product} with excellent features",
                "Premium {product} for professional use",
                "Standard {product} suitable for everyday use",
                "Economy {product} with basic functionality"
            ]
            
            products = ['product', 'item', 'device', 'tool', 'equipment']
            descriptions = []
            for _ in range(num_rows):
                template = random.choice(templates)
                product = random.choice(products)
                description = template.replace('{product}', product)
                
                # Add some variation
                if random.random() < 0.3:
                    description += ". Includes warranty."
                elif random.random() < 0.3:
                    description += ". Energy efficient."
                
                descriptions.append(description)
            
            return descriptions
        
        else:
            # Unknown type - try to mimic original pattern
            if original_data is not None and len(original_data) > 0:
                return self._mimic_pattern(original_data, num_rows)
            else:
                # Generate placeholder
                return [f"Data_{i}" for i in range(num_rows)]
    
    def _mimic_pattern(self, original_series, num_rows):
        """Generate data that looks like original"""
        samples = original_series.dropna().head(20).astype(str).tolist()
        
        if not samples:
            return [f"Value_{i}" for i in range(num_rows)]
        
        # Check pattern
        first_sample = samples[0]
        
        # Pattern 1: Alphanumeric codes
        if re.match(r'^[A-Z]+-\d+$', first_sample):
            # Format like "ABC-123"
            prefix = re.match(r'^[A-Z]+-', first_sample).group()
            return [f"{prefix}{random.randint(100, 999)}" for _ in range(num_rows)]
        
        # Pattern 2: Numbers only
        elif re.match(r'^\d+$', first_sample):
            length = len(first_sample)
            return [str(random.randint(10**(length-1), 10**length - 1)) 
                   for _ in range(num_rows)]
        
        # Pattern 3: Mixed with separators
        elif any(sep in first_sample for sep in ['_', '-', '.']):
            parts = re.split(r'[_\-.]+', first_sample)
            generated = []
            for _ in range(num_rows):
                part_list = []
                for part in parts:
                    if part.isdigit():
                        part_list.append(str(random.randint(0, 999)))
                    elif part.isalpha():
                        part_list.append(''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=len(part))))
                    else:
                        part_list.append(part)
                separator = first_sample[re.search(r'[_\-.]+', first_sample).start()]
                generated.append(separator.join(part_list))
            return generated
        
        # Default: Reuse samples
        else:
            return random.choices(samples, k=num_rows)

# =============================================================================
# MAIN GENERATOR (Orchestrates everything)
# =============================================================================
class SmartDataGenerator:
    def __init__(self):
        self.detector = SmartPatternDetector()
        self.generator = RealisticGenerator()
    
    def generate_realistic_data(self, original_df, num_rows, noise_level=0.1):
        """
        Main function: Generate realistic synthetic data
        """
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Understand data semantics (ONE LLM CALL)
        with st.spinner("🤔 Understanding your data (one-time analysis)..."):
            analysis = self.detector.detect_semantic_types(original_df.head(20))
        
        # Step 2: Generate each column realistically
        generated_data = {}
        
        for col in original_df.columns:
            semantic_type = analysis.get('columns', {}).get(col, 'OTHER')
            
            # Generate this column
            generated_data[col] = self.generator.generate_column(
                col_name=col,
                semantic_type=semantic_type,
                analysis=analysis.get(col, {}),
                num_rows=num_rows,
                original_data=original_df[col] if col in original_df.columns else None
            )
        
        # Step 3: Create DataFrame
        df_generated = pd.DataFrame(generated_data)
        
        # Step 4: Apply relationships
        df_generated = self._apply_relationships(df_generated, analysis)
        
        # Step 5: Add noise if requested
        if noise_level > 0:
            df_generated = self._add_controlled_noise(df_generated, noise_level)
        
        return df_generated
    
    def _apply_relationships(self, df, analysis):
        """Apply simple relationships for realism"""
        
        # Relationship 1: Email should match name if both exist
        name_cols = [c for c in df.columns if 'PERSON_DATA' in analysis.get('columns', {}).get(c, '')]
        email_cols = [c for c in df.columns if 'email' in c.lower()]
        
        if name_cols and email_cols:
            name_col = name_cols[0]
            email_col = email_cols[0]
            
            for i in range(len(df)):
                name = str(df.at[i, name_col])
                if ' ' in name:
                    first, last = name.split(' ', 1)
                    first = first.lower()
                    last = last.lower()
                    
                    # Update email to match name
                    current_email = str(df.at[i, email_col])
                    if '@' in current_email:
                        domain = current_email.split('@')[1]
                        # Create realistic email from name
                        email_pattern = random.choice([1, 2])
                        if email_pattern == 1:
                            new_email = f"{first}.{last}@{domain}"
                        else:
                            new_email = f"{first[0]}{last}@{domain}"
                        
                        df.at[i, email_col] = new_email
        
        # Relationship 2: Dates should be logical
        date_cols = [c for c in df.columns if 'DATES_TIMES' in analysis.get('columns', {}).get(c, '')]
        status_cols = [c for c in df.columns if any(x in c.lower() for x in ['status', 'state'])]
        
        if date_cols and status_cols:
            date_col = date_cols[0]
            status_col = status_cols[0]
            
            for i in range(len(df)):
                status = str(df.at[i, status_col]).lower()
                if 'pending' in status or 'processing' in status:
                    # Future dates for pending
                    date = pd.to_datetime(df.at[i, date_col], errors='coerce')
                    if pd.notna(date):
                        # Make it future (next 30 days)
                        if date < datetime.now():
                            new_date = datetime.now() + timedelta(days=random.randint(1, 30))
                            df.at[i, date_col] = new_date.strftime('%Y-%m-%d')
        
        return df
    
    def _add_controlled_noise(self, df, noise_level):
        """Add realistic variation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if noise_level > 0:
                # Add percentage-based noise
                noise = np.random.uniform(-noise_level, noise_level, len(df))
                df[col] = df[col] * (1 + noise)
                
                # Round appropriately
                if df[col].dtype in ['int64', 'int32']:
                    df[col] = df[col].round().astype(int)
                else:
                    df[col] = df[col].round(2)
        
        return df

# =============================================================================
# MAIN APP (Streamlit Interface)
# =============================================================================
def main():
    # Authentication check
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
    st.markdown("Generate **realistic, meaningful** data from your samples")
    
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
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "⚙️ Generate", "📊 Download"])
    
    with tab1:
        st.header("Upload Your Data")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                if df.empty:
                    st.error("Empty file")
                else:
                    st.session_state.original_data = df
                    
                    st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                    
                    # Show preview
                    with st.expander("Preview Data", expanded=True):
                        st.dataframe(df.head(10))
                    
                    # Show column types
                    st.subheader("📋 Column Analysis")
                    col_types = {}
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        sample = str(df[col].iloc[0])[:30] if len(df) > 0 else ""
                        col_types[col] = f"{dtype}: {sample}"
                    
                    for col, info in col_types.items():
                        st.write(f"**{col}**: `{info}`")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        if st.session_state.original_data is None:
            st.info("Please upload data first")
        else:
            st.header("Generate Synthetic Data")
            
            df = st.session_state.original_data
            
            # Settings
            col1, col2 = st.columns(2)
            
            with col1:
                num_rows = st.select_slider(
                    "Rows to generate",
                    options=[100, 500, 1000, 5000, 10000],
                    value=1000
                )
            
            with col2:
                noise = st.slider("Variation", 0.0, 0.5, 0.1, 0.05,
                                help="Adds realistic variation to data")
            
            # Generation button
            if st.button("🚀 Generate Realistic Data", type="primary", use_container_width=True):
                with st.spinner("Generating realistic data..."):
                    try:
                        generated = st.session_state.generator.generate_realistic_data(
                            original_df=df,
                            num_rows=num_rows,
                            noise_level=noise
                        )
                        
                        st.session_state.generated_data = generated
                        st.success(f"✅ Generated {len(generated)} realistic rows!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
    
    with tab3:
        if st.session_state.generated_data is None:
            st.info("Generate data first")
        else:
            st.header("Download Your Data")
            
            df_gen = st.session_state.generated_data
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df_gen):,}")
            with col2:
                st.metric("Columns", len(df_gen.columns))
            with col3:
                # Check data quality
                int_cols = len([c for c in df_gen.columns if df_gen[c].dtype in ['int64', 'int32']])
                st.metric("Integer Columns", int_cols)
            with col4:
                # Check for realistic emails
                email_cols = [c for c in df_gen.columns if 'email' in c.lower()]
                if email_cols:
                    sample_email = str(df_gen[email_cols[0]].iloc[0])
                    if '@' in sample_email and '.' in sample_email.split('@')[1]:
                        st.metric("Emails", "✅ Realistic")
            
            # Preview
            st.subheader("Preview")
            st.dataframe(df_gen.head(10))
            
            # Download buttons
            st.subheader("Download")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV
                csv = df_gen.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="synthetic_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON
                json_str = df_gen.to_json(orient='records', indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name="synthetic_data.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Reset
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.original_data = None
                st.session_state.generated_data = None
                st.rerun()

if __name__ == "__main__":
    main()
