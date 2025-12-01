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

# =============================================================================
# ULTIMATE LLM DATA GENERATOR
# =============================================================================

class UltimateLLMGenerator:
    """Ultimate generator with PERFECT prompting"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
        except:
            self.available = False
            st.warning("LLM not available")
    
    def generate_perfect_data(self, original_df, num_rows):
        """Generate PERFECT data using ultimate prompting"""
        if not self.available or original_df.empty:
            return self._smart_fallback(original_df, num_rows)
        
        # Get LLM to generate data directly with ULTIMATE prompt
        with st.spinner("🤖 LLM is generating PERFECT realistic data..."):
            llm_data = self._get_llm_generated_data(original_df, num_rows)
        
        if llm_data is not None:
            return llm_data
        
        return self._smart_fallback(original_df, num_rows)
    
    def _get_llm_generated_data(self, df, num_rows):
        """Get LLM to generate data directly with ULTIMATE prompt"""
        
        # Prepare data samples
        samples = self._prepare_perfect_samples(df)
        
        # ULTIMATE PROMPT - Forces LLM to generate perfect data
        prompt = f"""
        CRITICAL TASK: Generate {num_rows} rows of SYNTHETIC DATA that looks 100% REALISTIC.
        
        ORIGINAL DATA (5 sample rows):
        {json.dumps(samples, indent=2, default=str)}
        
        IMPORTANT OBSERVATIONS from original data:
        1. This is E-COMMERCE ORDER DATA
        2. OrderID: Sequential numbers starting from 1001
        3. CustomerName: Mix of Indian and Western names (e.g., Rahul Verma, Laura Adams)
        4. Age: Realistic ages (22-44 range)
        5. Email: Realistic email patterns (e.g., rahul.verma24@gmail.com, john.smith@company.com)
        6. Country: Real countries (India, USA, UK, Canada, Australia, Singapore, UAE)
        7. Product: REAL tech products (e.g., iPhone 15 Pro, Samsung Galaxy S24, Dell XPS Laptop)
        8. Quantity: Usually 1-3, sometimes more
        9. Price: REALISTIC prices (e.g., $999.99 for iPhone, $1499 for laptop, $199 for headphones)
        10. OrderDate: Recent dates (mostly 2024-2025) in DD-MM-YYYY format
        11. Status: Real order statuses (Shipped, Pending, Delivered, Cancelled, Returned)
        
        CRITICAL RULES for generation:
        
        A) ORDER IDs:
        - MUST continue sequence from original
        - Original ends around 10100, so start from 10101
        - Format: Just numbers, no prefixes
        
        B) CUSTOMER NAMES:
        - MUST be REALISTIC human names
        - Mix: 60% Western (John Smith, Laura Adams), 40% Indian (Rahul Verma, Priya Sharma)
        - Format: FirstName LastName (capitalized properly)
        - NO numbers, NO placeholders like "Customer_1"
        
        C) EMAILS:
        - MUST follow patterns: firstname.lastname##@domain or firstinitiallastname##@domain
        - Use realistic domains: gmail.com, yahoo.com, outlook.com, hotmail.com, company.com
        - Examples: john.smith24@gmail.com, rverma45@yahoo.com, l.adams@company.com
        
        D) COUNTRIES:
        - REAL countries only: India, USA, UK, Canada, Australia, Singapore, UAE, Germany, Japan
        - Distribution: 40% India, 30% USA, 30% others
        - NO nonsense values
        
        E) PRODUCTS:
        - REAL tech products only:
          * Smartphones: iPhone 15 Pro, Samsung Galaxy S24, Google Pixel 8, OnePlus 12
          * Laptops: Apple MacBook Pro, Dell XPS 15, HP Spectre x360, Lenovo ThinkPad
          * Tablets: iPad Pro 12.9", Samsung Galaxy Tab S9, Microsoft Surface Pro
          * Accessories: Sony WH-1000XM5 Headphones, Apple AirPods Pro, Logitech MX Master Mouse
        - Format: Brand + Model + Variant (e.g., "iPhone 15 Pro 256GB")
        
        F) PRICES:
        - REALISTIC prices for each product type:
          * Smartphones: $699-$1499 (often .99 endings)
          * Laptops: $899-$2999
          * Tablets: $399-$1299
          * Accessories: $49-$499
        - Common endings: .99, .95, .00
        - NO nonsense like $399 for a laptop or $25000 for headphones
        
        G) DATES:
        - Mostly recent (2024-2025)
        - Format: DD-MM-YYYY
        - Logical: Shipping dates after order dates, etc.
        
        H) STATUS:
        - Real statuses only: Shipped, Delivered, Pending, Processing, Cancelled, Returned
        - Distribution: 40% Shipped, 30% Delivered, 15% Pending, 10% Processing, 5% others
        
        I) QUANTITY:
        - Usually 1-3 items
        - Sometimes 4-5 for accessories
        - Rarely more than 5
        
        RETURN FORMAT: JSON array ONLY:
        [
            {{
                "OrderID": "10101",
                "CustomerName": "John Smith",
                "Age": 28,
                "Email": "john.smith24@gmail.com",
                "Country": "USA",
                "Product": "iPhone 15 Pro 256GB",
                "Quantity": 1,
                "Price": 1099.99,
                "OrderDate": "15-01-2024",
                "Status": "Shipped"
            }},
            ...
        ]
        
        Generate EXACTLY {num_rows} rows. Make EVERY value realistic and logical!
        """
        
        try:
            messages = [
                {"role": "system", "content": """You are an e-commerce data expert. You generate PERFECTLY realistic synthetic data.
                
                CRITICAL INSTRUCTIONS:
                1. ALL values MUST be realistic and make sense
                2. NO placeholder values (no Customer_1, Product_123, user123@email.com)
                3. Names MUST be real human names
                4. Products MUST be real tech products with realistic prices
                5. Emails MUST follow real email patterns
                6. Everything MUST be logically consistent
                
                If you generate nonsense data, real businesses will fail. Be PERFECT."""},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,  # Balanced for variety but consistency
                max_tokens=8000  # Need more tokens for good data
            )
            
            result = response.choices[0].message.content
            
            # Parse the response
            return self._parse_ultimate_response(result, df.columns, num_rows)
            
        except Exception as e:
            st.error(f"LLM generation failed: {str(e)}")
            return None
    
    def _prepare_perfect_samples(self, df):
        """Prepare perfect samples for LLM"""
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
        
        return samples
    
    def _parse_ultimate_response(self, response, expected_columns, num_rows):
        """Parse LLM's perfect response"""
        try:
            # Try to extract JSON array
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Validate data
                validated_data = []
                for item in data:
                    validated = {}
                    for col in expected_columns:
                        if col in item:
                            validated[col] = item[col]
                        else:
                            # Generate reasonable default
                            validated[col] = self._generate_default_value(col, len(validated_data))
                    validated_data.append(validated)
                
                df = pd.DataFrame(validated_data[:num_rows])
                
                # Apply final validation
                df = self._validate_and_fix_data(df, expected_columns)
                
                return df
            
            # If no JSON array found, try line-by-line parsing
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
                    key = parts[0].strip().replace('"', '').replace("'", "")
                    value = parts[1].strip().replace('"', '').replace("'", "")
                    current_obj[key] = value
                
                if line.endswith('},') or line.endswith('}'):
                    if current_obj:
                        data.append(current_obj.copy())
                        current_obj = {}
            
            if data:
                df = pd.DataFrame(data[:num_rows])
                df = self._validate_and_fix_data(df, expected_columns)
                return df
            
        except Exception as e:
            st.warning(f"Failed to parse LLM response: {str(e)}")
        
        return None
    
    def _validate_and_fix_data(self, df, expected_columns):
        """Validate and fix data quality"""
        # Ensure all columns exist
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        
        # Fix data types and quality
        for idx, row in df.iterrows():
            # Fix OrderID
            if 'OrderID' in df.columns:
                try:
                    df.at[idx, 'OrderID'] = str(int(df.at[idx, 'OrderID']))
                except:
                    df.at[idx, 'OrderID'] = str(10000 + idx)
            
            # Fix CustomerName
            if 'CustomerName' in df.columns:
                name = str(df.at[idx, 'CustomerName'])
                if any(x in name.lower() for x in ['customer', 'user', 'temp', 'test', 'data']):
                    df.at[idx, 'CustomerName'] = self._generate_real_name()
                elif not any(c.isalpha() for c in name):
                    df.at[idx, 'CustomerName'] = self._generate_real_name()
            
            # Fix Email
            if 'Email' in df.columns:
                email = str(df.at[idx, 'Email'])
                if '@' not in email:
                    name = str(df.at[idx, 'CustomerName']) if 'CustomerName' in df.columns else 'user'
                    df.at[idx, 'Email'] = self._generate_email_from_name(name, idx)
            
            # Fix Product
            if 'Product' in df.columns:
                product = str(df.at[idx, 'Product'])
                if any(x in product.lower() for x in ['product', 'item', 'goods', 'temp']):
                    df.at[idx, 'Product'] = self._generate_real_product()
                elif product.isdigit():
                    df.at[idx, 'Product'] = self._generate_real_product()
            
            # Fix Price
            if 'Price' in df.columns:
                try:
                    price = float(str(df.at[idx, 'Price']).replace('$', '').replace(',', ''))
                    # Ensure realistic price
                    if price > 10000 or price < 1:
                        df.at[idx, 'Price'] = round(random.uniform(49, 1499), 2)
                except:
                    df.at[idx, 'Price'] = round(random.uniform(49, 1499), 2)
            
            # Fix Age
            if 'Age' in df.columns:
                try:
                    age = int(float(str(df.at[idx, 'Age'])))
                    if age < 18 or age > 80:
                        df.at[idx, 'Age'] = random.randint(22, 44)
                except:
                    df.at[idx, 'Age'] = random.randint(22, 44)
            
            # Fix Country
            if 'Country' in df.columns:
                country = str(df.at[idx, 'Country'])
                real_countries = ['India', 'USA', 'UK', 'Canada', 'Australia', 'Singapore', 'UAE', 'Germany', 'Japan']
                if country not in real_countries:
                    df.at[idx, 'Country'] = random.choice(real_countries)
            
            # Fix Status
            if 'Status' in df.columns:
                status = str(df.at[idx, 'Status'])
                real_statuses = ['Shipped', 'Delivered', 'Pending', 'Processing', 'Cancelled', 'Returned']
                if status not in real_statuses:
                    df.at[idx, 'Status'] = random.choice(real_statuses)
            
            # Fix Quantity
            if 'Quantity' in df.columns:
                try:
                    qty = int(float(str(df.at[idx, 'Quantity'])))
                    if qty < 1 or qty > 10:
                        df.at[idx, 'Quantity'] = random.randint(1, 3)
                except:
                    df.at[idx, 'Quantity'] = random.randint(1, 3)
        
        # Reorder columns
        df = df[expected_columns]
        
        return df
    
    def _generate_default_value(self, col_name, idx):
        """Generate default value for missing column"""
        col_lower = col_name.lower()
        
        if 'id' in col_lower:
            return str(10000 + idx + 1)
        
        elif 'name' in col_lower:
            return self._generate_real_name()
        
        elif 'email' in col_lower:
            return self._generate_email_from_name("User", idx)
        
        elif 'product' in col_lower:
            return self._generate_real_product()
        
        elif 'price' in col_lower or 'amount' in col_lower or 'cost' in col_lower:
            return round(random.uniform(49, 1499), 2)
        
        elif 'age' in col_lower:
            return random.randint(22, 44)
        
        elif 'country' in col_lower:
            return random.choice(['India', 'USA', 'UK', 'Canada', 'Australia'])
        
        elif 'status' in col_lower:
            return random.choice(['Shipped', 'Delivered', 'Pending'])
        
        elif 'quantity' in col_lower or 'qty' in col_lower:
            return random.randint(1, 3)
        
        elif 'date' in col_lower:
            date = datetime.now() - timedelta(days=random.randint(1, 365))
            return date.strftime('%d-%m-%Y')
        
        else:
            return f"Value_{idx}"
    
    def _generate_real_name(self):
        """Generate realistic human name"""
        western_first = ['John', 'James', 'Michael', 'David', 'Robert', 'William', 'Mary', 'Patricia', 
                        'Jennifer', 'Linda', 'Elizabeth', 'Susan', 'Jessica', 'Sarah']
        indian_first = ['Rahul', 'Amit', 'Raj', 'Sanjay', 'Vikram', 'Priya', 'Neha', 'Anjali', 'Sneha']
        
        western_last = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis']
        indian_last = ['Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma']
        
        if random.random() < 0.6:  # 60% Western
            first = random.choice(western_first)
            last = random.choice(western_last)
        else:  # 40% Indian
            first = random.choice(indian_first)
            last = random.choice(indian_last)
        
        return f"{first} {last}"
    
    def _generate_email_from_name(self, name, idx):
        """Generate realistic email from name"""
        # Extract first and last name
        parts = str(name).split()
        if len(parts) >= 2:
            first = parts[0].lower()
            last = parts[-1].lower()
            
            patterns = [
                f"{first}.{last}{random.randint(1,99)}@gmail.com",
                f"{first[0]}{last}{random.randint(10,99)}@yahoo.com",
                f"{first}.{last}@company.com",
                f"{first}{random.randint(100,999)}@outlook.com"
            ]
        else:
            patterns = [
                f"user{idx+1000}@gmail.com",
                f"client{idx+500}@yahoo.com",
                f"customer{idx}@company.com"
            ]
        
        return random.choice(patterns)
    
    def _generate_real_product(self):
        """Generate realistic tech product"""
        products = [
            # Smartphones
            "iPhone 15 Pro 256GB",
            "Samsung Galaxy S24 Ultra",
            "Google Pixel 8 Pro",
            "OnePlus 12 512GB",
            "Xiaomi 14 Pro",
            
            # Laptops
            "Apple MacBook Pro M3",
            "Dell XPS 15 Laptop",
            "HP Spectre x360",
            "Lenovo ThinkPad X1 Carbon",
            "Microsoft Surface Laptop 5",
            
            # Tablets
            "iPad Pro 12.9\" M2",
            "Samsung Galaxy Tab S9 Ultra",
            "Microsoft Surface Pro 9",
            
            # Accessories
            "Sony WH-1000XM5 Headphones",
            "Apple AirPods Pro 2",
            "Logitech MX Master 3S Mouse",
            "Samsung Galaxy Watch 6",
            "Apple Watch Series 9"
        ]
        
        return random.choice(products)
    
    def _smart_fallback(self, df, num_rows):
        """Smart fallback generation"""
        generated = {}
        
        for col in df.columns:
            col_lower = col.lower()
            original_vals = df[col].dropna().tolist()
            
            if 'id' in col_lower:
                # Sequential IDs
                last_id = 10000
                if original_vals:
                    try:
                        ids = []
                        for val in original_vals[:10]:
                            num_part = re.sub(r'\D', '', str(val))
                            if num_part:
                                ids.append(int(num_part))
                        if ids:
                            last_id = max(ids)
                    except:
                        pass
                generated[col] = [str(last_id + i + 1) for i in range(num_rows)]
            
            elif 'name' in col_lower:
                # Realistic names
                names = []
                for i in range(num_rows):
                    if random.random() < 0.6:
                        # Western
                        first = random.choice(['John', 'James', 'Michael', 'David', 'Mary', 'Patricia', 'Linda', 'Susan'])
                        last = random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller'])
                    else:
                        # Indian
                        first = random.choice(['Rahul', 'Amit', 'Raj', 'Sanjay', 'Priya', 'Neha', 'Anjali'])
                        last = random.choice(['Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta'])
                    names.append(f"{first} {last}")
                generated[col] = names
            
            elif 'email' in col_lower:
                # Realistic emails
                emails = []
                for i in range(num_rows):
                    first = random.choice(['john', 'jane', 'alex', 'sam', 'mike', 'sara'])
                    last = random.choice(['smith', 'johnson', 'williams', 'brown', 'jones'])
                    num = random.randint(1, 99)
                    domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.com'])
                    emails.append(f"{first}.{last}{num}@{domain}")
                generated[col] = emails
            
            elif 'product' in col_lower:
                # Realistic products
                products = []
                for i in range(num_rows):
                    brands = ['Apple iPhone', 'Samsung Galaxy', 'Google Pixel', 'Dell XPS', 'HP Spectre']
                    models = ['15 Pro', 'S24 Ultra', '8 Pro', '15 Laptop', 'x360']
                    storage = ['128GB', '256GB', '512GB', '1TB']
                    products.append(f"{random.choice(brands)} {random.choice(models)} {random.choice(storage)}")
                generated[col] = products
            
            elif 'price' in col_lower or 'amount' in col_lower:
                # Realistic prices
                prices = []
                for i in range(num_rows):
                    base = random.uniform(49, 1499)
                    if random.random() < 0.3:
                        price = math.floor(base) + 0.99
                    elif random.random() < 0.5:
                        price = round(base, 2)
                    else:
                        price = round(base)
                    prices.append(float(price))
                generated[col] = prices
            
            elif 'country' in col_lower:
                # Real countries
                countries = ['India', 'USA', 'UK', 'Canada', 'Australia', 'Singapore', 'UAE', 'Germany']
                generated[col] = random.choices(countries, k=num_rows)
            
            elif 'status' in col_lower:
                # Real statuses
                statuses = ['Shipped', 'Delivered', 'Pending', 'Processing', 'Cancelled']
                weights = [0.4, 0.3, 0.15, 0.1, 0.05]
                generated[col] = random.choices(statuses, weights=weights, k=num_rows)
            
            elif 'age' in col_lower:
                # Realistic ages
                generated[col] = [random.randint(22, 44) for _ in range(num_rows)]
            
            elif 'quantity' in col_lower:
                # Realistic quantities
                generated[col] = [random.randint(1, 3) for _ in range(num_rows)]
            
            elif 'date' in col_lower:
                # Recent dates
                dates = []
                for i in range(num_rows):
                    days_ago = random.randint(1, 365)
                    date = datetime.now() - timedelta(days=days_ago)
                    dates.append(date.strftime('%d-%m-%Y'))
                generated[col] = dates
            
            else:
                # Use original patterns
                if original_vals and len(set(original_vals)) <= 10:
                    generated[col] = random.choices(original_vals, k=num_rows)
                else:
                    generated[col] = [f"{col}_{i}" for i in range(num_rows)]
        
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
        page_title="Perfect Data Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Header
    st.title("✨ Perfect Data Generator")
    st.markdown("**LLM-Powered • Realistic Values • No Nonsense Data**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize generator
    if 'ultimate_generator' not in st.session_state:
        st.session_state.ultimate_generator = UltimateLLMGenerator()
    
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload Dataset (CSV)", type=['csv'])
    
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
                st.subheader("⚙️ Generate Perfect Data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=200,  # Limit for quality
                                             value=100)
                
                with col2:
                    quality_level = st.select_slider(
                        "Quality Level",
                        options=["Good", "Better", "Best"],
                        value="Best"
                    )
                
                with col3:
                    if st.button("🚀 Generate PERFECT Data", type="primary"):
                        if not st.session_state.ultimate_generator.available:
                            st.error("LLM not available. Check API key.")
                        else:
                            with st.spinner("Generating PERFECT realistic data..."):
                                # FIX: Use the correct generator instance
                                generator = st.session_state.ultimate_generator
                                generated = generator.generate_perfect_data(df, int(num_rows))
                                st.session_state.generated_data = generated
                                if generated is not None:
                                    st.success(f"✅ Generated {len(generated)} PERFECT rows!")
                                    st.balloons()
                                else:
                                    st.error("Failed to generate data")
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📊 Generated Data (Perfect Quality)")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Tabs
                    tab1, tab2, tab3 = st.tabs(["Preview", "Quality Report", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                        
                        # Show sample
                        st.write("**Sample Row Analysis:**")
                        if len(df_gen) > 0:
                            sample = df_gen.iloc[0]
                            st.json(sample.to_dict())
                    
                    with tab2:
                        # Quality report
                        st.write("## 📈 Data Quality Report")
                        
                        # Check each column
                        perfect_cols = []
                        good_cols = []
                        needs_improvement = []
                        
                        for col in df_gen.columns:
                            sample = str(df_gen[col].iloc[0]) if len(df_gen) > 0 else ""
                            
                            # Check for nonsense
                            is_perfect = True
                            issues = []
                            
                            if 'name' in col.lower():
                                if any(x in sample.lower() for x in ['customer', 'user', 'temp', 'test']) or sample.isdigit():
                                    is_perfect = False
                                    issues.append("Contains placeholder")
                            
                            if 'email' in col.lower():
                                if '@' not in sample or any(x in sample.lower() for x in ['user', 'test', 'example']):
                                    is_perfect = False
                                    issues.append("Invalid email")
                            
                            if 'product' in col.lower():
                                if sample.isdigit() or any(x in sample.lower() for x in ['product', 'item', 'temp']):
                                    is_perfect = False
                                    issues.append("Not a real product")
                            
                            if 'price' in col.lower():
                                try:
                                    price = float(str(sample).replace('$', '').replace(',', ''))
                                    if price > 10000 or price < 1:
                                        is_perfect = False
                                        issues.append("Unrealistic price")
                                except:
                                    is_perfect = False
                                    issues.append("Invalid price")
                            
                            # Categorize
                            if is_perfect and len(df_gen[col].unique()) > len(df_gen) * 0.1:
                                perfect_cols.append((col, "✅ Perfect"))
                            elif len(issues) == 0:
                                good_cols.append((col, "👍 Good"))
                            else:
                                needs_improvement.append((col, f"⚠️ Issues: {', '.join(issues)}"))
                        
                        # Display report
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Perfect Columns", len(perfect_cols))
                            if perfect_cols:
                                st.write("**Perfect:**")
                                for col, status in perfect_cols[:5]:
                                    st.write(f"- {col}")
                        
                        with col2:
                            st.metric("Good Columns", len(good_cols))
                            if good_cols:
                                st.write("**Good:**")
                                for col, status in good_cols[:5]:
                                    st.write(f"- {col}")
                        
                        with col3:
                            st.metric("Needs Fix", len(needs_improvement))
                            if needs_improvement:
                                st.write("**Needs Fix:**")
                                for col, status in needs_improvement:
                                    st.write(f"- {col}")
                    
                    with tab3:
                        # Download
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "perfect_generated_data.csv",
                            "text/csv"
                        )
                        
                        # Regenerate
                        if st.button("🔄 Generate New Perfect Dataset"):
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__": 
    main()
