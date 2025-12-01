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
# UNIVERSAL LLM-DRIVEN GENERATOR
# =============================================================================

class UniversalLLMGenerator:
    """Universal generator where LLM creates logic for each dataset"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
        except:
            self.available = False
            st.warning("LLM not available")
    
    def generate_universal_data(self, original_df, num_rows):
        """Generate data using LLM-created logic"""
        if not self.available or original_df.empty:
            return self._statistical_fallback(original_df, num_rows)
        
        # Step 1: Get LLM to analyze and create generation logic
        with st.spinner("🤖 LLM is analyzing data patterns..."):
            generation_logic = self._get_generation_logic(original_df)
        
        if not generation_logic:
            return self._statistical_fallback(original_df, num_rows)
        
        # Step 2: Execute LLM's generation logic
        with st.spinner("⚡ Generating data with LLM logic..."):
            df_generated = self._execute_generation_logic(generation_logic, original_df, num_rows)
        
        return df_generated
    
    def _get_generation_logic(self, df):
        """Get LLM to create generation logic for this dataset"""
        
        # Prepare comprehensive data info
        data_info = self._prepare_comprehensive_info(df)
        
        prompt = f"""
        Analyze this dataset and create generation logic for synthetic data.
        
        DATASET INFO:
        {json.dumps(data_info, indent=2, default=str)}
        
        TASK: Create a GENERATION LOGIC JSON that tells HOW to generate realistic synthetic data.
        
        For EACH column, provide:
        1. **detected_type**: What type of data is this? (id, name, email, date, numeric, category, text, etc.)
        2. **patterns_observed**: What patterns do you see in the data?
        3. **generation_method**: How to generate this data? Options:
           - "sequential_id": Continue sequence
           - "random_id": Random IDs
           - "realistic_name": Realistic names
           - "realistic_email": Realistic emails
           - "realistic_date": Realistic dates in observed format
           - "numeric_range": Numbers within observed range
           - "numeric_distribution": Numbers with observed distribution
           - "category_from_samples": Use/extend observed categories
           - "pattern_based": Generate based on observed patterns
           - "text_based": Generate descriptive text
        4. **generation_parameters**: Specific parameters for generation
        5. **realism_rules**: Rules to make data realistic
        
        IMPORTANT: DO NOT predefine specific values. Instead, describe HOW to generate them.
        
        Return JSON format:
        {{
            "dataset_analysis": "Brief analysis of what this dataset represents",
            "generation_logic": {{
                "columns": {{
                    "column_name": {{
                        "detected_type": "type",
                        "patterns_observed": "patterns",
                        "generation_method": "method",
                        "generation_parameters": {{
                            "param1": "value1",
                            "param2": ["list", "of", "values"]
                        }},
                        "realism_rules": ["rule1", "rule2"]
                    }}
                }},
                "relationships": [
                    {{
                        "columns": ["col1", "col2"],
                        "relationship": "description",
                        "maintenance_rule": "how to maintain"
                    }}
                ],
                "global_rules": ["overall rules for realism"]
            }}
        }}
        
        EXAMPLE for email column:
        {{
            "detected_type": "email",
            "patterns_observed": "username@domain format, mix of personal and business emails",
            "generation_method": "realistic_email",
            "generation_parameters": {{
                "pattern_variety": ["name_dot_name", "initial_lastname", "random_word"],
                "domain_types": ["personal", "business", "country_specific"]
            }},
            "realism_rules": ["Ensure valid email format", "Mix personal and business domains"]
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a data generation expert. Analyze data patterns and create generation logic without predefining specific values."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            
            # Extract JSON
            try:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    logic = json.loads(json_match.group())
                    st.session_state.generation_logic = logic
                    return logic
            except json.JSONDecodeError as e:
                st.warning(f"JSON parse error: {e}")
                st.text("LLM response (first 500 chars):")
                st.text(result[:500])
            
        except Exception as e:
            st.warning(f"LLM logic creation failed: {str(e)}")
        
        return None
    
    def _prepare_comprehensive_info(self, df):
        """Prepare comprehensive data information"""
        info = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": {},
            "sample_data": []
        }
        
        # Column analysis
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "sample_values": col_data.head(5).tolist(),
                    "unique_count": col_data.nunique(),
                    "null_count": df[col].isna().sum(),
                    "unique_ratio": col_data.nunique() / len(col_data)
                }
                
                # Statistical analysis for numeric columns
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(numeric_data) > 0:
                        col_info["numeric_stats"] = {
                            "min": float(numeric_data.min()),
                            "max": float(numeric_data.max()),
                            "mean": float(numeric_data.mean()),
                            "std": float(numeric_data.std()) if len(numeric_data) > 1 else 0,
                            "is_integer": numeric_data.apply(lambda x: float(x).is_integer()).all()
                        }
                except:
                    pass
                
                # Text analysis
                if len(col_data) > 0:
                    str_samples = col_data.head(3).astype(str).tolist()
                    col_info["text_patterns"] = {
                        "avg_length": np.mean([len(s) for s in str_samples]),
                        "has_emails": any('@' in s for s in str_samples),
                        "has_dates": any(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', s) for s in str_samples),
                        "has_names": any(s[0].isupper() and ' ' in s for s in str_samples)
                    }
                
                info["columns"][col] = col_info
        
        # Sample rows
        for idx, row in df.head(3).iterrows():
            row_data = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_data[col] = None
                elif isinstance(val, (int, np.integer)):
                    row_data[col] = int(val)
                elif isinstance(val, (float, np.floating)):
                    row_data[col] = float(val)
                else:
                    row_data[col] = str(val)[:100]
            info["sample_data"].append(row_data)
        
        return info
    
    def _execute_generation_logic(self, logic, original_df, num_rows):
        """Execute the LLM-created generation logic"""
        
        if 'generation_logic' not in logic or 'columns' not in logic['generation_logic']:
            return self._statistical_fallback(original_df, num_rows)
        
        generation_logic = logic['generation_logic']
        columns_logic = generation_logic.get('columns', {})
        
        generated_data = {}
        
        for col in original_df.columns:
            if col in columns_logic:
                col_logic = columns_logic[col]
                method = col_logic.get('generation_method', 'pattern_based')
                params = col_logic.get('generation_parameters', {})
                original_series = original_df[col]
                
                # Generate based on method
                generated_data[col] = self._generate_by_method(
                    method=method,
                    col_name=col,
                    col_logic=col_logic,
                    original_series=original_series,
                    num_rows=num_rows,
                    params=params
                )
            else:
                # Use statistical fallback for this column
                generated_data[col] = self._generate_statistical(original_df[col], num_rows)
        
        # Create DataFrame
        df = pd.DataFrame(generated_data)
        
        # Apply relationships if specified
        relationships = generation_logic.get('relationships', [])
        df = self._apply_relationships(df, relationships)
        
        # Apply global rules
        global_rules = generation_logic.get('global_rules', [])
        df = self._apply_global_rules(df, global_rules)
        
        return df
    
    def _generate_by_method(self, method, col_name, col_logic, original_series, num_rows, params):
        """Generate data based on LLM-specified method"""
        
        detected_type = col_logic.get('detected_type', '').lower()
        original_values = original_series.dropna().tolist()
        
        # ID generation methods
        if method == 'sequential_id':
            return self._generate_sequential_id(col_name, original_values, num_rows)
        
        elif method == 'random_id':
            return self._generate_random_id(col_name, original_values, num_rows)
        
        # Name generation
        elif method == 'realistic_name':
            return self._generate_realistic_names(col_name, original_values, num_rows, params)
        
        # Email generation
        elif method == 'realistic_email':
            return self._generate_realistic_emails(col_name, original_values, num_rows, params)
        
        # Date generation
        elif method == 'realistic_date':
            return self._generate_realistic_dates(col_name, original_values, num_rows, params)
        
        # Numeric generation
        elif method in ['numeric_range', 'numeric_distribution']:
            return self._generate_numeric_data(col_name, original_values, num_rows, params, method)
        
        # Category generation
        elif method == 'category_from_samples':
            return self._generate_categories(col_name, original_values, num_rows, params)
        
        # Text generation
        elif method == 'text_based':
            return self._generate_text_data(col_name, original_values, num_rows, params)
        
        # Pattern-based (fallback)
        else:
            return self._generate_pattern_based(col_name, original_values, num_rows, params)
    
    def _generate_sequential_id(self, col_name, original_values, num_rows):
        """Generate sequential IDs"""
        # Find last ID
        last_id = 1000
        try:
            # Extract numbers from original values
            nums = []
            for val in original_values[:20]:
                if val is not None:
                    # Try to extract numeric part
                    num_str = re.sub(r'\D', '', str(val))
                    if num_str:
                        nums.append(int(num_str))
            if nums:
                last_id = max(nums)
        except:
            pass
        
        # Check for prefix
        prefix = ''
        if original_values:
            sample = str(original_values[0])
            match = re.match(r'^([A-Z]{2,4})', sample)
            if match:
                prefix = match.group(1)
        
        if prefix:
            return [f"{prefix}{last_id + i + 1}" for i in range(num_rows)]
        else:
            return [str(last_id + i + 1) for i in range(num_rows)]
    
    def _generate_random_id(self, col_name, original_values, num_rows):
        """Generate random IDs"""
        # Determine ID format from samples
        prefix = ''
        if original_values:
            sample = str(original_values[0])
            # Check for common patterns
            if re.match(r'^[A-Z]{2,4}', sample):
                prefix = re.match(r'^[A-Z]{2,4}', sample).group()
        
        if prefix:
            return [f"{prefix}{random.randint(10000, 99999)}" for _ in range(num_rows)]
        else:
            return [f"ID{random.randint(100000, 999999)}" for _ in range(num_rows)]
    
    def _generate_realistic_names(self, col_name, original_values, num_rows, params):
        """Generate realistic names based on patterns"""
        # Analyze name patterns from samples
        name_patterns = self._analyze_name_patterns(original_values)
        
        names = []
        for i in range(num_rows):
            if name_patterns['type'] == 'western_full':
                # Western: First Last
                first = self._generate_first_name(name_patterns)
                last = self._generate_last_name(name_patterns)
                names.append(f"{first} {last}")
            
            elif name_patterns['type'] == 'indian_full':
                # Indian: First Last
                first = self._generate_indian_first_name()
                last = self._generate_indian_last_name()
                names.append(f"{first} {last}")
            
            elif name_patterns['type'] == 'single':
                # Single name
                names.append(self._generate_single_name(name_patterns))
            
            else:
                # Mix of patterns
                if random.random() < 0.5:
                    first = self._generate_first_name(name_patterns)
                    last = self._generate_last_name(name_patterns)
                    names.append(f"{first} {last}")
                else:
                    names.append(self._generate_single_name(name_patterns))
        
        return names
    
    def _analyze_name_patterns(self, samples):
        """Analyze name patterns in samples"""
        if not samples:
            return {'type': 'mixed', 'has_middle': False}
        
        str_samples = [str(s) for s in samples[:5]]
        
        # Check patterns
        has_space = any(' ' in s for s in str_samples)
        has_dot = any('. ' in s for s in str_samples)
        has_comma = any(', ' in s for s in str_samples)
        
        if has_space:
            # Check if Western or Indian
            for sample in str_samples:
                if ' ' in sample:
                    parts = sample.split()
                    if len(parts) == 2:
                        # Check if Indian name pattern
                        if any(part.lower() in ['singh', 'kumar', 'patel', 'sharma'] for part in parts):
                            return {'type': 'indian_full', 'has_middle': False}
                        else:
                            return {'type': 'western_full', 'has_middle': False}
                    elif len(parts) == 3:
                        return {'type': 'western_full', 'has_middle': True}
        
        return {'type': 'mixed', 'has_middle': False}
    
    def _generate_first_name(self, patterns):
        """Generate first name"""
        # Common first names across cultures
        first_names = [
            # Western
            'John', 'James', 'Michael', 'David', 'Robert', 'William', 'Richard', 'Joseph',
            'Thomas', 'Charles', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth',
            'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy',
            # Indian
            'Rahul', 'Amit', 'Raj', 'Sanjay', 'Vikram', 'Arjun', 'Deepak', 'Karan',
            'Priya', 'Neha', 'Anjali', 'Sneha', 'Pooja', 'Divya', 'Meera',
            # Global
            'Carlos', 'Juan', 'Miguel', 'Mohammed', 'Ali', 'Chen', 'Wei', 'Hiroshi'
        ]
        
        return random.choice(first_names)
    
    def _generate_last_name(self, patterns):
        """Generate last name"""
        # Common last names across cultures
        last_names = [
            # Western
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Miller', 'Davis', 'Garcia',
            'Rodriguez', 'Wilson', 'Martinez', 'Anderson', 'Taylor', 'Thomas',
            # Indian
            'Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma', 'Reddy', 'Iyer',
            # Global
            'Garcia', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
            'Wang', 'Li', 'Zhang', 'Liu', 'Chen', 'Kim', 'Park', 'Nguyen'
        ]
        
        return random.choice(last_names)
    
    def _generate_indian_first_name(self):
        """Generate Indian first name"""
        indian_first = ['Rahul', 'Amit', 'Raj', 'Sanjay', 'Vikram', 'Arjun', 'Deepak',
                       'Karan', 'Vivek', 'Ankit', 'Priya', 'Neha', 'Anjali', 'Sneha',
                       'Pooja', 'Divya', 'Meera', 'Tanya', 'Riya', 'Swati']
        return random.choice(indian_first)
    
    def _generate_indian_last_name(self):
        """Generate Indian last name"""
        indian_last = ['Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma', 'Reddy',
                      'Iyer', 'Choudhary', 'Joshi', 'Desai', 'Mehta', 'Nair', 'Menon']
        return random.choice(indian_last)
    
    def _generate_single_name(self, patterns):
        """Generate single name"""
        single_names = ['Alex', 'Sam', 'Jordan', 'Taylor', 'Casey', 'Morgan', 'Riley',
                       'Avery', 'Quinn', 'Blake', 'Dakota', 'Emerson', 'Finley']
        return random.choice(single_names)
    
    def _generate_realistic_emails(self, col_name, original_values, num_rows, params):
        """Generate realistic emails"""
        emails = []
        
        # Email patterns from params or default
        pattern_variety = params.get('pattern_variety', ['name_dot_name', 'initial_lastname', 'random_word'])
        domain_types = params.get('domain_types', ['personal', 'business'])
        
        for i in range(num_rows):
            # Choose pattern
            pattern = random.choice(pattern_variety)
            
            if pattern == 'name_dot_name':
                # john.smith123@gmail.com
                first = random.choice(['john', 'jane', 'alex', 'sam', 'mike', 'sara'])
                last = random.choice(['smith', 'johnson', 'williams', 'brown', 'jones'])
                number = random.randint(1, 99) if random.random() < 0.5 else ''
                username = f"{first}.{last}{number}"
            
            elif pattern == 'initial_lastname':
                # jsmith45@gmail.com
                first_initial = random.choice('abcdefghijklmnopqrstuvwxyz')
                last = random.choice(['smith', 'johnson', 'williams', 'brown'])
                number = random.randint(1, 99) if random.random() < 0.5 else ''
                username = f"{first_initial}{last}{number}"
            
            elif pattern == 'random_word':
                # coolguy123@gmail.com
                adjectives = ['cool', 'happy', 'smart', 'fast', 'quiet', 'brave']
                nouns = ['guy', 'girl', 'coder', 'writer', 'runner', 'thinker']
                word = random.choice(adjectives) + random.choice(nouns)
                number = random.randint(1, 999) if random.random() < 0.5 else ''
                username = f"{word}{number}"
            
            else:
                # Simple username
                username = f"user{random.randint(1000, 9999)}"
            
            # Choose domain
            domain_type = random.choice(domain_types)
            
            if domain_type == 'personal':
                domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com'])
            elif domain_type == 'business':
                domain = random.choice(['company.com', 'corp.com', 'business.com', 'enterprise.com'])
            elif domain_type == 'country_specific':
                domain = random.choice(['co.in', 'co.uk', 'com.au', 'de', 'fr'])
            else:
                domain = random.choice(['gmail.com', 'yahoo.com'])
            
            emails.append(f"{username}@{domain}".lower())
        
        return emails
    
    def _generate_realistic_dates(self, col_name, original_values, num_rows, params):
        """Generate realistic dates"""
        dates = []
        
        # Determine format from samples
        format_str = '%Y-%m-%d'  # default
        
        if original_values:
            sample = str(original_values[0])
            if '/' in sample:
                if len(sample.split('/')[0]) == 4:
                    format_str = '%Y/%m/%d'
                else:
                    format_str = '%d/%m/%Y'
            elif '-' in sample and len(sample.split('-')[0]) == 2:
                format_str = '%d-%m-%Y'
            elif '.' in sample:
                format_str = '%d.%m.%Y'
        
        # Generate dates
        end_date = datetime.now()
        
        for i in range(num_rows):
            # Mix of recent and older dates
            if random.random() < 0.7:  # 70% recent (last 90 days)
                days_ago = random.randint(1, 90)
            else:  # 30% older
                days_ago = random.randint(91, 365*2)
            
            date = end_date - timedelta(days=days_ago)
            dates.append(date.strftime(format_str))
        
        return dates
    
    def _generate_numeric_data(self, col_name, original_values, num_rows, params, method):
        """Generate numeric data"""
        # Convert to numeric
        numeric_vals = []
        for val in original_values:
            try:
                if val is not None:
                    clean = str(val).replace('$', '').replace(',', '').strip()
                    num = float(clean)
                    numeric_vals.append(num)
            except:
                continue
        
        if not numeric_vals:
            # Default ranges based on column name
            if 'age' in col_name.lower():
                return [random.randint(18, 65) for _ in range(num_rows)]
            elif any(x in col_name.lower() for x in ['price', 'cost', 'amount']):
                # Realistic prices
                prices = []
                for _ in range(num_rows):
                    base = random.uniform(10, 1000)
                    if random.random() < 0.3:
                        price = math.floor(base) + 0.99
                    elif random.random() < 0.5:
                        price = round(base, 2)
                    else:
                        price = round(base)
                    prices.append(float(price))
                return prices
            elif 'quantity' in col_name.lower():
                return [random.randint(1, 10) for _ in range(num_rows)]
            else:
                return [random.randint(1, 100) for _ in range(num_rows)]
        
        # Calculate statistics
        min_val = min(numeric_vals)
        max_val = max(numeric_vals)
        mean_val = np.mean(numeric_vals)
        std_val = np.std(numeric_vals) if len(numeric_vals) > 1 else (max_val - min_val) / 4
        
        # Generate based on method
        if method == 'numeric_range':
            # Uniform distribution
            data = np.random.uniform(min_val, max_val, num_rows)
        else:  # numeric_distribution
            # Normal distribution
            data = np.random.normal(mean_val, std_val, num_rows)
            data = np.clip(data, min_val * 0.8, max_val * 1.2)
        
        # Check if original values were integers
        try:
            if all(v.is_integer() for v in numeric_vals if isinstance(v, float)):
                return [int(round(x)) for x in data]
        except:
            pass
        
        # Apply decimal places based on original
        decimal_places = 0
        for val in numeric_vals[:10]:
            if '.' in str(val):
                dec_part = str(val).split('.')[1]
                decimal_places = max(decimal_places, len(dec_part))
        
        return [float(round(x, decimal_places)) for x in data]
    
    def _generate_categories(self, col_name, original_values, num_rows, params):
        """Generate categorical data"""
        # Get unique values from original
        unique_vals = list(set(str(v) for v in original_values if v is not None))
        
        if not unique_vals:
            # Default categories based on column name
            if 'status' in col_name.lower():
                unique_vals = ['Active', 'Pending', 'Completed', 'Cancelled']
            elif 'type' in col_name.lower():
                unique_vals = ['Type A', 'Type B', 'Type C', 'Standard', 'Premium']
            elif 'country' in col_name.lower():
                unique_vals = ['USA', 'India', 'UK', 'Canada', 'Australia']
            else:
                unique_vals = ['Category 1', 'Category 2', 'Category 3', 'Other']
        
        # Generate with some distribution (not completely random)
        if len(unique_vals) <= 4:
            # Weight towards first few values
            weights = [0.4, 0.3, 0.2, 0.1][:len(unique_vals)]
            weights = [w/sum(weights) for w in weights]
            return random.choices(unique_vals, weights=weights, k=num_rows)
        else:
            # More uniform for many categories
            return random.choices(unique_vals, k=num_rows)
    
    def _generate_text_data(self, col_name, original_values, num_rows, params):
        """Generate text data"""
        texts = []
        
        # Analyze original text patterns
        if original_values:
            sample = str(original_values[0])
            # Check if it looks like product names
            if any(x in col_name.lower() for x in ['product', 'item', 'goods']) or \
               (len(sample.split()) >= 2 and any(c.isupper() for c in sample)):
                # Generate product-like names
                for i in range(num_rows):
                    adjectives = ['Smart', 'Pro', 'Ultra', 'Power', 'Fast', 'Premium']
                    nouns = ['Phone', 'Laptop', 'Tablet', 'Watch', 'Headphones', 'Speaker']
                    versions = ['Pro', 'Plus', 'Max', '2024', 'X', 'SE']
                    
                    if random.random() < 0.6:
                        text = f"{random.choice(adjectives)} {random.choice(nouns)} {random.choice(versions)}"
                    else:
                        text = f"{random.choice(nouns)} {random.randint(10, 99)}{random.choice(['GB', 'TB', 'inch'])}"
                    
                    texts.append(text)
                return texts
        
        # Generic text generation
        word_pool = ['Data', 'Record', 'Entry', 'Item', 'Element', 'Component',
                    'Module', 'Unit', 'System', 'Service', 'Product', 'Asset']
        
        for i in range(num_rows):
            num_words = random.randint(2, 4)
            words = random.choices(word_pool, k=num_words)
            text = ' '.join(words)
            
            if random.random() < 0.3:
                text += f" {random.randint(1, 100)}"
            
            texts.append(text)
        
        return texts
    
    def _generate_pattern_based(self, col_name, original_values, num_rows, params):
        """Generate data based on observed patterns"""
        if not original_values:
            return [f"Value_{i}" for i in range(num_rows)]
        
        # Analyze patterns in first sample
        sample = str(original_values[0])
        
        # Check for common patterns
        if '@' in sample:
            # Email-like
            return self._generate_realistic_emails(col_name, original_values, num_rows, {})
        
        elif re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', sample):
            # Date-like
            return self._generate_realistic_dates(col_name, original_values, num_rows, {})
        
        elif ' ' in sample and sample[0].isupper():
            # Name-like
            return self._generate_realistic_names(col_name, original_values, num_rows, {})
        
        else:
            # Try to use original values
            unique_vals = list(set(str(v) for v in original_values if v is not None))
            if unique_vals and len(unique_vals) <= 20:
                # Use existing values with variations
                results = []
                for _ in range(num_rows):
                    base = random.choice(unique_vals)
                    if random.random() < 0.3:
                        results.append(f"{base}_{random.randint(1, 100)}")
                    else:
                        results.append(base)
                return results
            else:
                # Generate similar patterns
                return [f"{col_name}_{i}" for i in range(num_rows)]
    
    def _generate_statistical(self, original_series, num_rows):
        """Statistical fallback generation"""
        values = original_series.dropna().tolist()
        
        if not values:
            return [f"Data_{i}" for i in range(num_rows)]
        
        # Check if numeric
        try:
            numeric_vals = []
            for val in values:
                clean = str(val).replace('$', '').replace(',', '').strip()
                num = float(clean)
                numeric_vals.append(num)
            
            if numeric_vals:
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
                mean_val = np.mean(numeric_vals)
                std_val = np.std(numeric_vals) if len(numeric_vals) > 1 else (max_val - min_val) / 4
                
                data = np.random.normal(mean_val, std_val, num_rows)
                data = np.clip(data, min_val * 0.8, max_val * 1.2)
                
                # Check if integers
                if all(v.is_integer() for v in numeric_vals if isinstance(v, float)):
                    return [int(round(x)) for x in data]
                else:
                    return [float(round(x, 2)) for x in data]
        except:
            pass
        
        # Text data - use unique values
        unique_vals = list(set(str(v) for v in values))
        if len(unique_vals) <= 20:
            return random.choices(unique_vals, k=num_rows)
        else:
            return [f"Item_{i}" for i in range(num_rows)]
    
    def _apply_relationships(self, df, relationships):
        """Apply relationships between columns"""
        for rel in relationships:
            cols = rel.get('columns', [])
            if len(cols) >= 2 and all(col in df.columns for col in cols):
                # Simple relationship: make col2 reference col1 values
                col1, col2 = cols[0], cols[1]
                unique_vals = df[col1].unique()
                df[col2] = np.random.choice(unique_vals, len(df))
        
        return df
    
    def _apply_global_rules(self, df, global_rules):
        """Apply global rules"""
        # Apply basic data quality rules
        for col in df.columns:
            # Ensure IDs are strings
            if 'id' in col.lower() and df[col].dtype != object:
                df[col] = df[col].astype(str)
            
            # Ensure emails have @
            if 'email' in col.lower():
                df[col] = df[col].apply(lambda x: x if '@' in str(x) else f"{x}@example.com")
        
        return df
    
    def _statistical_fallback(self, df, num_rows):
        """Fallback to statistical generation"""
        generated = {}
        for col in df.columns:
            generated[col] = self._generate_statistical(df[col], num_rows)
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
    st.title("🌍 Universal LLM-Powered Data Generator")
    st.markdown("**Analyzes ANY dataset • Creates custom logic • Generates realistic data**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize
    if 'generator' not in st.session_state:
        st.session_state.generator = UniversalLLMGenerator()
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload ANY Dataset (CSV)", type=['csv'])
    
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
                st.subheader("⚙️ Generate Universal Data")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=5000, 
                                             value=1000)
                
                with col2:
                    if st.button("🚀 Generate with LLM Logic", type="primary"):
                        if not st.session_state.generator.available:
                            st.error("LLM not available. Check API key.")
                        else:
                            with st.spinner("LLM is analyzing patterns and generating..."):
                                generated = st.session_state.generator.generate_universal_data(df, int(num_rows))
                                st.session_state.generated_data = generated
                                st.success(f"✅ Generated {len(generated)} realistic rows!")
                                st.balloons()
                
                # Show LLM analysis if available
                if 'generation_logic' in st.session_state:
                    with st.expander("🧠 LLM Analysis & Logic", expanded=False):
                        logic = st.session_state.generation_logic
                        
                        if 'dataset_analysis' in logic:
                            st.write(f"**Dataset Analysis:** {logic['dataset_analysis']}")
                        
                        if 'generation_logic' in logic and 'columns' in logic['generation_logic']:
                            st.write("**Column Generation Logic:**")
                            for col, col_logic in logic['generation_logic']['columns'].items():
                                with st.expander(f"**{col}**", expanded=False):
                                    st.write(f"**Detected Type:** `{col_logic.get('detected_type', 'N/A')}`")
                                    st.write(f"**Patterns:** {col_logic.get('patterns_observed', 'N/A')}")
                                    st.write(f"**Generation Method:** `{col_logic.get('generation_method', 'N/A')}`")
                                    
                                    params = col_logic.get('generation_parameters', {})
                                    if params:
                                        st.write("**Parameters:**")
                                        st.json(params, expanded=False)
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📊 Generated Data")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Tabs
                    tab1, tab2, tab3 = st.tabs(["Preview", "Quality Analysis", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                        
                        # Show statistics
                        st.write("**Data Statistics:**")
                        stats_cols = st.columns(3)
                        with stats_cols[0]:
                            st.metric("Rows", len(df_gen))
                        with stats_cols[1]:
                            st.metric("Columns", len(df_gen.columns))
                        with stats_cols[2]:
                            memory_mb = df_gen.memory_usage(deep=True).sum() / 1024**2
                            st.metric("Memory", f"{memory_mb:.2f} MB")
                    
                    with tab2:
                        # Quality metrics
                        st.write("**Quality Assessment:**")
                        
                        issues = []
                        for col in df_gen.columns:
                            # Check uniqueness
                            unique_ratio = df_gen[col].nunique() / len(df_gen)
                            if unique_ratio < 0.1 and len(df_gen) > 100:
                                issues.append(f"Low variety in '{col}' ({unique_ratio:.1%} unique)")
                            
                            # Check for nonsense values
                            sample = str(df_gen[col].iloc[0]) if len(df_gen) > 0 else ""
                            if any(x in sample.lower() for x in ['temp_', 'placeholder', 'gen_', 'test_']):
                                issues.append(f"Placeholder values in '{col}'")
                        
                        if issues:
                            st.warning("**Issues Found:**")
                            for issue in issues:
                                st.write(f"- {issue}")
                        else:
                            st.success("✅ Data quality looks good!")
                        
                        # Show sample comparisons
                        st.write("**Sample Comparison:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Sample:**")
                            if len(df) > 0:
                                orig_sample = {}
                                for col in df.columns[:4]:
                                    orig_sample[col] = df[col].iloc[0] if pd.notna(df[col].iloc[0]) else 'N/A'
                                st.json(orig_sample)
                        
                        with col2:
                            st.write("**Generated Sample:**")
                            if len(df_gen) > 0:
                                gen_sample = {}
                                for col in df_gen.columns[:4]:
                                    gen_sample[col] = df_gen[col].iloc[0] if pd.notna(df_gen[col].iloc[0]) else 'N/A'
                                st.json(gen_sample)
                    
                    with tab3:
                        # Download
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "universal_generated_data.csv",
                            "text/csv"
                        )
                        
                        # Regenerate
                        if st.button("🔄 Generate New Dataset"):
                            st.session_state.generated_data = None
                            st.session_state.generation_logic = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
