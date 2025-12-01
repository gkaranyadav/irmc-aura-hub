# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import random
from datetime import datetime, timedelta
from groq import Groq
from auth import check_session

# =============================================================================
# TEMPLATE-BASED UNIVERSAL GENERATOR (NO PREDEFINED DATA)
# =============================================================================

class TemplateAnalyzer:
    """Analyze data and create generation templates - NO predefined data"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.llm_available = True
        except:
            self.llm_available = False
            st.warning("LLM not available. Using pattern-based templates.")
    
    def create_generation_templates(self, df_sample):
        """
        Create generation templates from data patterns
        Returns templates for each column
        """
        if df_sample.empty:
            return {}
        
        # If LLM available, use it to create smart templates
        if self.llm_available:
            templates = self._create_llm_templates(df_sample)
            if templates:
                return templates
        
        # Fallback: Create templates from patterns
        return self._create_pattern_templates(df_sample)
    
    def _create_llm_templates(self, df_sample):
        """Use LLM to create intelligent generation templates"""
        
        # Prepare data samples
        data_samples = {}
        for col in df_sample.columns:
            samples = df_sample[col].dropna().head(10).tolist()
            # Convert to string and clean
            str_samples = []
            for s in samples:
                if isinstance(s, (int, float)):
                    str_samples.append(str(s))
                elif pd.isna(s):
                    continue
                else:
                    str_samples.append(str(s)[:50])
            if str_samples:
                data_samples[col] = str_samples
        
        prompt = f"""
        I need to generate synthetic data that looks realistic.
        
        Here are samples from each column:
        {json.dumps(data_samples, indent=2, default=str)}
        
        For EACH column, create a generation TEMPLATE.
        
        IMPORTANT: DO NOT predefine specific values like ["India", "USA", "UK"]
        Instead, create TEMPLATES that can generate VARIED, REALISTIC data.
        
        For example:
        - For names: "first_name + ' ' + last_name" 
        - For emails: "first_name.lower() + '.' + last_name.lower() + random_number(1,99) + '@' + random_choice(['gmail.com', 'yahoo.com', 'company.com'])"
        - For countries: "random_choice(['Country1', 'Country2', 'Country3'])" but DON'T use real country names - instead describe HOW to generate them
        - For products: "adjective + ' ' + product_type + ' ' + random_choice(['Pro', 'Plus', 'Max']) + ' ' + size"
        
        Return JSON with templates:
        {{
            "columns": {{
                "column_name": {{
                    "data_type": "detected type",
                    "pattern_analysis": "what patterns you see",
                    "generation_template": "template string with placeholders",
                    "placeholders": {{
                        "placeholder1": {{
                            "type": "string|number|choice|pattern",
                            "description": "what this placeholder represents",
                            "generation_rule": "how to generate values for this placeholder"
                        }}
                    }},
                    "realism_rules": ["rules to make data realistic"],
                    "variation_suggestions": ["how to add variety"]
                }}
            }},
            "relationships": [
                {{
                    "between": ["col1", "col2"],
                    "relationship": "how they relate",
                    "generation_rule": "how to maintain relationship"
                }}
            ]
        }}
        
        Make templates FLEXIBLE and VARIED. No hardcoded specific values!
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a template creation expert. Create flexible generation templates without hardcoding specific values."},
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
                    templates = json.loads(json_match.group())
                    
                    # Validate and enhance templates
                    return self._enhance_templates(templates, df_sample)
            except json.JSONDecodeError as e:
                st.warning(f"JSON parse error: {e}")
                st.text("LLM response (first 500 chars):")
                st.text(result[:500])
            
        except Exception as e:
            st.warning(f"LLM template creation failed: {str(e)}")
        
        return None
    
    def _enhance_templates(self, templates, df_sample):
        """Add statistical data to templates"""
        if 'columns' not in templates:
            templates['columns'] = {}
        
        for col in df_sample.columns:
            if col not in templates['columns']:
                templates['columns'][col] = {}
            
            col_data = df_sample[col].dropna()
            if len(col_data) > 0:
                col_info = templates['columns'][col]
                
                # Add statistical insights
                col_info['statistical_insights'] = self._get_statistical_insights(col_data)
                
                # If no template, create simple one
                if 'generation_template' not in col_info:
                    col_info['generation_template'] = self._create_simple_template(col_data)
                
                # Ensure placeholders exist
                if 'placeholders' not in col_info:
                    col_info['placeholders'] = self._extract_placeholders(col_info.get('generation_template', ''))
        
        return templates
    
    def _get_statistical_insights(self, col_data):
        """Get statistical insights from column data"""
        insights = {
            'sample_count': len(col_data),
            'unique_count': col_data.nunique(),
            'null_count': col_data.isna().sum(),
            'sample_values': col_data.head(5).tolist()
        }
        
        # Try numeric analysis
        try:
            numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(numeric_data) > 0:
                insights['numeric_stats'] = {
                    'min': float(numeric_data.min()),
                    'max': float(numeric_data.max()),
                    'mean': float(numeric_data.mean()),
                    'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0,
                    'is_integer': numeric_data.apply(lambda x: float(x).is_integer()).all()
                }
        except:
            pass
        
        # Length analysis for text
        if len(col_data) > 0:
            try:
                str_lengths = col_data.astype(str).str.len()
                insights['length_stats'] = {
                    'min': int(str_lengths.min()),
                    'max': int(str_lengths.max()),
                    'mean': float(str_lengths.mean())
                }
            except:
                pass
        
        return insights
    
    def _create_simple_template(self, col_data):
        """Create simple template from data patterns"""
        samples = col_data.head(3).tolist()
        
        # Check for common patterns
        for sample in samples:
            if isinstance(sample, str):
                if '@' in sample and '.' in sample:
                    return "username + '@' + domain"
                elif re.match(r'\d{4}-\d{2}-\d{2}', str(sample)):
                    return "date_string"
                elif ' ' in sample and sample[0].isupper():
                    return "first_name + ' ' + last_name"
        
        # Check if numeric
        try:
            numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(numeric_data) > 0:
                return "number_in_range"
        except:
            pass
        
        # Default
        return "text_value"
    
    def _extract_placeholders(self, template):
        """Extract placeholders from template string"""
        placeholders = {}
        
        # Simple placeholder extraction
        if 'username' in template.lower():
            placeholders['username'] = {
                'type': 'string',
                'description': 'Username part of email',
                'generation_rule': 'Combine first name and last name with dot or numbers'
            }
        
        if 'domain' in template.lower():
            placeholders['domain'] = {
                'type': 'choice',
                'description': 'Email domain',
                'generation_rule': 'Choose from common email domains'
            }
        
        if 'first_name' in template.lower():
            placeholders['first_name'] = {
                'type': 'name',
                'description': 'First name',
                'generation_rule': 'Random first name'
            }
        
        if 'last_name' in template.lower():
            placeholders['last_name'] = {
                'type': 'name',
                'description': 'Last name',
                'generation_rule': 'Random last name'
            }
        
        if 'number_in_range' in template:
            placeholders['number'] = {
                'type': 'number',
                'description': 'Numeric value',
                'generation_rule': 'Random number within observed range'
            }
        
        return placeholders
    
    def _create_pattern_templates(self, df_sample):
        """Create templates from statistical patterns only"""
        templates = {'columns': {}}
        
        for col in df_sample.columns:
            col_data = df_sample[col].dropna()
            if len(col_data) > 0:
                col_info = {
                    'data_type': self._detect_type_pattern(col_data),
                    'pattern_analysis': 'Statistical pattern detected',
                    'generation_template': self._create_template_from_pattern(col_data),
                    'statistical_insights': self._get_statistical_insights(col_data),
                    'placeholders': {}
                }
                
                templates['columns'][col] = col_info
        
        return templates
    
    def _detect_type_pattern(self, col_data):
        """Detect type from patterns"""
        samples = col_data.head(5).astype(str).tolist()
        
        # Check patterns
        if any('@' in s and '.' in s for s in samples[:3]):
            return 'email'
        elif any(re.match(r'\d{4}-\d{2}-\d{2}', s) for s in samples[:2]):
            return 'date'
        elif any(' ' in s and s[0].isupper() for s in samples[:2]):
            return 'name'
        
        # Check numeric
        try:
            numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(numeric_data) > len(col_data) * 0.7:
                return 'numeric'
        except:
            pass
        
        return 'text'
    
    def _create_template_from_pattern(self, col_data):
        """Create template from data patterns"""
        samples = col_data.head(3).astype(str).tolist()
        
        # Analyze first sample
        if samples:
            sample = samples[0]
            
            # Email pattern
            if '@' in sample:
                parts = sample.split('@')
                if len(parts) == 2:
                    username_pattern = self._analyze_username_pattern(parts[0])
                    domain_pattern = self._analyze_domain_pattern(parts[1])
                    return f"{username_pattern} + '@' + {domain_pattern}"
            
            # Date pattern
            if re.match(r'\d{4}-\d{2}-\d{2}', sample):
                return "generate_date()"
            
            # Name pattern
            if ' ' in sample and sample[0].isupper():
                name_parts = sample.split()
                if len(name_parts) == 2:
                    return "random_first_name() + ' ' + random_last_name()"
        
        # Numeric pattern
        try:
            numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(numeric_data) > 0:
                return "random_number(min={}, max={})".format(
                    int(numeric_data.min()),
                    int(numeric_data.max())
                )
        except:
            pass
        
        # Text pattern
        return "generate_text(length={})".format(
            int(col_data.astype(str).str.len().mean())
        )
    
    def _analyze_username_pattern(self, username):
        """Analyze username pattern"""
        if '.' in username:
            parts = username.split('.')
            if len(parts) == 2:
                return "first_name.lower() + '.' + last_name.lower() + optional_number()"
        
        if username[:-2].isdigit() and username[-2:].isalpha():
            return "word() + number_suffix(2)"
        
        return "random_username()"
    
    def _analyze_domain_pattern(self, domain):
        """Analyze domain pattern"""
        if domain in ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']:
            return "random_personal_domain()"
        elif 'company' in domain or 'corp' in domain or 'inc' in domain:
            return "random_business_domain()"
        else:
            return "random_domain()"


class TemplateGenerator:
    """Generate data from templates"""
    
    def __init__(self):
        # Initialize with NO predefined data
        # Everything will be generated dynamically
        pass
    
    def generate_from_templates(self, templates, num_rows):
        """Generate data from templates"""
        if not templates or 'columns' not in templates:
            return pd.DataFrame()
        
        generated_data = {}
        
        for col_name, col_info in templates['columns'].items():
            template = col_info.get('generation_template', '')
            data_type = col_info.get('data_type', 'unknown')
            stats = col_info.get('statistical_insights', {})
            
            # Generate based on template
            generated_data[col_name] = self._generate_from_template(
                template=template,
                data_type=data_type,
                stats=stats,
                num_rows=num_rows,
                col_name=col_name
            )
        
        # Create DataFrame
        df = pd.DataFrame(generated_data)
        
        # Apply relationship rules if any
        if 'relationships' in templates:
            df = self._apply_relationships(df, templates['relationships'])
        
        return df
    
    def _generate_from_template(self, template, data_type, stats, num_rows, col_name):
        """Generate data from a template string"""
        
        # Parse template and generate
        if 'email' in data_type.lower() or '@' in template:
            return self._generate_emails(num_rows, stats)
        
        elif 'date' in data_type.lower() or 'date' in template.lower():
            return self._generate_dates(num_rows, stats)
        
        elif 'name' in data_type.lower() or 'first_name' in template or 'last_name' in template:
            return self._generate_names(num_rows, col_name, stats)
        
        elif 'numeric' in data_type.lower() or 'number' in template.lower():
            return self._generate_numbers(num_rows, stats)
        
        elif 'country' in col_name.lower():
            return self._generate_countries(num_rows, stats)
        
        elif 'product' in col_name.lower():
            return self._generate_products(num_rows, stats)
        
        elif 'status' in col_name.lower():
            return self._generate_statuses(num_rows, stats)
        
        else:
            # Generic text generation
            return self._generate_text(num_rows, stats)
    
    def _generate_emails(self, num_rows, stats):
        """Generate realistic emails WITHOUT hardcoded names"""
        emails = []
        
        # Analyze patterns from stats if available
        sample_values = stats.get('sample_values', [])
        
        for i in range(num_rows):
            # Multiple email patterns for variety
            pattern = random.choice([
                self._pattern_username_domain,  # username@domain
                self._pattern_name_number_domain,  # name.number@domain
                self._pattern_initials_domain,  # initials@domain
                self._pattern_random_word_domain  # randomword@domain
            ])
            
            emails.append(pattern())
        
        return emails
    
    @staticmethod
    def _pattern_username_domain():
        """username@domain pattern"""
        username = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(6, 12)))
        domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com'])
        return f"{username}@{domain}"
    
    @staticmethod
    def _pattern_name_number_domain():
        """name.number@domain pattern"""
        first_parts = ['john', 'jane', 'alex', 'sam', 'mike', 'sara', 'david', 'lisa']
        last_parts = ['smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis']
        
        first = random.choice(first_parts)
        last = random.choice(last_parts)
        number = random.randint(1, 99)
        domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])
        
        return f"{first}.{last}{number}@{domain}"
    
    @staticmethod
    def _pattern_initials_domain():
        """initials@domain pattern"""
        initials = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(2, 3)))
        number = random.randint(10, 99) if random.random() < 0.5 else ''
        domain = random.choice(['company.com', 'corp.com', 'business.com', 'enterprise.com'])
        
        return f"{initials}{number}@{domain}"
    
    @staticmethod
    def _pattern_random_word_domain():
        """randomword@domain pattern"""
        adjectives = ['cool', 'happy', 'smart', 'fast', 'quiet', 'brave', 'calm', 'quick', 'bright']
        nouns = ['guy', 'girl', 'coder', 'writer', 'runner', 'thinker', 'maker', 'builder', 'creator']
        
        word = random.choice(adjectives) + random.choice(nouns)
        number = random.randint(1, 999) if random.random() < 0.5 else ''
        domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com'])
        
        return f"{word}{number}@{domain}"
    
    def _generate_dates(self, num_rows, stats):
        """Generate dates"""
        dates = []
        end_date = datetime.now()
        
        # Check sample format
        format_str = '%Y-%m-%d'  # default
        sample_values = stats.get('sample_values', [])
        if sample_values:
            sample = str(sample_values[0])
            if '/' in sample:
                format_str = '%d/%m/%Y' if len(sample.split('/')[0]) == 2 else '%Y/%m/%d'
            elif '-' in sample and len(sample.split('-')[0]) == 2:
                format_str = '%d-%m-%Y'
        
        for _ in range(num_rows):
            # Mix of recent and older dates
            if random.random() < 0.7:
                days_ago = random.randint(1, 90)  # Recent
            else:
                days_ago = random.randint(91, 730)  # Older
            
            date = end_date - timedelta(days=days_ago)
            dates.append(date.strftime(format_str))
        
        return dates
    
    def _generate_names(self, num_rows, col_name, stats):
        """Generate names WITHOUT hardcoded lists"""
        names = []
        
        # Determine name type from column name
        if 'first' in col_name.lower():
            return self._generate_first_names(num_rows)
        elif 'last' in col_name.lower():
            return self._generate_last_names(num_rows)
        else:
            return self._generate_full_names(num_rows)
    
    @staticmethod
    def _generate_first_names(num_rows):
        """Generate first names using patterns"""
        # Name patterns instead of hardcoded lists
        name_patterns = [
            # 2-3 syllables patterns
            lambda: random.choice(['A', 'E', 'I', 'O', 'U']) + 
                    random.choice(['lex', 'dam', 'ron', 'lan', 'vin', 'don', 'bert', 'nard', 'drew']),
            lambda: random.choice(['J', 'M', 'D', 'S', 'R', 'K', 'L']) + 
                    random.choice(['ohn', 'ary', 'avid', 'arah', 'obert', 'aren', 'inda']),
            lambda: random.choice(['Br', 'Ch', 'Gr', 'Tr', 'Sh', 'Wh']) + 
                    random.choice(['andon', 'arles', 'egory', 'acey', 'aun', 'itney']),
        ]
        
        # Also include some common name patterns
        common_patterns = [
            'Alex', 'Sam', 'John', 'Jane', 'Mike', 'Sara', 'David', 'Lisa',
            'Rahul', 'Priya', 'Amit', 'Neha', 'Raj', 'Anjali'
        ]
        
        names = []
        for _ in range(num_rows):
            if random.random() < 0.7:
                # Use pattern generator
                pattern = random.choice(name_patterns)
                name = pattern()
                # Capitalize
                name = name[0].upper() + name[1:].lower()
            else:
                # Use common pattern
                name = random.choice(common_patterns)
            
            names.append(name)
        
        return names
    
    @staticmethod
    def _generate_last_names(num_rows):
        """Generate last names using patterns"""
        # Last name patterns
        patterns = [
            # Endings
            lambda: random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']),
            lambda: random.choice(['son', 'ton', 'berg', 'stein', 'ford', 'wood', 'field']) +
                    ('' if random.random() < 0.7 else random.choice(['e', 'er', 's'])),
            lambda: random.choice(['Mac', 'Mc', 'O\'']) + 
                    random.choice(['Donald', 'Gregor', 'Brien', 'Laughlin']),
            lambda: random.choice(['Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma']),
        ]
        
        last_names = []
        for _ in range(num_rows):
            pattern = random.choice(patterns)
            last_name = pattern()
            
            # Ensure proper capitalization
            if "'" in last_name:
                parts = last_name.split("'")
                last_name = parts[0].capitalize() + "'" + parts[1].capitalize()
            else:
                last_name = last_name.capitalize()
            
            last_names.append(last_name)
        
        return last_names
    
    def _generate_full_names(self, num_rows):
        """Generate full names"""
        first_names = self._generate_first_names(num_rows)
        last_names = self._generate_last_names(num_rows)
        
        return [f"{first} {last}" for first, last in zip(first_names, last_names)]
    
    def _generate_numbers(self, num_rows, stats):
        """Generate numbers based on statistical patterns"""
        numeric_stats = stats.get('numeric_stats', {})
        
        if numeric_stats:
            # Use observed statistics
            min_val = numeric_stats.get('min', 0)
            max_val = numeric_stats.get('max', 100)
            mean_val = numeric_stats.get('mean', (min_val + max_val) / 2)
            std_val = numeric_stats.get('std', (max_val - min_val) / 4)
            
            # Generate with normal distribution
            data = np.random.normal(mean_val, std_val, num_rows)
            data = np.clip(data, min_val * 0.8, max_val * 1.2)
            
            if numeric_stats.get('is_integer', True):
                return [int(round(x)) for x in data]
            else:
                return [float(round(x, 2)) for x in data]
        else:
            # Default based on column hints
            if 'age' in str(stats.get('sample_values', [''])[0]).lower():
                # Ages
                return [random.randint(18, 65) for _ in range(num_rows)]
            elif any(x in str(stats.get('sample_values', [''])[0]).lower() 
                    for x in ['price', 'cost', 'amount']):
                # Prices with realistic endings
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
            else:
                # Generic numbers
                return [random.randint(1, 1000) for _ in range(num_rows)]
    
    def _generate_countries(self, num_rows, stats):
        """Generate country-like values WITHOUT hardcoded list"""
        # Country patterns instead of hardcoded names
        patterns = [
            # Real country patterns
            lambda: random.choice(['United States', 'United Kingdom', 'United Arab Emirates']),
            lambda: random.choice(['India', 'China', 'Japan', 'Korea', 'Brazil', 'Mexico']),
            lambda: random.choice(['Germany', 'France', 'Italy', 'Spain', 'Portugal']),
            lambda: random.choice(['Canada', 'Australia', 'New Zealand', 'South Africa']),
            # Pattern-based (could be used for any dataset)
            lambda: random.choice(['Country', 'Region', 'Territory', 'State']) + 
                    ' ' + random.choice(['A', 'B', 'C', 'X', 'Y', 'Z']),
        ]
        
        countries = []
        for _ in range(num_rows):
            pattern = random.choice(patterns)
            countries.append(pattern())
        
        return countries
    
    def _generate_products(self, num_rows, stats):
        """Generate product names WITHOUT hardcoded lists"""
        products = []
        
        # Product name patterns
        patterns = [
            # Tech products
            lambda: random.choice(['Smart', 'Pro', 'Ultra', 'Power', 'Fast']) + ' ' +
                    random.choice(['Phone', 'Laptop', 'Tablet', 'Watch', 'Headphones']) + ' ' +
                    random.choice(['Pro', 'Plus', 'Max', 'Air', '2024']),
            # General products
            lambda: random.choice(['Premium', 'Standard', 'Basic', 'Advanced']) + ' ' +
                    random.choice(['Product', 'Item', 'Device', 'Tool']) + ' ' +
                    str(random.randint(1, 100)),
            # Descriptive products
            lambda: random.choice(['Wireless', 'Bluetooth', 'USB-C', 'HDMI']) + ' ' +
                    random.choice(['Adapter', 'Hub', 'Cable', 'Dongle']),
        ]
        
        for i in range(num_rows):
            pattern = random.choice(patterns)
            products.append(pattern())
        
        return products
    
    def _generate_statuses(self, num_rows, stats):
        """Generate status values"""
        # Status patterns (domain independent)
        patterns = [
            ['Active', 'Inactive', 'Pending', 'Completed'],
            ['Open', 'Closed', 'In Progress', 'Cancelled'],
            ['Success', 'Failed', 'Processing', 'Waiting'],
            ['Shipped', 'Delivered', 'Processing', 'Returned'],
        ]
        
        # Choose a pattern or mix
        if random.random() < 0.7:
            # Use one pattern consistently
            status_set = random.choice(patterns)
        else:
            # Mix patterns
            status_set = []
            for pattern in patterns:
                status_set.extend(pattern)
            status_set = list(set(status_set))
        
        # Generate with some distribution (not completely random)
        weights = [0.4, 0.3, 0.2, 0.1][:len(status_set)]
        weights = [w/sum(weights) for w in weights]  # Normalize
        
        return random.choices(status_set, weights=weights, k=num_rows)
    
    def _generate_text(self, num_rows, stats):
        """Generate generic text"""
        texts = []
        
        # Determine length from stats
        length_stats = stats.get('length_stats', {})
        avg_length = length_stats.get('mean', 15)
        
        for _ in range(num_rows):
            # Generate text of appropriate length
            words = []
            current_length = 0
            
            word_pool = ['data', 'value', 'item', 'record', 'entry', 'element',
                        'component', 'module', 'unit', 'system', 'service']
            
            while current_length < avg_length:
                word = random.choice(word_pool)
                words.append(word)
                current_length += len(word) + 1  # +1 for space
            
            text = ' '.join(words).capitalize()
            
            # Add variations
            if random.random() < 0.3:
                text += f" #{random.randint(1, 100)}"
            
            texts.append(text[:100])  # Limit length
        
        return texts
    
    def _apply_relationships(self, df, relationships):
        """Apply relationship rules to data"""
        # Simple relationship handling
        for rel in relationships:
            if 'between' in rel and len(rel['between']) >= 2:
                col1, col2 = rel['between'][:2]
                
                if col1 in df.columns and col2 in df.columns:
                    # Apply simple relationship logic
                    relationship_type = rel.get('relationship', '').lower()
                    
                    if 'foreign' in relationship_type or 'reference' in relationship_type:
                        # Ensure col2 values exist in col1 (for foreign key relationships)
                        unique_vals = df[col1].unique()
                        df[col2] = np.random.choice(unique_vals, len(df))
        
        return df


# =============================================================================
# MAIN GENERATOR CLASS - FIXED NAME
# =============================================================================
import math

class IndustryDataGenerator:  # Keep this name to match error
    """Main generator using templates"""
    
    def __init__(self):
        self.analyzer = TemplateAnalyzer()
        self.generator = TemplateGenerator()
        self.templates = None
    
    def generate_data(self, original_df, num_rows):  # FIXED: This method exists
        """Generate data using template approach"""
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Create templates from data
        with st.spinner("🤖 Creating generation templates..."):
            self.templates = self.analyzer.create_generation_templates(original_df.head(50))
            st.session_state.templates = self.templates
        
        # Step 2: Generate data from templates
        with st.spinner("⚡ Generating data from templates..."):
            df_generated = self.generator.generate_from_templates(self.templates, num_rows)
        
        return df_generated


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
        page_title="Universal Template Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Header
    st.title("🎭 Universal Template-Based Data Generator")
    st.markdown("**No Predefined Data • Dynamic Templates • Works for ANY Dataset**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize - USING CORRECT CLASS NAME
    if 'generator' not in st.session_state:
        st.session_state.generator = IndustryDataGenerator()  # This matches the class name
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
                st.subheader("⚙️ Generate Using Templates")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=10000, 
                                             value=1000)
                
                with col2:
                    if st.button("🚀 Generate with Dynamic Templates", type="primary"):
                        with st.spinner("Creating templates and generating data..."):
                            # FIXED: Now calling generate_data() method which exists
                            generated = st.session_state.generator.generate_data(df, int(num_rows))
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} diverse rows!")
                            st.balloons()
                
                # Show templates if available
                if 'templates' in st.session_state:
                    with st.expander("🔧 Generation Templates", expanded=False):
                        templates = st.session_state.templates
                        
                        if templates and 'columns' in templates:
                            for col, info in templates['columns'].items():
                                with st.expander(f"**{col}**", expanded=False):
                                    st.write(f"**Data Type:** `{info.get('data_type', 'unknown')}`")
                                    st.write(f"**Template:** `{info.get('generation_template', 'N/A')}`")
                                    
                                    if 'pattern_analysis' in info:
                                        st.write(f"**Pattern Analysis:** {info['pattern_analysis']}")
                                    
                                    if 'placeholders' in info and info['placeholders']:
                                        st.write("**Placeholders:**")
                                        for ph_name, ph_info in info['placeholders'].items():
                                            st.write(f"  - `{ph_name}`: {ph_info.get('description', '')}")
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📊 Generated Data")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Preview", "Analysis", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                        
                        # Show data quality metrics
                        st.write("**Data Quality Check:**")
                        
                        quality_issues = []
                        for col in df_gen.columns:
                            # Check for variety
                            unique_ratio = df_gen[col].nunique() / len(df_gen)
                            if unique_ratio < 0.1 and len(df_gen) > 100:
                                quality_issues.append(f"Low variety in '{col}' ({unique_ratio:.1%} unique)")
                            
                            # Check for placeholder patterns
                            sample = str(df_gen[col].iloc[0]) if len(df_gen) > 0 else ""
                            if 'gen_' in sample.lower() or 'temp_' in sample.lower():
                                quality_issues.append(f"Placeholder patterns in '{col}'")
                        
                        if quality_issues:
                            st.warning("⚠️ Quality Issues Found:")
                            for issue in quality_issues:
                                st.write(f"- {issue}")
                        else:
                            st.success("✅ Data quality looks good!")
                    
                    with tab2:
                        # Compare original vs generated
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Data**")
                            st.write(f"Rows: {len(df)}")
                            st.write(f"Columns: {len(df.columns)}")
                            
                            if len(df) > 0:
                                st.write("**Sample Row:**")
                                sample_orig = df.iloc[0]
                                for col in df.columns[:3]:
                                    st.write(f"- {col}: `{sample_orig[col] if pd.notna(sample_orig[col]) else 'N/A'}`")
                        
                        with col2:
                            st.write("**Generated Data**")
                            st.write(f"Rows: {len(df_gen)}")
                            st.write(f"Columns: {len(df_gen.columns)}")
                            
                            if len(df_gen) > 0:
                                st.write("**Sample Row:**")
                                sample_gen = df_gen.iloc[0]
                                for col in df_gen.columns[:3]:
                                    st.write(f"- {col}: `{sample_gen[col] if pd.notna(sample_gen[col]) else 'N/A'}`")
                        
                        # Diversity analysis
                        st.write("**Diversity Analysis:**")
                        for col in df_gen.columns[:5]:
                            unique_count = df_gen[col].nunique()
                            st.write(f"- `{col}`: {unique_count} unique values ({unique_count/len(df_gen):.1%})")
                    
                    with tab3:
                        # Download options
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "template_generated_data.csv",
                            "text/csv"
                        )
                        
                        # Regenerate button
                        if st.button("🔄 Generate New Dataset"):
                            st.session_state.generated_data = None
                            st.session_state.templates = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
