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
# ENHANCED INDUSTRY-GRADE GENERATOR
# =============================================================================

class EnhancedColumnClassifier:
    """Enhanced rule-based column classification"""
    
    @staticmethod
    def classify_column(col_name, sample_values, df_stats=None):
        """
        Enhanced classification with better pattern detection
        """
        col_name_lower = str(col_name).lower()
        sample_strs = [str(v).lower() for v in sample_values[:10] if pd.notna(v)]
        
        # Enhanced ID detection
        if any(x in col_name_lower for x in ['id', 'code', 'num', 'no', 'number', 'ref', 'key', 'serial']):
            if EnhancedColumnClassifier._looks_like_id(sample_strs):
                return {
                    'category': 'identifier',
                    'subcategory': 'sequential_id' if EnhancedColumnClassifier._is_sequential(sample_strs) else 'random_id',
                    'confidence': 0.95,
                    'format_hint': EnhancedColumnClassifier._get_id_format(sample_strs)
                }
        
        # Enhanced date/time detection
        if any(x in col_name_lower for x in ['date', 'time', 'day', 'month', 'year', 'timestamp', 'datetime']):
            date_format = EnhancedColumnClassifier._detect_date_format(sample_strs)
            if date_format:
                return {
                    'category': 'temporal',
                    'subcategory': 'date_time' if any(':' in s for s in sample_strs[:3]) else 'date',
                    'confidence': 0.9,
                    'format_hint': date_format
                }
        
        # Enhanced name detection
        name_type = EnhancedColumnClassifier._detect_name_type(col_name_lower, sample_strs)
        if name_type:
            return {
                'category': 'person',
                'subcategory': name_type,
                'confidence': 0.85,
                'format_hint': 'mixed_names' if any(any(c.isalpha() and not c.isascii() for c in s) for s in sample_strs[:3]) else 'western_names'
            }
        
        # Enhanced email detection
        if any(x in col_name_lower for x in ['email', 'mail', 'contact', 'e-mail']):
            if any('@' in s for s in sample_strs[:3]):
                return {
                    'category': 'contact',
                    'subcategory': 'email',
                    'confidence': 0.98,
                    'format_hint': EnhancedColumnClassifier._get_email_pattern(sample_strs)
                }
        
        # Enhanced location detection
        location_type = EnhancedColumnClassifier._detect_location_type(col_name_lower, sample_strs)
        if location_type:
            return {
                'category': 'location',
                'subcategory': location_type,
                'confidence': 0.8,
                'format_hint': 'country_name' if location_type == 'country' else 'city_name'
            }
        
        # Enhanced categorical detection
        if any(x in col_name_lower for x in ['status', 'type', 'category', 'class', 'level', 'stage', 'phase', 'state']):
            unique_count = len(set(sample_strs))
            if unique_count <= 15:
                return {
                    'category': 'categorical',
                    'subcategory': 'status' if 'status' in col_name_lower else 'category',
                    'confidence': 0.85,
                    'unique_count': unique_count,
                    'values': list(set(sample_strs))[:10]
                }
        
        # Enhanced numeric detection with better type inference
        numeric_stats = EnhancedColumnClassifier._analyze_numeric(sample_strs)
        if numeric_stats['is_numeric']:
            # Determine numeric type
            num_type = EnhancedColumnClassifier._determine_numeric_type(col_name_lower, numeric_stats)
            
            return {
                'category': 'numeric',
                'subcategory': num_type,
                'confidence': 0.9,
                'stats': numeric_stats,
                'format_hint': 'currency' if any(x in col_name_lower for x in ['price', 'cost', 'amount', 'value']) else 'plain'
            }
        
        # Text/description detection
        if any(x in col_name_lower for x in ['desc', 'note', 'comment', 'detail', 'text', 'remark', 'description', 'title', 'name']):
            return {
                'category': 'text',
                'subcategory': 'product_name' if any(len(s.split()) >= 2 and any(c.isupper() for c in s) for s in sample_strs[:3]) else 'description',
                'confidence': 0.75,
                'avg_length': np.mean([len(s) for s in sample_strs]) if sample_strs else 0
            }
        
        # Default
        return {
            'category': 'unknown',
            'subcategory': 'generic',
            'confidence': 0.5
        }
    
    @staticmethod
    def _looks_like_id(values):
        """Enhanced ID detection"""
        if not values:
            return False
        
        # Check patterns
        patterns = [
            r'^[A-Z]{2,4}\d{4,8}$',  # ABCD12345
            r'^\d{6,12}$',  # 123456789
            r'^[A-Z]{2}\d{6}$',  # AB123456
            r'^\d{3}-\d{3}-\d{4}$',  # 123-456-7890
            r'^[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}$'  # UUID
        ]
        
        for pattern in patterns:
            if any(re.match(pattern, v.upper()) for v in values[:3]):
                return True
        
        # Check sequential
        try:
            nums = []
            for v in values[:10]:
                num_part = re.sub(r'\D', '', v)
                if num_part:
                    nums.append(int(num_part))
            
            if len(nums) >= 3:
                diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                if len(set(diffs)) == 1:  # All diffs equal
                    return True
        except:
            pass
        
        return False
    
    @staticmethod
    def _is_sequential(values):
        """Check if values are sequential"""
        try:
            nums = []
            for v in values[:5]:
                num_part = re.sub(r'\D', '', v)
                if num_part:
                    nums.append(int(num_part))
            
            if len(nums) >= 3:
                return all(nums[i+1] == nums[i] + 1 for i in range(len(nums)-1))
        except:
            return False
        return False
    
    @staticmethod
    def _get_id_format(values):
        """Get ID format pattern"""
        if not values:
            return "unknown"
        
        sample = str(values[0]).upper()
        
        if re.match(r'^[A-Z]{2,4}\d{4,8}$', sample):
            return "prefix_numeric"
        elif re.match(r'^\d{6,12}$', sample):
            return "numeric_only"
        elif re.match(r'^\d{3}-\d{3}-\d{4}$', sample):
            return "dashed_numeric"
        elif '-' in sample and len(sample) > 20:
            return "uuid"
        else:
            return "mixed"
    
    @staticmethod
    def _detect_date_format(values):
        """Detect date format"""
        if not values:
            return None
        
        sample = str(values[0])
        
        # Common date formats
        formats = {
            r'\d{4}-\d{2}-\d{2}': '%Y-%m-%d',
            r'\d{2}/\d{2}/\d{4}': '%d/%m/%Y',
            r'\d{2}-\d{2}-\d{4}': '%d-%m-%Y',
            r'\d{4}/\d{2}/\d{2}': '%Y/%m/%d',
            r'\d{2}\.\d{2}\.\d{4}': '%d.%m.%Y',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}': '%Y-%m-%d %H:%M:%S'
        }
        
        for pattern, date_format in formats.items():
            if re.match(pattern, sample):
                return date_format
        
        return None
    
    @staticmethod
    def _detect_name_type(col_name, values):
        """Detect name type"""
        if not values:
            return None
        
        # Check column name hints
        if any(x in col_name for x in ['first', 'given', 'fore']):
            return 'first_name'
        elif any(x in col_name for x in ['last', 'surname', 'family']):
            return 'last_name'
        elif any(x in col_name for x in ['full', 'complete', 'customer', 'person']):
            return 'full_name'
        
        # Check value patterns
        sample = str(values[0])
        words = sample.split()
        
        if len(words) == 1:
            # Check if it's a first name pattern
            if sample[0].isupper() and sample.isalpha():
                return 'first_name' if len(sample) <= 12 else 'single_name'
        elif len(words) == 2:
            if all(w[0].isupper() for w in words if w):
                return 'full_name'
        
        return None
    
    @staticmethod
    def _get_email_pattern(values):
        """Analyze email patterns"""
        if not values:
            return "generic"
        
        samples = values[:3]
        
        # Check patterns
        if any('.' in s.split('@')[0] for s in samples if '@' in s):
            if any(re.match(r'^[a-z]+\.[a-z]+\d*@', s) for s in samples):
                return "first.last"
            elif any(re.match(r'^[a-z]+\d*@', s) for s in samples):
                return "username"
        
        return "mixed"
    
    @staticmethod
    def _detect_location_type(col_name, values):
        """Detect location type"""
        if not values:
            return None
        
        # Column name hints
        if 'country' in col_name:
            return 'country'
        elif 'city' in col_name:
            return 'city'
        elif any(x in col_name for x in ['state', 'province', 'region']):
            return 'state'
        elif any(x in col_name for x in ['address', 'location', 'place']):
            return 'address'
        
        # Value pattern hints
        sample = str(values[0]).upper()
        if sample in ['USA', 'INDIA', 'UK', 'CANADA', 'AUSTRALIA', 'GERMANY']:
            return 'country'
        elif len(sample) <= 3 and sample.isalpha():
            return 'country_code'
        
        return 'general'
    
    @staticmethod
    def _analyze_numeric(values):
        """Analyze numeric patterns"""
        stats = {
            'is_numeric': False,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'is_integer': False,
            'has_decimal': False,
            'decimal_places': 0,
            'has_currency': False
        }
        
        numeric_vals = []
        for v in values:
            try:
                # Clean value
                clean_v = str(v).replace('$', '').replace(',', '').replace('€', '').replace('£', '').replace('¥', '')
                num = float(clean_v)
                numeric_vals.append(num)
                
                # Check for currency
                if any(c in str(v) for c in ['$', '€', '£', '¥']):
                    stats['has_currency'] = True
            except:
                continue
        
        if len(numeric_vals) >= 3:
            stats['is_numeric'] = True
            stats['min'] = float(min(numeric_vals))
            stats['max'] = float(max(numeric_vals))
            stats['mean'] = float(np.mean(numeric_vals))
            stats['std'] = float(np.std(numeric_vals)) if len(numeric_vals) > 1 else 0
            
            # Check if all integers
            stats['is_integer'] = all(v.is_integer() for v in numeric_vals)
            
            # Check decimal places
            decimal_counts = []
            for v in numeric_vals:
                if '.' in str(v):
                    stats['has_decimal'] = True
                    decimal_part = str(v).split('.')[1]
                    decimal_counts.append(len(decimal_part))
            
            if decimal_counts:
                stats['decimal_places'] = int(np.mean(decimal_counts))
        
        return stats
    
    @staticmethod
    def _determine_numeric_type(col_name, stats):
        """Determine numeric type"""
        col_lower = col_name.lower()
        
        if any(x in col_lower for x in ['age', 'years', 'old']):
            return 'age'
        elif any(x in col_lower for x in ['price', 'cost', 'amount', 'value', 'salary', 'income', 'revenue', 'fee']):
            return 'monetary'
        elif any(x in col_lower for x in ['quantity', 'count', 'qty', 'total', 'sum', 'number', 'items']):
            return 'count'
        elif any(x in col_lower for x in ['score', 'rating', 'grade', 'percentage', 'rate', 'ratio', 'percent']):
            return 'measurement'
        elif any(x in col_lower for x in ['weight', 'height', 'length', 'width', 'depth', 'size', 'volume']):
            return 'dimension'
        elif any(x in col_lower for x in ['temperature', 'pressure', 'humidity', 'speed', 'velocity']):
            return 'sensor_reading'
        else:
            return 'general'


class EnhancedDomainDetector:
    """Enhanced domain detection with better inference"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
        except:
            self.available = False
    
    def detect_domain_with_context(self, column_names, sample_data, column_types):
        """Enhanced domain detection with context"""
        if not self.available:
            return self._enhanced_rule_based_domain(column_names, column_types)
        
        # Prepare context
        context = {
            "columns": column_names,
            "sample_data": sample_data,
            "column_types": column_types
        }
        
        prompt = f"""
        Analyze this dataset and determine its domain.
        
        Context:
        {json.dumps(context, indent=2, default=str)}
        
        DOMAIN OPTIONS:
        1. ecommerce - Online shopping, orders, products, customers, payments
        2. finance - Banking, transactions, accounts, investments, loans
        3. healthcare - Medical records, patients, treatments, hospitals, prescriptions
        4. human_resources - Employees, salaries, departments, performance, attendance
        5. education - Students, courses, grades, teachers, schools
        6. iot_sensors - Sensor data, devices, measurements, timestamps, readings
        7. logistics - Shipping, inventory, warehouses, delivery, tracking
        8. social_media - Users, posts, likes, comments, followers
        9. real_estate - Properties, listings, prices, locations, features
        10. manufacturing - Production, quality, equipment, maintenance, output
        11. telecom - Customers, calls, data usage, plans, devices
        12. marketing - Campaigns, leads, conversions, ads, metrics
        13. other - None of the above
        
        Return JSON with:
        {{
            "domain": "chosen_domain",
            "confidence": 0.0-1.0,
            "reason": "brief explanation",
            "suggestions": ["suggestions for realistic data generation"]
        }}
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            try:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
            
            # Fallback
            return {
                "domain": self._enhanced_rule_based_domain(column_names, column_types),
                "confidence": 0.7,
                "reason": "Rule-based detection",
                "suggestions": ["Generate data matching observed patterns"]
            }
            
        except:
            return {
                "domain": self._enhanced_rule_based_domain(column_names, column_types),
                "confidence": 0.6,
                "reason": "Fallback detection",
                "suggestions": ["Use statistical patterns from original data"]
            }
    
    def _enhanced_rule_based_domain(self, column_names, column_types):
        """Enhanced rule-based domain detection"""
        col_names_lower = [str(c).lower() for c in column_names]
        col_types = [t.get('category', '') for t in column_types.values()]
        
        # Domain patterns with weights
        patterns = {
            'ecommerce': {
                'keywords': ['order', 'product', 'customer', 'price', 'quantity', 'shipping', 'payment', 'cart', 'invoice'],
                'required_types': ['identifier', 'person', 'numeric', 'categorical'],
                'score': 0
            },
            'finance': {
                'keywords': ['account', 'transaction', 'balance', 'amount', 'credit', 'debit', 'bank', 'loan', 'interest'],
                'required_types': ['identifier', 'numeric', 'temporal'],
                'score': 0
            },
            'healthcare': {
                'keywords': ['patient', 'doctor', 'diagnosis', 'treatment', 'hospital', 'medical', 'prescription', 'test', 'result'],
                'required_types': ['person', 'categorical', 'temporal'],
                'score': 0
            },
            'human_resources': {
                'keywords': ['employee', 'salary', 'department', 'hire', 'position', 'manager', 'performance', 'review', 'attendance'],
                'required_types': ['person', 'numeric', 'categorical'],
                'score': 0
            }
        }
        
        # Calculate scores
        for domain, pattern in patterns.items():
            # Keyword matching
            for keyword in pattern['keywords']:
                if any(keyword in col for col in col_names_lower):
                    pattern['score'] += 2
            
            # Type matching
            for req_type in pattern['required_types']:
                if req_type in col_types:
                    pattern['score'] += 1
        
        # Find best domain
        if patterns:
            best_domain = max(patterns.items(), key=lambda x: x[1]['score'])
            if best_domain[1]['score'] > 3:
                return best_domain[0]
        
        return 'other'


class EnhancedSmartGenerator:
    """Enhanced generator with realistic data patterns"""
    
    # Enhanced domain data
    DOMAIN_DATA = {
        'ecommerce': {
            'countries': ['India', 'USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'China', 'Brazil'],
            'statuses': ['Shipped', 'Pending', 'Delivered', 'Cancelled', 'Processing', 'Returned', 'Refunded', 'On Hold'],
            'products': {
                'categories': ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty', 'Sports', 'Toys', 'Automotive'],
                'electronics': {
                    'types': ['Smartphone', 'Laptop', 'Tablet', 'Smartwatch', 'Headphones', 'Speaker', 'Monitor', 'Keyboard', 'Mouse'],
                    'brands': ['Apple', 'Samsung', 'Dell', 'HP', 'Lenovo', 'Sony', 'Bose', 'Logitech'],
                    'models': ['Pro', 'Plus', 'Max', 'Air', 'SE', 'X', 'Ultra', 'Lite', '2024', '2025'],
                    'specs': ['256GB', '512GB', '1TB', '16GB RAM', '32GB RAM', '8-core', '12-core', '4K', 'OLED']
                },
                'price_ranges': {
                    'Smartphone': (200, 1500),
                    'Laptop': (500, 3000),
                    'Tablet': (100, 1200),
                    'Smartwatch': (100, 800),
                    'Headphones': (50, 500),
                    'Speaker': (30, 400),
                    'Monitor': (150, 2000)
                }
            },
            'quantity_range': (1, 5)  # Most orders have 1-5 items
        },
        'finance': {
            'transaction_types': ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Refund', 'Interest', 'Fee', 'Dividend'],
            'currencies': ['USD', 'EUR', 'GBP', 'INR', 'JPY', 'CAD', 'AUD', 'CHF'],
            'status': ['Completed', 'Pending', 'Failed', 'Cancelled', 'Processing', 'Reversed'],
            'amount_ranges': {
                'Deposit': (100, 10000),
                'Withdrawal': (50, 5000),
                'Transfer': (20, 2000),
                'Payment': (10, 1000)
            }
        }
    }
    
    # Universal data pools
    UNIVERSAL_DATA = {
        'first_names': {
            'western': ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth',
                       'David', 'Susan', 'Richard', 'Jessica', 'Joseph', 'Sarah', 'Thomas', 'Karen', 'Charles', 'Nancy',
                       'Christopher', 'Lisa', 'Daniel', 'Margaret', 'Matthew', 'Sandra', 'Anthony', 'Ashley', 'Donald', 'Kimberly'],
            'indian': ['Rahul', 'Priya', 'Aarav', 'Ananya', 'Vikram', 'Sneha', 'Arjun', 'Diya', 'Raj', 'Neha',
                      'Sanjay', 'Pooja', 'Amit', 'Riya', 'Deepak', 'Swati', 'Karan', 'Meera', 'Vivek', 'Tanya'],
            'global': ['Carlos', 'Maria', 'Juan', 'Ana', 'Mohammed', 'Fatima', 'Ali', 'Aisha', 'Chen', 'Li',
                      'Wang', 'Zhang', 'Hiroshi', 'Yuki', 'Kenji', 'Sakura', 'Dmitry', 'Olga', 'Ivan', 'Natalia']
        },
        'last_names': {
            'common': ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                      'Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma', 'Khan', 'Chen', 'Wang', 'Li', 'Kim', 'Park', 'Nguyen']
        },
        'email_domains': {
            'personal': ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com'],
            'corporate': ['company.com', 'business.com', 'enterprise.com', 'corp.com', 'inc.com'],
            'country': ['co.in', 'co.uk', 'ca', 'com.au', 'de', 'fr', 'jp']
        },
        'companies': ['TechCorp', 'GlobalBiz', 'InnovateInc', 'MegaCorp', 'NextGen', 'FutureTech', 'SmartSolutions', 'DigitalWorks']
    }
    
    @staticmethod
    def generate_enhanced_data(col_name, col_info, domain_info, original_values, num_rows):
        """Enhanced data generation with better realism"""
        
        category = col_info.get('category', 'unknown')
        subcategory = col_info.get('subcategory', 'generic')
        domain = domain_info.get('domain', 'other')
        
        # Enhanced generation based on category
        if category == 'identifier':
            return EnhancedSmartGenerator._generate_enhanced_id(col_name, col_info, original_values, num_rows)
        
        elif category == 'person':
            return EnhancedSmartGenerator._generate_enhanced_person(col_name, col_info, domain, num_rows)
        
        elif category == 'contact':
            return EnhancedSmartGenerator._generate_enhanced_email(col_name, col_info, num_rows)
        
        elif category == 'location':
            return EnhancedSmartGenerator._generate_enhanced_location(col_name, col_info, domain, num_rows)
        
        elif category == 'categorical':
            return EnhancedSmartGenerator._generate_enhanced_categorical(col_name, col_info, domain, original_values, num_rows)
        
        elif category == 'numeric':
            return EnhancedSmartGenerator._generate_enhanced_numeric(col_name, col_info, domain, original_values, num_rows)
        
        elif category == 'text':
            if subcategory == 'product_name':
                return EnhancedSmartGenerator._generate_enhanced_products(domain, original_values, num_rows)
            else:
                return EnhancedSmartGenerator._generate_enhanced_text(col_info, num_rows)
        
        elif category == 'temporal':
            return EnhancedSmartGenerator._generate_enhanced_dates(col_info, original_values, num_rows)
        
        else:
            return EnhancedSmartGenerator._generate_from_enhanced_patterns(original_values, num_rows)
    
    @staticmethod
    def _generate_enhanced_id(col_name, col_info, original_values, num_rows):
        """Generate enhanced IDs"""
        # Get format hint
        format_hint = col_info.get('format_hint', 'unknown')
        
        # Determine start based on original
        start_id = 1000
        if original_values:
            try:
                nums = []
                for val in original_values[:10]:
                    if pd.notna(val):
                        num_part = re.sub(r'\D', '', str(val))
                        if num_part:
                            nums.append(int(num_part))
                if nums:
                    start_id = max(nums) + 1
            except:
                pass
        
        # Generate based on format
        if format_hint == 'prefix_numeric':
            prefix = 'ORD' if 'order' in col_name.lower() else 'CUST' if 'customer' in col_name.lower() else 'PROD'
            return [f"{prefix}{start_id + i:05d}" for i in range(num_rows)]
        
        elif format_hint == 'numeric_only':
            return [str(start_id + i) for i in range(num_rows)]
        
        elif format_hint == 'uuid':
            return [str(uuid.uuid4()) for _ in range(num_rows)]
        
        else:
            # Mixed format
            prefixes = ['ID', 'REF', 'KEY', 'CODE']
            return [f"{random.choice(prefixes)}{start_id + i:06d}" for i in range(num_rows)]
    
    @staticmethod
    def _generate_enhanced_person(col_name, col_info, domain, num_rows):
        """Generate enhanced person names"""
        subcategory = col_info.get('subcategory', 'full_name')
        format_hint = col_info.get('format_hint', 'western_names')
        
        # Choose name pool based on format hint
        if 'indian' in str(format_hint).lower() or domain == 'ecommerce':
            # Mix of Indian and Western names for e-commerce
            first_names = EnhancedSmartGenerator.UNIVERSAL_DATA['first_names']['indian'] + \
                         EnhancedSmartGenerator.UNIVERSAL_DATA['first_names']['western'][:10]
        elif 'global' in str(format_hint).lower():
            first_names = EnhancedSmartGenerator.UNIVERSAL_DATA['first_names']['global']
        else:
            first_names = EnhancedSmartGenerator.UNIVERSAL_DATA['first_names']['western']
        
        last_names = EnhancedSmartGenerator.UNIVERSAL_DATA['last_names']['common']
        
        # Generate names
        if subcategory == 'first_name':
            return random.choices(first_names, k=num_rows)
        elif subcategory == 'last_name':
            return random.choices(last_names, k=num_rows)
        else:  # full_name
            names = []
            for _ in range(num_rows):
                first = random.choice(first_names)
                last = random.choice(last_names)
                
                # Occasionally add middle initial
                if random.random() < 0.2:
                    middle = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    names.append(f"{first} {middle}. {last}")
                else:
                    names.append(f"{first} {last}")
            
            return names
    
    @staticmethod
    def _generate_enhanced_email(col_name, col_info, num_rows):
        """Generate enhanced realistic emails"""
        format_hint = col_info.get('format_hint', 'generic')
        
        emails = []
        for i in range(num_rows):
            # Choose email pattern
            rand = random.random()
            
            if rand < 0.4:  # first.last pattern
                first = random.choice(['john', 'jane', 'alex', 'sam', 'mike', 'sara', 'david', 'lisa'])
                last = random.choice(['smith', 'johnson', 'williams', 'brown', 'jones', 'miller', 'davis'])
                num = random.randint(1, 99) if random.random() < 0.5 else ''
                username = f"{first}.{last}{num}"
            
            elif rand < 0.7:  # first initial + last
                first = random.choice(['j', 'a', 's', 'm', 'd', 'l', 'r', 'p'])
                last = random.choice(['smith', 'johnson', 'williams', 'brown', 'kumar', 'sharma', 'patel'])
                num = random.randint(1, 99) if random.random() < 0.3 else ''
                username = f"{first}{last}{num}"
            
            elif rand < 0.85:  # descriptive username
                adjectives = ['cool', 'happy', 'smart', 'fast', 'quiet', 'brave', 'calm']
                nouns = ['guy', 'girl', 'coder', 'writer', 'runner', 'thinker', 'maker']
                num = random.randint(1, 999)
                username = f"{random.choice(adjectives)}{random.choice(nouns)}{num}"
            
            else:  # simple username
                username = f"user{random.randint(1000, 9999)}"
            
            # Choose domain
            if random.random() < 0.7:
                domain = random.choice(EnhancedSmartGenerator.UNIVERSAL_DATA['email_domains']['personal'])
            else:
                if random.random() < 0.5:
                    domain = random.choice(EnhancedSmartGenerator.UNIVERSAL_DATA['email_domains']['corporate'])
                else:
                    domain = random.choice(EnhancedSmartGenerator.UNIVERSAL_DATA['email_domains']['country'])
            
            emails.append(f"{username}@{domain}".lower())
        
        return emails
    
    @staticmethod
    def _generate_enhanced_location(col_name, col_info, domain, num_rows):
        """Generate enhanced locations"""
        subcategory = col_info.get('subcategory', 'general')
        
        if subcategory == 'country':
            domain_data = EnhancedSmartGenerator.DOMAIN_DATA.get(domain, {})
            countries = domain_data.get('countries', EnhancedSmartGenerator.UNIVERSAL_DATA.get('countries', ['USA', 'India', 'UK']))
            
            # Generate with some distribution (not completely random)
            weights = [0.3, 0.25, 0.2, 0.15, 0.1] + [0.01] * (len(countries) - 5)
            weights = weights[:len(countries)]
            weights = [w/sum(weights) for w in weights]  # Normalize
            
            return random.choices(countries, weights=weights, k=num_rows)
        
        elif subcategory == 'city':
            cities = ['New York', 'London', 'Tokyo', 'Mumbai', 'Sydney', 'Berlin', 'Paris', 'Toronto',
                     'Singapore', 'Dubai', 'Hong Kong', 'Shanghai', 'São Paulo', 'Moscow', 'Seoul',
                     'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
            return random.choices(cities, k=num_rows)
        
        else:
            return ['Location {}'.format(random.randint(1, 100)) for _ in range(num_rows)]
    
    @staticmethod
    def _generate_enhanced_categorical(col_name, col_info, domain, original_values, num_rows):
        """Generate enhanced categorical data"""
        col_lower = col_name.lower()
        
        # Use original values if they make sense
        unique_vals = list(set(str(v) for v in original_values if pd.notna(v)))
        
        if unique_vals and len(unique_vals) <= 15:
            # Check if original values are meaningful
            if all(len(v) > 1 and any(c.isalpha() for c in v) for v in unique_vals[:5]):
                # Add some variations
                all_vals = unique_vals.copy()
                if len(all_vals) < 8:
                    # Add some common values
                    if 'status' in col_lower:
                        common = ['Active', 'Inactive', 'Pending', 'Completed', 'Failed', 'Cancelled']
                        all_vals.extend([v for v in common if v not in all_vals])
                
                return random.choices(all_vals, k=num_rows)
        
        # Domain-specific values
        domain_data = EnhancedSmartGenerator.DOMAIN_DATA.get(domain, {})
        
        if 'status' in col_lower and 'statuses' in domain_data:
            values = domain_data['statuses']
        elif 'type' in col_lower or 'category' in col_lower:
            values = ['Type A', 'Type B', 'Type C', 'Standard', 'Premium', 'Basic', 'Advanced']
        else:
            values = ['Value {}'.format(i) for i in range(1, 6)]
        
        # Add some randomness but keep distribution realistic
        if len(values) >= 3:
            # Weight towards first few values
            weights = [0.4, 0.3, 0.15] + [0.15/(len(values)-3)] * (len(values)-3)
            return random.choices(values, weights=weights[:len(values)], k=num_rows)
        else:
            return random.choices(values, k=num_rows)
    
    @staticmethod
    def _generate_enhanced_numeric(col_name, col_info, domain, original_values, num_rows):
        """Generate enhanced numeric data with realistic values"""
        subcategory = col_info.get('subcategory', 'general')
        stats = col_info.get('stats', {})
        format_hint = col_info.get('format_hint', 'plain')
        
        # Extract numeric values from original
        numeric_vals = []
        for val in original_values:
            try:
                if pd.notna(val):
                    clean = str(val).replace('$', '').replace(',', '').strip()
                    num = float(clean)
                    numeric_vals.append(num)
            except:
                continue
        
        # Generate based on subcategory
        if subcategory == 'age':
            # Realistic age distribution
            if numeric_vals:
                mean_age = np.mean(numeric_vals)
                std_age = np.std(numeric_vals) if len(numeric_vals) > 1 else 8
            else:
                mean_age = 35
                std_age = 12
            
            # Generate with normal distribution, clipped to realistic range
            ages = np.random.normal(mean_age, std_age, num_rows)
            ages = np.clip(ages, 18, 80)
            return [int(round(age)) for age in ages]
        
        elif subcategory == 'monetary':
            # Realistic prices with common endings
            if numeric_vals:
                mean_val = np.mean(numeric_vals)
                std_val = np.std(numeric_vals) if len(numeric_vals) > 1 else mean_val * 0.5
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
            else:
                # Default realistic ranges
                if 'price' in col_name.lower():
                    mean_val = 500
                    std_val = 400
                    min_val = 10
                    max_val = 3000
                else:
                    mean_val = 1000
                    std_val = 800
                    min_val = 50
                    max_val = 10000
            
            # Generate prices
            prices = []
            for _ in range(num_rows):
                # Base price with normal distribution
                base = np.random.normal(mean_val, std_val)
                base = np.clip(base, min_val * 0.5, max_val * 1.5)
                
                # Apply realistic price psychology
                rand = random.random()
                if rand < 0.3:  # .99 ending
                    price = math.floor(base) + 0.99
                elif rand < 0.5:  # .95 ending
                    price = math.floor(base) + 0.95
                elif rand < 0.6:  # .50 ending
                    price = math.floor(base) + 0.50
                elif rand < 0.7:  # .00 ending
                    price = round(base)
                else:  # Random cents
                    price = round(base, 2)
                
                # Ensure reasonable values
                if price < 1:
                    price = round(random.uniform(1, 100), 2)
                
                prices.append(float(price))
            
            return prices
        
        elif subcategory == 'count':
            # Realistic counts (often small numbers)
            if numeric_vals:
                mean_count = np.mean(numeric_vals)
                max_count = max(numeric_vals)
            else:
                mean_count = 2
                max_count = 10
            
            # Generate with Poisson-like distribution
            counts = []
            for _ in range(num_rows):
                # Poisson distribution for counts
                lam = min(mean_count, 10)  # Lambda for Poisson
                count = np.random.poisson(lam) + 1
                count = min(count, max_count * 2)
                counts.append(int(count))
            
            return counts
        
        else:
            # General numeric
            if numeric_vals:
                mean_val = np.mean(numeric_vals)
                std_val = np.std(numeric_vals) if len(numeric_vals) > 1 else mean_val * 0.3
                min_val = min(numeric_vals)
                max_val = max(numeric_vals)
                
                data = np.random.normal(mean_val, std_val, num_rows)
                data = np.clip(data, min_val * 0.8, max_val * 1.2)
                
                if stats.get('is_integer', False):
                    return [int(round(x)) for x in data]
                else:
                    decimals = stats.get('decimal_places', 2)
                    return [float(round(x, decimals)) for x in data]
            else:
                return [random.randint(1, 100) for _ in range(num_rows)]
    
    @staticmethod
    def _generate_enhanced_products(domain, original_values, num_rows):
        """Generate enhanced product names with variety"""
        if domain != 'ecommerce':
            # Generic products for other domains
            items = ['Item', 'Product', 'Service', 'Component', 'Module', 'Tool']
            types = ['Standard', 'Premium', 'Basic', 'Advanced', 'Professional']
            return [f"{random.choice(types)} {random.choice(items)} {random.randint(1, 100)}" 
                   for _ in range(num_rows)]
        
        # E-commerce specific products
        domain_data = EnhancedSmartGenerator.DOMAIN_DATA.get('ecommerce', {})
        products_data = domain_data.get('products', {})
        
        electronics = products_data.get('electronics', {})
        categories = products_data.get('categories', ['Electronics'])
        
        products = []
        used_combinations = set()
        
        for i in range(num_rows):
            # Ensure variety - don't repeat same combination too often
            for attempt in range(10):  # Try up to 10 times to get unique combination
                # Choose product type
                if random.random() < 0.8:  # 80% electronics
                    product_type = random.choice(electronics['types'])
                    brand = random.choice(electronics['brands'])
                    model = random.choice(electronics['models'])
                    spec = random.choice(electronics['specs']) if random.random() < 0.6 else ''
                    
                    combo = (product_type, brand, model)
                    if combo not in used_combinations or len(used_combinations) > 50:
                        used_combinations.add(combo)
                        product_name = f"{brand} {product_type} {model} {spec}".strip()
                        break
                else:  # 20% other categories
                    category = random.choice(categories)
                    if category == 'Electronics':
                        continue  # Try again
                    
                    # Generic product
                    adjectives = ['Premium', 'Luxury', 'Essential', 'Everyday', 'Professional']
                    product_name = f"{random.choice(adjectives)} {category} Item {random.randint(1, 100)}"
                    break
            else:
                # Fallback if no unique combo found
                product_name = f"Product {i+1000}"
            
            products.append(product_name)
        
        return products
    
    @staticmethod
    def _generate_enhanced_text(col_info, num_rows):
        """Generate enhanced text"""
        avg_length = col_info.get('avg_length', 20)
        
        texts = []
        for _ in range(num_rows):
            # Generate text of appropriate length
            words = ['report', 'analysis', 'summary', 'review', 'document', 'note', 'comment',
                    'description', 'details', 'information', 'data', 'results', 'findings']
            
            num_words = max(2, int(avg_length / 5))
            text_words = random.choices(words, k=num_words)
            text = ' '.join(word.capitalize() for word in text_words)
            
            # Add some variations
            if random.random() < 0.3:
                text += f" #{random.randint(1, 100)}"
            
            texts.append(text)
        
        return texts
    
    @staticmethod
    def _generate_enhanced_dates(col_info, original_values, num_rows):
        """Generate enhanced dates"""
        format_hint = col_info.get('format_hint', '%Y-%m-%d')
        
        # Parse format
        try:
            date_format = format_hint
        except:
            date_format = '%Y-%m-%d'
        
        # Generate dates (mostly recent with some older)
        dates = []
        end_date = datetime.now()
        
        for _ in range(num_rows):
            # 60% recent (last 60 days), 30% medium (61-365 days), 10% older
            rand = random.random()
            if rand < 0.6:
                days_ago = random.randint(1, 60)
            elif rand < 0.9:
                days_ago = random.randint(61, 365)
            else:
                days_ago = random.randint(366, 365*2)
            
            date = end_date - timedelta(days=days_ago)
            dates.append(date.strftime(date_format))
        
        return dates
    
    @staticmethod
    def _generate_from_enhanced_patterns(original_values, num_rows):
        """Enhanced pattern-based generation"""
        if not original_values:
            return [f"Data_{i:04d}" for i in range(num_rows)]
        
        # Analyze patterns
        samples = [str(v) for v in original_values[:20] if pd.notna(v)]
        
        if len(set(samples)) <= 10:
            # Use existing values with some variations
            base_values = list(set(samples))
            results = []
            for _ in range(num_rows):
                base = random.choice(base_values)
                if random.random() < 0.3:
                    results.append(f"{base}_{random.randint(1, 100)}")
                else:
                    results.append(base)
            return results
        else:
            # Generate similar patterns
            pattern = EnhancedSmartGenerator._analyze_pattern(samples[0] if samples else "Sample")
            return [EnhancedSmartGenerator._generate_from_pattern(pattern) for _ in range(num_rows)]
    
    @staticmethod
    def _analyze_pattern(sample):
        """Analyze pattern in a sample"""
        # Simple pattern analysis
        pattern = []
        for char in sample:
            if char.isdigit():
                pattern.append('#')
            elif char.isalpha():
                pattern.append('@')
            else:
                pattern.append(char)
        return ''.join(pattern)
    
    @staticmethod
    def _generate_from_pattern(pattern):
        """Generate from pattern"""
        result = []
        for char in pattern:
            if char == '#':
                result.append(str(random.randint(0, 9)))
            elif char == '@':
                result.append(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
            else:
                result.append(char)
        return ''.join(result)


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================
import math
import uuid

class EnhancedIndustryGenerator:
    """Main enhanced generator class"""
    
    def __init__(self):
        self.classifier = EnhancedColumnClassifier()
        self.domain_detector = EnhancedDomainDetector()
        self.generator = EnhancedSmartGenerator()
    
    def generate(self, original_df, num_rows):
        """Enhanced generation pipeline"""
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Enhanced column classification
        column_analysis = {}
        for col in original_df.columns:
            samples = original_df[col].dropna().head(30).tolist()
            analysis = self.classifier.classify_column(col, samples)
            column_analysis[col] = analysis
        
        # Step 2: Enhanced domain detection
        sample_row = {}
        for col in original_df.columns[:5]:
            if len(original_df[col].dropna()) > 0:
                sample_row[col] = original_df[col].iloc[0]
        
        domain_info = self.domain_detector.detect_domain_with_context(
            list(original_df.columns),
            sample_row,
            column_analysis
        )
        
        # Store for display
        st.session_state.column_analysis = column_analysis
        st.session_state.domain_info = domain_info
        
        # Step 3: Enhanced generation
        generated_data = {}
        
        for col in original_df.columns:
            analysis = column_analysis[col]
            original_values = original_df[col].dropna().tolist()
            
            generated_data[col] = self.generator.generate_enhanced_data(
                col_name=col,
                col_info=analysis,
                domain_info=domain_info,
                original_values=original_values,
                num_rows=num_rows
            )
        
        # Step 4: Create DataFrame
        df_generated = pd.DataFrame(generated_data)
        
        # Step 5: Post-processing for realism
        df_generated = self._apply_realism_checks(df_generated, column_analysis, domain_info)
        
        return df_generated
    
    def _apply_realism_checks(self, df, column_analysis, domain_info):
        """Apply final realism checks"""
        # Ensure IDs are unique and sequential if needed
        for col in df.columns:
            if col in column_analysis:
                info = column_analysis[col]
                if info.get('category') == 'identifier' and info.get('subcategory') == 'sequential_id':
                    # Ensure sequential order
                    try:
                        # Extract numbers and sort
                        nums = []
                        for val in df[col]:
                            num_part = re.sub(r'\D', '', str(val))
                            if num_part:
                                nums.append(int(num_part))
                        
                        if nums and len(set(nums)) == len(nums):
                            # Already unique
                            pass
                        else:
                            # Make unique and sequential
                            start = 1000
                            df[col] = [str(start + i) for i in range(len(df))]
                    except:
                        pass
        
        return df


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
        page_title="Enhanced Data Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Header
    st.title("🚀 Enhanced Industry-Grade Data Generator")
    st.markdown("**Realistic • Diverse • High-Quality** - Production-ready data generation")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize
    if 'generator' not in st.session_state:
        st.session_state.generator = EnhancedIndustryGenerator()
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
                    st.write(f"**Data Types:**")
                    for col in df.columns:
                        st.write(f"- {col}: {df[col].dtype}")
                
                # Generation controls
                st.subheader("⚙️ Generate High-Quality Data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=10000, 
                                             value=1000)
                
                with col2:
                    realism_level = st.select_slider(
                        "Realism Level",
                        options=["Basic", "Standard", "Enhanced", "Premium"],
                        value="Enhanced"
                    )
                
                with col3:
                    if st.button("🚀 Generate Premium Data", type="primary"):
                        with st.spinner("Analyzing patterns and generating high-quality data..."):
                            generated = st.session_state.generator.generate(df, int(num_rows))
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} premium quality rows!")
                            st.balloons()
                
                # Show analysis
                if 'column_analysis' in st.session_state:
                    with st.expander("🔍 Advanced Analysis", expanded=False):
                        analysis = st.session_state.column_analysis
                        domain_info = st.session_state.domain_info
                        
                        st.write(f"**Detected Domain:** `{domain_info.get('domain', 'unknown')}`")
                        st.write(f"**Confidence:** `{domain_info.get('confidence', 0):.2f}`")
                        st.write(f"**Reason:** {domain_info.get('reason', '')}")
                        
                        if 'suggestions' in domain_info:
                            st.write("**AI Suggestions:**")
                            for suggestion in domain_info['suggestions'][:3]:
                                st.write(f"- {suggestion}")
                        
                        st.write("---")
                        st.write("**Column Analysis:**")
                        for col, info in analysis.items():
                            with st.expander(f"{col}", expanded=False):
                                st.write(f"**Category:** `{info.get('category', 'unknown')}`")
                                st.write(f"**Subcategory:** `{info.get('subcategory', 'generic')}`")
                                st.write(f"**Confidence:** `{info.get('confidence', 0):.2f}`")
                                
                                if 'format_hint' in info:
                                    st.write(f"**Format:** `{info['format_hint']}`")
                                
                                if 'stats' in info and info['stats'].get('is_numeric'):
                                    stats = info['stats']
                                    st.write("**Numeric Stats:**")
                                    st.write(f"- Min: `{stats.get('min', 'N/A')}`")
                                    st.write(f"- Max: `{stats.get('max', 'N/A')}`")
                                    st.write(f"- Mean: `{stats.get('mean', 'N/A')}`")
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📊 Generated Data Analysis")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Tabs for different views
                    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Quality Check", "Statistics", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                        st.write(f"**Shape:** {len(df_gen)} rows × {len(df_gen.columns)} columns")
                        
                        # Show sample row
                        st.write("**Sample Row:**")
                        sample_idx = random.randint(0, min(5, len(df_gen)-1))
                        sample = df_gen.iloc[sample_idx]
                        st.json(sample.to_dict())
                    
                    with tab2:
                        st.write("**Data Quality Assessment:**")
                        
                        # Check for nonsense values
                        issues = []
                        for col in df_gen.columns:
                            # Check for placeholder values
                            sample = str(df_gen[col].iloc[0]) if len(df_gen) > 0 else ""
                            if 'GEN_' in sample or 'DATA_' in sample or 'VALUE_' in sample:
                                issues.append(f"Column '{col}' has placeholder values")
                            
                            # Check for reasonable diversity
                            unique_ratio = df_gen[col].nunique() / len(df_gen)
                            if unique_ratio < 0.01 and len(df_gen) > 100:
                                issues.append(f"Column '{col}' has low diversity ({unique_ratio:.1%} unique)")
                        
                        if issues:
                            st.warning("**Issues Found:**")
                            for issue in issues:
                                st.write(f"- {issue}")
                        else:
                            st.success("✅ Data looks realistic and diverse!")
                        
                        # Show value distributions for key columns
                        st.write("**Value Distributions:**")
                        key_cols = [col for col in df_gen.columns if df_gen[col].nunique() < 20][:3]
                        for col in key_cols[:3]:
                            if df_gen[col].nunique() < 20:
                                st.write(f"**{col}:**")
                                counts = df_gen[col].value_counts().head(10)
                                st.bar_chart(counts)
                    
                    with tab3:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Data Statistics**")
                            st.write(f"- Rows: {len(df)}")
                            st.write(f"- Columns: {len(df.columns)}")
                            st.write(f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                            
                            if len(df) > 0:
                                st.write("**Sample Values:**")
                                for col in df.columns[:3]:
                                    unique_count = df[col].nunique()
                                    st.write(f"- {col}: {df[col].iloc[0] if pd.notna(df[col].iloc[0]) else 'N/A'} (Unique: {unique_count})")
                        
                        with col2:
                            st.write("**Generated Data Statistics**")
                            st.write(f"- Rows: {len(df_gen)}")
                            st.write(f"- Columns: {len(df_gen.columns)}")
                            st.write(f"- Memory: {df_gen.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                            
                            if len(df_gen) > 0:
                                st.write("**Sample Values:**")
                                for col in df_gen.columns[:3]:
                                    unique_count = df_gen[col].nunique()
                                    st.write(f"- {col}: {df_gen[col].iloc[0] if pd.notna(df_gen[col].iloc[0]) else 'N/A'} (Unique: {unique_count})")
                    
                    with tab4:
                        # Download options
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "enhanced_synthetic_data.csv",
                            "text/csv",
                            key="download-csv"
                        )
                        
                        # JSON download
                        json_str = df_gen.to_json(orient='records', indent=2, default=str)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            "enhanced_synthetic_data.json",
                            "application/json",
                            key="download-json"
                        )
                        
                        # Excel download
                        try:
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_gen.to_excel(writer, index=False, sheet_name='Generated_Data')
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                "📥 Download Excel",
                                excel_data,
                                "enhanced_synthetic_data.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download-excel"
                            )
                        except:
                            st.info("Excel download requires openpyxl. Install with: `pip install openpyxl`")
                        
                        # Regenerate button
                        if st.button("🔄 Generate New Dataset", type="secondary"):
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
