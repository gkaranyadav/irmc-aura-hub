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
# INDUSTRY-GRADE UNIVERSAL GENERATOR
# =============================================================================

class ColumnClassifier:
    """Rule-based column classification - NO LLM, works for any dataset"""
    
    @staticmethod
    def classify_column(col_name, sample_values):
        """
        Classify columns based on name hints and value patterns
        Returns: (category, subcategory, confidence)
        """
        col_name_lower = str(col_name).lower()
        sample_strs = [str(v).lower() for v in sample_values[:10] if pd.notna(v)]
        
        # Check for ID patterns
        if any(x in col_name_lower for x in ['id', 'code', 'num', 'no', 'number', 'ref', 'key']):
            if ColumnClassifier._looks_like_id(sample_strs):
                return ('identifier', 'sequential_id' if ColumnClassifier._is_sequential(sample_strs) else 'random_id', 0.9)
        
        # Check for date/time
        if any(x in col_name_lower for x in ['date', 'time', 'day', 'month', 'year', 'timestamp']):
            if ColumnClassifier._looks_like_date(sample_strs):
                return ('temporal', 'date' if any('/' in s or '-' in s for s in sample_strs[:3]) else 'timestamp', 0.85)
        
        # Check for names
        if any(x in col_name_lower for x in ['name', 'person', 'customer', 'user', 'patient', 'employee']):
            if ColumnClassifier._looks_like_name(sample_strs):
                return ('person', 'full_name' if any(' ' in s for s in sample_strs[:3]) else 'single_name', 0.8)
        
        # Check for emails
        if any(x in col_name_lower for x in ['email', 'mail', 'contact']):
            if any('@' in s for s in sample_strs[:3]):
                return ('contact', 'email', 0.95)
        
        # Check for locations
        if any(x in col_name_lower for x in ['country', 'city', 'state', 'region', 'location', 'address']):
            if ColumnClassifier._looks_like_location(sample_strs):
                return ('location', 'country' if 'country' in col_name_lower else 'general', 0.7)
        
        # Check for categories/status
        if any(x in col_name_lower for x in ['status', 'type', 'category', 'class', 'level', 'stage']):
            if len(set(sample_strs)) <= 10:  # Limited unique values
                return ('categorical', 'status', 0.8)
        
        # Check for numeric
        numeric_count = sum(ColumnClassifier._is_numeric_like(v) for v in sample_strs[:10])
        if numeric_count > len(sample_strs) * 0.7:  # Mostly numeric
            # Check what kind of numeric
            if any(x in col_name_lower for x in ['age', 'years', 'old']):
                return ('numeric', 'age', 0.85)
            elif any(x in col_name_lower for x in ['price', 'cost', 'amount', 'value', 'salary', 'income']):
                return ('numeric', 'monetary', 0.85)
            elif any(x in col_name_lower for x in ['quantity', 'count', 'qty', 'total', 'sum']):
                return ('numeric', 'count', 0.8)
            elif any(x in col_name_lower for x in ['score', 'rating', 'grade', 'percentage', 'rate']):
                return ('numeric', 'measurement', 0.8)
            else:
                return ('numeric', 'general', 0.7)
        
        # Check for text/descriptions
        if any(x in col_name_lower for x in ['desc', 'note', 'comment', 'detail', 'text', 'remark']):
            return ('text', 'description', 0.7)
        
        # Default: unknown
        return ('unknown', 'generic', 0.5)
    
    @staticmethod
    def _looks_like_id(values):
        if not values:
            return False
        # Check for sequential numbers or codes
        try:
            nums = [int(re.sub(r'\D', '', v)) for v in values if re.sub(r'\D', '', v)]
            if len(nums) > 2:
                sorted_nums = sorted(nums)[:10]
                diffs = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
                if all(d == diffs[0] for d in diffs):
                    return True
        except:
            pass
        
        # Check for common ID patterns
        patterns = [r'^[A-Z]{2,3}\d{4,6}$', r'^\d{6,10}$', r'^[A-Z]{2}\d{6}$']
        for pattern in patterns:
            if any(re.match(pattern, v.upper()) for v in values[:3]):
                return True
        return False
    
    @staticmethod
    def _is_sequential(values):
        try:
            nums = [int(re.sub(r'\D', '', v)) for v in values[:5] if re.sub(r'\D', '', v)]
            if len(nums) > 2:
                return all(nums[i+1] > nums[i] for i in range(len(nums)-1))
        except:
            return False
    
    @staticmethod
    def _looks_like_date(values):
        date_patterns = [
            r'\d{4}[-/]\d{2}[-/]\d{2}',
            r'\d{2}[-/]\d{2}[-/]\d{4}',
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        ]
        for pattern in date_patterns:
            if any(re.match(pattern, v) for v in values[:3]):
                return True
        return False
    
    @staticmethod
    def _looks_like_name(values):
        # Names usually have letters, may have spaces, first letter often capital
        for val in values[:3]:
            if isinstance(val, str) and len(val) > 1:
                words = val.split()
                if 1 <= len(words) <= 3:
                    if all(w[0].isupper() for w in words if w):
                        return True
        return False
    
    @staticmethod
    def _looks_like_location(values):
        # Locations are usually proper nouns
        for val in values[:3]:
            if isinstance(val, str) and val[0].isupper() and len(val) > 2:
                if val.upper() != val:  # Not all caps
                    return True
        return False
    
    @staticmethod
    def _is_numeric_like(value):
        try:
            float(value)
            return True
        except:
            # Check for currency symbols
            if re.match(r'^[$€£¥]\d+\.?\d*$', value.replace(',', '')):
                return True
            return False


class DomainDetector:
    """Detect domain with minimal LLM usage"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
        except:
            self.available = False
    
    def detect_domain(self, column_names, sample_row):
        """Detect domain with ONE LLM call"""
        if not self.available:
            return self._rule_based_domain(column_names)
        
        prompt = f"""
        Given these column names: {column_names}
        And this sample data: {json.dumps(sample_row, default=str)}
        
        What DOMAIN is this data from? Choose ONE from:
        - ecommerce (online shopping, orders, products)
        - finance (banking, transactions, accounts)
        - healthcare (medical, patient records, treatments)
        - human_resources (employees, salaries, departments)
        - education (students, courses, grades)
        - iot_sensors (sensor readings, devices, measurements)
        - logistics (shipping, tracking, inventory)
        - social_media (users, posts, interactions)
        - real_estate (properties, listings, prices)
        - manufacturing (production, quality, equipment)
        - other (if none of the above)
        
        Return ONLY the domain name (e.g., "ecommerce") and a BRIEF reason.
        Format: domain|reason
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            if '|' in result:
                domain, reason = result.split('|', 1)
                return domain.strip().lower(), reason.strip()
            return result.lower(), "Detected by LLM"
            
        except:
            return self._rule_based_domain(column_names), "Rule-based fallback"
    
    def _rule_based_domain(self, column_names):
        """Rule-based domain detection"""
        col_names_lower = [str(c).lower() for c in column_names]
        
        # Domain patterns
        patterns = {
            'ecommerce': ['order', 'product', 'customer', 'price', 'quantity', 'shipping'],
            'finance': ['account', 'transaction', 'balance', 'amount', 'credit', 'debit'],
            'healthcare': ['patient', 'doctor', 'diagnosis', 'treatment', 'hospital', 'medical'],
            'human_resources': ['employee', 'salary', 'department', 'hire', 'position', 'manager'],
            'education': ['student', 'course', 'grade', 'teacher', 'school', 'class'],
            'iot_sensors': ['sensor', 'device', 'reading', 'measurement', 'temperature', 'pressure'],
            'logistics': ['shipment', 'tracking', 'warehouse', 'inventory', 'delivery', 'supply']
        }
        
        scores = {}
        for domain, keywords in patterns.items():
            score = sum(1 for keyword in keywords if any(keyword in col for col in col_names_lower))
            scores[domain] = score
        
        if scores:
            best_domain = max(scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:
                return best_domain[0]
        
        return 'other'


class SmartGenerator:
    """Main generator with domain-specific logic"""
    
    # Domain-specific data libraries (minimal but effective)
    DOMAIN_DATA = {
        'ecommerce': {
            'countries': ['India', 'USA', 'UK', 'Canada', 'Australia', 'Germany', 'Japan', 'China'],
            'statuses': ['Shipped', 'Pending', 'Delivered', 'Cancelled', 'Processing', 'Returned'],
            'products': {
                'prefix': ['Smart', 'Pro', 'Ultra', 'Power', 'Fast', 'Mini', 'Max', 'Lite'],
                'type': ['Phone', 'Laptop', 'Tablet', 'Watch', 'Headphones', 'Speaker', 'Monitor'],
                'suffix': ['Pro', 'Plus', 'Max', 'Air', 'SE', 'X', '2024']
            }
        },
        'healthcare': {
            'conditions': ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Migraine', 'Allergy'],
            'treatments': ['Medication', 'Therapy', 'Surgery', 'Exercise', 'Diet', 'Rest'],
            'status': ['Stable', 'Critical', 'Recovering', 'In Treatment', 'Discharged']
        },
        'finance': {
            'transaction_types': ['Deposit', 'Withdrawal', 'Transfer', 'Payment', 'Refund'],
            'currencies': ['USD', 'EUR', 'GBP', 'INR', 'JPY', 'CAD', 'AUD'],
            'status': ['Completed', 'Pending', 'Failed', 'Cancelled', 'Processing']
        },
        'other': {
            'generic_status': ['Active', 'Inactive', 'Pending', 'Completed', 'Cancelled'],
            'generic_categories': ['Type A', 'Type B', 'Type C', 'Standard', 'Premium']
        }
    }
    
    @staticmethod
    def generate_for_category(category, subcategory, domain, original_values, num_rows, col_name=None):
        """Generate data based on category and domain"""
        
        # Convert original values to analyze patterns
        orig_clean = [v for v in original_values if pd.notna(v)]
        
        if category == 'identifier':
            return SmartGenerator._generate_id(subcategory, orig_clean, num_rows, col_name)
        
        elif category == 'temporal':
            return SmartGenerator._generate_temporal(subcategory, orig_clean, num_rows)
        
        elif category == 'person':
            return SmartGenerator._generate_person(subcategory, domain, num_rows)
        
        elif category == 'contact':
            return SmartGenerator._generate_contact(subcategory, domain, num_rows)
        
        elif category == 'location':
            return SmartGenerator._generate_location(subcategory, domain, num_rows)
        
        elif category == 'categorical':
            return SmartGenerator._generate_categorical(subcategory, domain, orig_clean, num_rows, col_name)
        
        elif category == 'numeric':
            return SmartGenerator._generate_numeric(subcategory, orig_clean, num_rows)
        
        elif category == 'text':
            return SmartGenerator._generate_text(subcategory, domain, orig_clean, num_rows)
        
        else:
            # Unknown category - use original patterns
            return SmartGenerator._generate_from_patterns(orig_clean, num_rows)
    
    @staticmethod
    def _generate_id(subcategory, orig_values, num_rows, col_name):
        """Generate IDs"""
        if subcategory == 'sequential_id':
            # Find starting point
            start = 1000
            try:
                nums = [int(re.sub(r'\D', '', str(v))) for v in orig_values[:10] if re.sub(r'\D', '', str(v))]
                if nums:
                    start = max(nums) + 1
            except:
                pass
            
            # Check for prefix patterns
            prefix = ''
            if orig_values:
                sample = str(orig_values[0])
                match = re.match(r'^([A-Z]{2,3})', sample)
                if match:
                    prefix = match.group(1)
            
            if prefix:
                return [f"{prefix}{start + i:05d}" for i in range(num_rows)]
            else:
                return [str(start + i) for i in range(num_rows)]
        
        else:  # random_id
            prefix = ''
            if col_name:
                col_lower = col_name.lower()
                if 'customer' in col_lower:
                    prefix = 'CUST'
                elif 'order' in col_lower:
                    prefix = 'ORD'
                elif 'product' in col_lower:
                    prefix = 'PROD'
            
            if prefix:
                return [f"{prefix}{random.randint(10000, 99999)}" for _ in range(num_rows)]
            else:
                return [f"ID{random.randint(100000, 999999)}" for _ in range(num_rows)]
    
    @staticmethod
    def _generate_temporal(subcategory, orig_values, num_rows):
        """Generate dates/times"""
        # Analyze format from original
        format_str = '%Y-%m-%d'  # default
        
        if orig_values:
            sample = str(orig_values[0])
            if '/' in sample:
                if len(sample.split('/')[0]) == 4:
                    format_str = '%Y/%m/%d'
                else:
                    format_str = '%d/%m/%Y'
            elif '-' in sample and len(sample.split('-')[0]) == 2:
                format_str = '%d-%m-%Y'
        
        # Generate dates (mostly recent)
        dates = []
        end_date = datetime.now()
        
        for _ in range(num_rows):
            # 70% recent (last 90 days), 30% older
            if random.random() < 0.7:
                days_ago = random.randint(1, 90)
            else:
                days_ago = random.randint(91, 365*2)
            
            date = end_date - timedelta(days=days_ago)
            dates.append(date.strftime(format_str))
        
        return dates
    
    @staticmethod
    def _generate_person(subcategory, domain, num_rows):
        """Generate person names"""
        # Universal name pools (works for all domains)
        first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
                      'William', 'Elizabeth', 'David', 'Susan', 'Richard', 'Jessica', 'Joseph', 'Sarah',
                      'Rahul', 'Priya', 'Aarav', 'Ananya', 'Vikram', 'Sneha', 'Arjun', 'Diya',
                      'Carlos', 'Maria', 'Juan', 'Ana', 'Mohammed', 'Fatima', 'Ali', 'Aisha']
        
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                     'Singh', 'Kumar', 'Patel', 'Sharma', 'Gupta', 'Verma', 'Reddy', 'Khan',
                     'Martinez', 'Lopez', 'Gonzalez', 'Rodriguez', 'Perez', 'Sanchez', 'Ramirez',
                     'Chen', 'Wang', 'Li', 'Zhang', 'Liu', 'Yang', 'Huang', 'Kim']
        
        if subcategory == 'full_name':
            return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(num_rows)]
        else:
            return [random.choice(first_names) for _ in range(num_rows)]
    
    @staticmethod
    def _generate_contact(subcategory, domain, num_rows):
        """Generate contact info"""
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'company.com']
        
        emails = []
        for i in range(num_rows):
            # Various email patterns
            if random.random() < 0.4:
                # first.last
                first = random.choice(['john', 'jane', 'alex', 'sam', 'mike', 'sara'])
                last = random.choice(['smith', 'johnson', 'williams', 'brown', 'jones'])
                email = f"{first}.{last}{random.randint(1,99)}@{random.choice(domains)}"
            elif random.random() < 0.7:
                # username + numbers
                email = f"user{random.randint(1000, 9999)}@{random.choice(domains)}"
            else:
                # random letters
                email = f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}@{random.choice(domains)}"
            
            emails.append(email.lower())
        
        return emails
    
    @staticmethod
    def _generate_location(subcategory, domain, num_rows):
        """Generate location data"""
        # Universal location pool
        countries = ['India', 'USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan',
                    'China', 'Brazil', 'Mexico', 'Italy', 'Spain', 'Russia', 'South Korea']
        
        cities = ['New York', 'London', 'Tokyo', 'Mumbai', 'Sydney', 'Berlin', 'Paris', 'Toronto',
                 'Singapore', 'Dubai', 'Hong Kong', 'Shanghai', 'São Paulo', 'Moscow', 'Seoul']
        
        if subcategory == 'country' or 'country' in str(subcategory):
            return random.choices(countries, k=num_rows)
        else:
            return random.choices(cities, k=num_rows)
    
    @staticmethod
    def _generate_categorical(subcategory, domain, orig_values, num_rows, col_name):
        """Generate categorical data"""
        # Use domain-specific values if available
        domain_data = SmartGenerator.DOMAIN_DATA.get(domain, SmartGenerator.DOMAIN_DATA['other'])
        
        col_lower = str(col_name).lower() if col_name else ''
        
        # Try to extract from original values first
        unique_vals = list(set(str(v) for v in orig_values if pd.notna(v)))
        
        if len(unique_vals) > 0 and len(unique_vals) <= 10:
            # Use original categories if limited
            return random.choices(unique_vals, k=num_rows)
        
        # Use domain-specific defaults
        if 'status' in col_lower:
            if 'statuses' in domain_data:
                return random.choices(domain_data['statuses'], k=num_rows)
            else:
                return random.choices(['Active', 'Inactive', 'Pending', 'Completed', 'Failed'], k=num_rows)
        
        elif 'type' in col_lower or 'category' in col_lower:
            return random.choices(['Type A', 'Type B', 'Type C', 'Standard', 'Premium', 'Basic'], k=num_rows)
        
        else:
            # Generic categories
            return [f"Category {random.randint(1, 5)}" for _ in range(num_rows)]
    
    @staticmethod
    def _generate_numeric(subcategory, orig_values, num_rows):
        """Generate numeric data"""
        # Extract numeric values
        numeric_vals = []
        for val in orig_values:
            try:
                if pd.notna(val):
                    numeric_vals.append(float(val))
            except:
                continue
        
        if numeric_vals:
            min_val = min(numeric_vals)
            max_val = max(numeric_vals)
            mean_val = np.mean(numeric_vals)
            
            # Adjust based on subcategory
            if subcategory == 'age':
                min_val = max(18, min_val)
                max_val = min(100, max_val)
                # Generate ages with normal distribution
                data = np.random.normal(mean_val, (max_val - min_val) / 6, num_rows)
                data = np.clip(data, min_val, max_val)
                return [int(round(x)) for x in data]
            
            elif subcategory == 'monetary':
                # Common price endings
                data = np.random.normal(mean_val, (max_val - min_val) / 4, num_rows)
                data = np.clip(data, min_val * 0.5, max_val * 1.5)
                
                prices = []
                for x in data:
                    if random.random() < 0.3:
                        price = round(x) + 0.99
                    elif random.random() < 0.5:
                        price = round(x, 2)
                    else:
                        price = round(x)
                    prices.append(float(price))
                return prices
            
            elif subcategory == 'count':
                # Usually integers, often small
                data = np.random.normal(mean_val, max(1, mean_val/2), num_rows)
                data = np.clip(data, max(1, min_val), max_val)
                return [int(round(x)) for x in data]
            
            else:
                # General numeric
                data = np.random.normal(mean_val, (max_val - min_val) / 4, num_rows)
                data = np.clip(data, min_val, max_val)
                # Check if original values were integers
                if all(v.is_integer() for v in numeric_vals if isinstance(v, float)):
                    return [int(round(x)) for x in data]
                else:
                    return [float(round(x, 2)) for x in data]
        
        else:
            # Default ranges based on subcategory
            if subcategory == 'age':
                return [random.randint(18, 65) for _ in range(num_rows)]
            elif subcategory == 'monetary':
                return [round(random.uniform(10, 1000), 2) for _ in range(num_rows)]
            elif subcategory == 'count':
                return [random.randint(1, 10) for _ in range(num_rows)]
            else:
                return [random.randint(1, 100) for _ in range(num_rows)]
    
    @staticmethod
    def _generate_text(subcategory, domain, orig_values, num_rows):
        """Generate text data"""
        # Analyze original patterns
        samples = [str(v) for v in orig_values[:5] if pd.notna(v)]
        
        if not samples:
            # Generate generic text
            words = ['Report', 'Analysis', 'Summary', 'Review', 'Evaluation', 'Assessment']
            return [f"{random.choice(words)} {random.randint(1, 100)}" for _ in range(num_rows)]
        
        # Check if product-like
        if any(len(s.split()) >= 2 and any(c.isupper() for c in s) for s in samples[:3]):
            # Generate product-like names
            prefixes = ['Smart', 'Pro', 'Ultra', 'Power', 'Fast', 'Mini', 'Max']
            types = ['Device', 'System', 'Tool', 'Unit', 'Module', 'Component']
            suffixes = ['Pro', 'Plus', 'Max', '2024', 'X', 'SE']
            
            products = []
            for _ in range(num_rows):
                if random.random() < 0.6:
                    name = f"{random.choice(prefixes)} {random.choice(types)} {random.choice(suffixes)}"
                else:
                    name = f"{random.choice(types)} {random.randint(10, 99)}{random.choice(['GB', 'TB', 'MB'])}"
                products.append(name)
            return products
        
        else:
            # Generic text generation
            return [f"Item {random.randint(100, 999)}" for _ in range(num_rows)]
    
    @staticmethod
    def _generate_from_patterns(orig_values, num_rows):
        """Fallback: generate from patterns in original data"""
        if not orig_values:
            return [f"Value_{i}" for i in range(num_rows)]
        
        # Get unique values
        unique_vals = list(set(str(v) for v in orig_values if pd.notna(v)))
        
        if len(unique_vals) <= 20:
            # Use existing values
            return random.choices(unique_vals, k=num_rows)
        else:
            # Generate similar patterns
            samples = [str(v) for v in orig_values[:10] if pd.notna(v)]
            return [f"{random.choice(samples)}_{random.randint(1, 100)}" for _ in range(num_rows)]


# =============================================================================
# MAIN GENERATOR CLASS (RENAMED TO MATCH)
# =============================================================================
class IndustryDataGenerator:  # Renamed from UniversalDataGenerator
    """Main class combining all components"""
    
    def __init__(self):
        self.classifier = ColumnClassifier()
        self.domain_detector = DomainDetector()
        self.generator = SmartGenerator()
    
    def generate(self, original_df, num_rows):  # Fixed method name
        """Main generation pipeline"""
        if original_df.empty:
            return pd.DataFrame()
        
        # Step 1: Classify all columns (rule-based, no LLM)
        column_analysis = {}
        for col in original_df.columns:
            samples = original_df[col].dropna().head(20).tolist()
            category, subcategory, confidence = self.classifier.classify_column(col, samples)
            column_analysis[col] = {
                'category': category,
                'subcategory': subcategory,
                'confidence': confidence,
                'samples': samples[:5]
            }
        
        # Step 2: Detect domain (ONE LLM call)
        sample_row = {}
        for col in original_df.columns[:5]:  # First 5 columns
            if len(original_df[col].dropna()) > 0:
                sample_row[col] = original_df[col].iloc[0]
        
        domain, domain_reason = self.domain_detector.detect_domain(
            list(original_df.columns),
            sample_row
        )
        
        # Store analysis in session for display
        st.session_state.column_analysis = column_analysis
        st.session_state.domain_info = {'domain': domain, 'reason': domain_reason}
        
        # Step 3: Generate each column
        generated_data = {}
        
        for col in original_df.columns:
            analysis = column_analysis[col]
            original_values = original_df[col].dropna().tolist()
            
            generated_data[col] = self.generator.generate_for_category(
                category=analysis['category'],
                subcategory=analysis['subcategory'],
                domain=domain,
                original_values=original_values,
                num_rows=num_rows,
                col_name=col
            )
        
        # Step 4: Create DataFrame
        df_generated = pd.DataFrame(generated_data)
        
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
        page_title="Industry-Grade Data Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Header
    st.title("🏭 Industry-Grade Synthetic Data Generator")
    st.markdown("**Universal • Realistic • Cost-Effective** - Works for ANY dataset")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize - FIXED CLASS NAME
    if 'generator' not in st.session_state:
        st.session_state.generator = IndustryDataGenerator()  # Changed to IndustryDataGenerator
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
                st.subheader("⚙️ Generate Synthetic Data")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=10000, 
                                             value=1000)
                
                with col2:
                    if st.button("🚀 Generate Industry-Quality Data", type="primary"):
                        with st.spinner("Analyzing and generating..."):
                            # FIXED: Using correct method name 'generate'
                            generated = st.session_state.generator.generate(df, int(num_rows))
                            st.session_state.generated_data = generated
                            st.success(f"✅ Generated {len(generated)} realistic rows!")
                            st.balloons()
                
                # Show analysis if available
                if 'column_analysis' in st.session_state:
                    with st.expander("🔍 Analysis Results", expanded=False):
                        analysis = st.session_state.column_analysis
                        domain_info = st.session_state.domain_info
                        
                        st.write(f"**Detected Domain:** `{domain_info['domain']}`")
                        st.write(f"**Reason:** {domain_info['reason']}")
                        st.write("---")
                        
                        st.write("**Column Classification:**")
                        for col, info in analysis.items():
                            st.write(f"**{col}**:")
                            st.write(f"  • Category: `{info['category']}`")
                            st.write(f"  • Subcategory: `{info['subcategory']}`")
                            st.write(f"  • Confidence: `{info['confidence']:.2f}`")
                            st.write(f"  • Samples: `{info['samples'][:3] if info['samples'] else 'None'}`")
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    st.subheader("📋 Generated Data Preview")
                    
                    df_gen = st.session_state.generated_data
                    
                    # Show in tabs
                    tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Download"])
                    
                    with tab1:
                        st.dataframe(df_gen.head(20))
                        st.write(f"**Shape:** {len(df_gen)} rows × {len(df_gen.columns)} columns")
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Data**")
                            st.write(f"Rows: {len(df)}")
                            st.write(f"Columns: {len(df.columns)}")
                            if len(df) > 0:
                                for col in df.columns[:3]:
                                    st.write(f"- {col}: {df[col].iloc[0] if pd.notna(df[col].iloc[0]) else 'N/A'}")
                        
                        with col2:
                            st.write("**Generated Data**")
                            st.write(f"Rows: {len(df_gen)}")
                            st.write(f"Columns: {len(df_gen.columns)}")
                            if len(df_gen) > 0:
                                for col in df_gen.columns[:3]:
                                    st.write(f"- {col}: {df_gen[col].iloc[0] if pd.notna(df_gen[col].iloc[0]) else 'N/A'}")
                    
                    with tab3:
                        # Download options
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "industry_generated_data.csv",
                            "text/csv",
                            key="download-csv"
                        )
                        
                        # Generate more button
                        if st.button("🔄 Generate New Dataset"):
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
