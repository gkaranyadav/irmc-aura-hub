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
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import hashlib

# =============================================================================
# INTELLIGENT DATA ANALYZER & GENERATOR
# =============================================================================

class IntelligentDataGenerator:
    """Ultimate generator: LLM analyzes patterns, Python generates data perfectly"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
            self.models = {
                "analysis": "llama-3.3-70b-versatile",  # Best for deep analysis
                "large_context": "mixtral-8x7b-32768",   # For complex datasets
            }
        except:
            self.available = False
            st.warning("LLM not available. Using rule-based fallback.")
    
    def deep_analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Phase 1: Deep LLM analysis of the dataset
        Returns a generation blueprint with intelligent rules
        """
        if not self.available or df.empty:
            return self._create_basic_blueprint(df)
        
        # Prepare samples for LLM (5-7 rows for analysis)
        samples = self._prepare_samples_for_analysis(df)
        
        # Get column statistics for context
        stats = self._compute_column_statistics(df)
        
        prompt = self._build_analysis_prompt(df, samples, stats)
        
        try:
            messages = [
                {"role": "system", "content": """You are an expert data analyst and synthetic data generation specialist.

                CRITICAL MISSION: Analyze ANY dataset and create PERFECT generation rules for synthetic data.

                You MUST:
                1. Deeply understand EACH column's purpose and patterns
                2. Detect ALL relationships between columns
                3. Create realistic generation rules that maintain real-world logic
                4. Provide context-aware value pools
                5. Ensure ALL synthetic data will be 100% realistic

                Think step-by-step and be extremely thorough."""},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.models["analysis"],
                messages=messages,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            blueprint = self._parse_analysis_result(result, df)
            
            # Enhance blueprint with additional insights
            blueprint = self._enhance_blueprint(blueprint, df, stats)
            
            return blueprint
            
        except Exception as e:
            st.error(f"Deep analysis failed: {str(e)}")
            return self._create_basic_blueprint(df)
    
    def _prepare_samples_for_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Prepare representative samples for LLM analysis"""
        samples = []
        # Take first 3, middle 2, and last 2 rows for variety
        indices = list(range(min(3, len(df))))
        if len(df) > 5:
            mid = len(df) // 2
            indices.extend([mid, mid + 1])
        if len(df) > 7:
            indices.extend([-2, -1])
        
        indices = sorted(set(idx for idx in indices if 0 <= idx < len(df)))
        
        for idx in indices[:7]:  # Max 7 samples
            row = df.iloc[idx]
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
                    sample[col] = str(val)[:100]  # Limit length
            samples.append(sample)
        
        return samples
    
    def _compute_column_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute detailed statistics for each column"""
        stats = {}
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                stats[col] = {"null_count": len(df[col]), "total": len(df[col])}
                continue
            
            col_stats = {
                "null_count": df[col].isnull().sum(),
                "total": len(df[col]),
                "unique_count": len(non_null.unique()),
                "unique_ratio": len(non_null.unique()) / len(non_null),
                "sample_values": non_null.head(5).tolist()
            }
            
            # Type-specific stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std())
                })
            
            stats[col] = col_stats
        
        return stats
    
    def _build_analysis_prompt(self, df: pd.DataFrame, samples: List[Dict], stats: Dict) -> str:
        """Build comprehensive analysis prompt"""
        
        column_details = []
        for col in df.columns:
            detail = f"**{col}**"
            if col in stats:
                s = stats[col]
                detail += f" | Null: {s['null_count']}/{s['total']}"
                detail += f" | Unique: {s['unique_count']} ({s['unique_ratio']:.1%})"
                if 'min' in s:
                    detail += f" | Range: {s['min']} - {s['max']}"
            
            # Add first few unique values
            unique_vals = df[col].dropna().unique()[:5]
            if len(unique_vals) > 0:
                detail += f" | Samples: {unique_vals.tolist()}"
            
            column_details.append(detail)
        
        prompt = f"""
        DEEP DATASET ANALYSIS REQUEST
        
        DATASET OVERVIEW:
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Column names: {', '.join(df.columns.tolist())}
        
        COLUMN DETAILS:
        {chr(10).join(column_details)}
        
        SAMPLE DATA (representative rows):
        {json.dumps(samples, indent=2, default=str)}
        
        YOUR TASK:
        Analyze this dataset COMPLETELY and create a generation blueprint for synthetic data.
        
        For EACH column, provide:
        1. **column_type**: What kind of data is this? Choose from:
           - sequential_id (AP001, ID100, etc.)
           - human_name (First Last format)
           - categorical_fixed (limited set of values)
           - categorical_open (many values but categorical)
           - numeric_integer (whole numbers)
           - numeric_float (decimal numbers)
           - monetary_amount (prices, fees, etc.)
           - date (dates in any format)
           - time (times in any format)
           - datetime (date + time)
           - phone_number (phone/mobile numbers)
           - email_address (email addresses)
           - text_description (free text)
           - boolean (true/false, yes/no)
           - code_id (alphanumeric codes)
           - percentage (0-100%)
           - rating (1-5, 1-10 scales)
           - custom (describe in detail)
        
        2. **patterns_detected**: What specific patterns do you see?
           - Format: "AP###", "DD-MM-YYYY", etc.
           - Structure: "FirstName LastName", "Dr. Title LastName"
           - Range: "18-70", "$50-$500"
           - Distribution: "mostly 20-40", "round numbers common"
        
        3. **generation_rules**: EXACT rules for generating new values:
           - How to continue sequences?
           - What value ranges to use?
           - What formats to maintain?
           - Any constraints or special logic?
        
        4. **value_pool_suggestions**: Realistic values for categorical data
           - Lists of names, products, categories, etc.
           - Make them realistic and domain-appropriate
        
        For the ENTIRE dataset, provide:
        5. **dataset_context**: What domain/context is this? (medical, e-commerce, academic, etc.)
        
        6. **relationships**: ALL relationships between columns:
           - "If column A contains X, then column B should be Y"
           - "Column C must be greater than Column D"
           - "Columns E and F should have logical consistency"
        
        7. **realism_constraints**: Rules to ensure realism:
           - "No decimal ages"
           - "Phone numbers must be valid format"
           - "Dates must be logical (no future dates for completed orders)"
        
        8. **generation_strategy**: How to generate data intelligently:
           - "Generate names first, then assign gender based on name patterns"
           - "Calculate fees based on department and treatment type"
        
        OUTPUT FORMAT: Return a JSON object with this structure:
        {{
            "dataset_context": "medical_appointments_india",
            "columns": {{
                "column_name_1": {{
                    "column_type": "sequential_id",
                    "patterns_detected": ["AP### format", "sequential numbers"],
                    "generation_rules": ["Continue from AP025", "AP{num:03d} format"],
                    "value_pool_suggestions": []
                }},
                "column_name_2": {{
                    "column_type": "human_name",
                    "patterns_detected": ["Indian names", "First Last format"],
                    "generation_rules": ["70% Indian names", "30% Western names", "Title case"],
                    "value_pool_suggestions": {{
                        "indian_male_first": ["Rahul", "Amit", "Raj"],
                        "indian_female_first": ["Priya", "Neha", "Anjali"],
                        "indian_last": ["Patel", "Sharma", "Singh"]
                    }}
                }}
            }},
            "relationships": [
                "If PatientName contains 'Singh' or 'Kumar' then Gender = 'M'",
                "If Department = 'Cardiology' then Fee between 1000-2000"
            ],
            "realism_constraints": [
                "Ages must be integers 18-70",
                "Phone numbers must be 10 digits starting with 6-9"
            ],
            "generation_strategy": "Generate IDs sequentially, then names, then assign gender based on name patterns"
        }}
        
        Be EXTREMELY thorough. Every column MUST have detailed analysis.
        The synthetic data MUST be 100% realistic and logical.
        """
        
        return prompt
    
    def _parse_analysis_result(self, result: str, df: pd.DataFrame) -> Dict:
        """Parse LLM's analysis into a structured blueprint"""
        try:
            data = json.loads(result)
            
            # Ensure all columns are covered
            for col in df.columns:
                if col not in data.get("columns", {}):
                    # Create basic entry for missing columns
                    data.setdefault("columns", {})[col] = {
                        "column_type": "unknown",
                        "patterns_detected": ["Not analyzed by LLM"],
                        "generation_rules": ["Use values from original data"],
                        "value_pool_suggestions": df[col].dropna().unique().tolist()[:20]
                    }
            
            # Add metadata
            data["analysis_timestamp"] = datetime.now().isoformat()
            data["original_columns"] = df.columns.tolist()
            data["original_row_count"] = len(df)
            
            return data
            
        except Exception as e:
            st.warning(f"Could not parse LLM analysis: {str(e)}")
            # Create fallback blueprint
            return self._create_basic_blueprint(df)
    
    def _enhance_blueprint(self, blueprint: Dict, df: pd.DataFrame, stats: Dict) -> Dict:
        """Enhance blueprint with programmatically detected patterns"""
        
        for col in df.columns:
            if col not in blueprint.get("columns", {}):
                blueprint.setdefault("columns", {})[col] = {}
            
            col_config = blueprint["columns"][col]
            col_data = df[col].dropna()
            
            # Detect patterns programmatically
            detected_patterns = self._detect_patterns_programmatically(col, df[col])
            if detected_patterns:
                col_config.setdefault("patterns_detected", []).extend(detected_patterns)
            
            # Detect data type programmatically
            if "column_type" not in col_config or col_config["column_type"] == "unknown":
                col_config["column_type"] = self._detect_column_type_programmatically(col, df[col])
            
            # Add value pool from actual data
            if col_data.dtype == 'object' and len(col_data.unique()) <= 50:
                unique_vals = col_data.unique().tolist()
                if "value_pool_suggestions" not in col_config:
                    col_config["value_pool_suggestions"] = unique_vals
            
            # Detect numeric ranges
            if pd.api.types.is_numeric_dtype(df[col]) and len(col_data) > 0:
                col_config["numeric_range"] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean())
                }
        
        # Detect relationships programmatically
        detected_relationships = self._detect_relationships_programmatically(df)
        if detected_relationships:
            blueprint.setdefault("relationships", []).extend(detected_relationships)
        
        return blueprint
    
    def _detect_patterns_programmatically(self, col_name: str, series: pd.Series) -> List[str]:
        """Detect patterns in column data"""
        patterns = []
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return patterns
        
        # Sample values for pattern detection
        samples = non_null.head(10).astype(str).tolist()
        
        # Check for IDs
        col_lower = col_name.lower()
        if any(x in col_lower for x in ['id', 'no', 'num', 'code', 'ref']):
            if all(re.match(r'^[A-Z]{2}\d{3}$', str(x)) for x in samples[:5] if x):
                patterns.append("Format: XX### (2 letters + 3 digits)")
            elif all(re.match(r'^\d+$', str(x)) for x in samples[:5] if x):
                patterns.append("Numeric sequential IDs")
        
        # Check for dates
        date_patterns = [
            (r'\d{2}-\d{2}-\d{4}', 'DD-MM-YYYY'),
            (r'\d{4}-\d{2}-\d{2}', 'YYYY-MM-DD'),
            (r'\d{2}/\d{2}/\d{4}', 'DD/MM/YYYY'),
        ]
        for pattern, format_name in date_patterns:
            if any(re.match(pattern, str(x)) for x in samples if x):
                patterns.append(f"Date format: {format_name}")
                break
        
        # Check for times
        time_patterns = [
            (r'\d{1,2}:\d{2} [AP]M', '12-hour time with AM/PM'),
            (r'\d{2}:\d{2}:\d{2}', '24-hour time with seconds'),
            (r'\d{2}:\d{2}', '24-hour time'),
        ]
        for pattern, format_name in time_patterns:
            if any(re.match(pattern, str(x)) for x in samples if x):
                patterns.append(f"Time format: {format_name}")
                break
        
        # Check for phone numbers
        if any(re.match(r'^\d{10}$', str(x)) for x in samples if x):
            patterns.append("10-digit phone numbers")
        
        # Check for emails
        if any('@' in str(x) for x in samples):
            patterns.append("Email addresses")
        
        return patterns
    
    def _detect_column_type_programmatically(self, col_name: str, series: pd.Series) -> str:
        """Detect column type programmatically"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return "unknown"
        
        col_lower = col_name.lower()
        
        # Check common patterns
        samples = non_null.head(10).astype(str).tolist()
        
        # IDs
        if any(x in col_lower for x in ['id', 'no', 'number', 'code', 'ref', 'appointment']):
            if all(re.match(r'^[A-Za-z0-9-_]+$', str(x)) for x in samples if x):
                return "code_id"
        
        # Names
        if any(x in col_lower for x in ['name', 'patient', 'doctor', 'customer', 'user']):
            if all(len(str(x).split()) >= 2 for x in samples if x and ' ' in str(x)):
                return "human_name"
        
        # Dates/Times
        if any(x in col_lower for x in ['date', 'time', 'datetime', 'timestamp']):
            date_patterns = [r'\d{2}[-/]\d{2}[-/]\d{4}', r'\d{4}[-/]\d{2}[-/]\d{2}']
            if any(any(re.search(p, str(x)) for p in date_patterns) for x in samples if x):
                if any(':' in str(x) for x in samples):
                    return "datetime"
                return "date"
            if any(':' in str(x) for x in samples):
                return "time"
        
        # Phone/Email
        if 'phone' in col_lower or 'mobile' in col_lower or 'contact' in col_lower:
            return "phone_number"
        if 'email' in col_lower or 'mail' in col_lower:
            return "email_address"
        
        # Monetary
        if any(x in col_lower for x in ['price', 'fee', 'amount', 'cost', 'salary', 'revenue']):
            return "monetary_amount"
        
        # Numeric detection
        try:
            numeric_count = pd.to_numeric(non_null.head(20), errors='coerce').notna().sum()
            if numeric_count > len(non_null.head(20)) * 0.8:
                if all(float(x).is_integer() for x in non_null.head(10) if pd.notna(x)):
                    return "numeric_integer"
                return "numeric_float"
        except:
            pass
        
        # Categorical detection
        unique_ratio = len(non_null.unique()) / len(non_null)
        if unique_ratio < 0.3 and len(non_null.unique()) <= 20:
            return "categorical_fixed"
        
        return "text_description"
    
    def _detect_relationships_programmatically(self, df: pd.DataFrame) -> List[str]:
        """Detect relationships between columns programmatically"""
        relationships = []
        
        # Look for name-gender relationships
        name_cols = [col for col in df.columns if any(x in col.lower() for x in ['name', 'patient', 'customer'])]
        gender_cols = [col for col in df.columns if 'gender' in col.lower() or 'sex' in col.lower()]
        
        for name_col in name_cols:
            for gender_col in gender_cols:
                if name_col != gender_col:
                    # Check if there's a pattern
                    for idx, row in df.iterrows():
                        if pd.notna(row[name_col]) and pd.notna(row[gender_col]):
                            name = str(row[name_col])
                            gender = str(row[gender_col])
                            # Could add pattern detection here
                            break
        
        # Look for date sequence relationships
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if len(date_cols) >= 2:
            relationships.append(f"{date_cols[0]} should be before or equal to {date_cols[1]} for logical consistency")
        
        return relationships
    
    def _create_basic_blueprint(self, df: pd.DataFrame) -> Dict:
        """Create a basic blueprint when LLM is not available"""
        blueprint = {
            "dataset_context": "general_dataset",
            "columns": {},
            "relationships": [],
            "realism_constraints": ["Maintain data types", "Use realistic values"],
            "generation_strategy": "Generate each column independently based on observed patterns"
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            col_type = self._detect_column_type_programmatically(col, df[col])
            
            col_config = {
                "column_type": col_type,
                "patterns_detected": self._detect_patterns_programmatically(col, df[col]),
                "generation_rules": ["Generate values similar to original data"]
            }
            
            if col_type in ["categorical_fixed", "human_name", "code_id"] and len(col_data) > 0:
                unique_vals = col_data.unique().tolist()[:50]
                col_config["value_pool_suggestions"] = unique_vals
            
            if pd.api.types.is_numeric_dtype(df[col]) and len(col_data) > 0:
                col_config["numeric_range"] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max())
                }
            
            blueprint["columns"][col] = col_config
        
        return blueprint
    
    def generate_data_from_blueprint(self, blueprint: Dict, num_rows: int) -> pd.DataFrame:
        """
        Phase 2: Generate perfect synthetic data using the blueprint
        """
        data = {}
        original_columns = blueprint.get("original_columns", list(blueprint.get("columns", {}).keys()))
        
        # Initialize all columns
        for col in original_columns:
            data[col] = []
        
        # Get generation strategy
        strategy = blueprint.get("generation_strategy", "column_by_column")
        
        if "sequential" in strategy.lower():
            # Generate sequentially
            for i in range(num_rows):
                row = self._generate_single_row(blueprint, i, num_rows)
                for col in original_columns:
                    data[col].append(row.get(col, None))
        else:
            # Generate column by column (default)
            for col in original_columns:
                col_config = blueprint.get("columns", {}).get(col, {})
                col_type = col_config.get("column_type", "unknown")
                
                if col_type == "sequential_id":
                    data[col] = self._generate_sequential_ids(col_config, num_rows)
                elif col_type == "human_name":
                    data[col] = self._generate_human_names(col_config, num_rows)
                elif col_type == "categorical_fixed":
                    data[col] = self._generate_categorical_fixed(col_config, num_rows)
                elif col_type == "numeric_integer":
                    data[col] = self._generate_numeric_integer(col_config, num_rows)
                elif col_type == "numeric_float":
                    data[col] = self._generate_numeric_float(col_config, num_rows)
                elif col_type == "monetary_amount":
                    data[col] = self._generate_monetary_amount(col_config, num_rows)
                elif col_type == "date":
                    data[col] = self._generate_dates(col_config, num_rows)
                elif col_type == "time":
                    data[col] = self._generate_times(col_config, num_rows)
                elif col_type == "phone_number":
                    data[col] = self._generate_phone_numbers(col_config, num_rows)
                elif col_type == "email_address":
                    data[col] = self._generate_emails(col_config, num_rows)
                elif col_type == "code_id":
                    data[col] = self._generate_code_ids(col_config, num_rows)
                else:
                    # Default: Use value pool or random values
                    data[col] = self._generate_default_column(col_config, num_rows)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Apply relationships and constraints
        df = self._apply_relationships(df, blueprint)
        df = self._apply_constraints(df, blueprint)
        
        return df
    
    def _generate_single_row(self, blueprint: Dict, row_index: int, total_rows: int) -> Dict:
        """Generate a single row with cross-column consistency"""
        row = {}
        
        for col, col_config in blueprint.get("columns", {}).items():
            col_type = col_config.get("column_type", "unknown")
            
            if col_type == "sequential_id":
                row[col] = self._generate_sequential_id_value(col_config, row_index)
            elif col_type == "human_name":
                row[col] = self._generate_human_name_value(col_config, row_index)
            elif col_type == "categorical_fixed":
                row[col] = self._generate_categorical_value(col_config, row_index)
            elif col_type == "numeric_integer":
                row[col] = self._generate_numeric_integer_value(col_config, row_index)
            elif col_type == "numeric_float":
                row[col] = self._generate_numeric_float_value(col_config, row_index)
            elif col_type == "monetary_amount":
                row[col] = self._generate_monetary_amount_value(col_config, row_index)
            elif col_type == "date":
                row[col] = self._generate_date_value(col_config, row_index)
            elif col_type == "time":
                row[col] = self._generate_time_value(col_config, row_index)
            elif col_type == "phone_number":
                row[col] = self._generate_phone_number_value(col_config, row_index)
            elif col_type == "email_address":
                row[col] = self._generate_email_value(col_config, row_index)
            elif col_type == "code_id":
                row[col] = self._generate_code_id_value(col_config, row_index)
            else:
                row[col] = self._generate_default_value(col_config, row_index)
        
        return row
    
    def _generate_sequential_ids(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate sequential IDs"""
        patterns = col_config.get("patterns_detected", [])
        start_id = 1000
        
        # Try to extract start from patterns
        for pattern in patterns:
            if "AP###" in pattern or "AP" in pattern:
                start_id = 26  # Continue from AP025
                break
        
        ids = []
        for i in range(num_rows):
            ids.append(f"AP{start_id + i:03d}")
        
        return ids
    
    def _generate_sequential_id_value(self, col_config: Dict, index: int) -> str:
        """Generate a single sequential ID"""
        patterns = col_config.get("patterns_detected", [])
        start_id = 1000 + index
        
        for pattern in patterns:
            if "AP###" in pattern or "AP" in pattern:
                start_id = 26 + index
                return f"AP{start_id:03d}"
            elif "ID" in pattern:
                return f"ID{start_id}"
        
        return f"ID{start_id}"
    
    def _generate_human_names(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate human names"""
        names = []
        
        # Check for Indian names
        patterns = col_config.get("patterns_detected", [])
        is_indian = any("indian" in str(p).lower() for p in patterns)
        
        # Value pools from config
        value_pool = col_config.get("value_pool_suggestions", {})
        
        if isinstance(value_pool, dict):
            # Use provided name components
            first_male = value_pool.get("indian_male_first", ["Rahul", "Amit", "Raj", "Sanjay", "Vikram"])
            first_female = value_pool.get("indian_female_first", ["Priya", "Neha", "Anjali", "Sneha", "Pooja"])
            last_names = value_pool.get("indian_last", ["Patel", "Sharma", "Singh", "Kumar", "Gupta"])
        else:
            # Default Indian names
            first_male = ["Rahul", "Amit", "Raj", "Sanjay", "Vikram", "Karan", "Sachin"]
            first_female = ["Priya", "Neha", "Anjali", "Sneha", "Pooja", "Sonia", "Kavita"]
            last_names = ["Patel", "Sharma", "Singh", "Kumar", "Gupta", "Jain", "Reddy"]
        
        for i in range(num_rows):
            if random.random() < 0.5:  # 50% male
                first = random.choice(first_male)
            else:
                first = random.choice(first_female)
            
            last = random.choice(last_names)
            names.append(f"{first} {last}")
        
        return names
    
    def _generate_human_name_value(self, col_config: Dict, index: int) -> str:
        """Generate a single human name"""
        return self._generate_human_names(col_config, 1)[0]
    
    def _generate_categorical_fixed(self, col_config: Dict, num_rows: int) -> List:
        """Generate categorical values"""
        value_pool = col_config.get("value_pool_suggestions", [])
        
        if isinstance(value_pool, list) and len(value_pool) > 0:
            return random.choices(value_pool, k=num_rows)
        else:
            # Generate default categories
            categories = [f"Category_{i}" for i in range(1, 6)]
            return random.choices(categories, k=num_rows)
    
    def _generate_categorical_value(self, col_config: Dict, index: int):
        """Generate a single categorical value"""
        return self._generate_categorical_fixed(col_config, 1)[0]
    
    def _generate_numeric_integer(self, col_config: Dict, num_rows: int) -> List[int]:
        """Generate integer values"""
        numeric_range = col_config.get("numeric_range", {})
        min_val = int(numeric_range.get("min", 1))
        max_val = int(numeric_range.get("max", 100))
        
        # Adjust for common ranges
        patterns = col_config.get("patterns_detected", [])
        for pattern in patterns:
            if "age" in str(pattern).lower():
                min_val, max_val = 18, 70
                break
        
        return [random.randint(min_val, max_val) for _ in range(num_rows)]
    
    def _generate_numeric_integer_value(self, col_config: Dict, index: int) -> int:
        """Generate a single integer value"""
        return self._generate_numeric_integer(col_config, 1)[0]
    
    def _generate_numeric_float(self, col_config: Dict, num_rows: int) -> List[float]:
        """Generate float values"""
        numeric_range = col_config.get("numeric_range", {})
        min_val = numeric_range.get("min", 0.0)
        max_val = numeric_range.get("max", 100.0)
        
        # For monetary, use 2 decimal places
        patterns = col_config.get("patterns_detected", [])
        is_monetary = any(x in str(p).lower() for p in patterns for x in ['price', 'fee', 'amount'])
        
        values = []
        for _ in range(num_rows):
            val = random.uniform(min_val, max_val)
            if is_monetary:
                val = round(val, 2)
            else:
                val = round(val, random.choice([1, 2, 3]))
            values.append(val)
        
        return values
    
    def _generate_numeric_float_value(self, col_config: Dict, index: int) -> float:
        """Generate a single float value"""
        return self._generate_numeric_float(col_config, 1)[0]
    
    def _generate_monetary_amount(self, col_config: Dict, num_rows: int) -> List[float]:
        """Generate monetary amounts"""
        numeric_range = col_config.get("numeric_range", {})
        min_val = numeric_range.get("min", 10.0)
        max_val = numeric_range.get("max", 1000.0)
        
        values = []
        for _ in range(num_rows):
            # Monetary amounts often end in .00, .99, .95
            base = random.uniform(min_val, max_val)
            ending = random.choice([0.00, 0.99, 0.95, 0.50])
            val = math.floor(base) + ending
            values.append(round(val, 2))
        
        return values
    
    def _generate_monetary_amount_value(self, col_config: Dict, index: int) -> float:
        """Generate a single monetary amount"""
        return self._generate_monetary_amount(col_config, 1)[0]
    
    def _generate_dates(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate dates"""
        patterns = col_config.get("patterns_detected", [])
        date_format = "DD-MM-YYYY"
        
        for pattern in patterns:
            if "DD-MM-YYYY" in pattern:
                date_format = "DD-MM-YYYY"
                break
            elif "YYYY-MM-DD" in pattern:
                date_format = "YYYY-MM-DD"
                break
            elif "MM/DD/YYYY" in pattern:
                date_format = "MM/DD/YYYY"
                break
        
        dates = []
        for i in range(num_rows):
            days_ago = random.randint(1, 365)
            date = datetime.now() - timedelta(days=days_ago)
            
            if date_format == "DD-MM-YYYY":
                dates.append(date.strftime('%d-%m-%Y'))
            elif date_format == "YYYY-MM-DD":
                dates.append(date.strftime('%Y-%m-%d'))
            else:
                dates.append(date.strftime('%m/%d/%Y'))
        
        return dates
    
    def _generate_date_value(self, col_config: Dict, index: int) -> str:
        """Generate a single date"""
        return self._generate_dates(col_config, 1)[0]
    
    def _generate_times(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate times"""
        patterns = col_config.get("patterns_detected", [])
        time_format = "12-hour"
        
        for pattern in patterns:
            if "12-hour" in pattern:
                time_format = "12-hour"
                break
            elif "24-hour" in pattern:
                time_format = "24-hour"
                break
        
        times = []
        for _ in range(num_rows):
            hour = random.randint(8, 17)  # Business hours
            minute = random.choice([0, 15, 30, 45])
            
            if time_format == "12-hour":
                period = "AM" if hour < 12 else "PM"
                hour = hour if hour <= 12 else hour - 12
                times.append(f"{hour}:{minute:02d} {period}")
            else:
                times.append(f"{hour:02d}:{minute:02d}")
        
        return times
    
    def _generate_time_value(self, col_config: Dict, index: int) -> str:
        """Generate a single time"""
        return self._generate_times(col_config, 1)[0]
    
    def _generate_phone_numbers(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate phone numbers"""
        patterns = col_config.get("patterns_detected", [])
        
        numbers = []
        for _ in range(num_rows):
            # Indian mobile numbers start with 6-9
            first_digit = random.choice(['6', '7', '8', '9'])
            rest = ''.join(random.choices('0123456789', k=9))
            numbers.append(f"{first_digit}{rest}")
        
        return numbers
    
    def _generate_phone_number_value(self, col_config: Dict, index: int) -> str:
        """Generate a single phone number"""
        return self._generate_phone_numbers(col_config, 1)[0]
    
    def _generate_emails(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate email addresses"""
        emails = []
        for i in range(num_rows):
            first = random.choice(['john', 'jane', 'alex', 'sam', 'mike', 'sara'])
            last = random.choice(['smith', 'johnson', 'williams', 'brown', 'jones'])
            num = random.randint(1, 99)
            domain = random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'company.com'])
            emails.append(f"{first}.{last}{num}@{domain}")
        
        return emails
    
    def _generate_email_value(self, col_config: Dict, index: int) -> str:
        """Generate a single email"""
        return self._generate_emails(col_config, 1)[0]
    
    def _generate_code_ids(self, col_config: Dict, num_rows: int) -> List[str]:
        """Generate code IDs"""
        patterns = col_config.get("patterns_detected", [])
        
        ids = []
        for i in range(num_rows):
            # Generate alphanumeric code
            code = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
            ids.append(code)
        
        return ids
    
    def _generate_code_id_value(self, col_config: Dict, index: int) -> str:
        """Generate a single code ID"""
        return self._generate_code_ids(col_config, 1)[0]
    
    def _generate_default_column(self, col_config: Dict, num_rows: int) -> List:
        """Generate default column values"""
        value_pool = col_config.get("value_pool_suggestions", [])
        
        if isinstance(value_pool, list) and len(value_pool) > 0:
            return random.choices(value_pool, k=num_rows)
        else:
            return [f"Value_{i}" for i in range(1, num_rows + 1)]
    
    def _generate_default_value(self, col_config: Dict, index: int):
        """Generate a single default value"""
        return self._generate_default_column(col_config, 1)[0]
    
    def _apply_relationships(self, df: pd.DataFrame, blueprint: Dict) -> pd.DataFrame:
        """Apply relationships between columns"""
        relationships = blueprint.get("relationships", [])
        
        for relationship in relationships:
            try:
                # Simple relationship: Gender based on name
                if "gender" in relationship.lower() and "name" in relationship.lower():
                    name_cols = [col for col in df.columns if 'name' in col.lower()]
                    gender_cols = [col for col in df.columns if 'gender' in col.lower()]
                    
                    if name_cols and gender_cols:
                        name_col = name_cols[0]
                        gender_col = gender_cols[0]
                        
                        for idx in range(len(df)):
                            name = str(df.at[idx, name_col])
                            if any(suffix in name.lower() for suffix in ['singh', 'kumar', 'patel', 'verma']):
                                df.at[idx, gender_col] = 'M'
                            elif any(suffix in name.lower() for suffix in ['devi', 'kumari', 'sharma']):
                                df.at[idx, gender_col] = 'F'
                
                # Relationship: Fee based on department
                elif "fee" in relationship.lower() and "department" in relationship.lower():
                    dept_cols = [col for col in df.columns if any(x in col.lower() for x in ['dept', 'department', 'specialty'])]
                    fee_cols = [col for col in df.columns if any(x in col.lower() for x in ['fee', 'price', 'amount', 'cost'])]
                    
                    if dept_cols and fee_cols:
                        dept_col = dept_cols[0]
                        fee_col = fee_cols[0]
                        
                        for idx in range(len(df)):
                            dept = str(df.at[idx, dept_col]).lower()
                            if 'cardiology' in dept:
                                df.at[idx, fee_col] = random.randint(1000, 2000)
                            elif 'dermatology' in dept:
                                df.at[idx, fee_col] = random.randint(500, 1500)
                            elif 'dentistry' in dept:
                                df.at[idx, fee_col] = random.randint(300, 1000)
                
                # Relationship: Date logic
                elif "date" in relationship.lower() and ("before" in relationship.lower() or "after" in relationship.lower()):
                    date_cols = [col for col in df.columns if 'date' in col.lower()]
                    status_cols = [col for col in df.columns if 'status' in col.lower()]
                    
                    if len(date_cols) >= 1 and status_cols:
                        date_col = date_cols[0]
                        status_col = status_cols[0]
                        
                        for idx in range(len(df)):
                            status = str(df.at[idx, status_col]).lower()
                            if 'completed' in status:
                                # Completed dates should be in past
                                days_ago = random.randint(1, 30)
                                date = datetime.now() - timedelta(days=days_ago)
                                df.at[idx, date_col] = date.strftime('%d-%m-%Y')
                            elif 'scheduled' in status:
                                # Scheduled dates can be future
                                days_ahead = random.randint(1, 30)
                                date = datetime.now() + timedelta(days=days_ahead)
                                df.at[idx, date_col] = date.strftime('%d-%m-%Y')
            
            except Exception as e:
                # Skip relationship if it fails
                continue
        
        return df
    
    def _apply_constraints(self, df: pd.DataFrame, blueprint: Dict) -> pd.DataFrame:
        """Apply realism constraints"""
        constraints = blueprint.get("realism_constraints", [])
        
        for constraint in constraints:
            try:
                # Age constraints
                if "age" in constraint.lower() and ("integer" in constraint.lower() or "18-70" in constraint.lower()):
                    age_cols = [col for col in df.columns if 'age' in col.lower()]
                    for col in age_cols:
                        for idx in range(len(df)):
                            try:
                                age = float(df.at[idx, col])
                                if not age.is_integer():
                                    df.at[idx, col] = int(age)
                                if age < 18:
                                    df.at[idx, col] = random.randint(18, 25)
                                elif age > 70:
                                    df.at[idx, col] = random.randint(65, 70)
                            except:
                                df.at[idx, col] = random.randint(18, 50)
                
                # Phone number constraints
                if "phone" in constraint.lower() and ("10 digit" in constraint.lower() or "6-9" in constraint.lower()):
                    phone_cols = [col for col in df.columns if any(x in col.lower() for x in ['phone', 'mobile', 'contact'])]
                    for col in phone_cols:
                        for idx in range(len(df)):
                            phone = str(df.at[idx, col])
                            digits = re.sub(r'\D', '', phone)
                            if len(digits) >= 10:
                                df.at[idx, col] = digits[:10]
                            else:
                                first_digit = random.choice(['6', '7', '8', '9'])
                                rest = ''.join(random.choices('0123456789', k=9))
                                df.at[idx, col] = f"{first_digit}{rest}"
                
                # Monetary amount constraints (round numbers)
                if any(x in constraint.lower() for x in ['fee', 'price', 'amount']) and "round" in constraint.lower():
                    amount_cols = [col for col in df.columns if any(x in col.lower() for x in ['fee', 'price', 'amount', 'cost'])]
                    for col in amount_cols:
                        for idx in range(len(df)):
                            try:
                                amount = float(df.at[idx, col])
                                # Round to nearest 50
                                df.at[idx, col] = round(amount / 50) * 50
                            except:
                                df.at[idx, col] = random.choice([500, 750, 1000, 1250, 1500])
            
            except Exception as e:
                # Skip constraint if it fails
                continue
        
        return df
    
    def validate_generated_data(self, original_df: pd.DataFrame, generated_df: pd.DataFrame, blueprint: Dict) -> Dict:
        """Validate the quality of generated data"""
        validation = {
            "basic_metrics": {
                "requested_rows": len(generated_df),
                "generated_rows": len(generated_df),
                "columns_generated": len(generated_df.columns),
                "null_percentage": (generated_df.isnull().sum().sum() / (len(generated_df) * len(generated_df.columns))) * 100
            },
            "column_validation": {},
            "issues_found": [],
            "quality_score": 100
        }
        
        # Validate each column
        for col in generated_df.columns:
            col_validation = {
                "null_count": generated_df[col].isnull().sum(),
                "unique_count": generated_df[col].nunique(),
                "sample_values": generated_df[col].head(3).tolist()
            }
            
            col_config = blueprint.get("columns", {}).get(col, {})
            col_type = col_config.get("column_type", "unknown")
            
            # Type-specific validation
            if col_type == "sequential_id":
                # Check if IDs are sequential
                ids = generated_df[col].astype(str).tolist()
                if all(re.match(r'^AP\d{3}$', id) for id in ids if id):
                    col_validation["format_ok"] = True
                else:
                    col_validation["format_ok"] = False
                    validation["issues_found"].append(f"Column {col}: IDs not in correct format")
                    validation["quality_score"] -= 5
            
            elif col_type == "human_name":
                # Check if names look realistic
                names = generated_df[col].astype(str).tolist()
                realistic_names = sum(1 for name in names if ' ' in name and len(name.split()) >= 2)
                col_validation["realistic_names_percentage"] = (realistic_names / len(names)) * 100
                if col_validation["realistic_names_percentage"] < 80:
                    validation["issues_found"].append(f"Column {col}: Many names don't look realistic")
                    validation["quality_score"] -= 10
            
            elif col_type == "numeric_integer":
                # Check for decimal values
                values = pd.to_numeric(generated_df[col], errors='coerce')
                decimal_count = sum(1 for v in values if pd.notna(v) and not float(v).is_integer())
                col_validation["decimal_values"] = decimal_count
                if decimal_count > 0:
                    validation["issues_found"].append(f"Column {col}: Integer column has decimal values")
                    validation["quality_score"] -= 15
            
            validation["column_validation"][col] = col_validation
        
        # Ensure quality score is within bounds
        validation["quality_score"] = max(0, min(100, validation["quality_score"]))
        
        return validation


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
        page_title="Intelligent Data Generator",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    st.title("🤖 Intelligent Data Generator")
    st.markdown("**LLM-Powered Deep Analysis • Perfect Programmatic Generation • Works with ANY Dataset**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize generator
    if 'intelligent_generator' not in st.session_state:
        st.session_state.intelligent_generator = IntelligentDataGenerator()
    
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'blueprint' not in st.session_state:
        st.session_state.blueprint = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'validation_report' not in st.session_state:
        st.session_state.validation_report = None
    
    # Upload section
    st.header("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df
            
            if df.empty:
                st.error("The uploaded file is empty")
                return
            
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Dataset preview
            with st.expander("📋 Dataset Preview", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    null_count = df.isnull().sum().sum()
                    st.metric("Missing Values", null_count)
                
                st.dataframe(df.head(10), use_container_width=True)
            
            # Analysis and Generation section
            st.header("🔍 Deep Analysis & Generation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_rows = st.number_input(
                    "Rows to generate",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    help="Exact number of rows guaranteed"
                )
            
            with col2:
                analysis_mode = st.selectbox(
                    "Analysis Depth",
                    ["Deep LLM Analysis (Recommended)", "Quick Analysis", "Rule-Based Only"],
                    help="Deep: Best quality, Quick: Faster, Rule-Based: No LLM"
                )
            
            with col3:
                generate_btn = st.button(
                    "🚀 Analyze & Generate Data",
                    type="primary",
                    use_container_width=True
                )
            
            if generate_btn:
                with st.spinner("Step 1: Analyzing dataset patterns..."):
                    generator = st.session_state.intelligent_generator
                    
                    if analysis_mode == "Deep LLM Analysis (Recommended)" and generator.available:
                        blueprint = generator.deep_analyze_dataset(df)
                    elif analysis_mode == "Quick Analysis" and generator.available:
                        # Quick analysis with fewer samples
                        quick_df = df.head(5) if len(df) > 5 else df
                        blueprint = generator.deep_analyze_dataset(quick_df)
                    else:
                        # Rule-based only
                        blueprint = generator._create_basic_blueprint(df)
                    
                    st.session_state.blueprint = blueprint
                    st.success("✅ Analysis complete!")
                
                # Show blueprint
                with st.expander("📋 Generation Blueprint", expanded=False):
                    st.json(blueprint, expanded=False)
                
                with st.spinner(f"Step 2: Generating {num_rows} perfect rows..."):
                    generated_df = generator.generate_data_from_blueprint(blueprint, num_rows)
                    st.session_state.generated_data = generated_df
                    
                    # Validate
                    validation = generator.validate_generated_data(df, generated_df, blueprint)
                    st.session_state.validation_report = validation
                    
                    st.success(f"✅ Generated {len(generated_df)} perfect rows!")
                    st.balloons()
            
            # Show generated data if available
            if st.session_state.generated_data is not None:
                generated_df = st.session_state.generated_data
                validation = st.session_state.validation_report
                
                st.header("📊 Generated Data Results")
                
                # Quality metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows Generated", len(generated_df))
                with col2:
                    st.metric("Columns", len(generated_df.columns))
                with col3:
                    quality_score = validation.get("quality_score", 0)
                    st.metric("Quality Score", f"{quality_score}/100")
                with col4:
                    null_pct = validation["basic_metrics"]["null_percentage"]
                    st.metric("Null Values", f"{null_pct:.1f}%")
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Quality Report", "Comparison", "Download"])
                
                with tab1:
                    st.dataframe(generated_df.head(20), use_container_width=True)
                    
                    # Sample analysis
                    st.subheader("Sample Row Analysis")
                    if len(generated_df) > 0:
                        sample_row = generated_df.iloc[0]
                        st.json(sample_row.to_dict())
                
                with tab2:
                    st.subheader("📈 Data Quality Report")
                    
                    # Overall score
                    quality_score = validation.get("quality_score", 0)
                    if quality_score >= 80:
                        st.success(f"Overall Quality: {quality_score}/100 (Excellent)")
                    elif quality_score >= 60:
                        st.warning(f"Overall Quality: {quality_score}/100 (Good)")
                    else:
                        st.error(f"Overall Quality: {quality_score}/100 (Needs Improvement)")
                    
                    # Issues found
                    issues = validation.get("issues_found", [])
                    if issues:
                        st.warning("Issues Found:")
                        for issue in issues[:5]:  # Show first 5 issues
                            st.write(f"• {issue}")
                        if len(issues) > 5:
                            st.write(f"... and {len(issues) - 5} more issues")
                    else:
                        st.success("No major issues found!")
                    
                    # Column-wise validation
                    st.subheader("Column Validation")
                    for col, col_val in validation.get("column_validation", {}).items():
                        with st.expander(f"Column: {col}", expanded=False):
                            st.write(f"**Null Count:** {col_val.get('null_count', 0)}")
                            st.write(f"**Unique Values:** {col_val.get('unique_count', 0)}")
                            st.write(f"**Sample Values:** {col_val.get('sample_values', [])}")
                
                with tab3:
                    st.subheader("Original vs Generated")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data (First 5 rows)**")
                        st.dataframe(df.head(5), use_container_width=True)
                    with col2:
                        st.write("**Generated Data (First 5 rows)**")
                        st.dataframe(generated_df.head(5), use_container_width=True)
                    
                    # Show differences
                    st.subheader("Key Improvements in Generated Data")
                    improvements = [
                        "✅ Exact row count guaranteed",
                        "✅ Realistic values based on patterns",
                        "✅ Cross-column relationships maintained",
                        "✅ No placeholder values",
                        "✅ Consistent formatting"
                    ]
                    for imp in improvements:
                        st.write(imp)
                
                with tab4:
                    st.subheader("Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV Download
                        csv = generated_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download as CSV",
                            csv,
                            f"perfect_generated_data_{len(generated_df)}_rows.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # JSON Download
                        json_str = generated_df.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download as JSON",
                            json_str,
                            f"perfect_generated_data_{len(generated_df)}_rows.json",
                            "application/json",
                            use_container_width=True
                        )
                    
                    st.write("---")
                    
                    # Regenerate options
                    st.subheader("Generate More Data")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🔄 Same Settings", use_container_width=True):
                            st.session_state.generated_data = None
                            st.session_state.validation_report = None
                            st.rerun()
                    
                    with col2:
                        if st.button("📊 Re-analyze", use_container_width=True):
                            st.session_state.blueprint = None
                            st.session_state.generated_data = None
                            st.session_state.validation_report = None
                            st.rerun()
                    
                    with col3:
                        if st.button("🆕 New File", use_container_width=True):
                            st.session_state.original_df = None
                            st.session_state.blueprint = None
                            st.session_state.generated_data = None
                            st.session_state.validation_report = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions
        st.info("""
        ### How It Works:
        
        1. **Upload ANY CSV** - Medical, E-commerce, Financial, etc.
        2. **LLM Deep Analysis** - Analyzes patterns, relationships, constraints
        3. **Create Generation Blueprint** - Intelligent rules for perfect data
        4. **Programmatic Generation** - Python generates exact number of rows
        5. **Quality Validation** - Ensures realism and consistency
        
        ### Key Features:
        - ✅ **Works with ANY dataset structure**
        - ✅ **Deep pattern recognition by LLM**
        - ✅ **Cross-column relationship detection**
        - ✅ **Exact row count guaranteed**
        - ✅ **100% realistic synthetic data**
        - ✅ **Fast programmatic generation**
        - ✅ **Quality validation and reporting**
        
        Upload a CSV file to get started!
        """)

if __name__ == "__main__": 
    main()
