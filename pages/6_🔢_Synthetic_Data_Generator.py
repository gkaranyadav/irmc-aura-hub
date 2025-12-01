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
from typing import Dict, List, Any, Optional

# =============================================================================
# UNIVERSAL LLM DATA GENERATOR - FIXED WITH CHUNKED GENERATION
# =============================================================================

class UniversalDataGenerator:
    """Universal generator that adapts to ANY dataset"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
            # Available models - try different ones based on needs
            self.models = {
                "fast": "llama-3.1-8b-instant",  # Fast, good for small data
                "balanced": "llama-3.3-70b-versatile",  # Balanced quality/speed
                "large": "mixtral-8x7b-32768",  # Larger context window
                "best": "llama-3-groq-70b-8192-tool-use-preview"  # Best quality
            }
        except:
            self.available = False
            st.warning("LLM not available")
    
    def analyze_dataset(self, df):
        """Analyze dataset to understand structure and patterns"""
        analysis = {
            "columns": {},
            "types": {},
            "patterns": {},
            "sample_data": {},
            "stats": {}
        }
        
        for col in df.columns:
            # Get sample values
            non_null_vals = df[col].dropna()
            if len(non_null_vals) > 0:
                sample_vals = non_null_vals.head(5).tolist()
                analysis["sample_data"][col] = sample_vals
            
            # Detect data type
            col_type = self._detect_column_type(df[col], col)
            analysis["types"][col] = col_type
            
            # Get patterns based on type
            if col_type == "categorical":
                analysis["patterns"][col] = {
                    "type": "categorical",
                    "values": df[col].dropna().unique().tolist()[:20]  # Limit to 20 unique values
                }
            elif col_type == "numeric":
                if len(non_null_vals) > 0:
                    analysis["patterns"][col] = {
                        "type": "numeric",
                        "min": float(non_null_vals.min()),
                        "max": float(non_null_vals.max()),
                        "mean": float(non_null_vals.mean())
                    }
            elif col_type == "id":
                analysis["patterns"][col] = {
                    "type": "id",
                    "pattern": self._detect_id_pattern(df[col].head(10).tolist())
                }
            elif col_type == "name":
                analysis["patterns"][col] = {
                    "type": "name",
                    "has_middle": any(len(str(x).split()) > 2 for x in non_null_vals.head(5))
                }
            elif col_type == "date":
                analysis["patterns"][col] = {
                    "type": "date",
                    "format": self._detect_date_format(non_null_vals.head(5).tolist())
                }
            elif col_type == "phone":
                analysis["patterns"][col] = {
                    "type": "phone",
                    "pattern": self._detect_phone_pattern(non_null_vals.head(5).tolist())
                }
        
        return analysis
    
    def _detect_column_type(self, series, col_name):
        """Detect the type of a column"""
        col_lower = col_name.lower()
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return "unknown"
        
        # Check for IDs
        if any(x in col_lower for x in ['id', 'code', 'num', 'no', 'number', 'ref', 'appointment']):
            return "id"
        
        # Check for names
        if any(x in col_lower for x in ['name', 'patient', 'doctor', 'person']):
            return "name"
        
        # Check for emails
        if any(x in col_lower for x in ['email', 'mail']):
            return "email"
        
        # Check for phones
        if any(x in col_lower for x in ['phone', 'mobile', 'contact', 'number']):
            return "phone"
        
        # Check for dates
        if any(x in col_lower for x in ['date', 'time', 'appointment']):
            date_patterns = [r'\d{2}[-/]\d{2}[-/]\d{4}', r'\d{4}[-/]\d{2}[-/]\d{2}']
            samples = non_null.head(10).astype(str).tolist()
            if any(any(re.search(pattern, str(x)) for pattern in date_patterns) for x in samples):
                return "date"
        
        # Check for categorical
        unique_ratio = len(non_null.unique()) / len(non_null) if len(non_null) > 0 else 1
        if unique_ratio < 0.3 and len(non_null.unique()) <= 50:
            return "categorical"
        
        # Check for numeric
        try:
            numeric_vals = pd.to_numeric(non_null.head(20), errors='coerce')
            if numeric_vals.notna().sum() > len(numeric_vals) * 0.8:  # 80% are numeric
                return "numeric"
        except:
            pass
        
        # Default to text
        return "text"
    
    def _detect_id_pattern(self, samples):
        """Detect ID pattern"""
        if not samples:
            return "sequential"
        
        first_sample = str(samples[0])
        if first_sample.startswith(('AP', 'ID', 'REF', 'CUST')):
            return "prefixed_sequential"
        elif all(str(x).isdigit() for x in samples if x and str(x).strip()):
            return "numeric_sequential"
        else:
            return "mixed"
    
    def _detect_date_format(self, samples):
        """Detect date format"""
        if not samples:
            return "DD-MM-YYYY"
        
        sample = str(samples[0])
        if re.match(r'\d{2}-\d{2}-\d{4}', sample):
            return "DD-MM-YYYY"
        elif re.match(r'\d{4}-\d{2}-\d{2}', sample):
            return "YYYY-MM-DD"
        elif re.match(r'\d{2}/\d{2}/\d{4}', sample):
            return "DD/MM/YYYY"
        else:
            return "DD-MM-YYYY"
    
    def _detect_phone_pattern(self, samples):
        """Detect phone pattern"""
        if not samples:
            return "10_digit"
        
        sample = str(samples[0])
        if len(sample) == 10 and sample.isdigit():
            return "10_digit"
        elif '+' in sample:
            return "international"
        else:
            return "variable"
    
    def generate_perfect_data(self, original_df, num_rows):
        """Generate PERFECT data that matches ANY dataset structure"""
        if not self.available or original_df.empty:
            return self._smart_fallback(original_df, num_rows)
        
        # Analyze the dataset first
        analysis = self.analyze_dataset(original_df)
        
        # For large datasets, use chunked generation
        if num_rows > 50:
            return self._chunked_generation(original_df, num_rows, analysis)
        
        # For small datasets, use single request
        with st.spinner(f"🤖 LLM is generating {num_rows} perfect synthetic rows..."):
            llm_data = self._get_llm_generated_data(original_df, num_rows, analysis)
        
        if llm_data is not None and len(llm_data) >= num_rows:
            return llm_data.head(num_rows)
        
        # If LLM failed, use enhanced fallback
        st.warning(f"LLM generated only {len(llm_data) if llm_data is not None else 0} rows. Using enhanced fallback...")
        return self._enhanced_fallback(original_df, num_rows, analysis)
    
    def _chunked_generation(self, df, num_rows, analysis):
        """Generate data in chunks to handle large row counts"""
        chunks_needed = math.ceil(num_rows / 30)  # 30 rows per chunk (safe limit)
        chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for chunk_num in range(chunks_needed):
            chunk_rows = min(30, num_rows - len(chunks) * 30)
            status_text.text(f"Generating chunk {chunk_num + 1}/{chunks_needed} ({chunk_rows} rows)...")
            
            # Update start ID for each chunk
            start_id = 1000 + (chunk_num * 30)
            
            # Generate chunk
            chunk_data = self._get_llm_generated_chunk(df, chunk_rows, analysis, start_id, chunk_num)
            
            if chunk_data is not None and len(chunk_data) > 0:
                chunks.append(chunk_data)
            
            # Update progress
            progress_bar.progress((chunk_num + 1) / chunks_needed)
        
        status_text.text("Combining chunks...")
        
        if chunks:
            # Combine all chunks
            combined_df = pd.concat(chunks, ignore_index=True)
            
            # Ensure we have exactly the requested number of rows
            if len(combined_df) < num_rows:
                missing = num_rows - len(combined_df)
                st.info(f"Generated {len(combined_df)} rows. Adding {missing} more...")
                additional_df = self._generate_missing_rows(combined_df, missing, analysis)
                combined_df = pd.concat([combined_df, additional_df], ignore_index=True)
            
            # Trim if too many
            combined_df = combined_df.head(num_rows)
            
            status_text.text(f"✅ Successfully generated {len(combined_df)} rows!")
            progress_bar.empty()
            
            return combined_df
        
        # If chunked generation failed, use enhanced fallback
        progress_bar.empty()
        status_text.text("Chunked generation failed, using fallback...")
        return self._enhanced_fallback(df, num_rows, analysis)
    
    def _get_llm_generated_chunk(self, df, chunk_rows, analysis, start_id, chunk_num):
        """Get LLM to generate a chunk of data"""
        
        # Prepare samples
        samples = []
        for i, (idx, row) in enumerate(df.head(3).iterrows()):
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
        
        # Build prompt for chunk
        prompt = self._build_chunk_prompt(df, chunk_rows, samples, analysis, start_id, chunk_num)
        
        try:
            # Use appropriate model based on chunk size
            model_to_use = self.models["large"] if chunk_rows > 20 else self.models["balanced"]
            
            messages = [
                {"role": "system", "content": """You are a data generation expert. Generate realistic synthetic data.
                
                IMPORTANT RULES:
                1. Generate EXACTLY the requested number of rows
                2. Follow the exact column order and names
                3. Maintain data types and formats
                4. Make all values realistic
                5. Return ONLY JSON array, nothing else"""},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.3,
                max_tokens=4000,  # Reasonable for chunk
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # Parse the response
            return self._parse_llm_response(result, df.columns, chunk_rows, analysis, start_id)
            
        except Exception as e:
            st.warning(f"Chunk generation failed: {str(e)}")
            return None
    
    def _build_chunk_prompt(self, df, chunk_rows, samples, analysis, start_id, chunk_num):
        """Build prompt for chunk generation"""
        
        column_info = []
        for col in df.columns:
            col_type = analysis["types"].get(col, "text")
            samples_list = analysis["sample_data"].get(col, [])
            
            info = f"Column: '{col}' (Type: {col_type})"
            if samples_list:
                info += f" | Samples: {samples_list[:3]}"
            
            if col_type == "id" and "AP" in str(samples_list[0] if samples_list else ""):
                info += " | Pattern: AP### (Appointment IDs starting with AP)"
            elif col_type == "name":
                info += " | Format: Indian names like 'Rahul Patel', 'Priya Sharma'"
            elif col_type == "phone":
                info += " | Format: 10-digit Indian mobile numbers"
            elif col_type == "date":
                info += " | Format: DD-MM-YYYY"
            
            column_info.append(info)
        
        prompt = f"""
        Generate EXACTLY {chunk_rows} rows of synthetic data matching this dataset structure.
        
        DATASET STRUCTURE:
        - Columns: {', '.join(df.columns.tolist())}
        - Total columns: {len(df.columns)}
        
        COLUMN DETAILS:
        {chr(10).join(f'{i+1}. {info}' for i, info in enumerate(column_info))}
        
        SAMPLE DATA (first 3 rows):
        {json.dumps(samples, indent=2, default=str)}
        
        IMPORTANT INSTRUCTIONS for this chunk (Chunk #{chunk_num + 1}):
        1. Generate EXACTLY {chunk_rows} rows, no less
        2. Use the EXACT same column names and order
        3. ID numbers should start from AP{start_id:03d} and continue sequentially
        4. Follow these patterns:
           - IDs: AP{start_id:03d}, AP{start_id+1:03d}, etc.
           - Names: Real Indian names (First Last format)
           - Ages: Realistic ages (18-70)
           - Gender: M or F only
           - Phone: 10-digit numbers (e.g., 9876543210)
           - Department: Medical departments like Cardiology, Neurology, etc.
           - Doctor: Indian doctor names (Dr. First Last)
           - Date: DD-MM-YYYY format, recent dates
           - Time: HH:MM AM/PM format
           - Diagnosis: Real medical conditions
           - Fee: Reasonable fees (500-1500)
           - Status: 'Completed' or similar statuses
        
        OUTPUT FORMAT: Return ONLY a JSON object with key "data" containing the array:
        {{
            "data": [
                {{
                    "{df.columns[0]}": "AP{start_id:03d}",
                    "{df.columns[1]}": "Amit Kumar",
                    ...
                }},
                {{
                    "{df.columns[0]}": "AP{start_id+1:03d}",
                    "{df.columns[1]}": "Priya Sharma",
                    ...
                }},
                ...
            ]
        }}
        
        Generate EXACTLY {chunk_rows} rows. DO NOT include any other text.
        """
        
        return prompt
    
    def _get_llm_generated_data(self, df, num_rows, analysis):
        """Get LLM to generate data for ANY dataset (single request)"""
        
        # Prepare samples
        samples = []
        for i, (idx, row) in enumerate(df.head(3).iterrows()):
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
        
        # Build dynamic prompt
        prompt = self._build_single_prompt(df, num_rows, samples, analysis)
        
        try:
            # Choose model based on size
            model_to_use = self.models["large"] if num_rows > 20 else self.models["balanced"]
            
            messages = [
                {"role": "system", "content": """You are a data generation expert. Generate realistic synthetic data.
                
                IMPORTANT RULES:
                1. Generate EXACTLY the requested number of rows
                2. Follow the exact column order and names
                3. Maintain data types and formats
                4. Make all values realistic
                5. Return ONLY JSON array, nothing else"""},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.3,
                max_tokens=8000,
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            
            # Parse the response
            return self._parse_llm_response(result, df.columns, num_rows, analysis)
            
        except Exception as e:
            st.error(f"LLM generation failed: {str(e)}")
            return None
    
    def _build_single_prompt(self, df, num_rows, samples, analysis):
        """Build prompt for single request generation"""
        
        column_info = []
        for col in df.columns:
            col_type = analysis["types"].get(col, "text")
            samples_list = analysis["sample_data"].get(col, [])
            
            info = f"Column: '{col}' (Type: {col_type})"
            if samples_list:
                info += f" | Samples: {samples_list[:3]}"
            
            if col_type == "id" and "AP" in str(samples_list[0] if samples_list else ""):
                info += " | Pattern: AP### (Appointment IDs starting with AP)"
            elif col_type == "name":
                info += " | Format: Indian names like 'Rahul Patel', 'Priya Sharma'"
            elif col_type == "phone":
                info += " | Format: 10-digit Indian mobile numbers"
            elif col_type == "date":
                info += " | Format: DD-MM-YYYY"
            
            column_info.append(info)
        
        prompt = f"""
        Generate EXACTLY {num_rows} rows of synthetic data matching this dataset structure.
        
        DATASET STRUCTURE:
        - Columns: {', '.join(df.columns.tolist())}
        - Total columns: {len(df.columns)}
        
        COLUMN DETAILS:
        {chr(10).join(f'{i+1}. {info}' for i, info in enumerate(column_info))}
        
        SAMPLE DATA (first 3 rows):
        {json.dumps(samples, indent=2, default=str)}
        
        IMPORTANT INSTRUCTIONS:
        1. Generate EXACTLY {num_rows} rows, no less
        2. Use the EXACT same column names and order
        3. Follow these specific patterns:
           - IDs: Continue from AP025 -> AP026, AP027, etc.
           - Names: Real Indian names (First Last format)
           - Ages: Realistic ages (18-70)
           - Gender: M or F only
           - Phone: 10-digit numbers (e.g., 9876543210)
           - Department: Medical departments like Cardiology, Neurology, etc.
           - Doctor: Indian doctor names (Dr. First Last)
           - Date: DD-MM-YYYY format, recent dates
           - Time: HH:MM AM/PM format
           - Diagnosis: Real medical conditions
           - Fee: Reasonable fees (500-1500)
           - Status: 'Completed' or similar statuses
        
        4. Make data realistic and consistent
        5. Ensure all {num_rows} rows are generated
        
        OUTPUT FORMAT: Return ONLY a JSON object with key "data" containing the array:
        {{
            "data": [
                {{
                    "{df.columns[0]}": "AP026",
                    "{df.columns[1]}": "Amit Kumar",
                    ...
                }},
                ...
            ]
        }}
        
        Generate EXACTLY {num_rows} rows. DO NOT include any other text.
        """
        
        return prompt
    
    def _parse_llm_response(self, response, expected_columns, num_rows, analysis, start_id=1000):
        """Parse LLM's response"""
        try:
            # First try to parse as JSON
            data = json.loads(response)
            
            # Check if data is in "data" key
            if "data" in data and isinstance(data["data"], list):
                rows = data["data"]
            elif isinstance(data, list):
                rows = data
            else:
                # Try to find any array in the response
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
                if json_match:
                    rows = json.loads(json_match.group())
                else:
                    return None
            
            # Convert to DataFrame
            if rows:
                df = pd.DataFrame(rows)
                
                # Ensure we have all columns
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = None
                
                # Reorder columns
                df = df[expected_columns]
                
                # Apply validation
                df = self._validate_data(df, analysis)
                
                # Fix IDs if needed
                df = self._fix_ids(df, analysis, start_id)
                
                return df.head(num_rows)  # Ensure exact number
            
        except Exception as e:
            st.warning(f"JSON parsing failed: {str(e)}")
        
        return None
    
    def _validate_data(self, df, analysis):
        """Validate and clean generated data"""
        for col in df.columns:
            col_type = analysis["types"].get(col, "text")
            
            # Clean column
            df[col] = df[col].apply(lambda x: self._clean_value(x, col_type, col))
        
        return df
    
    def _fix_ids(self, df, analysis, start_id=1000):
        """Fix ID sequences"""
        for col in df.columns:
            col_type = analysis["types"].get(col, "text")
            if col_type == "id":
                # Check if IDs need fixing
                for idx in range(len(df)):
                    current_id = str(df.at[idx, col])
                    if 'AP' in current_id:
                        # Extract and fix number
                        num_match = re.search(r'\d+', current_id)
                        if num_match:
                            expected_num = start_id + idx
                            df.at[idx, col] = f"AP{expected_num:03d}"
                        else:
                            df.at[idx, col] = f"AP{start_id + idx:03d}"
        return df
    
    def _clean_value(self, value, col_type, col_name):
        """Clean a single value based on column type"""
        if pd.isna(value):
            return self._generate_default_value(col_type, col_name)
        
        str_val = str(value).strip()
        
        if col_type == "id":
            # Clean IDs
            if col_name.lower().startswith('appointment') or 'AP' in str_val:
                # Extract number part
                num_match = re.search(r'\d+', str_val)
                if num_match:
                    return f"AP{num_match.group().zfill(3)}"
                return f"AP{random.randint(26, 999):03d}"
            return str_val
        
        elif col_type == "name":
            # Clean names
            if any(x in str_val.lower() for x in ['user', 'test', 'dummy', 'null']):
                return self._generate_indian_name()
            return str_val.title()
        
        elif col_type == "numeric":
            # Clean numbers
            try:
                # Remove non-numeric characters except decimal point
                clean = re.sub(r'[^\d.-]', '', str_val)
                return float(clean)
            except:
                return random.randint(1, 100)
        
        elif col_type == "phone":
            # Clean phone numbers
            digits = re.sub(r'\D', '', str_val)
            if len(digits) >= 10:
                return digits[:10]
            return ''.join(random.choices('0123456789', k=10))
        
        elif col_type == "date":
            # Clean dates
            if re.match(r'\d{2}[-/]\d{2}[-/]\d{4}', str_val):
                return str_val
            # Generate recent date
            days_ago = random.randint(1, 365)
            date = datetime.now() - timedelta(days=days_ago)
            return date.strftime('%d-%m-%Y')
        
        return str_val
    
    def _generate_default_value(self, col_type, col_name):
        """Generate default value for a column type"""
        if col_type == "id":
            if 'appointment' in col_name.lower() or 'AP' in col_name:
                return f"AP{random.randint(100, 999):03d}"
            return f"ID{random.randint(1000, 9999)}"
        
        elif col_type == "name":
            return self._generate_indian_name()
        
        elif col_type == "numeric":
            return random.randint(1, 1000)
        
        elif col_type == "phone":
            return ''.join(random.choices('9876543210', k=10))
        
        elif col_type == "date":
            days_ago = random.randint(1, 365)
            date = datetime.now() - timedelta(days=days_ago)
            return date.strftime('%d-%m-%Y')
        
        elif col_type == "gender":
            return random.choice(['M', 'F'])
        
        return "N/A"
    
    def _generate_indian_name(self):
        """Generate realistic Indian name"""
        first_names = ['Rahul', 'Amit', 'Raj', 'Sanjay', 'Vikram', 'Karan', 'Sachin', 'Saurabh', 
                      'Priya', 'Neha', 'Sonia', 'Sneha', 'Kavita', 'Pooja', 'Surbhi', 'Anjali']
        last_names = ['Patel', 'Sharma', 'Singh', 'Kumar', 'Gupta', 'Jain', 'Reddy', 'Iyer', 'Nair']
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_missing_rows(self, existing_df, num_rows, analysis):
        """Generate additional rows when LLM doesn't provide enough"""
        generated = {}
        
        for col in existing_df.columns:
            col_type = analysis["types"].get(col, "text")
            existing_vals = existing_df[col].dropna().tolist()
            
            if col_type == "id" and existing_vals:
                # Continue ID sequence
                last_id = existing_vals[-1]
                num_match = re.search(r'\d+', str(last_id))
                if num_match:
                    start_num = int(num_match.group()) + 1
                    generated[col] = [f"AP{start_num + i:03d}" for i in range(num_rows)]
                else:
                    generated[col] = [f"AP{100 + i:03d}" for i in range(num_rows)]
            
            elif col_type == "name":
                generated[col] = [self._generate_indian_name() for _ in range(num_rows)]
            
            elif col_type == "numeric" and existing_vals:
                # Generate similar numbers
                nums = [float(x) for x in existing_vals if isinstance(x, (int, float, np.number))]
                if nums:
                    min_val = min(nums)
                    max_val = max(nums)
                    generated[col] = [random.uniform(min_val, max_val) for _ in range(num_rows)]
                else:
                    generated[col] = [random.randint(1, 1000) for _ in range(num_rows)]
            
            elif col_type == "phone":
                generated[col] = [''.join(random.choices('9876543210', k=10)) for _ in range(num_rows)]
            
            elif col_type == "date":
                generated[col] = [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%d-%m-%Y') 
                                for _ in range(num_rows)]
            
            elif col_type == "gender":
                generated[col] = random.choices(['M', 'F'], k=num_rows)
            
            elif col_type == "categorical" and existing_vals:
                # Use existing categories
                generated[col] = random.choices(existing_vals, k=num_rows)
            
            else:
                # Generate based on samples
                if existing_vals:
                    generated[col] = random.choices(existing_vals, k=num_rows)
                else:
                    generated[col] = [f"Value_{i}" for i in range(num_rows)]
        
        return pd.DataFrame(generated)
    
    def _enhanced_fallback(self, df, num_rows, analysis):
        """Enhanced fallback generation"""
        generated = {}
        
        for col in df.columns:
            col_type = analysis["types"].get(col, "text")
            existing_vals = df[col].dropna().tolist()
            
            if col_type == "id":
                # Generate sequential IDs
                if existing_vals and any('AP' in str(x) for x in existing_vals[:5]):
                    # Extract max number from AP IDs
                    max_num = 25
                    for val in existing_vals:
                        if isinstance(val, str) and 'AP' in val:
                            num_match = re.search(r'\d+', val)
                            if num_match:
                                num = int(num_match.group())
                                if num > max_num:
                                    max_num = num
                    generated[col] = [f"AP{i:03d}" for i in range(max_num + 1, max_num + num_rows + 1)]
                else:
                    generated[col] = [f"ID{1000 + i}" for i in range(num_rows)]
            
            elif col_type == "name":
                generated[col] = [self._generate_indian_name() for _ in range(num_rows)]
            
            elif col_type == "numeric":
                if existing_vals:
                    nums = [float(x) for x in existing_vals if isinstance(x, (int, float, np.number))]
                    if nums:
                        min_val = min(nums) * 0.8
                        max_val = max(nums) * 1.2
                        generated[col] = [round(random.uniform(min_val, max_val), 2) for _ in range(num_rows)]
                    else:
                        generated[col] = [random.randint(500, 1500) for _ in range(num_rows)]
                else:
                    generated[col] = [random.randint(1, 1000) for _ in range(num_rows)]
            
            elif col_type == "gender":
                generated[col] = random.choices(['M', 'F'], k=num_rows, weights=[0.55, 0.45])
            
            elif col_type == "phone":
                generated[col] = [''.join(random.choices('9876543210', k=10)) for _ in range(num_rows)]
            
            elif col_type == "date":
                generated[col] = [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%d-%m-%Y') 
                                for _ in range(num_rows)]
            
            elif col_type == "time":
                times = []
                for _ in range(num_rows):
                    hour = random.randint(8, 17)  # 8 AM to 5 PM
                    minute = random.choice([0, 15, 30, 45])
                    period = "AM" if hour < 12 else "PM"
                    hour = hour if hour <= 12 else hour - 12
                    times.append(f"{hour}:{minute:02d} {period}")
                generated[col] = times
            
            elif col_type == "categorical" and existing_vals:
                # Use weighted sampling if we can detect patterns
                unique_vals = list(set(existing_vals))
                if len(unique_vals) <= 10:
                    # Count frequencies for weighted sampling
                    from collections import Counter
                    counts = Counter(existing_vals)
                    total = sum(counts.values())
                    weights = [counts[val]/total for val in unique_vals]
                    generated[col] = random.choices(unique_vals, weights=weights, k=num_rows)
                else:
                    generated[col] = random.choices(existing_vals, k=num_rows)
            
            else:
                # Text or unknown
                if existing_vals:
                    generated[col] = random.choices(existing_vals, k=num_rows)
                else:
                    generated[col] = [f"Data_{i}" for i in range(num_rows)]
        
        return pd.DataFrame(generated)
    
    def _smart_fallback(self, df, num_rows):
        """Original smart fallback for compatibility"""
        return self._enhanced_fallback(df, num_rows, self.analyze_dataset(df))


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
    st.title("✨ Universal Data Generator")
    st.markdown("**LLM-Powered • Works with ANY Dataset • Guaranteed Row Count**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize generator
    if 'universal_generator' not in st.session_state:
        st.session_state.universal_generator = UniversalDataGenerator()
    
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    if 'data_analysis' not in st.session_state:
        st.session_state.data_analysis = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload ANY Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df
            
            if df.empty:
                st.error("Empty file")
            else:
                st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
                
                # Show dataset info
                with st.expander("📋 Dataset Information", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Total Columns", len(df.columns))
                    with col3:
                        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
                    
                    st.write("**Columns:**")
                    cols_per_row = 4
                    cols = df.columns.tolist()
                    for i in range(0, len(cols), cols_per_row):
                        col_group = cols[i:i+cols_per_row]
                        col_metrics = st.columns(cols_per_row)
                        for j, col in enumerate(col_group):
                            with col_metrics[j]:
                                st.code(col)
                                st.caption(f"Type: {df[col].dtype}")
                                st.caption(f"Unique: {df[col].nunique()}")
                
                # Preview
                with st.expander("👀 Data Preview", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Generation controls
                st.subheader("⚙️ Generate Synthetic Data")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_rows = st.number_input("Rows to generate", 
                                             min_value=10, 
                                             max_value=1000,
                                             value=100,
                                             help="Guaranteed to generate exactly this many rows")
                
                with col2:
                    generation_mode = st.selectbox(
                        "Generation Mode",
                        ["Smart LLM (Chunked)", "Enhanced Fallback", "Fast Basic"],
                        help="LLM: Best quality (chunked for large datasets) | Fallback: Good quality | Basic: Fastest"
                    )
                
                with col3:
                    if st.button("🚀 Generate Data", type="primary", use_container_width=True):
                        with st.spinner(f"Generating {num_rows} rows..."):
                            generator = st.session_state.universal_generator
                            
                            # Analyze dataset if needed
                            if st.session_state.data_analysis is None:
                                st.session_state.data_analysis = generator.analyze_dataset(df)
                            
                            # Generate based on mode
                            if generation_mode == "Smart LLM (Chunked)" and generator.available:
                                generated = generator.generate_perfect_data(df, int(num_rows))
                            elif generation_mode == "Enhanced Fallback":
                                analysis = st.session_state.data_analysis
                                generated = generator._enhanced_fallback(df, int(num_rows), analysis)
                            else:
                                # Fast basic
                                generated = generator._smart_fallback(df, int(num_rows))
                            
                            # Ensure we have exactly the requested number of rows
                            if generated is not None:
                                if len(generated) < num_rows:
                                    st.warning(f"Generated only {len(generated)} rows. Generating more...")
                                    # Generate additional rows
                                    analysis = st.session_state.data_analysis
                                    additional = generator._generate_missing_rows(
                                        generated, 
                                        num_rows - len(generated), 
                                        analysis
                                    )
                                    generated = pd.concat([generated, additional], ignore_index=True)
                                
                                # Trim if too many
                                generated = generated.head(num_rows)
                                
                                st.session_state.generated_data = generated
                                st.success(f"✅ Successfully generated {len(generated)} rows!")
                                st.balloons()
                            else:
                                st.error("Failed to generate data")
                
                # Show generated data
                if st.session_state.generated_data is not None:
                    df_gen = st.session_state.generated_data
                    
                    st.subheader(f"📊 Generated Data ({len(df_gen)} rows)")
                    
                    # Tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Quality Check", "Download"])
                    
                    with tab1:
                        # Show data with pagination
                        page_size = 20
                        total_pages = max(1, (len(df_gen) + page_size - 1) // page_size)
                        
                        page = st.number_input("Page", 1, total_pages, 1)
                        start_idx = (page - 1) * page_size
                        end_idx = min(start_idx + page_size, len(df_gen))
                        
                        st.dataframe(df_gen.iloc[start_idx:end_idx], use_container_width=True)
                        st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {len(df_gen)}")
                    
                    with tab2:
                        # Statistics
                        st.write("### Dataset Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(df_gen))
                            st.metric("Total Columns", len(df_gen.columns))
                        
                        with col2:
                            null_count = df_gen.isnull().sum().sum()
                            null_percent = (null_count / (len(df_gen) * len(df_gen.columns))) * 100
                            st.metric("Null Values", f"{null_count:,}")
                            st.metric("Null %", f"{null_percent:.1f}%")
                        
                        with col3:
                            duplicate_rows = df_gen.duplicated().sum()
                            duplicate_percent = (duplicate_rows / len(df_gen)) * 100
                            st.metric("Duplicate Rows", duplicate_rows)
                            st.metric("Duplicate %", f"{duplicate_percent:.1f}%")
                        
                        # Column-wise stats
                        st.write("### Column Statistics")
                        stats_df = pd.DataFrame({
                            'Column': df_gen.columns,
                            'Type': [str(df_gen[col].dtype) for col in df_gen.columns],
                            'Unique Values': [df_gen[col].nunique() for col in df_gen.columns],
                            'Null Count': [df_gen[col].isnull().sum() for col in df_gen.columns],
                            'Sample Value': [df_gen[col].iloc[0] if len(df_gen) > 0 else 'N/A' for col in df_gen.columns]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                    
                    with tab3:
                        # Quality metrics
                        st.write("### Data Quality Report")
                        
                        quality_scores = []
                        for col in df_gen.columns:
                            score = 100
                            issues = []
                            
                            # Check nulls
                            null_pct = (df_gen[col].isnull().sum() / len(df_gen)) * 100
                            if null_pct > 10:
                                score -= 20
                                issues.append(f"{null_pct:.1f}% nulls")
                            
                            # Check uniqueness
                            unique_pct = (df_gen[col].nunique() / len(df_gen)) * 100
                            if unique_pct < 5 and len(df_gen) > 20:
                                score -= 10
                                issues.append("Low diversity")
                            
                            # Check for placeholders
                            if df_gen[col].dtype == 'object':
                                sample = str(df_gen[col].iloc[0]) if len(df_gen) > 0 else ""
                                if any(x in sample.lower() for x in ['test', 'dummy', 'temp', 'null', 'na', 'none']):
                                    score -= 15
                                    issues.append("Contains placeholders")
                            
                            quality_scores.append({
                                'Column': col,
                                'Quality Score': f"{score:.0f}/100",
                                'Issues': ', '.join(issues) if issues else 'Good',
                                'Status': '✅ Good' if score >= 80 else '⚠️ Needs Review' if score >= 60 else '❌ Poor'
                            })
                        
                        quality_df = pd.DataFrame(quality_scores)
                        st.dataframe(quality_df, use_container_width=True)
                        
                        # Overall score
                        avg_score = sum([int(x['Quality Score'].split('/')[0]) for x in quality_scores]) / len(quality_scores)
                        st.metric("Overall Quality Score", f"{avg_score:.1f}/100")
                    
                    with tab4:
                        # Download options
                        st.write("### Download Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV Download
                            csv = df_gen.to_csv(index=False)
                            st.download_button(
                                "📥 Download as CSV",
                                csv,
                                f"synthetic_data_{len(df_gen)}_rows.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # JSON Download
                            json_str = df_gen.to_json(orient='records', indent=2)
                            st.download_button(
                                "📥 Download as JSON",
                                json_str,
                                f"synthetic_data_{len(df_gen)}_rows.json",
                                "application/json",
                                use_container_width=True
                            )
                        
                        st.write("---")
                        
                        # Regenerate options
                        st.write("### Generate More Data")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("🔄 Same Settings", use_container_width=True):
                                st.session_state.generated_data = None
                                st.rerun()
                        
                        with col2:
                            if st.button("📊 New Analysis", use_container_width=True):
                                st.session_state.generated_data = None
                                st.session_state.data_analysis = None
                                st.rerun()
                        
                        with col3:
                            if st.button("🆕 New File", use_container_width=True):
                                st.session_state.generated_data = None
                                st.session_state.data_analysis = None
                                st.session_state.original_df = None
                                st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

if __name__ == "__main__": 
    main()
