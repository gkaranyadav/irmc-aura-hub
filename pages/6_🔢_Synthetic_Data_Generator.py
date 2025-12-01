# pages/6_🔢_Synthetic_Data_Generator.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import random
from datetime import datetime
from groq import Groq
from auth import check_session

# =============================================================================
# PURE LLM DATA GENERATOR - NO FALLBACK, NO PREDEFINED RULES
# =============================================================================

class PureLLMGenerator:
    """100% LLM-based generator - LLM does EVERYTHING"""
    
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            self.available = True
        except:
            self.available = False
            st.error("❌ LLM not available. Check API key in secrets.")
    
    def generate_perfect_data(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """
        Pure LLM generation - LLM analyzes AND generates perfect data
        """
        if not self.available or df.empty:
            st.error("LLM not available or empty dataset")
            return None
        
        # Get comprehensive samples
        samples = self._get_rich_samples(df)
        
        # Build ULTIMATE prompt
        prompt = self._build_ultimate_prompt(df, samples, num_rows)
        
        with st.spinner("🤖 LLM is DEEPLY analyzing and generating PERFECT data..."):
            try:
                messages = [
                    {
                        "role": "system", 
                        "content": """You are the ULTIMATE data analysis and generation expert. 
                        You analyze ANY dataset with DEEP understanding and generate PERFECT synthetic data.
                        
                        CRITICAL INSTRUCTIONS:
                        1. FIRST, analyze the dataset COMPLETELY - understand EVERYTHING about it
                        2. Look at ALL patterns, relationships, constraints, domain knowledge
                        3. Understand cultural context, regional patterns, business logic
                        4. THEN generate synthetic data that is 100% realistic and logical
                        5. Every single value MUST make real-world sense
                        6. Maintain ALL relationships between columns
                        7. Generate EXACTLY the requested number of rows
                        
                        Your generated data should be INDISTINGUISHABLE from real data."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.1,  # Low temp for consistency
                    max_tokens=16000,  # More tokens for large generation
                    response_format={"type": "json_object"}
                )
                
                result = response.choices[0].message.content
                
                # Parse and validate
                generated_df = self._parse_llm_generation(result, df.columns, num_rows)
                
                if generated_df is not None and len(generated_df) == num_rows:
                    return generated_df
                else:
                    st.error(f"LLM generated {len(generated_df) if generated_df else 0} rows, expected {num_rows}")
                    return None
                
            except Exception as e:
                st.error(f"LLM generation failed: {str(e)}")
                return None
    
    def _get_rich_samples(self, df: pd.DataFrame) -> List[Dict]:
        """Get rich samples showing full diversity"""
        samples = []
        
        # Get samples from different parts of the dataset
        sample_indices = []
        
        # First 5 rows
        sample_indices.extend(range(min(5, len(df))))
        
        # Middle rows if dataset is large
        if len(df) > 10:
            mid = len(df) // 2
            sample_indices.extend([mid-1, mid, mid+1])
        
        # Last rows
        if len(df) > 8:
            sample_indices.extend([-3, -2, -1])
        
        # Remove duplicates and ensure valid indices
        sample_indices = sorted(set(idx for idx in sample_indices if 0 <= idx < len(df)))
        
        # Take up to 10 samples
        for idx in sample_indices[:10]:
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
                    sample[col] = str(val)
            samples.append(sample)
        
        return samples
    
    def _build_ultimate_prompt(self, df: pd.DataFrame, samples: List[Dict], num_rows: int) -> str:
        """Build the ULTIMATE prompt for deep analysis and generation"""
        
        # Calculate basic stats
        column_stats = []
        for col in df.columns:
            non_null = df[col].dropna()
            stats = f"**{col}**"
            
            # Add data type
            stats += f" | Type: {df[col].dtype}"
            
            # Add unique count
            stats += f" | Unique: {len(non_null.unique())}/{len(non_null)}"
            
            # Add sample values
            sample_vals = non_null.head(3).tolist()
            if sample_vals:
                stats += f" | Samples: {sample_vals}"
            
            column_stats.append(stats)
        
        prompt = f"""
        # ULTIMATE DATA GENERATION TASK
        
        ## DATASET TO ANALYZE AND REPLICATE:
        
        **BASIC INFO:**
        - Rows in original: {len(df)}
        - Columns: {len(df.columns)}
        - Column names: {', '.join(df.columns)}
        
        **COLUMN STATISTICS:**
        {chr(10).join(column_stats)}
        
        **COMPREHENSIVE SAMPLE DATA (showing full diversity):**
        ```json
        {json.dumps(samples, indent=2, default=str)}
        ```
        
        ## YOUR MISSION:
        
        ### PHASE 1: DEEP ANALYSIS
        FIRST, analyze this dataset DEEPLY like ChatGPT would:
        
        1. **WHAT is this data?**
           - What domain/industry? (Medical, E-commerce, Financial, etc.)
           - What country/region? (India, USA, etc.)
           - What specific purpose does it serve?
        
        2. **UNDERSTAND EACH COLUMN:**
           For EACH column ({', '.join(df.columns)}):
           - What does this column represent in REAL WORLD?
           - What patterns/formats/ranges does it have?
           - What are realistic values for this column?
           - What constraints exist?
        
        3. **UNDERSTAND RELATIONSHIPS:**
           - How are columns related to each other?
           - What business rules/logic exists?
           - What would make data look FAKE? (avoid these!)
        
        4. **DETECT SPECIFICS:**
           - Name patterns (Indian/Western/Other)
           - Phone number formats
           - Date/time formats
           - ID patterns
           - Amount patterns
           - Status values
           - Department categories
           - Diagnosis/Product categories
        
        ### PHASE 2: PERFECT GENERATION
        Generate **EXACTLY {num_rows} rows** of synthetic data that:
        
        1. **IS 100% REALISTIC** - Every value makes real-world sense
        2. **MAINTAINS ALL PATTERNS** from original data
        3. **RESPECTS ALL RELATIONSHIPS** between columns
        4. **FOLLOWS DOMAIN KNOWLEDGE** (medical logic, e-commerce logic, etc.)
        5. **HAS NO PLACEHOLDER VALUES** (no Value_1, Category_1, etc.)
        6. **IS CULTURALLY APPROPRIATE** (names, phones, formats match region)
        
        ### CRITICAL RULES FOR YOUR GENERATION:
        
        **FOR MEDICAL DATA (if this is medical):**
        - Names: Real Indian names if Indian data, proper gender alignment
        - Gender: Must match name patterns (Singh/Kumar → M, Devi/Kumari → F)
        - Age: Realistic ages (18-70 for adults), integers only
        - Phone: Valid Indian mobile numbers (10 digits, starts 6-9)
        - Department: Real medical departments (Cardiology, Neurology, etc.)
        - Doctor: Real doctor names with "Dr." prefix
        - Date: Recent dates in DD-MM-YYYY format
        - Time: Business hours (9AM-5PM) in proper format
        - Diagnosis: Must match department (ENT → ear/nose/throat issues)
        - Fee: Realistic medical fees (500-2000), often round numbers
        - Status: Real statuses (Completed, Scheduled, Cancelled)
        
        **FOR E-COMMERCE DATA (if this is e-commerce):**
        - Product names: Real products with brands
        - Prices: Realistic prices for product types
        - Customers: Real names and emails
        - Dates: Logical sequence (order before delivery)
        
        **GENERAL RULES (for any data):**
        - NO placeholder text (Value_1, Category_1, etc.)
        - NO unrealistic values (120-year-olds, $10 iPhones, etc.)
        - NO relationship violations (gender mismatches, etc.)
        - EVERY value must be logical and realistic
        
        ## OUTPUT FORMAT:
        
        Return a JSON object with this EXACT structure:
        ```json
        {{
            "analysis_summary": {{
                "dataset_type": "medical_appointments_india",
                "domain_description": "Patient appointment records in Indian hospital/clinic",
                "key_insights": [
                    "Appointment IDs follow AP### pattern, last is AP025",
                    "Indian names with gender indicators in surnames",
                    "Indian mobile numbers (10 digits, start 6-9)",
                    "Medical departments with appropriate diagnoses",
                    "Recent appointment dates in DD-MM-YYYY format"
                ],
                "generation_rules_used": [
                    "Gender assigned based on name patterns",
                    "Diagnosis matched to department",
                    "Phone numbers validated as Indian mobile",
                    "Ages kept realistic (18-70)"
                ]
            }},
            
            "generated_data": [
                {{
                    "{df.columns[0] if len(df.columns) > 0 else 'col1'}": "AP026",
                    "{df.columns[1] if len(df.columns) > 1 else 'col2'}": "Rahul Patel",
                    "{df.columns[2] if len(df.columns) > 2 else 'col3'}": 32,
                    "{df.columns[3] if len(df.columns) > 3 else 'col4'}": "M",
                    "{df.columns[4] if len(df.columns) > 4 else 'col5'}": "9876543210",
                    "{df.columns[5] if len(df.columns) > 5 else 'col6'}": "Cardiology",
                    "{df.columns[6] if len(df.columns) > 6 else 'col7'}": "Dr. S. Jain",
                    "{df.columns[7] if len(df.columns) > 7 else 'col8'}": "15-01-2024",
                    "{df.columns[8] if len(df.columns) > 8 else 'col9'}": "2:15 PM",
                    "{df.columns[9] if len(df.columns) > 9 else 'col10'}": "Chest Pain",
                    "{df.columns[10] if len(df.columns) > 10 else 'col11'}": 1200,
                    "{df.columns[11] if len(df.columns) > 11 else 'col12'}": "Completed"
                }},
                {{
                    "{df.columns[0] if len(df.columns) > 0 else 'col1'}": "AP027",
                    "{df.columns[1] if len(df.columns) > 1 else 'col2'}": "Priya Sharma",
                    "{df.columns[2] if len(df.columns) > 2 else 'col3'}": 28,
                    "{df.columns[3] if len(df.columns) > 3 else 'col4'}": "F",
                    "{df.columns[4] if len(df.columns) > 4 else 'col5'}": "9876543211",
                    "{df.columns[5] if len(df.columns) > 5 else 'col6'}": "Dermatology",
                    "{df.columns[6] if len(df.columns) > 6 else 'col7'}": "Dr. A. Kumar",
                    "{df.columns[7] if len(df.columns) > 7 else 'col8'}": "16-01-2024",
                    "{df.columns[8] if len(df.columns) > 8 else 'col9'}": "10:00 AM",
                    "{df.columns[9] if len(df.columns) > 9 else 'col10'}": "Skin Allergy",
                    "{df.columns[10] if len(df.columns) > 10 else 'col11'}": 800,
                    "{df.columns[11] if len(df.columns) > 11 else 'col12'}": "Completed"
                }},
                // ... Generate EXACTLY {num_rows} rows in total
                // EVERY row must be 100% realistic and follow ALL rules above
            ]
        }}
        ```
        
        ## FINAL INSTRUCTIONS:
        1. **ANALYZE DEEPLY FIRST** - understand EVERYTHING about the data
        2. **GENERATE EXACTLY {num_rows} rows** - no more, no less
        3. **USE PROPER COLUMN NAMES** - same as original: {', '.join(df.columns)}
        4. **EVERY VALUE MUST BE REALISTIC** - no placeholders, no nonsense
        5. **MAINTAIN ALL RELATIONSHIPS** - gender matches name, diagnosis matches department, etc.
        6. **MAKE DATA PERFECT** - indistinguishable from real data
        
        Remember: You are generating data that will be used in REAL applications.
        If you generate nonsense data, real businesses will fail.
        Be PERFECT.
        """
        
        return prompt
    
    def _parse_llm_generation(self, result: str, expected_columns: List[str], expected_rows: int) -> pd.DataFrame:
        """Parse LLM's generated data"""
        try:
            data = json.loads(result)
            
            # Check if generated_data exists
            if "generated_data" not in data:
                st.error("LLM didn't return generated_data in expected format")
                return None
            
            generated_rows = data["generated_data"]
            
            if not isinstance(generated_rows, list):
                st.error("Generated data is not a list")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(generated_rows)
            
            # Ensure we have all expected columns
            missing_cols = set(expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_columns)
            
            if missing_cols:
                st.warning(f"Missing columns in generated data: {missing_cols}")
                for col in missing_cols:
                    df[col] = None
            
            if extra_cols:
                st.warning(f"Extra columns in generated data: {extra_cols}")
            
            # Reorder columns to match original
            df = df[expected_columns]
            
            # Validate row count
            if len(df) != expected_rows:
                st.warning(f"Generated {len(df)} rows, expected {expected_rows}")
                # If we got more rows, truncate; if fewer, we'll handle outside
                if len(df) > expected_rows:
                    df = df.head(expected_rows)
            
            # Show analysis summary
            if "analysis_summary" in data:
                st.success(f"✅ LLM Analysis: {data['analysis_summary'].get('dataset_type', 'Unknown')}")
            
            return df
            
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse LLM response as JSON: {str(e)}")
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if "generated_data" in data:
                        df = pd.DataFrame(data["generated_data"])
                        df = df[expected_columns] if all(col in df.columns for col in expected_columns) else df
                        return df
                except:
                    pass
            
            st.error("Could not extract data from LLM response")
            return None
        except Exception as e:
            st.error(f"Error parsing generated data: {str(e)}")
            return None


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
        page_title="Pure LLM Data Generator",
        page_icon="🧠",
        layout="wide"
    )
    
    # Header
    st.title("🧠 Pure LLM Data Generator")
    st.markdown("**100% LLM Intelligence • No Fallback • No Predefined Rules**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize generator
    if 'pure_llm_generator' not in st.session_state:
        st.session_state.pure_llm_generator = PureLLMGenerator()
    
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload ANY Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df
            
            if df.empty:
                st.error("Empty file")
                return
            
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Preview
            with st.expander("📋 Original Data Preview", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    null_count = df.isnull().sum().sum()
                    st.metric("Missing Values", null_count)
                
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show column info
                st.write("**Columns:**")
                cols_per_row = 4
                cols = df.columns.tolist()
                for i in range(0, len(cols), cols_per_row):
                    col_group = cols[i:i+cols_per_row]
                    col_displays = st.columns(cols_per_row)
                    for j, col in enumerate(col_group):
                        with col_displays[j]:
                            st.code(col)
                            unique = df[col].nunique()
                            dtype = df[col].dtype
                            st.caption(f"{dtype} | Unique: {unique}")
            
            # Generation controls
            st.subheader("⚙️ Generate with Pure LLM Intelligence")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_rows = st.number_input(
                    "Rows to generate",
                    min_value=10,
                    max_value=200,  # Keep reasonable for LLM
                    value=50,
                    help="LLM will generate EXACTLY this many rows"
                )
            
            with col2:
                model_info = st.info("Using: llama-3.3-70b-versatile")
            
            with col3:
                if st.button("🚀 Generate with Pure LLM", type="primary", use_container_width=True):
                    if not st.session_state.pure_llm_generator.available:
                        st.error("LLM not available. Check API key.")
                    else:
                        with st.spinner("🤖 LLM is deeply analyzing and generating perfect data..."):
                            generator = st.session_state.pure_llm_generator
                            generated = generator.generate_perfect_data(df, int(num_rows))
                            
                            if generated is not None:
                                st.session_state.generated_data = generated
                                st.success(f"✅ LLM generated {len(generated)} PERFECT rows!")
                                st.balloons()
                            else:
                                st.error("LLM failed to generate data")
                
                # Retry button
                if st.session_state.generated_data is not None:
                    if st.button("🔄 Retry Generation", type="secondary"):
                        st.session_state.generated_data = None
                        st.rerun()
            
            # Show generated data
            if st.session_state.generated_data is not None:
                df_gen = st.session_state.generated_data
                
                st.subheader(f"📊 LLM-Generated Data ({len(df_gen)} rows)")
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["Data Preview", "Quality Check", "Download"])
                
                with tab1:
                    st.dataframe(df_gen.head(20), use_container_width=True)
                    
                    # Show sample analysis
                    if len(df_gen) > 0:
                        st.subheader("🧐 Sample Row Analysis")
                        sample = df_gen.iloc[0]
                        st.json(sample.to_dict())
                        
                        # Check quality
                        st.subheader("✅ Quality Check")
                        issues = []
                        
                        # Check for placeholders
                        for col in df_gen.columns:
                            sample_val = str(df_gen[col].iloc[0])
                            if any(placeholder in sample_val.lower() for placeholder in 
                                  ['value_', 'category_', 'temp_', 'dummy_', 'test_']):
                                issues.append(f"Column '{col}' has placeholder: {sample_val}")
                        
                        if issues:
                            st.warning("⚠️ Issues found:")
                            for issue in issues:
                                st.write(f"- {issue}")
                        else:
                            st.success("✓ No placeholder values found")
                        
                        # Check name-gender alignment (if applicable)
                        name_cols = [col for col in df_gen.columns if 'name' in col.lower()]
                        gender_cols = [col for col in df_gen.columns if 'gender' in col.lower()]
                        
                        if name_cols and gender_cols:
                            name_col = name_cols[0]
                            gender_col = gender_cols[0]
                            
                            mismatches = 0
                            for idx in range(min(10, len(df_gen))):
                                name = str(df_gen.at[idx, name_col]).lower()
                                gender = str(df_gen.at[idx, gender_col]).upper()
                                
                                # Check for obvious mismatches
                                if ('rahul' in name or 'amit' in name or 'raj' in name or 
                                    'singh' in name or 'kumar' in name) and gender == 'F':
                                    mismatches += 1
                                elif ('priya' in name or 'neha' in name or 'anjali' in name or
                                      'devi' in name or 'kumari' in name) and gender == 'M':
                                    mismatches += 1
                            
                            if mismatches > 0:
                                st.warning(f"⚠️ Found {mismatches} gender-name mismatches in first 10 rows")
                            else:
                                st.success("✓ All name-gender pairs look correct")
                
                with tab2:
                    st.subheader("📈 Data Quality Report")
                    
                    # Basic stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df_gen))
                    with col2:
                        st.metric("Total Columns", len(df_gen.columns))
                    with col3:
                        null_pct = (df_gen.isnull().sum().sum() / (len(df_gen) * len(df_gen.columns))) * 100
                        st.metric("Null %", f"{null_pct:.1f}%")
                    
                    # Column analysis
                    st.subheader("Column Analysis")
                    for col in df_gen.columns:
                        with st.expander(f"Column: {col}", expanded=False):
                            st.write(f"**Type:** {df_gen[col].dtype}")
                            st.write(f"**Unique values:** {df_gen[col].nunique()}")
                            st.write(f"**Sample values:** {df_gen[col].head(5).tolist()}")
                            
                            # Check for common issues
                            issues = []
                            
                            # Check for placeholders
                            sample = str(df_gen[col].iloc[0]) if len(df_gen) > 0 else ""
                            if any(ph in sample.lower() for ph in ['value_', 'category_', 'temp_']):
                                issues.append("Contains placeholder values")
                            
                            # Check for unrealistic values
                            if df_gen[col].dtype in ['int64', 'float64']:
                                if df_gen[col].min() < 0 and 'age' in col.lower():
                                    issues.append("Negative age")
                                elif df_gen[col].max() > 1000 and 'age' in col.lower():
                                    issues.append("Unrealistic age (>1000)")
                            
                            if issues:
                                st.warning("Issues: " + ", ".join(issues))
                            else:
                                st.success("✓ Looks good")
                
                with tab3:
                    st.subheader("📥 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV
                        csv = df_gen.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"llm_generated_data_{len(df_gen)}_rows.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # JSON
                        json_str = df_gen.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            f"llm_generated_data_{len(df_gen)}_rows.json",
                            "application/json",
                            use_container_width=True
                        )
                    
                    st.write("---")
                    
                    # Regenerate options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Generate New", use_container_width=True):
                            st.session_state.generated_data = None
                            st.rerun()
                    
                    with col2:
                        if st.button("🆕 New File", use_container_width=True):
                            st.session_state.original_df = None
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions
        st.info("""
        ## 🧠 **Pure LLM Data Generator**
        
        ### **How It Works:**
        
        1. **Upload ANY CSV** - The LLM will analyze it DEEPLY
        2. **LLM Understands Everything** - Domain, patterns, relationships, constraints
        3. **LLM Generates Perfect Data** - 100% realistic, logical, consistent
        4. **No Fallback, No Predefined Rules** - Pure LLM intelligence only
        
        ### **What the LLM Does:**
        
        **For Medical Data:**
        - Understands Indian names and gender patterns
        - Knows medical departments and appropriate diagnoses
        - Generates valid Indian phone numbers
        - Creates realistic appointment dates/times
        - Sets appropriate medical fees
        - Ensures gender matches names
        
        **For E-commerce Data:**
        - Understands product categories and realistic prices
        - Generates valid customer information
        - Creates logical order sequences
        - Maintains order-shipping relationships
        
        **For ANY Data:**
        - Analyzes patterns and relationships
        - Understands cultural/regional context
        - Applies domain knowledge
        - Generates 100% realistic synthetic data
        
        ### **Key Features:**
        - ✅ **100% LLM Intelligence** - No fallback logic
        - ✅ **Deep Semantic Analysis** - Understands meaning, not just patterns
        - ✅ **Cross-Column Relationships** - Maintains all logical connections
        - ✅ **Cultural Awareness** - Respects regional patterns
        - ✅ **Domain Knowledge** - Applies industry-specific logic
        - ✅ **Perfect Data Quality** - No placeholders, no nonsense
        
        **Upload a CSV to experience pure LLM intelligence!**
        """)

if __name__ == "__main__": 
    main()
