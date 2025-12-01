# pages/6_🔢_Synthetic_Data_Generator.py - LLM-POWERED VERSION
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import json
import re

# =============================================================================
# LLM ANALYZER FOR SMART CONSTRAINTS
# =============================================================================

class LLMAnalyzer:
    """Use LLM to analyze data and create smart constraints"""
    
    @staticmethod
    def analyze_with_llm(df: pd.DataFrame, sample_size: int = 10) -> Dict:
        """
        Analyze data using LLM to understand relationships and constraints
        """
        try:
            # Use available LLM (Groq, OpenAI, or local)
            api_key = st.secrets.get("GROQ_API_KEY", "")
            
            if not api_key:
                # If no API key, use rule-based analysis
                return LLMAnalyzer._rule_based_analysis(df)
            
            from groq import Groq
            client = Groq(api_key=api_key)
            
            # Prepare sample data
            sample_df = df.head(sample_size) if len(df) > sample_size else df
            
            # Build comprehensive prompt
            prompt = f"""
            You are a data quality expert. Analyze this dataset and identify:
            
            1. **Column Types**: Identify each column's type (ID, name, age, gender, phone, specialty, doctor, date, time, symptom, cost, status)
            2. **Data Relationships**: Find logical relationships between columns (e.g., doctor → specialty, symptom → specialty, gender → name patterns)
            3. **Data Quality Issues**: Identify any inconsistencies or errors
            4. **Business Rules**: Discover domain-specific rules (e.g., medical: chest pain → cardiology, pregnancy → gynecology)
            5. **Constraints Needed**: What constraints should be enforced during synthetic data generation?
            
            Dataset Shape: {df.shape}
            Columns: {list(df.columns)}
            
            Sample Data (first {len(sample_df)} rows):
            {sample_df.to_string()}
            
            Statistical Summary:
            {df.describe(include='all').to_string()}
            
            Column Details:
            {LLMAnalyzer._get_column_details(df)}
            
            Return JSON with this structure:
            {{
                "dataset_type": "medical_appointments",
                "column_analysis": {{
                    "column_name": {{
                        "type": "categorical/numeric/id/text/date",
                        "expected_pattern": "regex pattern if any",
                        "unique_values_count": 0,
                        "should_be_unique": true/false,
                        "allowed_values": ["list", "if", "applicable"]
                    }}
                }},
                "relationships": [
                    {{
                        "type": "mapping",
                        "from_column": "Doctor",
                        "to_column": "Specialty",
                        "constraint": "One doctor should have one specialty"
                    }}
                ],
                "domain_rules": [
                    "Chest pain should go to Cardiology, not ENT",
                    "Male patients should not have pregnancy-related symptoms"
                ],
                "constraints_needed": [
                    "Unique constraint on patient_id",
                    "Age should be between 0-120",
                    "Phone numbers should be 10 digits"
                ],
                "data_quality_issues": [
                    "Inconsistent doctor-specialty mapping",
                    "Invalid symptom-specialty combinations"
                ],
                "generation_recommendations": [
                    "Use CTGAN with more epochs",
                    "Add custom constraints for symptom-specialty mapping"
                ]
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a data scientist and domain expert."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Store analysis in session state
            st.session_state.llm_analysis = analysis
            return analysis
            
        except Exception as e:
            st.warning(f"LLM analysis failed: {e}. Using rule-based analysis.")
            return LLMAnalyzer._rule_based_analysis(df)
    
    @staticmethod
    def _get_column_details(df: pd.DataFrame) -> str:
        """Get detailed column information"""
        details = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().unique()[:5].tolist()
            null_count = df[col].isnull().sum()
            
            details.append(f"""
            - {col}:
              Type: {dtype}
              Unique values: {unique_count}
              Null values: {null_count}
              Sample: {sample_values}
            """)
        
        return "\n".join(details)
    
    @staticmethod
    def _rule_based_analysis(df: pd.DataFrame) -> Dict:
        """Fallback rule-based analysis"""
        analysis = {
            "dataset_type": "generic",
            "column_analysis": {},
            "relationships": [],
            "domain_rules": [],
            "constraints_needed": [],
            "data_quality_issues": [],
            "generation_recommendations": []
        }
        
        # Auto-detect column types
        for col in df.columns:
            col_info = {
                "type": "unknown",
                "unique_values_count": df[col].nunique(),
                "should_be_unique": False,
                "expected_pattern": None
            }
            
            # Detect column type by name and content
            col_lower = col.lower()
            
            # Common patterns
            if any(word in col_lower for word in ['id', 'code', 'ref']):
                col_info["type"] = "id"
                col_info["should_be_unique"] = True
            elif any(word in col_lower for word in ['name', 'patient', 'doctor']):
                col_info["type"] = "text"
            elif 'age' in col_lower:
                col_info["type"] = "numeric"
            elif any(word in col_lower for word in ['gender', 'sex']):
                col_info["type"] = "categorical"
                col_info["allowed_values"] = ["M", "F", "Male", "Female"]
            elif any(word in col_lower for word in ['phone', 'mobile', 'contact']):
                col_info["type"] = "text"
                col_info["expected_pattern"] = r'^\d{10}$'
            elif any(word in col_lower for word in ['date']):
                col_info["type"] = "date"
            elif any(word in col_lower for word in ['time']):
                col_info["type"] = "time"
            elif any(word in col_lower for word in ['cost', 'price', 'amount', 'fee']):
                col_info["type"] = "numeric"
            elif any(word in col_lower for word in ['status', 'result']):
                col_info["type"] = "categorical"
            
            analysis["column_analysis"][col] = col_info
        
        return analysis
    
    @staticmethod
    def create_sdv_constraints(analysis: Dict, df: pd.DataFrame):
        """Create SDV constraints based on LLM analysis"""
        try:
            from sdv.constraints import Unique, GreaterThan, FixedIncrements, CustomConstraint
            
            constraints = []
            
            # 1. Unique constraints for ID columns
            for col, info in analysis.get("column_analysis", {}).items():
                if info.get("should_be_unique", False) and col in df.columns:
                    if df[col].nunique() == len(df):  # Already unique in training data
                        constraints.append(Unique(column_names=[col]))
            
            # 2. Numeric range constraints
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            for col in numeric_cols:
                if 'age' in col.lower():
                    # Age constraint: 0-120
                    constraints.append(GreaterThan(column_name=col, low_bound=0))
                    # Could add UpperBound if available in SDV version
                elif any(word in col.lower() for word in ['cost', 'price', 'amount']):
                    # Positive cost constraint
                    constraints.append(GreaterThan(column_name=col, low_bound=0))
            
            # 3. Categorical value constraints (if we know allowed values)
            for col, info in analysis.get("column_analysis", {}).items():
                if col in df.columns and info.get("type") == "categorical":
                    allowed_vals = info.get("allowed_values")
                    if allowed_vals and len(allowed_vals) > 0:
                        # Create custom constraint for allowed values
                        class AllowedValuesConstraint(CustomConstraint):
                            def __init__(self, column_name, allowed_values):
                                self.column_name = column_name
                                self.allowed_values = allowed_values
                                super().__init__(constraint_columns=[column_name])
                            
                            def _fit(self, table_data):
                                pass
                            
                            def _transform(self, table_data):
                                return table_data
                            
                            def _reverse_transform(self, table_data):
                                # During generation, ensure values are from allowed set
                                col_data = table_data[self.column_name]
                                # Replace invalid values with random allowed ones
                                invalid_mask = ~col_data.isin(self.allowed_values)
                                if invalid_mask.any():
                                    random_vals = np.random.choice(self.allowed_values, size=invalid_mask.sum())
                                    col_data[invalid_mask] = random_vals
                                return table_data
                        
                        constraints.append(AllowedValuesConstraint(col, allowed_vals))
            
            return constraints
            
        except Exception as e:
            st.warning(f"Could not create constraints: {e}")
            return []

# =============================================================================
# ENHANCED SDV GENERATOR WITH LLM ANALYSIS
# =============================================================================

class SmartSDVGenerator:
    """SDV generator enhanced with LLM analysis"""
    
    def __init__(self):
        self.available = self._check_sdv()
        self.llm_analyzer = LLMAnalyzer()
    
    def _check_sdv(self):
        try:
            from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
            from sdv.metadata import SingleTableMetadata
            return True
        except:
            return False
    
    def generate_smart_data(self, df: pd.DataFrame, num_rows: int, method: str = "ctgan") -> Optional[pd.DataFrame]:
        """
        Generate synthetic data with LLM-guided constraints
        """
        if not self.available:
            st.error("SDV not available")
            return None
        
        try:
            # Step 1: LLM Analysis
            with st.spinner("🧠 LLM analyzing data patterns and relationships..."):
                analysis = self.llm_analyzer.analyze_with_llm(df)
                
                # Display analysis results
                st.subheader("📋 LLM Analysis Results")
                
                with st.expander("Dataset Type", expanded=True):
                    st.write(f"**Type**: {analysis.get('dataset_type', 'Unknown')}")
                
                with st.expander("Data Quality Issues", expanded=False):
                    issues = analysis.get('data_quality_issues', [])
                    if issues:
                        for issue in issues:
                            st.write(f"❌ {issue}")
                    else:
                        st.write("✅ No major issues found")
                
                with st.expander("Domain Rules Discovered", expanded=False):
                    rules = analysis.get('domain_rules', [])
                    if rules:
                        for rule in rules:
                            st.write(f"📝 {rule}")
                    else:
                        st.write("No specific domain rules identified")
                
                with st.expander("Recommended Constraints", expanded=False):
                    constraints = analysis.get('constraints_needed', [])
                    if constraints:
                        for constraint in constraints:
                            st.write(f"🔒 {constraint}")
                    else:
                        st.write("No constraints recommended")
            
            # Step 2: Create SDV metadata with constraints
            with st.spinner("⚙️ Setting up SDV model with constraints..."):
                from sdv.metadata import SingleTableMetadata
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data=df)
                
                # Add constraints based on LLM analysis
                constraints = self.llm_analyzer.create_sdv_constraints(analysis, df)
                
                if constraints:
                    st.info(f"✅ Added {len(constraints)} constraints to model")
                    for constraint in constraints:
                        st.write(f"   - {type(constraint).__name__}")
            
            # Step 3: Train model
            with st.spinner(f"🤖 Training {method.upper()} model..."):
                if method == "gaussian":
                    from sdv.single_table import GaussianCopulaSynthesizer
                    model = GaussianCopulaSynthesizer(
                        metadata=metadata,
                        constraints=constraints if constraints else None,
                        default_distribution='gamma'
                    )
                elif method == "tvae":
                    from sdv.single_table import TVAESynthesizer
                    model = TVAESynthesizer(
                        metadata=metadata,
                        constraints=constraints if constraints else None,
                        epochs=200,  # More epochs for better learning
                        batch_size=min(50, len(df))
                    )
                else:  # ctgan (default)
                    from sdv.single_table import CTGANSynthesizer
                    model = CTGANSynthesizer(
                        metadata=metadata,
                        constraints=constraints if constraints else None,
                        epochs=300,  # More epochs for small datasets
                        batch_size=min(32, len(df)),  # Smaller batch for small data
                        verbose=False
                    )
                
                model.fit(df)
            
            # Step 4: Generate data
            with st.spinner("🎯 Generating synthetic data..."):
                synthetic_data = model.sample(num_rows=num_rows)
            
            # Step 5: Post-process based on LLM rules
            synthetic_data = self._apply_llm_rules(synthetic_data, analysis)
            
            # Step 6: Quality check
            self._check_quality(synthetic_data, df, analysis)
            
            return synthetic_data
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def _apply_llm_rules(self, synthetic_data: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Apply post-processing rules from LLM analysis"""
        df = synthetic_data.copy()
        
        rules = analysis.get('domain_rules', [])
        
        for rule in rules:
            rule_lower = rule.lower()
            
            # Medical-specific rules
            if 'chest pain' in rule_lower and 'cardiology' in rule_lower:
                # Ensure chest pain goes to cardiology
                if 'Symptoms' in df.columns and 'Specialty' in df.columns:
                    chest_pain_mask = df['Symptoms'].astype(str).str.lower().str.contains('chest pain')
                    if chest_pain_mask.any():
                        df.loc[chest_pain_mask, 'Specialty'] = 'Cardiology'
            
            if 'pregnancy' in rule_lower and ('male' in rule_lower or 'm ' in rule_lower):
                # Ensure male patients don't have pregnancy symptoms
                if 'Gender' in df.columns and 'Symptoms' in df.columns:
                    male_mask = df['Gender'].astype(str).str.upper().isin(['M', 'MALE'])
                    pregnancy_mask = df['Symptoms'].astype(str).str.lower().str.contains('pregnancy')
                    invalid_mask = male_mask & pregnancy_mask
                    if invalid_mask.any():
                        # Change symptom to something else
                        df.loc[invalid_mask, 'Symptoms'] = 'General Checkup'
        
        return df
    
    def _check_quality(self, synthetic: pd.DataFrame, original: pd.DataFrame, analysis: Dict):
        """Check quality of generated data"""
        st.subheader("✅ Quality Check Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            issues_fixed = 0
            rules = analysis.get('domain_rules', [])
            for rule in rules:
                if 'chest pain' in rule.lower():
                    if 'Symptoms' in synthetic.columns and 'Specialty' in synthetic.columns:
                        chest_pain = synthetic[synthetic['Symptoms'].astype(str).str.lower().str.contains('chest pain')]
                        if not chest_pain.empty:
                            correct_specialty = chest_pain['Specialty'].astype(str).str.lower().str.contains('cardiology').all()
                            if correct_specialty:
                                issues_fixed += 1
            
            st.metric("Rules Enforced", issues_fixed)
        
        with col2:
            # Check column consistency
            matching_cols = sum(1 for col in original.columns if col in synthetic.columns)
            st.metric("Columns Preserved", f"{matching_cols}/{len(original.columns)}")
        
        with col3:
            # Check data types
            type_matches = 0
            for col in original.columns:
                if col in synthetic.columns:
                    if str(original[col].dtype) == str(synthetic[col].dtype):
                        type_matches += 1
            st.metric("Data Types Matched", f"{type_matches}/{len(original.columns)}")
        
        # Show sample of generated data
        with st.expander("🔍 Generated Data Sample", expanded=True):
            st.dataframe(synthetic.head(10), use_container_width=True)
            
            # Highlight potential issues
            issues = []
            if 'Symptoms' in synthetic.columns and 'Specialty' in synthetic.columns:
                wrong_mappings = []
                symptom_specialty_pairs = [
                    ('chest pain', 'cardiology'),
                    ('migraine', 'neurology'),
                    ('ear infection', 'ent'),
                    ('eye pain', 'ophthalmology')
                ]
                
                for symptom, expected_specialty in symptom_specialty_pairs:
                    mask = synthetic['Symptoms'].astype(str).str.lower().str.contains(symptom)
                    if mask.any():
                        actual_specialties = synthetic.loc[mask, 'Specialty'].unique()
                        if expected_specialty.lower() not in [s.lower() for s in actual_specialties]:
                            wrong_mappings.append(f"{symptom} → {actual_specialties[0]} (should be {expected_specialty})")
                
                if wrong_mappings:
                    st.warning("⚠️ Potential incorrect mappings:")
                    for mapping in wrong_mappings:
                        st.write(f"  - {mapping}")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Smart SDV Generator",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Smart SDV Generator with LLM Analysis")
    st.markdown("**LLM analyzes your data → Creates smart constraints → SDV generates quality synthetic data**")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Your Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Quick preview
        with st.expander("📋 Data Preview", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Data Quality", "Good" if len(df) > 50 else "Limited")
            
            st.dataframe(df.head(), use_container_width=True)
        
        # Generation controls
        st.subheader("⚙️ Smart Generation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),  # At least as many as original
                max_value=len(df) * 20,  # Max 20x original
                value=len(df) * 5,  # Default 5x original
                help=f"With {len(df)} real rows, recommend {len(df) * 5}-{len(df) * 10} synthetic rows"
            )
        
        with col2:
            method = st.selectbox(
                "SDV Method",
                ["ctgan", "tvae", "gaussian"],
                help="CTGAN: Best for complex patterns, TVAE: Good balance, Gaussian: Fastest"
            )
        
        # LLM enhancement option
        use_llm = st.checkbox(
            "🧠 Enable LLM Analysis (Highly Recommended)",
            value=True,
            help="LLM will analyze data relationships and create smart constraints"
        )
        
        # Generate button
        if st.button("🚀 Generate Smart Synthetic Data", type="primary", use_container_width=True):
            if len(df) < 30:
                st.warning(f"⚠️ Only {len(df)} rows. Quality may be limited. Consider collecting more data.")
            
            if use_llm:
                generator = SmartSDVGenerator()
                with st.spinner("Generating with LLM intelligence..."):
                    synthetic = generator.generate_smart_data(df, int(num_rows), method)
            else:
                # Basic SDV without LLM
                st.info("Using basic SDV (no LLM analysis)")
                # ... basic SDV code here ...
            
            if 'synthetic' in locals() and synthetic is not None:
                st.session_state.generated_data = synthetic
                st.balloons()
        
        # Show results if generated
        if 'generated_data' in st.session_state and st.session_state.generated_data is not None:
            synthetic = st.session_state.generated_data
            
            st.subheader(f"📊 Generated Data ({len(synthetic)} rows)")
            
            tab1, tab2, tab3 = st.tabs(["Preview", "Analysis", "Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
            
            with tab2:
                # Compare distributions
                if 'Specialty' in df.columns and 'Specialty' in synthetic.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Specialty Distribution**")
                        st.bar_chart(df['Specialty'].value_counts())
                    with col2:
                        st.write("**Generated Specialty Distribution**")
                        st.bar_chart(synthetic['Specialty'].value_counts())
            
            with tab3:
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"smart_synthetic_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    else:
        st.info("""
        ## 🧠 How This Works
        
    
        """)

if __name__ == "__main__":
    main()
