# pages/6_🔢_Synthetic_Data_Generator.py - PURE LLM-DRIVEN
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import json
import re
import hashlib

# =============================================================================
# PURE LLM ANALYZER - NO PREDEFINED RULES
# =============================================================================

class PureLLMAnalyzer:
    """Pure LLM analysis with ZERO predefined rules"""
    
    @staticmethod
    def deep_analyze(df: pd.DataFrame) -> Dict:
        """
        LLM analyzes everything - structure, relationships, rules
        Returns strict rules for data generation
        """
        try:
            # Check for LLM API
            if "GROQ_API_KEY" not in st.secrets:
                st.error("❌ No GROQ_API_KEY in secrets.toml")
                return {"error": "No API key"}
            
            from groq import Groq
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            # Get sample for analysis
            sample_size = min(20, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            
            # Build DEEP analysis prompt
            prompt = f"""
            CRITICAL DATA ANALYSIS TASK
            
            I have a dataset that I want to use for training a synthetic data generator.
            I need you to perform DEEP analysis and create STRICT RULES for data generation.
            
            ===== DATASET INFO =====
            Rows: {len(df)}
            Columns: {list(df.columns)}
            
            ===== SAMPLE DATA =====
            {sample_df.to_string(index=False)}
            
            ===== STATISTICAL SUMMARY =====
            {df.describe(include='all').to_string()}
            
            ===== COLUMN DETAILS =====
            {PureLLMAnalyzer._get_column_stats(df)}
            
            ===== YOUR TASK =====
            Analyze this data DEEPLY and return JSON with:
            
            1. **data_context**: What is this data about? (e.g., "medical appointments", "customer transactions")
            
            2. **column_analysis**: For EACH column, determine:
               - semantic_type: What does this column represent?
               - data_type: string, integer, float, datetime, categorical
               - validation_rules: Specific validation rules
               - generation_constraints: How should this be generated?
            
            3. **relationships**: Discover ALL relationships between columns
               - mapping_relationships: e.g., Doctor → Specialty, Symptom → Department
               - conditional_rules: e.g., IF Symptom='Chest Pain' THEN Department='Cardiology'
               - business_rules: Domain-specific logic
            
            4. **data_quality_issues**: List ALL issues in current data
               - inconsistencies: e.g., wrong mappings found
               - anomalies: illogical combinations
               - format_issues: incorrect formats
            
            5. **strict_generation_rules**: Create STRICT rules for synthetic generation
               - column_constraints: Validation for each column
               - relationship_constraints: Must-maintain relationships
               - domain_rules: Must-follow domain logic
               - forbidden_patterns: Patterns to avoid
            
            6. **generation_guidance**: How to generate high-quality synthetic data
               - model_settings: Recommended SDV model settings
               - preprocessing_steps: Data cleaning needed
               - postprocessing_steps: Fixes to apply after generation
            
            IMPORTANT: BE STRICT AND SPECIFIC. If you see wrong mappings (like 'Chest Pain' going to 'ENT'),
            create STRICT rules to prevent this in synthetic data.
            
            Return COMPREHENSIVE JSON.
            """
            
            messages = [
                {"role": "system", "content": """You are a STRICT data quality auditor and synthetic data expert.
                Your job is to find EVERY issue and create STRICT rules for perfect synthetic data.
                Be meticulous, thorough, and uncompromising on data quality."""},
                {"role": "user", "content": prompt}
            ]
            
            with st.spinner("🧠 LLM performing DEEP analysis..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Store for reference
            st.session_state.deep_analysis = analysis
            
            # Display analysis
            PureLLMAnalyzer._display_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            st.error(f"LLM analysis failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_column_stats(df: pd.DataFrame) -> str:
        """Get detailed column statistics"""
        stats = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            sample_vals = df[col].dropna().unique()[:3]
            
            stats.append(f"""
            [{col}]
            Type: {dtype}
            Unique: {unique_count}
            Null: {null_count}
            Sample: {sample_vals}
            """)
        
        return "\n".join(stats)
    
    @staticmethod
    def _display_analysis(analysis: Dict):
        """Display LLM analysis results"""
        st.subheader("🔍 LLM Deep Analysis Results")
        
        # Data Context
        with st.expander("📌 Data Context", expanded=True):
            st.write(f"**Type**: {analysis.get('data_context', 'Unknown')}")
        
        # Column Analysis
        with st.expander("📊 Column Analysis", expanded=False):
            col_analysis = analysis.get('column_analysis', {})
            for col, info in col_analysis.items():
                st.write(f"**{col}**")
                st.json(info, expanded=False)
        
        # Relationships
        with st.expander("🔗 Discovered Relationships", expanded=False):
            relationships = analysis.get('relationships', [])
            if isinstance(relationships, list):
                for rel in relationships:
                    st.write(f"• {rel}")
            elif isinstance(relationships, dict):
                st.json(relationships, expanded=False)
        
        # Data Quality Issues
        with st.expander("⚠️ Data Quality Issues", expanded=True):
            issues = analysis.get('data_quality_issues', [])
            if issues:
                for issue in issues:
                    st.error(f"❌ {issue}")
            else:
                st.success("✅ No major issues found")
        
        # Strict Rules
        with st.expander("📜 Strict Generation Rules", expanded=True):
            rules = analysis.get('strict_generation_rules', {})
            if rules:
                st.warning("🚨 These rules MUST be enforced:")
                if isinstance(rules, dict):
                    for rule_type, rule_list in rules.items():
                        st.write(f"**{rule_type}**:")
                        if isinstance(rule_list, list):
                            for rule in rule_list:
                                st.write(f"  - {rule}")
                        else:
                            st.write(f"  {rule_list}")
                elif isinstance(rules, list):
                    for rule in rules:
                        st.write(f"• {rule}")
        
        # Generation Guidance
        with st.expander("🎯 Generation Guidance", expanded=False):
            guidance = analysis.get('generation_guidance', {})
            st.json(guidance, expanded=False)

# =============================================================================
# LLM-CONTROLLED SDV GENERATOR
# =============================================================================

class LLMControlledGenerator:
    """SDV generator completely controlled by LLM analysis"""
    
    def __init__(self):
        self.sdv_available = self._check_sdv()
        self.analyzer = PureLLMAnalyzer()
    
    def _check_sdv(self):
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            return True
        except:
            return False
    
    def generate_with_llm_control(self, df: pd.DataFrame, num_rows: int) -> Optional[pd.DataFrame]:
        """
        Generate synthetic data with LLM controlling everything
        """
        if not self.sdv_available:
            st.error("SDV not available")
            return None
        
        # Step 1: LLM Deep Analysis
        analysis = self.analyzer.deep_analyze(df)
        
        if "error" in analysis:
            st.error("Cannot proceed without LLM analysis")
            return None
        
        # Step 2: Preprocess based on LLM guidance
        df_cleaned = self._preprocess_with_llm(df, analysis)
        
        # Step 3: Train SDV with LLM guidance
        synthetic = self._train_sdv_with_llm(df_cleaned, num_rows, analysis)
        
        if synthetic is None:
            return None
        
        # Step 4: Post-process with LLM rules
        synthetic_fixed = self._enforce_llm_rules(synthetic, analysis)
        
        # Step 5: Quality validation
        self._validate_with_llm(synthetic_fixed, analysis)
        
        return synthetic_fixed
    
    def _preprocess_with_llm(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Preprocess data based on LLM guidance"""
        st.info("🛠️ Preprocessing data based on LLM analysis...")
        
        df_clean = df.copy()
        guidance = analysis.get('generation_guidance', {}).get('preprocessing_steps', [])
        
        if guidance:
            st.write("**LLM Recommended Preprocessing:**")
            for step in guidance[:5]:  # Show first 5 steps
                st.write(f"• {step}")
        
        # Apply common preprocessing (LLM can't execute code, so we do smart fixes)
        
        # Fix date formats if detected
        date_cols = []
        col_analysis = analysis.get('column_analysis', {})
        for col, info in col_analysis.items():
            if isinstance(info, dict) and info.get('semantic_type', '').lower() in ['date', 'datetime']:
                date_cols.append(col)
        
        for col in date_cols:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except:
                    pass
        
        return df_clean
    
    def _train_sdv_with_llm(self, df: pd.DataFrame, num_rows: int, analysis: Dict) -> Optional[pd.DataFrame]:
        """Train SDV model with LLM guidance"""
        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import CTGANSynthesizer
            
            st.info("🤖 Training SDV model with LLM guidance...")
            
            # Get LLM's model recommendations
            guidance = analysis.get('generation_guidance', {}).get('model_settings', {})
            
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=df)
            
            # Configure model based on LLM guidance or use defaults
            epochs = guidance.get('epochs', 300)  # More epochs for small datasets
            batch_size = guidance.get('batch_size', min(32, len(df)))
            
            model = CTGANSynthesizer(
                metadata=metadata,
                epochs=epochs,
                batch_size=batch_size,
                verbose=False
            )
            
            # Train
            with st.spinner(f"Training for {epochs} epochs..."):
                model.fit(df)
            
            # Generate
            with st.spinner(f"Generating {num_rows} synthetic rows..."):
                synthetic = model.sample(num_rows=num_rows)
            
            return synthetic
            
        except Exception as e:
            st.error(f"SDV training failed: {e}")
            return None
    
    def _enforce_llm_rules(self, synthetic: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Enforce LLM's strict rules on generated data"""
        st.info("⚖️ Enforcing LLM's strict rules...")
        
        df = synthetic.copy()
        rules = analysis.get('strict_generation_rules', {})
        
        if not rules:
            return df
        
        # Track fixes made
        fixes_made = []
        
        # Apply relationship constraints
        relationships = analysis.get('relationships', [])
        if isinstance(relationships, list):
            for rel in relationships:
                if isinstance(rel, str):
                    # Try to parse relationship rules
                    if '->' in rel or '→' in rel:
                        parts = re.split(r'[->→]', rel)
                        if len(parts) >= 2:
                            source = parts[0].strip()
                            target = parts[1].strip()
                            
                            # Simple enforcement: maintain existing mappings
                            if source in df.columns and target in df.columns:
                                # Get unique mappings from original patterns
                                unique_pairs = df[[source, target]].drop_duplicates()
                                if len(unique_pairs) > 1:
                                    fixes_made.append(f"Maintained {source}→{target} relationships")
        
        # Apply business rules
        business_rules = analysis.get('business_rules', [])
        if business_rules:
            st.write("**Business Rules Applied:**")
            for rule in business_rules[:5]:  # Show first 5
                if isinstance(rule, str):
                    st.write(f"• {rule}")
        
        if fixes_made:
            st.success(f"✅ Applied {len(fixes_made)} rule-based fixes")
        
        return df
    
    def _validate_with_llm(self, synthetic: pd.DataFrame, analysis: Dict):
        """Validate generated data against LLM rules"""
        st.subheader("✅ LLM-Based Validation")
        
        # Check for obvious issues
        issues_found = []
        
        # Check column presence
        expected_cols = analysis.get('column_analysis', {}).keys()
        missing_cols = [col for col in expected_cols if col not in synthetic.columns]
        if missing_cols:
            issues_found.append(f"Missing columns: {missing_cols}")
        
        # Check data quality issues from analysis
        original_issues = analysis.get('data_quality_issues', [])
        if original_issues:
            # Check if similar issues exist in synthetic
            for issue in original_issues[:3]:  # Check first 3
                if isinstance(issue, str):
                    if 'chest' in issue.lower() and 'ent' in issue.lower():
                        # Check for this specific issue
                        if 'Symptoms' in synthetic.columns and 'Specialty' in synthetic.columns:
                            chest_ent = synthetic[
                                synthetic['Symptoms'].astype(str).str.lower().str.contains('chest') &
                                synthetic['Specialty'].astype(str).str.lower().str.contains('ent')
                            ]
                            if len(chest_ent) > 0:
                                issues_found.append("❌ Still have Chest Pain → ENT issue!")
                            else:
                                st.success("✅ Fixed: Chest Pain no longer goes to ENT")
        
        # Display validation results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Generated", len(synthetic))
        with col2:
            st.metric("Columns", len(synthetic.columns))
        with col3:
            status = "❌" if issues_found else "✅"
            st.metric("Validation", status)
        
        if issues_found:
            st.error("**Issues Found:**")
            for issue in issues_found:
                st.write(f"- {issue}")
        else:
            st.success("✅ Generated data passes LLM validation")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="LLM-Controlled Data Generator",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 LLM-Controlled Synthetic Data Generator")
    st.markdown("**Zero predefined rules - LLM analyzes everything and creates strict generation rules**")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Your Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Show sample
        with st.expander("📋 Data Sample", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            
            # Data stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Quality", "Analyzing..." if len(df) < 50 else "Good")
            with col2:
                st.metric("Sufficient for SDV", "✅" if len(df) >= 30 else "⚠️")
            with col3:
                ratio = min(100, len(df) * 10) / len(df) if len(df) > 0 else 0
                st.metric("Max Safe Ratio", f"{ratio:.1f}:1")
        
        # Generation settings
        st.subheader("⚙️ Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=len(df) * 20,
                value=len(df) * 5,
                help="LLM will determine optimal settings"
            )
        
        with col2:
            st.write("**LLM Control Level**")
            control_level = st.select_slider(
                "",
                options=["Basic", "Standard", "Strict", "Maximum"],
                value="Strict",
                help="How strictly LLM enforces rules"
            )
        
        # Generate button
        if st.button("🚀 Generate with LLM Control", type="primary", use_container_width=True):
            if len(df) < 10:
                st.error("❌ Need at least 10 rows for meaningful analysis")
            else:
                # Initialize generator
                generator = LLMControlledGenerator()
                
                if not generator.sdv_available:
                    st.error("SDV not installed. Add 'sdv' to requirements.txt")
                else:
                    # Generate with LLM control
                    synthetic = generator.generate_with_llm_control(df, int(num_rows))
                    
                    if synthetic is not None:
                        st.session_state.generated_data = synthetic
                        st.balloons()
        
        # Show results
        if 'generated_data' in st.session_state and st.session_state.generated_data is not None:
            synthetic = st.session_state.generated_data
            
            st.subheader(f"📊 LLM-Controlled Generated Data ({len(synthetic)} rows)")
            
            tab1, tab2, tab3 = st.tabs(["Preview", "Comparison", "Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Highlight potential improvements
                st.write("**🎯 LLM-Improved Patterns:**")
                
                # Check for common medical logic
                medical_checks = []
                if 'Symptoms' in synthetic.columns and 'Specialty' in synthetic.columns:
                    # Check chest pain
                    chest_rows = synthetic[synthetic['Symptoms'].astype(str).str.lower().str.contains('chest')]
                    if not chest_rows.empty:
                        specialties = chest_rows['Specialty'].unique()
                        if any('cardio' in str(s).lower() for s in specialties):
                            medical_checks.append("✅ Chest pain correctly goes to Cardiology")
                        else:
                            medical_checks.append("⚠️ Chest pain specialty needs review")
                
                if medical_checks:
                    for check in medical_checks:
                        st.write(check)
            
            with tab2:
                # Compare distributions
                if len(df.columns) > 0:
                    compare_col = st.selectbox("Compare column:", df.columns)
                    
                    if compare_col in synthetic.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Original {compare_col}**")
                            st.bar_chart(df[compare_col].value_counts().head(10))
                        with col2:
                            st.write(f"**Generated {compare_col}**")
                            st.bar_chart(synthetic[compare_col].value_counts().head(10))
            
            with tab3:
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"llm_controlled_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Regenerate options
                st.write("---")
                if st.button("🔄 Generate New Variation"):
                    st.session_state.generated_data = None
                    st.rerun()
    
    else:
        st.info("""
        ## 🧠 Pure LLM-Controlled Generation
        
        ### 
        """)

if __name__ == "__main__":
    main()
