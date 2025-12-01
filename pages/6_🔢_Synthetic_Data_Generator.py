# pages/6_🔢_Synthetic_Data_Generator.py - LLM TO SDV TRANSLATOR
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import json
import re

# =============================================================================
# LLM TO SDV TRANSLATOR
# =============================================================================

class LLMToSDVTranslator:
    """Translate LLM's natural language analysis to SDV constraints"""
    
    @staticmethod
    def get_llm_analysis(df: pd.DataFrame) -> str:
        """Get natural language analysis from LLM"""
        try:
            if "GROQ_API_KEY" not in st.secrets:
                return "No API key available"
            
            from groq import Groq
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            sample = df.head(min(15, len(df)))
            
            # Simple free-form analysis
            prompt = f"""
            Analyze this dataset. What patterns and relationships do you see?
            
            Columns: {list(df.columns)}
            
            Sample data:
            {sample.to_string(index=False)}
            
            Describe the patterns in natural language.
            """
            
            messages = [
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": prompt}
            ]
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Analysis failed: {e}"
    
    @staticmethod
    def translate_to_sdv_constraints(llm_analysis: str, df: pd.DataFrame) -> Dict:
        """
        Translate LLM's natural language to actionable SDV guidance
        """
        constraints_info = {
            "detected_relationships": [],
            "quality_issues": [],
            "generation_guidance": {},
            "column_specific_rules": {}
        }
        
        # Extract column mentions from analysis
        mentioned_columns = []
        for col in df.columns:
            if col in llm_analysis:
                mentioned_columns.append(col)
        
        # Look for relationship patterns in the text
        lines = llm_analysis.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Look for relationship mentions
            if any(word in line_lower for word in ['correlation', 'relationship', 'association', 'link', 'tied to', 'related to']):
                # Try to extract column names from this line
                for col1 in df.columns:
                    if col1.lower() in line_lower:
                        for col2 in df.columns:
                            if col2.lower() in line_lower and col1 != col2:
                                constraints_info["detected_relationships"].append({
                                    "type": "correlation",
                                    "columns": [col1, col2],
                                    "description": line.strip()
                                })
            
            # Look for data quality issues
            if any(word in line_lower for word in ['bias', 'skew', 'imbalance', 'missing', 'inconsistent', 'wrong', 'error']):
                constraints_info["quality_issues"].append(line.strip())
            
            # Look for specific column rules
            for col in df.columns:
                if col.lower() in line_lower:
                    if 'format' in line_lower or 'pattern' in line_lower:
                        if col not in constraints_info["column_specific_rules"]:
                            constraints_info["column_specific_rules"][col] = []
                        constraints_info["column_specific_rules"][col].append(line.strip())
        
        # Generate SDV guidance based on analysis
        constraints_info["generation_guidance"] = LLMToSDVTranslator._generate_sdv_guidance(llm_analysis, df)
        
        return constraints_info
    
    @staticmethod
    def _generate_sdv_guidance(llm_analysis: str, df: pd.DataFrame) -> Dict:
        """Generate SDV-specific guidance from LLM analysis"""
        guidance = {
            "model_settings": {},
            "data_preprocessing": [],
            "post_processing": []
        }
        
        # Adjust model settings based on data characteristics
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Dynamic model configuration
        if n_rows < 50:
            guidance["model_settings"] = {
                "epochs": 300,
                "batch_size": min(32, n_rows),
                "verbose": False,
                "recommendation": "Small dataset - use more epochs"
            }
        elif n_rows < 200:
            guidance["model_settings"] = {
                "epochs": 200,
                "batch_size": 50,
                "verbose": False,
                "recommendation": "Medium dataset - standard training"
            }
        else:
            guidance["model_settings"] = {
                "epochs": 100,
                "batch_size": 100,
                "verbose": False,
                "recommendation": "Large dataset - efficient training"
            }
        
        # Look for specific guidance in LLM analysis
        if 'bias' in llm_analysis.lower() or 'skew' in llm_analysis.lower():
            guidance["data_preprocessing"].append("Check for class imbalance in categorical columns")
        
        if 'correlation' in llm_analysis.lower():
            guidance["model_settings"]["enforce_relationships"] = True
        
        return guidance

# =============================================================================
# SMART SDV GENERATOR WITH TRANSLATED CONSTRAINTS
# =============================================================================

class SmartSDVGeneratorWithTranslation:
    """SDV generator that uses translated LLM constraints"""
    
    def __init__(self):
        self.sdv_available = self._check_sdv()
        self.translator = LLMToSDVTranslator()
    
    def _check_sdv(self):
        try:
            from sdv.single_table import CTGANSynthesizer
            return True
        except:
            return False
    
    def generate_with_translated_constraints(self, df: pd.DataFrame, num_rows: int) -> Optional[pd.DataFrame]:
        """Generate using translated LLM constraints"""
        if not self.sdv_available:
            st.error("SDV not available")
            return None
        
        # Step 1: Get LLM analysis
        with st.spinner("🧠 LLM analyzing data..."):
            llm_analysis = self.translator.get_llm_analysis(df)
        
        # Step 2: Translate to SDV constraints
        with st.spinner("🔄 Translating to SDV guidance..."):
            constraints_info = self.translator.translate_to_sdv_constraints(llm_analysis, df)
        
        # Display translation results
        self._display_translation(llm_analysis, constraints_info)
        
        # Step 3: Train SDV with guidance
        try:
            from sdv.single_table import CTGANSynthesizer
            from sdv.metadata import SingleTableMetadata
            
            with st.spinner("🤖 Training SDV with translated guidance..."):
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data=df)
                
                # Get guidance
                guidance = constraints_info.get("generation_guidance", {}).get("model_settings", {})
                
                # Train model with translated settings
                model = CTGANSynthesizer(
                    metadata=metadata,
                    epochs=guidance.get("epochs", 200),
                    batch_size=guidance.get("batch_size", 50),
                    verbose=False
                )
                
                model.fit(df)
            
            # Step 4: Generate
            with st.spinner(f"🎯 Generating {num_rows} rows..."):
                synthetic = model.sample(num_rows=num_rows)
            
            # Step 5: Apply post-processing based on LLM analysis
            synthetic = self._apply_post_processing(synthetic, constraints_info, df)
            
            # Step 6: Validate
            self._validate_with_constraints(synthetic, constraints_info)
            
            return synthetic
            
        except Exception as e:
            st.error(f"Generation failed: {e}")
            return None
    
    def _display_translation(self, llm_analysis: str, constraints_info: Dict):
        """Display the translation process"""
        st.subheader("🔍 LLM Analysis → SDV Translation")
        
        # Show original LLM analysis
        with st.expander("📝 LLM's Natural Language Analysis", expanded=True):
            st.write(llm_analysis)
        
        # Show translated constraints
        with st.expander("🔄 Translated to SDV Guidance", expanded=True):
            
            # Detected relationships
            relationships = constraints_info.get("detected_relationships", [])
            if relationships:
                st.write("**Detected Relationships:**")
                for rel in relationships[:5]:  # Show first 5
                    cols = rel.get("columns", [])
                    if len(cols) >= 2:
                        st.write(f"• {cols[0]} ↔ {cols[1]}: {rel.get('description', '')[:100]}...")
            
            # Quality issues
            issues = constraints_info.get("quality_issues", [])
            if issues:
                st.write("**Data Quality Issues:**")
                for issue in issues[:3]:
                    st.error(f"⚠️ {issue}")
            
            # Generation guidance
            guidance = constraints_info.get("generation_guidance", {})
            if guidance.get("model_settings"):
                st.write("**SDV Model Settings:**")
                st.json(guidance["model_settings"], expanded=False)
    
    def _apply_post_processing(self, synthetic: pd.DataFrame, constraints_info: Dict, original: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing based on LLM analysis"""
        df = synthetic.copy()
        
        # Get detected relationships
        relationships = constraints_info.get("detected_relationships", [])
        
        # Apply simple relationship preservation
        fixes_applied = []
        
        for rel in relationships:
            cols = rel.get("columns", [])
            if len(cols) >= 2:
                col1, col2 = cols[0], cols[1]
                
                if col1 in df.columns and col2 in df.columns:
                    # Check if relationship exists in original data
                    if col1 in original.columns and col2 in original.columns:
                        # Get unique combinations from original
                        unique_combos = original[[col1, col2]].drop_duplicates()
                        
                        if len(unique_combos) > 0 and len(unique_combos) < len(original) * 0.5:
                            # Relationship exists, try to preserve it
                            # Sample approach: For some rows, enforce original combos
                            n_to_fix = min(20, len(df))
                            if n_to_fix > 0:
                                # Get random combos from original
                                sample_combos = unique_combos.sample(n_to_fix, replace=True).reset_index(drop=True)
                                
                                # Apply to random rows in synthetic
                                random_indices = np.random.choice(df.index, n_to_fix, replace=False)
                                df.loc[random_indices, [col1, col2]] = sample_combos.values
                                
                                fixes_applied.append(f"Preserved {col1}↔{col2} relationship")
        
        if fixes_applied:
            st.info(f"✅ Applied {len(fixes_applied)} relationship-preserving fixes")
        
        return df
    
    def _validate_with_constraints(self, synthetic: pd.DataFrame, constraints_info: Dict):
        """Validate against translated constraints"""
        st.subheader("✅ Validation Against LLM Insights")
        
        relationships = constraints_info.get("detected_relationships", [])
        
        if relationships:
            st.write("**Relationship Preservation Check:**")
            
            for rel in relationships[:3]:  # Check first 3
                cols = rel.get("columns", [])
                if len(cols) >= 2:
                    col1, col2 = cols[0], cols[1]
                    
                    if col1 in synthetic.columns and col2 in synthetic.columns:
                        # Calculate relationship strength
                        unique_pairs = synthetic[[col1, col2]].drop_duplicates().shape[0]
                        total_rows = len(synthetic)
                        
                        if unique_pairs < total_rows * 0.8:
                            st.success(f"✅ {col1}↔{col2}: Strong relationship preserved ({unique_pairs} unique pairs)")
                        else:
                            st.warning(f"⚠️ {col1}↔{col2}: Weak relationship ({unique_pairs} unique pairs)")

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="LLM→SDV Translator",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 LLM → SDV Translator")
    st.markdown("**LLM analyzes in natural language → Translated to SDV constraints → Generate quality data**")
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload Your Data (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} rows")
        
        # Preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
        
        # Generation settings
        st.subheader("⚙️ Translation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=len(df) * 10,
                value=len(df) * 3
            )
        
        with col2:
            translation_depth = st.select_slider(
                "Translation Depth",
                options=["Basic", "Moderate", "Detailed"],
                value="Moderate"
            )
        
        # Generate button
        if st.button("🚀 Generate with Translated Constraints", type="primary", use_container_width=True):
            if len(df) < 20:
                st.warning("Small dataset - LLM may have limited patterns to analyze")
            
            generator = SmartSDVGeneratorWithTranslation()
            
            if not generator.sdv_available:
                st.error("SDV not available")
            else:
                synthetic = generator.generate_with_translated_constraints(df, int(num_rows))
                
                if synthetic is not None:
                    st.session_state.generated_data = synthetic
                    st.balloons()
        
        # Show results
        if 'generated_data' in st.session_state and st.session_state.generated_data is not None:
            synthetic = st.session_state.generated_data
            
            st.subheader(f"📊 Generated Data ({len(synthetic)} rows)")
            
            tab1, tab2, tab3 = st.tabs(["Data", "Comparison", "Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
            
            with tab2:
                # Compare key columns
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
                    "📥 Download",
                    csv,
                    f"translated_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    else:
        st.info("""
        ## 🔄 LLM → SDV Translation Pipeline
       !**
        """)

if __name__ == "__main__":
    main()
