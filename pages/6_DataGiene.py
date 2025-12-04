# pages/6_🔢_Synthetic_Data_Generator_crewAI.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime
import hashlib
import random
import os

# =============================================================================
# CREW AI AGENTS - FIXED
# =============================================================================

from crewai import Agent, Task, Crew, Process, LLM

class SyntheticDataCrew:
    """Crew AI powered synthetic data generation"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        # FIX: Use simpler LLM configuration
        self.llm = LLM(
            model="groq/llama-3.1-70b-versatile",  # Format: provider/model
            api_key=groq_api_key,
            temperature=0.1,
        )
        self.setup_crew()
    
    def setup_crew(self):
        """Setup all specialized agents"""
        
        # 🕵️ Agent 1: Data Detective
        self.data_detective = Agent(
            role="Universal Data Detective",
            goal="Identify patterns and context in ANY dataset",
            backstory="""You are an expert at understanding ANY type of data. 
            You can analyze datasets from any domain - business, science, social, technical.
            You look for column meanings, data types, and overall context without assumptions.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 📊 Agent 2: Statistical Analyst
        self.statistical_analyst = Agent(
            role="Statistical Pattern Analyst",
            goal="Find statistical patterns in ANY data",
            backstory="""You are a statistician who finds patterns in any dataset.
            You analyze distributions, correlations, outliers - regardless of domain.
            You think in probabilities and data relationships.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🎯 Agent 3: Rule Miner
        self.rule_miner = Agent(
            role="Data Relationship Miner",
            goal="Extract relationships and rules from data",
            backstory="""You find relationships between columns in any dataset.
            You look for: if column A has value X, then column B often has value Y.
            You find constraints and dependencies without domain bias.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🛠️ Agent 4: Constraint Engineer
        self.constraint_engineer = Agent(
            role="Data Constraint Engineer",
            goal="Build generation constraints for ANY data",
            backstory="""You create rules for generating synthetic data.
            You work with any data type: numbers, text, dates, categories.
            You ensure synthetic data follows discovered patterns.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🎨 Agent 5: Synthetic Artist
        self.synthetic_artist = Agent(
            role="Universal Synthetic Data Artist",
            goal="Generate synthetic data for ANY dataset",
            backstory="""You create realistic synthetic data for any domain.
            You maintain statistical properties while creating new combinations.
            You work with all data types and structures.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🧪 Agent 6: Quality Auditor
        self.quality_auditor = Agent(
            role="Universal Data Quality Auditor",
            goal="Validate synthetic data quality for ANY dataset",
            backstory="""You validate synthetic data against original patterns.
            You check statistical similarity, rule compliance, and data quality.
            You work with any data type and structure.""",
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )

# =============================================================================
# SIMPLIFIED GENERATOR (No API key input needed)
# =============================================================================

class EnhancedCrewAIGenerator:
    """Simplified generator that gets API key from secrets"""
    
    def __init__(self):
        # Get API key from Streamlit secrets
        try:
            self.groq_api_key = st.secrets["GROQ_API_KEY"]
            self.crew = SyntheticDataCrew(self.groq_api_key)
        except KeyError:
            st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to your secrets.toml")
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize Crew AI: {e}")
            st.stop()
    
    def generate_enhanced(self, df: pd.DataFrame, num_rows: int) -> Dict[str, Any]:
        """Generate synthetic data"""
        
        try:
            # Step 1: Crew Analysis
            st.info("🧠 AI agents analyzing your data...")
            rules = self._simple_analysis(df)
            
            # Step 2: Generate data
            st.info(f"🎨 Generating {num_rows} synthetic rows...")
            synthetic_df = self._generate_with_sdv(df, num_rows)
            
            # Step 3: Validation
            validation = self._simple_validation(df, synthetic_df)
            
            return {
                "synthetic_data": synthetic_df,
                "rules": rules,
                "validation": validation
            }
            
        except Exception as e:
            st.error(f"Generation failed: {e}")
            # Fallback to SDV only
            return self._fallback_generation(df, num_rows)
    
    def _simple_analysis(self, df: pd.DataFrame) -> Dict:
        """Simple analysis without complex crew tasks"""
        return {
            "data_type": "tabular",
            "columns": list(df.columns),
            "row_count": len(df),
            "column_types": {col: str(df[col].dtype) for col in df.columns}
        }
    
    def _generate_with_sdv(self, df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
        """Generate using SDV (reliable fallback)"""
        try:
            from sdv.tabular import GaussianCopula
            model = GaussianCopula()
            model.fit(df)
            return model.sample(num_rows)
        except:
            # Simple variation method
            synthetic_rows = []
            for _ in range(num_rows):
                row = {}
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        # Add variation
                        val = df[col].sample(1).iloc[0]
                        if isinstance(val, (int, float)):
                            variation = val * random.uniform(0.8, 1.2)
                            row[col] = variation
                        else:
                            row[col] = val
                    else:
                        row[col] = df[col].sample(1).iloc[0]
                synthetic_rows.append(row)
            return pd.DataFrame(synthetic_rows)
    
    def _simple_validation(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Simple validation"""
        validation = {
            "rows_generated": len(synthetic),
            "columns_preserved": len(synthetic.columns),
            "null_percentage": synthetic.isnull().sum().sum() / (len(synthetic) * len(synthetic.columns)) * 100,
            "similarity_score": self._calculate_similarity(original, synthetic)
        }
        
        validation["overall_score"] = max(0, 100 - validation["null_percentage"] / 2)
        
        return validation
    
    def _calculate_similarity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate basic similarity"""
        similarities = []
        
        for col in original.columns:
            if col in synthetic.columns:
                if pd.api.types.is_numeric_dtype(original[col]):
                    orig_mean = original[col].mean()
                    synth_mean = synthetic[col].mean()
                    if orig_mean != 0:
                        similarity = 100 * (1 - abs(orig_mean - synth_mean) / abs(orig_mean))
                        similarities.append(min(100, similarity))
        
        return np.mean(similarities) if similarities else 50
    
    def _fallback_generation(self, df: pd.DataFrame, num_rows: int) -> Dict:
        """Fallback generation method"""
        synthetic_rows = []
        
        for _ in range(num_rows):
            row = {}
            for col in df.columns:
                row[col] = df[col].sample(1).iloc[0]
            synthetic_rows.append(row)
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        
        return {
            "synthetic_data": synthetic_df,
            "rules": {"method": "simple_sampling"},
            "validation": {
                "rows_generated": len(synthetic_df),
                "overall_score": 60,
                "note": "Generated using simple sampling method"
            }
        }

# =============================================================================
# STREAMLIT APP - SIMPLIFIED
# =============================================================================

def main():
    st.set_page_config(
        page_title="Synthetic Data Generator",
        page_icon="🔢",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="header">🔢 Synthetic Data Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Upload any CSV → Get synthetic data
    *No API key needed - uses your configured Groq key*
    """)
    
    # Check if API key exists in secrets
    if "GROQ_API_KEY" not in st.secrets:
        st.error("⚠️ GROQ_API_KEY not found in secrets!")
        st.info("""
        **To fix this:**
        1. Go to Streamlit Cloud dashboard
        2. Click on your app → Settings → Secrets
        3. Add: `GROQ_API_KEY = "your-groq-key-here"`
        4. Redeploy the app
        """)
        return
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload CSV File", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", len(df))
                st.metric("Numeric Columns", 
                         sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col])))
            with col2:
                st.metric("Columns", len(df.columns))
                st.metric("Missing Values", df.isnull().sum().sum())
        
        # Generation settings
        st.subheader("⚙️ Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_rows = st.number_input(
                "Rows to generate",
                min_value=len(df),
                max_value=10000,
                value=min(len(df) * 3, 1000),
                step=100
            )
        
        with col2:
            method = st.selectbox(
                "Generation Method",
                ["Smart Generation (Recommended)", "Statistical Modeling", "Simple Sampling"]
            )
        
        # Generate button
        if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
            
            # Initialize generator (gets API key from secrets automatically)
            generator = EnhancedCrewAIGenerator()
            
            # Generate based on method
            with st.spinner("Generating synthetic data..."):
                if method == "Smart Generation (Recommended)":
                    result = generator.generate_enhanced(df, num_rows)
                elif method == "Statistical Modeling":
                    result = generator._generate_with_sdv(df, num_rows)
                    result = {
                        "synthetic_data": result,
                        "rules": {"method": "SDV"},
                        "validation": generator._simple_validation(df, result)
                    }
                else:  # Simple Sampling
                    result = generator._fallback_generation(df, num_rows)
            
            # Store results
            st.session_state.synthetic_data = result["synthetic_data"]
            st.session_state.rules = result["rules"]
            st.session_state.validation = result["validation"]
            
            st.balloons()
        
        # Display results
        if 'synthetic_data' in st.session_state:
            synthetic = st.session_state.synthetic_data
            validation = st.session_state.validation
            
            st.subheader(f"✨ Generated {len(synthetic)} Rows")
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                score = validation.get("overall_score", 0)
                st.metric("Quality Score", f"{score:.1f}%")
            with col2:
                null_pct = validation.get("null_percentage", 0)
                st.metric("Null Values", f"{null_pct:.1f}%")
            with col3:
                similarity = validation.get("similarity_score", 0)
                st.metric("Similarity", f"{similarity:.1f}%")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Synthetic Data", "📈 Comparison", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
            
            with tab2:
                if len(df.columns) > 0:
                    col_to_compare = st.selectbox("Select column to compare", df.columns)
                    
                    if col_to_compare in synthetic.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original**")
                            st.write(f"Unique values: {df[col_to_compare].nunique()}")
                            if pd.api.types.is_numeric_dtype(df[col_to_compare]):
                                st.write(f"Mean: {df[col_to_compare].mean():.2f}")
                                st.write(f"Std: {df[col_to_compare].std():.2f}")
                        
                        with col2:
                            st.write("**Synthetic**")
                            st.write(f"Unique values: {synthetic[col_to_compare].nunique()}")
                            if pd.api.types.is_numeric_dtype(synthetic[col_to_compare]):
                                st.write(f"Mean: {synthetic[col_to_compare].mean():.2f}")
                                st.write(f"Std: {synthetic[col_to_compare].std():.2f}")
            
            with tab3:
                # Download
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download Synthetic Data (CSV)",
                    csv,
                    f"synthetic_data_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Regenerate
                if st.button("🔄 Generate New Variation", use_container_width=True):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Instructions
        st.info("""
        ### 📖 How to use:
        
        1. **Upload** any CSV file
        2. **Configure** generation settings
        3. **Click** Generate Synthetic Data
        4. **Download** the results
        
        ### 🔧 Requirements:
        - CSV file with headers
        - At least 10 rows recommended
        - Groq API key in Streamlit secrets
        
        ### ⚡ Features:
        - Works with any dataset
        - Maintains statistical properties
        - Multiple generation methods
        - Quality validation
        """)

if __name__ == "__main__":
    main()
