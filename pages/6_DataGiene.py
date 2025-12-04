# pages/6_🔢_Synthetic_Data_Generator_crewAI.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime
import hashlib
import random

# =============================================================================
# CREW AI AGENTS - FIXED FOR ANY DATASET
# =============================================================================

from crewai import Agent, Task, Crew, Process, LLM
from langchain_groq import ChatGroq
import os

class SyntheticDataCrew:
    """Crew AI powered synthetic data generation - WORKS WITH ANY DATASET"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        # CRITICAL: Use LLM directly to force Groq
        self.llm = LLM(
            model="llama-3.1-70b-versatile",
            temperature=0.1,
            api_key=groq_api_key,
            provider="groq"  # Force Groq provider
        )
        self.setup_crew()
    
    def setup_crew(self):
        """Setup all specialized agents - GENERIC FOR ANY DATA"""
        
        # 🕵️ Agent 1: Data Detective (GENERIC)
        self.data_detective = Agent(
            role="Universal Data Detective",
            goal="Identify patterns and context in ANY dataset",
            backstory="""You are an expert at understanding ANY type of data. 
            You can analyze datasets from any domain - business, science, social, technical.
            You look for column meanings, data types, and overall context without assumptions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 📊 Agent 2: Statistical Analyst (GENERIC)
        self.statistical_analyst = Agent(
            role="Statistical Pattern Analyst",
            goal="Find statistical patterns in ANY data",
            backstory="""You are a statistician who finds patterns in any dataset.
            You analyze distributions, correlations, outliers - regardless of domain.
            You think in probabilities and data relationships.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🎯 Agent 3: Rule Miner (GENERIC)
        self.rule_miner = Agent(
            role="Data Relationship Miner",
            goal="Extract relationships and rules from data",
            backstory="""You find relationships between columns in any dataset.
            You look for: if column A has value X, then column B often has value Y.
            You find constraints and dependencies without domain bias.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🛠️ Agent 4: Constraint Engineer (GENERIC)
        self.constraint_engineer = Agent(
            role="Data Constraint Engineer",
            goal="Build generation constraints for ANY data",
            backstory="""You create rules for generating synthetic data.
            You work with any data type: numbers, text, dates, categories.
            You ensure synthetic data follows discovered patterns.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🎨 Agent 5: Synthetic Artist (GENERIC)
        self.synthetic_artist = Agent(
            role="Universal Synthetic Data Artist",
            goal="Generate synthetic data for ANY dataset",
            backstory="""You create realistic synthetic data for any domain.
            You maintain statistical properties while creating new combinations.
            You work with all data types and structures.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # 🧪 Agent 6: Quality Auditor (GENERIC)
        self.quality_auditor = Agent(
            role="Universal Data Quality Auditor",
            goal="Validate synthetic data quality for ANY dataset",
            backstory="""You validate synthetic data against original patterns.
            You check statistical similarity, rule compliance, and data quality.
            You work with any data type and structure.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full crew analysis of ANY dataset"""
        
        # Sample data for analysis
        sample_data = df.head(10).to_string()
        
        # Task 1: Understand the Data (GENERIC)
        task1 = Task(
            description=f"""Analyze this dataset WITHOUT domain assumptions:
            
            Columns: {list(df.columns)}
            
            First 10 rows:
            {sample_data}
            
            Analyze:
            1. What type of data is this? (business, scientific, transactional, etc.)
            2. What does each column likely represent?
            3. What are the data types? (numeric, text, date, categorical)
            4. Any obvious patterns or structures?
            
            Return JSON with: data_type, column_analysis, data_types""",
            agent=self.data_detective,
            expected_output="JSON analysis"
        )
        
        # Task 2: Statistical Analysis (GENERIC)
        task2 = Task(
            description=f"""Perform statistical analysis:
            
            Dataset shape: {df.shape}
            Columns: {list(df.columns)}
            
            Analyze:
            1. Basic statistics for numeric columns
            2. Value distributions for categorical columns
            3. Missing value patterns
            4. Any correlations between columns
            5. Outliers or unusual values
            
            Return JSON with: statistics, distributions, correlations, missing_data""",
            agent=self.statistical_analyst,
            expected_output="JSON statistical analysis",
            context=[task1]
        )
        
        # Task 3: Relationship Mining (GENERIC)
        task3 = Task(
            description=f"""Find relationships in the data:
            
            Based on data understanding and statistics, find:
            1. If-then relationships between columns
            2. Value constraints (e.g., age > 0, dates chronological)
            3. Column dependencies
            4. Unique value patterns
            
            Return JSON with: relationships, constraints, dependencies""",
            agent=self.rule_miner,
            expected_output="JSON relationships",
            context=[task1, task2]
        )
        
        # Task 4: Constraint Building (GENERIC)
        task4 = Task(
            description=f"""Create generation constraints:
            
            Based on all previous analysis, create:
            1. Value ranges for numeric columns
            2. Allowed values for categorical columns
            3. Relationship rules to maintain
            4. Data type constraints
            
            Return JSON with: constraints, rules, generation_guidelines""",
            agent=self.constraint_engineer,
            expected_output="JSON constraints",
            context=[task1, task2, task3]
        )
        
        # Create Crew
        analysis_crew = Crew(
            agents=[
                self.data_detective,
                self.statistical_analyst,
                self.rule_miner,
                self.constraint_engineer
            ],
            tasks=[task1, task2, task3, task4],
            verbose=False,
            process=Process.sequential
        )
        
        # Execute analysis
        analysis_result = analysis_crew.kickoff()
        
        # Parse results
        return self._parse_crew_output(analysis_result)
    
    def _parse_crew_output(self, crew_output: str) -> Dict[str, Any]:
        """Parse crew output into structured rules"""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', crew_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: Basic statistical rules
        return {
            "data_type": "tabular",
            "constraints": {},
            "relationships": []
        }
    
    def generate_synthetic_data(self, df: pd.DataFrame, rules: Dict, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data for ANY dataset"""
        
        # Task 5: Data Generation (GENERIC)
        task5 = Task(
            description=f"""Generate {num_rows} synthetic rows:
            
            Original data: {df.shape[0]} rows × {df.shape[1]} columns
            
            Requirements:
            1. Create NEW data that follows original patterns
            2. Maintain statistical properties
            3. Follow discovered constraints
            4. Ensure realistic value combinations
            
            Rules to follow: {json.dumps(rules, indent=2)[:1500]}...
            
            Return JSON array of synthetic rows with same columns.""",
            agent=self.synthetic_artist,
            expected_output="JSON array of synthetic data",
            context=[]
        )
        
        # Task 6: Quality Validation (GENERIC)
        task6 = Task(
            description=f"""Validate synthetic data:
            
            Check if synthetic data:
            1. Maintains statistical similarity
            2. Follows constraints and rules
            3. Has realistic value combinations
            4. Maintains data types
            
            Return JSON with: validation_score, issues, similarity_metrics""",
            agent=self.quality_auditor,
            expected_output="JSON validation report",
            context=[task5]
        )
        
        # Generation Crew
        generation_crew = Crew(
            agents=[self.synthetic_artist, self.quality_auditor],
            tasks=[task5, task6],
            verbose=False,
            process=Process.sequential
        )
        
        # Generate data
        generation_result = generation_crew.kickoff()
        
        # Extract synthetic data
        synthetic_data = self._extract_synthetic_data(generation_result, df)
        
        return synthetic_data
    
    def _extract_synthetic_data(self, crew_output: str, original_df: pd.DataFrame) -> pd.DataFrame:
        """Extract synthetic data from crew output"""
        try:
            # Try to get JSON array from output
            import re
            json_match = re.search(r'\[.*\]', crew_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data and isinstance(data, list):
                    return pd.DataFrame(data)
        except:
            pass
        
        # Fallback: Use SDV for generation
        try:
            from sdv.tabular import GaussianCopula
            model = GaussianCopula()
            model.fit(original_df)
            return model.sample(len(original_df) * 2)
        except:
            # Last fallback: Simple sampling with variation
            synthetic_rows = []
            for _ in range(min(1000, len(original_df) * 3)):
                row = {}
                for col in original_df.columns:
                    if original_df[col].dtype in ['int64', 'float64']:
                        # Add some variation to numeric columns
                        mean = original_df[col].mean()
                        std = original_df[col].std()
                        row[col] = np.random.normal(mean, std)
                    else:
                        # Sample from categorical/text
                        row[col] = original_df[col].sample(1).iloc[0]
                synthetic_rows.append(row)
            return pd.DataFrame(synthetic_rows)

# =============================================================================
# GENERIC GENERATOR (WORKS WITH ANY DATASET)
# =============================================================================

class EnhancedCrewAIGenerator:
    """Enhanced generator that works with ANY dataset"""
    
    def __init__(self, groq_api_key: str):
        self.crew = SyntheticDataCrew(groq_api_key)
    
    def generate_enhanced(self, df: pd.DataFrame, num_rows: int) -> Dict[str, Any]:
        """Enhanced generation for ANY dataset"""
        
        # Step 1: Crew Analysis (GENERIC)
        rules = self.crew.analyze_dataset(df)
        
        # Step 2: Generate with Crew (GENERIC)
        synthetic_df = self.crew.generate_synthetic_data(df, rules, num_rows)
        
        # Step 3: Validation (GENERIC)
        validation = self._validate_any_data(df, synthetic_df, rules)
        
        return {
            "synthetic_data": synthetic_df,
            "rules": rules,
            "validation": validation
        }
    
    def _validate_any_data(self, original: pd.DataFrame, synthetic: pd.DataFrame, rules: Dict) -> Dict:
        """Generic validation for ANY dataset"""
        
        validation = {
            "basic_checks": self._basic_data_checks(synthetic),
            "statistical_similarity": self._statistical_comparison(original, synthetic),
            "data_quality": self._data_quality_metrics(synthetic)
        }
        
        # Calculate overall score
        scores = []
        if validation["basic_checks"].get("score"):
            scores.append(validation["basic_checks"]["score"])
        if validation["statistical_similarity"].get("score"):
            scores.append(validation["statistical_similarity"]["score"])
        if validation["data_quality"].get("quality_score"):
            scores.append(validation["data_quality"]["quality_score"])
        
        validation["overall_score"] = np.mean(scores) if scores else 0
        
        return validation
    
    def _basic_data_checks(self, df: pd.DataFrame) -> Dict:
        """Basic checks for ANY data"""
        checks = {
            "has_data": len(df) > 0,
            "has_columns": len(df.columns) > 0,
            "no_empty_df": not df.empty,
            "columns_match_types": {},
            "issues": []
        }
        
        # Check each column
        for col in df.columns:
            null_percent = df[col].isnull().sum() / len(df) * 100
            if null_percent > 50:
                checks["issues"].append(f"High null values in {col}: {null_percent:.1f}%")
            
            # Check data type consistency
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    checks["columns_match_types"][col] = "numeric"
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    checks["columns_match_types"][col] = "datetime"
                else:
                    checks["columns_match_types"][col] = "categorical/text"
            except:
                checks["columns_match_types"][col] = "unknown"
        
        # Calculate score
        issue_count = len(checks["issues"])
        checks["score"] = max(0, 100 - (issue_count * 10))
        
        return checks
    
    def _statistical_comparison(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Compare statistical properties"""
        
        comparison = {
            "column_comparisons": {},
            "similarity_scores": []
        }
        
        for col in original.columns:
            if col in synthetic.columns:
                orig_series = original[col].dropna()
                synth_series = synthetic[col].dropna()
                
                if len(orig_series) > 0 and len(synth_series) > 0:
                    if pd.api.types.is_numeric_dtype(orig_series):
                        # Compare numeric stats
                        mean_diff = abs(orig_series.mean() - synth_series.mean()) / max(1, abs(orig_series.mean()))
                        std_diff = abs(orig_series.std() - synth_series.std()) / max(1, abs(orig_series.std()))
                        similarity = 100 * (1 - (mean_diff + std_diff) / 2)
                        
                        comparison["column_comparisons"][col] = {
                            "type": "numeric",
                            "original_mean": float(orig_series.mean()),
                            "synthetic_mean": float(synth_series.mean()),
                            "similarity": similarity
                        }
                    
                    else:
                        # Compare categorical/text distributions
                        orig_counts = orig_series.value_counts(normalize=True)
                        synth_counts = synth_series.value_counts(normalize=True)
                        
                        common = set(orig_counts.index) & set(synth_counts.index)
                        if len(common) > 0:
                            total_diff = sum(abs(orig_counts.get(c, 0) - synth_counts.get(c, 0)) for c in common)
                            similarity = 100 * (1 - total_diff)
                        else:
                            similarity = 0
                        
                        comparison["column_comparisons"][col] = {
                            "type": "categorical",
                            "original_unique": len(orig_counts),
                            "synthetic_unique": len(synth_counts),
                            "similarity": similarity
                        }
                    
                    comparison["similarity_scores"].append(similarity)
        
        if comparison["similarity_scores"]:
            comparison["score"] = np.mean(comparison["similarity_scores"])
        else:
            comparison["score"] = 0
        
        return comparison
    
    def _data_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics"""
        
        quality = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "null_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": df.duplicated().sum() / len(df) * 100
        }
        
        # Quality score (higher is better)
        quality["quality_score"] = max(0, 100 - quality["null_percentage"] - quality["duplicate_percentage"])
        
        return quality

# =============================================================================
# STREAMLIT APP - GENERIC FOR ANY DATASET
# =============================================================================

def main():
    st.set_page_config(
        page_title="Universal Synthetic Data Generator",
        page_icon="🌍",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .universal-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        .agent-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            border-left: 4px solid #667eea;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="universal-header">🌍 Universal Synthetic Data Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Works with **ANY** Dataset - No Domain Restrictions!
    
    **🕵️ Data Detective** → **📊 Statistical Analyst** → **🎯 Rule Miner**  
    **🛠️ Constraint Engineer** → **🎨 Synthetic Artist** → **🧪 Quality Auditor**
    """)
    
    # API Key
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key to use Crew AI")
        st.info("Get free API key from: https://console.groq.com/keys")
        return
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload ANY CSV Dataset", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return
        
        # Preview
        with st.expander("📋 Data Preview"):
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                numeric_cols = sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))
                st.metric("Numeric Columns", numeric_cols)
        
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
            generation_mode = st.selectbox(
                "Generation Mode",
                ["Smart (Crew AI)", "Statistical (SDV)", "Simple (Sampling)"]
            )
        
        # Agent visualization
        st.subheader("👥 AI Agents Ready for ANY Data")
        
        agents = [
            {"emoji": "🕵️", "name": "Data Detective", "role": "Understands any data"},
            {"emoji": "📊", "name": "Statistical Analyst", "role": "Finds patterns"},
            {"emoji": "🎯", "name": "Rule Miner", "role": "Extracts relationships"},
            {"emoji": "🛠️", "name": "Constraint Engineer", "role": "Builds rules"},
            {"emoji": "🎨", "name": "Synthetic Artist", "role": "Generates data"},
            {"emoji": "🧪", "name": "Quality Auditor", "role": "Validates quality"},
        ]
        
        cols = st.columns(6)
        for idx, (col, agent) in enumerate(zip(cols, agents)):
            with col:
                st.markdown(f"""
                <div class="agent-card">
                    <h3>{agent['emoji']} {agent['name']}</h3>
                    <p>{agent['role']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Generate button
        if st.button("🚀 Generate Synthetic Data", type="primary", use_container_width=True):
            
            try:
                # Initialize generator
                generator = EnhancedCrewAIGenerator(groq_api_key)
                
                # Generate based on mode
                with st.spinner("🤖 AI agents analyzing your data..."):
                    if generation_mode == "Smart (Crew AI)":
                        result = generator.generate_enhanced(df, num_rows)
                    elif generation_mode == "Statistical (SDV)":
                        # Use SDV directly
                        from sdv.tabular import GaussianCopula
                        model = GaussianCopula()
                        model.fit(df)
                        synthetic_df = model.sample(num_rows)
                        result = {
                            "synthetic_data": synthetic_df,
                            "rules": {"method": "SDV GaussianCopula"},
                            "validation": generator._validate_any_data(df, synthetic_df, {})
                        }
                    else:  # Simple sampling
                        synthetic_rows = []
                        for _ in range(num_rows):
                            row = {}
                            for col in df.columns:
                                row[col] = df[col].sample(1).iloc[0]
                            synthetic_df = pd.DataFrame(synthetic_rows)
                        result = {
                            "synthetic_data": synthetic_df,
                            "rules": {"method": "Simple Sampling"},
                            "validation": generator._validate_any_data(df, synthetic_df, {})
                        }
                
                # Store results
                st.session_state.synthetic_data = result["synthetic_data"]
                st.session_state.rules = result["rules"]
                st.session_state.validation = result["validation"]
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
        
        # Display results
        if 'synthetic_data' in st.session_state:
            synthetic = st.session_state.synthetic_data
            rules = st.session_state.rules
            validation = st.session_state.validation
            
            st.subheader(f"✨ Generated {len(synthetic)} Rows")
            
            # Overall score
            overall_score = validation.get("overall_score", 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                score_color = "🟢" if overall_score >= 90 else "🟡" if overall_score >= 70 else "🔴"
                st.metric("Overall Score", f"{overall_score:.1f}%")
            
            with col2:
                stats = validation.get("statistical_similarity", {}).get("score", 0)
                st.metric("Statistical Similarity", f"{stats:.1f}%")
            
            with col3:
                quality = validation.get("data_quality", {}).get("quality_score", 0)
                st.metric("Data Quality", f"{quality:.1f}%")
            
            with col4:
                null_pct = validation.get("data_quality", {}).get("null_percentage", 0)
                st.metric("Null Values", f"{null_pct:.1f}%")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Synthetic Data", "🧠 AI Analysis", "✅ Validation", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Quick comparison
                if len(df.columns) > 0:
                    compare_col = st.selectbox("Compare column", df.columns, key="compare")
                    if compare_col in synthetic.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original Distribution**")
                            try:
                                st.bar_chart(df[compare_col].value_counts().head(10))
                            except:
                                st.write("Cannot display chart for this column type")
                        with col2:
                            st.write("**Synthetic Distribution**")
                            try:
                                st.bar_chart(synthetic[compare_col].value_counts().head(10))
                            except:
                                st.write("Cannot display chart for this column type")
            
            with tab2:
                # Show analysis
                st.write("### 🧠 AI Analysis Results")
                
                # Data type
                data_type = rules.get("data_type", "Unknown")
                st.write(f"**Data Type Detected:** {data_type}")
                
                # Basic stats
                if "constraints" in rules:
                    st.write("**Constraints Found:**")
                    for key, value in list(rules["constraints"].items())[:5]:
                        st.write(f"- {key}: {value}")
            
            with tab3:
                # Detailed validation
                st.write("### ✅ Validation Report")
                
                # Basic checks
                basic = validation.get("basic_checks", {})
                if basic.get("issues"):
                    st.write("**Issues Found:**")
                    for issue in basic["issues"][:5]:
                        st.write(f"- ⚠️ {issue}")
                
                # Statistical comparison
                stats = validation.get("statistical_similarity", {})
                if stats.get("column_comparisons"):
                    st.write("**Column Comparisons (Top 10):**")
                    for col, comp in list(stats["column_comparisons"].items())[:10]:
                        similarity = comp.get("similarity", 0)
                        st.write(f"- {col}: {similarity:.1f}% similarity")
            
            with tab4:
                # Download
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download Synthetic Data (CSV)",
                    csv,
                    f"synthetic_data_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Download analysis
                if st.button("📥 Download AI Analysis (JSON)", use_container_width=True):
                    analysis_json = json.dumps(rules, indent=2)
                    st.download_button(
                        "Download",
                        analysis_json,
                        "ai_analysis.json",
                        "application/json"
                    )
                
                # Regenerate
                if st.button("🔄 Generate New Variation", use_container_width=True):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome
        st.info("""
        ### 🎯 Works with ANY Dataset:
        
        - **Business data** (sales, customers, transactions)
        - **Scientific data** (experiments, measurements)
        - **Social data** (surveys, demographics)
        - **Technical data** (logs, metrics, IoT)
        - **Healthcare, Finance, Education, etc.**
        
        **No domain restrictions!** The AI adapts to your data.
        """)
        
        # Example datasets
        with st.expander("📚 Try These Examples"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Example 1: Customer Data**")
                st.code("""customer_id,age,city,purchase_amount,date
001,25,NYC,150.50,2024-01-15
002,32,LA,89.99,2024-01-16
003,41,CHI,200.00,2024-01-17""")
            
            with col2:
                st.write("**Example 2: Sensor Data**")
                st.code("""timestamp,temperature,humidity,pressure
2024-01-01 10:00,22.5,65,1013
2024-01-01 11:00,23.1,63,1012
2024-01-01 12:00,24.0,60,1011""")

if __name__ == "__main__":
    main()
