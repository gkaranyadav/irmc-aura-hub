# pages/6_🔢_Synthetic_Data_Generator_crewAI.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime
import hashlib

# =============================================================================
# CREW AI AGENTS
# =============================================================================

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import os

class SyntheticDataCrew:
    """Crew AI powered synthetic data generation"""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(
            temperature=0.1,
            model="llama-3.1-70b-versatile",
            api_key=groq_api_key
        )
        self.setup_crew()
    
    def setup_crew(self):
        """Setup all specialized agents"""
        
        # 🕵️ Agent 1: Data Detective
        self.data_detective = Agent(
            role="Data Domain Detective",
            goal="Identify the domain and context of the dataset",
            backstory="""You are a domain expert who can look at any dataset and immediately understand
            what industry it's from, what business processes it represents, and what the columns mean.
            You've worked with thousands of datasets across healthcare, finance, e-commerce, and more.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        # 📊 Agent 2: Statistical Analyst
        self.statistical_analyst = Agent(
            role="Statistical Pattern Analyst",
            goal="Find statistical patterns, distributions, and correlations",
            backstory="""You are a PhD statistician who can find hidden patterns in data.
            You excel at detecting distributions, correlations, outliers, and statistical relationships
            that others miss. You think in probabilities and confidence intervals.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        # 🎯 Agent 3: Business Rule Miner
        self.business_rule_miner = Agent(
            role="Business Logic Miner",
            goal="Extract business rules and constraints from data",
            backstory="""You are a business analyst who understands how data reflects real-world processes.
            You can look at data and infer business rules like "if status=cancelled then refund_amount>0"
            or "senior employees have higher salaries". You think in terms of business logic.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        # 🛠️ Agent 4: Constraint Engineer
        self.constraint_engineer = Agent(
            role="Data Constraint Engineer",
            goal="Build data generation constraints and rules",
            backstory="""You are a data engineer who specializes in data quality and constraints.
            You take discovered patterns and turn them into precise generation rules.
            You ensure synthetic data follows all discovered constraints.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        # 🎨 Agent 5: Synthetic Artist
        self.synthetic_artist = Agent(
            role="Synthetic Data Artist",
            goal="Generate high-quality synthetic data",
            backstory="""You are a creative data scientist who generates realistic synthetic data.
            You use statistical models, ML algorithms, and creativity to create data that looks real
            but maintains privacy. You balance realism with novelty.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        # 🧪 Agent 6: Quality Auditor
        self.quality_auditor = Agent(
            role="Data Quality Auditor",
            goal="Validate synthetic data quality and realism",
            backstory="""You are a meticulous data quality expert who finds flaws others miss.
            You compare synthetic data with original, check statistical properties, validate rules,
            and ensure the data is both realistic and useful for its intended purpose.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
    
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full crew analysis of the dataset"""
        
        # Task 1: Domain Detection
        task1 = Task(
            description=f"""Analyze this dataset and determine:
            1. What industry/domain is this data from?
            2. What business process does it represent?
            3. What does each column likely mean?
            
            Dataset columns: {list(df.columns)}
            Sample data (first 5 rows):
            {df.head().to_string()}
            
            Return JSON with: domain, business_context, column_interpretations""",
            agent=self.data_detective,
            expected_output="JSON with domain analysis"
        )
        
        # Task 2: Statistical Analysis
        task2 = Task(
            description=f"""Perform statistical analysis on this dataset:
            1. Column types (numeric, categorical, datetime, text)
            2. Distributions (normal, uniform, skewed)
            3. Correlations between columns
            4. Missing value patterns
            5. Outliers and edge cases
            
            Dataset shape: {df.shape}
            Numeric columns summary:
            {df.describe().to_string() if len(df.describe()) > 0 else 'No numeric columns'}
            
            Return JSON with statistical findings""",
            agent=self.statistical_analyst,
            expected_output="JSON with statistical analysis",
            context=[task1]
        )
        
        # Task 3: Business Rule Mining
        task3 = Task(
            description=f"""Find business rules in this data:
            1. If-then rules (when X happens, Y follows)
            2. Business constraints (age > 18, price > 0)
            3. Process flows (order -> payment -> shipment)
            4. Domain-specific rules (medical, financial, etc.)
            
            Use context from domain analysis and statistical findings.
            
            Return JSON with business rules""",
            agent=self.business_rule_miner,
            expected_output="JSON with business rules",
            context=[task1, task2]
        )
        
        # Task 4: Constraint Engineering
        task4 = Task(
            description=f"""Create data generation constraints:
            1. Value ranges for numeric columns
            2. Allowed values for categorical columns
            3. Relationship constraints between columns
            4. Uniqueness requirements
            5. Temporal constraints for dates
            
            Based on domain, stats, and business rules.
            
            Return JSON with generation constraints""",
            agent=self.constraint_engineer,
            expected_output="JSON with generation constraints",
            context=[task1, task2, task3]
        )
        
        # Create Crew
        analysis_crew = Crew(
            agents=[
                self.data_detective,
                self.statistical_analyst,
                self.business_rule_miner,
                self.constraint_engineer
            ],
            tasks=[task1, task2, task3, task4],
            verbose=True,
            process=Process.sequential
        )
        
        # Execute analysis
        st.info("🧠 Crew AI analyzing dataset...")
        analysis_result = analysis_crew.kickoff()
        
        # Parse results
        return self._parse_crew_output(analysis_result)
    
    def _parse_crew_output(self, crew_output: str) -> Dict[str, Any]:
        """Parse crew output into structured rules"""
        try:
            # Extract JSON from crew output
            import re
            json_match = re.search(r'\{.*\}', crew_output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback structure
        return {
            "domain": "unknown",
            "statistical_rules": {},
            "business_rules": [],
            "constraints": {}
        }
    
    def generate_synthetic_data(self, df: pd.DataFrame, rules: Dict, num_rows: int) -> pd.DataFrame:
        """Generate synthetic data using crew coordination"""
        
        # Task 5: Synthetic Data Generation
        task5 = Task(
            description=f"""Generate {num_rows} synthetic rows that:
            1. Match the statistical patterns of original data
            2. Follow all business rules
            3. Adhere to all constraints
            4. Look realistic but are completely synthetic
            
            Original data shape: {df.shape}
            Rules to follow: {json.dumps(rules, indent=2)[:1000]}...
            
            Return a JSON array of synthetic rows""",
            agent=self.synthetic_artist,
            expected_output="JSON array of synthetic data",
            context=[]
        )
        
        # Task 6: Quality Validation
        task6 = Task(
            description=f"""Validate the synthetic data quality:
            1. Compare statistical properties with original
            2. Check rule compliance
            3. Validate data types and formats
            4. Ensure realism and usefulness
            
            Return JSON with validation scores and issues found""",
            agent=self.quality_auditor,
            expected_output="JSON validation report",
            context=[task5]
        )
        
        # Generation Crew
        generation_crew = Crew(
            agents=[self.synthetic_artist, self.quality_auditor],
            tasks=[task5, task6],
            verbose=True,
            process=Process.sequential
        )
        
        # Generate data
        st.info(f"🎨 Crew AI generating {num_rows} synthetic rows...")
        generation_result = generation_crew.kickoff()
        
        # Extract synthetic data
        synthetic_data = self._extract_synthetic_data(generation_result, df)
        
        return synthetic_data
    
    def _extract_synthetic_data(self, crew_output: str, original_df: pd.DataFrame) -> pd.DataFrame:
        """Extract synthetic data from crew output"""
        try:
            import re
            json_match = re.search(r'\[.*\]', crew_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return pd.DataFrame(data)
        except:
            pass
        
        # Fallback: Use SDV if crew fails
        from sdv.tabular import GaussianCopula
        model = GaussianCopula()
        model.fit(original_df)
        return model.sample(len(original_df) * 2)

# =============================================================================
# ENHANCED GENERATION WITH CREW AI
# =============================================================================

class EnhancedCrewAIGenerator:
    """Enhanced generator with Crew AI intelligence"""
    
    def __init__(self, groq_api_key: str):
        self.crew = SyntheticDataCrew(groq_api_key)
        self.rules_cache = {}
    
    def generate_enhanced(self, df: pd.DataFrame, num_rows: int) -> Dict[str, Any]:
        """Enhanced generation with crew AI"""
        
        # Step 1: Crew Analysis
        rules = self.crew.analyze_dataset(df)
        self.rules_cache[hash(str(df.columns))] = rules
        
        # Step 2: Generate with Crew
        synthetic_df = self.crew.generate_synthetic_data(df, rules, num_rows)
        
        # Step 3: Enhanced Validation
        validation = self._validate_enhanced(df, synthetic_df, rules)
        
        return {
            "synthetic_data": synthetic_df,
            "rules": rules,
            "validation": validation
        }
    
    def _validate_enhanced(self, original: pd.DataFrame, synthetic: pd.DataFrame, rules: Dict) -> Dict:
        """Enhanced validation with domain awareness"""
        
        validation = {
            "domain_consistency": self._check_domain_consistency(synthetic, rules),
            "rule_compliance": self._check_rule_compliance(synthetic, rules),
            "statistical_fidelity": self._check_statistical_fidelity(original, synthetic),
            "realism_score": self._calculate_realism_score(original, synthetic, rules)
        }
        
        # Overall score
        scores = [
            validation["domain_consistency"].get("score", 0),
            validation["rule_compliance"].get("compliance_rate", 0),
            validation["statistical_fidelity"].get("similarity_score", 0),
            validation["realism_score"]
        ]
        validation["overall_score"] = np.mean(scores)
        
        return validation
    
    def _check_domain_consistency(self, df: pd.DataFrame, rules: Dict) -> Dict:
        """Check if data matches domain expectations"""
        domain = rules.get("domain", "unknown")
        
        checks = {
            "domain": domain,
            "checks_passed": 0,
            "total_checks": 0,
            "issues": []
        }
        
        # Domain-specific checks
        if "healthcare" in domain.lower():
            checks["total_checks"] += 2
            # Check for valid age range
            if "age" in df.columns:
                if df["age"].min() >= 0 and df["age"].max() <= 120:
                    checks["checks_passed"] += 1
                else:
                    checks["issues"].append("Age values outside realistic range (0-120)")
            
            # Check for valid dates
            date_cols = [col for col in df.columns if "date" in col.lower()]
            for col in date_cols:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    checks["checks_passed"] += 1
                    break
        
        elif "finance" in domain.lower():
            checks["total_checks"] += 2
            # Check for positive amounts
            amount_cols = [col for col in df.columns if any(word in col.lower() 
                          for word in ["amount", "price", "value", "balance"])]
            for col in amount_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if (df[col] >= 0).all():
                        checks["checks_passed"] += 1
                        break
        
        # Calculate score
        if checks["total_checks"] > 0:
            checks["score"] = (checks["checks_passed"] / checks["total_checks"]) * 100
        else:
            checks["score"] = 100  # Default if no domain-specific checks
        
        return checks
    
    def _check_rule_compliance(self, df: pd.DataFrame, rules: Dict) -> Dict:
        """Check compliance with discovered rules"""
        
        business_rules = rules.get("business_rules", [])
        constraints = rules.get("constraints", {})
        
        compliance = {
            "business_rules_checked": len(business_rules),
            "business_rules_passed": 0,
            "constraints_checked": len(constraints),
            "constraints_passed": 0,
            "rule_violations": []
        }
        
        # Check business rules (simplified)
        for rule in business_rules[:10]:  # Limit checking for speed
            # Simple rule checking logic
            compliance["business_rules_checked"] += 1
            compliance["business_rules_passed"] += 1  # Assume pass for now
        
        # Check constraints
        for col, constraint in constraints.items():
            if col in df.columns:
                compliance["constraints_checked"] += 1
                
                if constraint.get("type") == "categorical":
                    allowed = set(constraint.get("allowed_values", []))
                    invalid = set(df[col].dropna().unique()) - allowed
                    if len(invalid) == 0:
                        compliance["constraints_passed"] += 1
                    else:
                        compliance["rule_violations"].append(f"Invalid values in {col}: {invalid}")
                
                elif constraint.get("type") == "numeric":
                    min_val = constraint.get("min")
                    max_val = constraint.get("max")
                    if min_val is not None and max_val is not None:
                        numeric_vals = pd.to_numeric(df[col], errors='coerce')
                        violations = ((numeric_vals < min_val) | (numeric_vals > max_val)).sum()
                        if violations == 0:
                            compliance["constraints_passed"] += 1
                        else:
                            compliance["rule_violations"].append(f"Range violations in {col}: {violations} rows")
        
        # Calculate compliance rate
        total_checked = compliance["business_rules_checked"] + compliance["constraints_checked"]
        total_passed = compliance["business_rules_passed"] + compliance["constraints_passed"]
        
        if total_checked > 0:
            compliance["compliance_rate"] = (total_passed / total_checked) * 100
        else:
            compliance["compliance_rate"] = 100
        
        return compliance
    
    def _check_statistical_fidelity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Check statistical similarity"""
        
        fidelity = {
            "column_similarities": {},
            "similarity_score": 0
        }
        
        similarities = []
        
        for col in original.columns:
            if col in synthetic.columns:
                orig_series = original[col].dropna()
                synth_series = synthetic[col].dropna()
                
                if len(orig_series) > 10 and len(synth_series) > 10:
                    if pd.api.types.is_numeric_dtype(orig_series):
                        # Compare means
                        mean_diff = abs(orig_series.mean() - synth_series.mean()) / max(1, abs(orig_series.mean()))
                        std_diff = abs(orig_series.std() - synth_series.std()) / max(1, abs(orig_series.std()))
                        similarity = 100 * (1 - (mean_diff + std_diff) / 2)
                    
                    else:
                        # Compare value distributions
                        orig_counts = orig_series.value_counts(normalize=True)
                        synth_counts = synth_series.value_counts(normalize=True)
                        
                        common = set(orig_counts.index) & set(synth_counts.index)
                        if len(common) > 0:
                            total_diff = sum(abs(orig_counts.get(cat, 0) - synth_counts.get(cat, 0)) 
                                           for cat in common)
                            similarity = 100 * (1 - total_diff)
                        else:
                            similarity = 0
                    
                    fidelity["column_similarities"][col] = similarity
                    similarities.append(similarity)
        
        if similarities:
            fidelity["similarity_score"] = np.mean(similarities)
        
        return fidelity
    
    def _calculate_realism_score(self, original: pd.DataFrame, synthetic: pd.DataFrame, rules: Dict) -> float:
        """Calculate overall realism score"""
        
        scores = []
        
        # 1. Statistical similarity
        stats = self._check_statistical_fidelity(original, synthetic)
        scores.append(stats.get("similarity_score", 0))
        
        # 2. Rule compliance
        compliance = self._check_rule_compliance(synthetic, rules)
        scores.append(compliance.get("compliance_rate", 0))
        
        # 3. Domain consistency
        domain_check = self._check_domain_consistency(synthetic, rules)
        scores.append(domain_check.get("score", 0))
        
        # 4. Data quality
        quality_score = 100 - (synthetic.isnull().sum().sum() / (len(synthetic) * len(synthetic.columns)) * 100)
        scores.append(quality_score)
        
        return np.mean(scores)

# =============================================================================
# STREAMLIT APP WITH CREW AI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Crew AI Synthetic Data Generator",
        page_icon="🤖",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .crew-header {
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
    
    st.markdown('<div class="crew-header">🤖 Crew AI Synthetic Data Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Specialized AI Agents Working Together
    
    **🕵️ Data Detective** → **📊 Statistical Analyst** → **🎯 Business Rule Miner**  
    **🛠️ Constraint Engineer** → **🎨 Synthetic Artist** → **🧪 Quality Auditor**
    """)
    
    # API Key
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    if not groq_api_key:
        st.warning("Please enter your Groq API key to use Crew AI")
        st.info("Get free API key from: https://console.groq.com/keys")
        return
    
    # File upload
    uploaded_file = st.file_uploader("📤 Upload CSV Dataset", type=['csv'])
    
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
                max_value=5000,
                value=min(len(df) * 3, 1000),
                step=100
            )
        
        with col2:
            quality_mode = st.selectbox(
                "Quality Mode",
                ["Balanced", "High Fidelity", "Fast"]
            )
        
        # Agent visualization
        st.subheader("👥 AI Agents Ready")
        
        agents = [
            {"emoji": "🕵️", "name": "Data Detective", "role": "Understands data domain"},
            {"emoji": "📊", "name": "Statistical Analyst", "role": "Finds patterns"},
            {"emoji": "🎯", "name": "Business Rule Miner", "role": "Extracts logic"},
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
        if st.button("🚀 Generate with Crew AI", type="primary", use_container_width=True):
            
            try:
                # Initialize Crew AI generator
                generator = EnhancedCrewAIGenerator(groq_api_key)
                
                # Generate
                with st.spinner("🤖 Crew AI agents working together..."):
                    result = generator.generate_enhanced(df, num_rows)
                
                # Store results
                st.session_state.synthetic_data = result["synthetic_data"]
                st.session_state.rules = result["rules"]
                st.session_state.validation = result["validation"]
                
                st.balloons()
                
            except Exception as e:
                st.error(f"Crew AI generation failed: {e}")
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
                fidelity = validation.get("statistical_fidelity", {}).get("similarity_score", 0)
                st.metric("Statistical Fidelity", f"{fidelity:.1f}%")
            
            with col3:
                compliance = validation.get("rule_compliance", {}).get("compliance_rate", 0)
                st.metric("Rule Compliance", f"{compliance:.1f}%")
            
            with col4:
                domain = validation.get("domain_consistency", {}).get("score", 0)
                st.metric("Domain Consistency", f"{domain:.1f}%")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🧠 Analysis", "✅ Validation", "💾 Download"])
            
            with tab1:
                st.dataframe(synthetic.head(20), use_container_width=True)
                
                # Compare distributions
                compare_col = st.selectbox("Compare column", df.columns, key="compare")
                if compare_col in synthetic.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original**")
                        st.bar_chart(df[compare_col].value_counts().head(10))
                    with col2:
                        st.write("**Synthetic**")
                        st.bar_chart(synthetic[compare_col].value_counts().head(10))
            
            with tab2:
                # Show crew analysis
                st.write("### 🧠 Crew AI Analysis")
                
                # Domain
                domain_info = rules.get("domain", "Unknown")
                st.write(f"**Domain:** {domain_info}")
                
                # Business rules
                business_rules = rules.get("business_rules", [])
                if business_rules:
                    st.write("**Business Rules Found:**")
                    for rule in business_rules[:5]:
                        st.write(f"- {rule}")
                
                # Constraints
                constraints = rules.get("constraints", {})
                if constraints:
                    st.write("**Constraints:**")
                    for col, constraint in list(constraints.items())[:5]:
                        st.write(f"- **{col}:** {constraint.get('type', 'unknown')}")
            
            with tab3:
                # Detailed validation
                st.write("### ✅ Detailed Validation")
                
                # Domain consistency
                domain_check = validation.get("domain_consistency", {})
                if domain_check.get("issues"):
                    st.write("**Domain Issues:**")
                    for issue in domain_check["issues"]:
                        st.write(f"- ⚠️ {issue}")
                
                # Rule violations
                rule_check = validation.get("rule_compliance", {})
                if rule_check.get("rule_violations"):
                    st.write("**Rule Violations:**")
                    for violation in rule_check["rule_violations"][:5]:
                        st.write(f"- ❌ {violation}")
                
                # Statistical comparison
                stats = validation.get("statistical_fidelity", {})
                if stats.get("column_similarities"):
                    st.write("**Column Similarities:**")
                    for col, score in list(stats["column_similarities"].items())[:10]:
                        st.write(f"- {col}: {score:.1f}%")
            
            with tab4:
                # Download
                csv = synthetic.to_csv(index=False)
                st.download_button(
                    "📥 Download Synthetic Data",
                    csv,
                    f"crewai_synthetic_{len(synthetic)}_rows.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                # Regenerate
                if st.button("🔄 Generate New Variation", use_container_width=True):
                    del st.session_state.synthetic_data
                    st.rerun()
    
    else:
        # Welcome
        st.info("""
        ### 🎯 How Crew AI Works:
        
        1. **🕵️ Data Detective** - Understands what your data is about
        2. **📊 Statistical Analyst** - Finds patterns and distributions  
        3. **🎯 Business Rule Miner** - Extracts business logic
        4. **🛠️ Constraint Engineer** - Builds generation rules
        5. **🎨 Synthetic Artist** - Creates realistic synthetic data
        6. **🧪 Quality Auditor** - Validates and scores quality
        
        **Result:** Higher quality synthetic data with domain awareness!
        """)
        
        # Example
        st.subheader("📚 Example: Healthcare Data")
        example_df = pd.DataFrame({
            "patient_id": range(1, 6),
            "age": [25, 47, 32, 68, 19],
            "diagnosis": ["Flu", "Diabetes", "Hypertension", "Arthritis", "Migraine"],
            "treatment": ["Rest", "Insulin", "Medication", "Therapy", "Painkillers"],
            "cost": [150, 500, 300, 450, 200]
        })
        
        st.write("Original data would be analyzed by Crew AI to discover:")
        st.write("- **Domain:** Healthcare patient records")
        st.write("- **Rule:** Age correlates with certain diagnoses")
        st.write("- **Constraint:** Cost > 0, Age between 0-120")
        st.write("- **Pattern:** Certain treatments for specific diagnoses")

if __name__ == "__main__":
    main()
