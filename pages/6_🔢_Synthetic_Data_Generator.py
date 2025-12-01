# pages/6_🔢_Synthetic_Data_Generator.py - CLEAN VERSION
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
import json

# =============================================================================
# CHECK SDV AVAILABILITY
# =============================================================================
def check_sdv_availability():
    """Check if SDV is available"""
    try:
        # SDV 1.29.1 imports
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import SingleTableMetadata
        st.session_state.sdv_available = True
        return True
    except ImportError as e:
        st.session_state.sdv_available = False
        st.error(f"❌ SDV is not installed. Please add 'sdv>=1.29.0' to requirements.txt")
        st.info("To install manually: `pip install sdv>=1.29.0`")
        return False

# =============================================================================
# REAL SDV-BASED GENERATOR (Updated for SDV 1.29.1)
# =============================================================================

class SDVDataGenerator:
    """REAL synthetic data generation using SDV library"""
    
    def __init__(self):
        self.available = check_sdv_availability()
        
    def generate_with_sdv(self, df: pd.DataFrame, num_rows: int, method: str = "ctgan") -> Optional[pd.DataFrame]:
        """
        Generate synthetic data using SDV methods
        """
        if not self.available:
            st.error("SDV not available")
            return None
        
        try:
            with st.spinner(f"🤖 Training {method.upper()} model on your data..."):
                # Step 1: Create metadata from dataframe
                from sdv.metadata import SingleTableMetadata
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(data=df)
                
                # Step 2: Choose model based on method
                if method == "gaussian":
                    from sdv.single_table import GaussianCopulaSynthesizer
                    model = GaussianCopulaSynthesizer(
                        metadata=metadata,
                        default_distribution='gamma'
                    )
                elif method == "tvae":
                    from sdv.single_table import TVAESynthesizer
                    model = TVAESynthesizer(
                        metadata=metadata,
                        epochs=100,
                        batch_size=50
                    )
                else:  # ctgan (default)
                    from sdv.single_table import CTGANSynthesizer
                    model = CTGANSynthesizer(
                        metadata=metadata,
                        epochs=100,
                        batch_size=50,
                        verbose=False
                    )
                
                # Step 3: Fit model
                model.fit(df)
                
                # Step 4: Generate synthetic data
                st.info("🎯 Generating synthetic data...")
                synthetic_data = model.sample(num_rows=num_rows)
                
                # Step 5: Evaluate quality
                try:
                    from sdv.evaluation.single_table import evaluate_quality
                    quality_report = evaluate_quality(
                        real_data=df,
                        synthetic_data=synthetic_data,
                        metadata=metadata
                    )
                    
                    # Get overall quality score
                    quality_score = quality_report.get_score()
                    st.success(f"✅ Data Quality Score: {quality_score:.3f}/1.0")
                    
                    # Show detailed scores
                    with st.expander("📊 View Detailed Quality Metrics"):
                        scores = quality_report.get_properties()
                        for prop_name, prop_score in scores.items():
                            st.write(f"**{prop_name}**: {prop_score.get('score', 0):.3f}")
                            
                except Exception as e:
                    st.info(f"⚠️ Quality evaluation skipped: {str(e)[:100]}")
                
                return synthetic_data
                
        except Exception as e:
            st.error(f"SDV generation failed: {str(e)}")
            return None
    
    def compare_methods(self, df: pd.DataFrame, num_rows: int = 50) -> Dict[str, Any]:
        """
        Compare different SDV methods
        """
        results = {}
        
        methods = ["gaussian", "ctgan", "tvae"]
        
        for method in methods:
            with st.spinner(f"Testing {method.upper()}..."):
                try:
                    synthetic = self.generate_with_sdv(df, num_rows, method)
                    if synthetic is not None:
                        # Basic quality checks
                        quality_metrics = {
                            "rows_generated": len(synthetic),
                            "columns": len(synthetic.columns),
                            "null_percentage": (synthetic.isnull().sum().sum() / (len(synthetic) * len(synthetic.columns))) * 100,
                            "data_types_match": all(str(synthetic[col].dtype) == str(df[col].dtype) 
                                                   for col in df.columns if col in synthetic.columns)
                        }
                        results[method] = quality_metrics
                except Exception as e:
                    results[method] = {"error": str(e)}
        
        return results

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    # Bypass auth for now - uncomment when auth is ready
    # from auth import check_session
    # if not check_session():
    #     st.warning("Please login first")
    #     st.stop()
    
    # Page config
    st.set_page_config(
        page_title="SDV Data Generator",
        page_icon="🧪",
        layout="wide"
    )
    
    # Header
    st.title("🧪 SDV Data Generator")
    st.markdown("**Using Synthetic Data Vault (SDV) v1.29.1 - REAL Generative AI for Tabular Data**")
    
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # Check SDV availability first
    if 'sdv_available' not in st.session_state:
        check_sdv_availability()
    
    if not st.session_state.get('sdv_available', False):
        st.error("""
        ## ❌ SDV is not available
        
        Please make sure SDV is installed:
        
        1. **Check requirements.txt** contains:
        ```
        sdv>=1.29.0
        ```
        
        2. **Install manually**:
        ```bash
        pip install sdv>=1.29.0
        ```
        
        3. **Restart the app** after installation
        """)
        st.stop()
    
    # Show SDV info
    with st.expander("📚 About SDV Technology", expanded=True):
        st.markdown("""
        ### **Synthetic Data Vault (SDV) v1.29.1** - Production-Grade Synthetic Data
        
        SDV uses **real generative AI models** for tabular data:
        
        **1. Gaussian Copula** 📊
        - Statistical model using copula functions
        - Preserves correlations between columns
        - Fast and good for simple datasets
        
        **2. CTGAN** 🧠
        - **Conditional Tabular Generative Adversarial Network**
        - Deep learning model specifically for tables
        - Learns complex patterns and distributions
        - Best for complex, real-world data
        
        **3. TVAE** ⚡
        - **Tabular Variational Autoencoder**
        - Neural network that learns latent representations
        - Good balance of quality and speed
        """)
    
    # Initialize generator
    if 'sdv_generator' not in st.session_state:
        st.session_state.sdv_generator = SDVDataGenerator()
    
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'generated_data' not in st.session_state:
        st.session_state.generated_data = None
    
    # Upload
    uploaded_file = st.file_uploader("📤 Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.original_df = df
            
            if df.empty:
                st.error("Empty file")
                return
            
            st.success(f"✅ Loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Preview
            with st.expander("📋 Data Preview", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    nulls = df.isnull().sum().sum()
                    st.metric("Missing Values", nulls)
                
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show data types
                st.write("**Data Types:**")
                type_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': [str(df[col].dtype) for col in df.columns],
                    'Unique': [df[col].nunique() for col in df.columns],
                    'Nulls': [df[col].isnull().sum() for col in df.columns]
                })
                st.dataframe(type_df, use_container_width=True, hide_index=True)
            
            # Generation controls
            st.subheader("⚙️ Generate with SDV")
            
            col1, col2 = st.columns(2)
            with col1:
                num_rows = st.number_input(
                    "Rows to generate",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    help="SDV can generate any number of rows"
                )
            
            with col2:
                method = st.selectbox(
                    "SDV Method",
                    ["ctgan", "gaussian", "tvae"],
                    help="CTGAN: Best for complex data, Gaussian: Fastest, TVAE: Balanced"
                )
            
            # Generate button
            if st.button("🚀 Generate with SDV", type="primary", use_container_width=True):
                generator = st.session_state.sdv_generator
                
                if not generator.available:
                    st.error("SDV is not available.")
                else:
                    generated = generator.generate_with_sdv(df, int(num_rows), method)
                    
                    if generated is not None:
                        st.session_state.generated_data = generated
                        st.success(f"✅ Generated {len(generated)} synthetic rows!")
                        st.balloons()
                    else:
                        st.error("Failed to generate data")
            
            # Compare methods button
            if len(df) >= 50 and len(df) <= 1000:
                if st.button("🔍 Compare SDV Methods", type="secondary"):
                    generator = st.session_state.sdv_generator
                    sample_df = df.sample(min(200, len(df))) if len(df) > 200 else df
                    results = generator.compare_methods(sample_df, 50)
                    
                    st.subheader("📊 Method Comparison (50 rows each)")
                    
                    for method, metrics in results.items():
                        with st.expander(f"{method.upper()} Results", expanded=False):
                            if "error" in metrics:
                                st.error(f"Error: {metrics['error']}")
                            else:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Rows", metrics["rows_generated"])
                                with col2:
                                    st.metric("Null %", f"{metrics['null_percentage']:.1f}%")
                                with col3:
                                    st.metric("Columns", metrics["columns"])
                                with col4:
                                    status = "✅" if metrics["data_types_match"] else "⚠️"
                                    st.metric("Types Match", status)
            
            # Show generated data
            if st.session_state.generated_data is not None:
                generated_df = st.session_state.generated_data
                
                st.subheader(f"📊 SDV-Generated Data ({len(generated_df)} rows)")
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Download"])
                
                with tab1:
                    st.dataframe(generated_df.head(20), use_container_width=True)
                    
                    # Compare with original
                    st.subheader("🔍 Comparison with Original")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data (Sample)**")
                        st.dataframe(df.head(5), use_container_width=True)
                    with col2:
                        st.write("**Generated Data (Sample)**")
                        st.dataframe(generated_df.head(5), use_container_width=True)
                
                with tab2:
                    # Statistical comparison
                    st.subheader("📈 Statistical Analysis")
                    
                    # Numeric columns comparison
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.write("**Numeric Columns Comparison:**")
                        
                        for col in numeric_cols[:3]:
                            if col in generated_df.columns:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Original {col}:**")
                                    st.write(f"Mean: {df[col].mean():.2f}")
                                    st.write(f"Std: {df[col].std():.2f}")
                                    st.write(f"Min: {df[col].min():.2f}")
                                    st.write(f"Max: {df[col].max():.2f}")
                                
                                with col2:
                                    st.write(f"**Generated {col}:**")
                                    st.write(f"Mean: {generated_df[col].mean():.2f}")
                                    st.write(f"Std: {generated_df[col].std():.2f}")
                                    st.write(f"Min: {generated_df[col].min():.2f}")
                                    st.write(f"Max: {generated_df[col].max():.2f}")
                    
                    # Categorical columns comparison
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        st.write("**Categorical Columns Distribution:**")
                        
                        for col in categorical_cols[:2]:
                            if col in generated_df.columns:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Original {col} Top Values:**")
                                    top_original = df[col].value_counts().head(5)
                                    for val, count in top_original.items():
                                        st.write(f"{val}: {count}")
                                
                                with col2:
                                    st.write(f"**Generated {col} Top Values:**")
                                    top_generated = generated_df[col].value_counts().head(5)
                                    for val, count in top_generated.items():
                                        st.write(f"{val}: {count}")
                
                with tab3:
                    st.subheader("📥 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = generated_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"sdv_generated_{len(generated_df)}_rows.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        json_str = generated_df.to_json(orient='records', indent=2)
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            f"sdv_generated_{len(generated_df)}_rows.json",
                            "application/json",
                            use_container_width=True
                        )
                    
                    st.write("---")
                    
                    # Regenerate options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Generate More", use_container_width=True):
                            st.session_state.generated_data = None
                            st.rerun()
                    
                    with col2:
                        if st.button("🆕 New File", use_container_width=True):
                            st.session_state.original_df = None
                            st.session_state.generated_data = None
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:
        # Instructions
        st.info("""
        ## 🧪 **SDV (Synthetic Data Vault) v1.29.1 Generator**
        
        ✅ **SDV is installed and ready!**
        
        ### **How It Works:**
        1. **Upload** a CSV file with your real data
        2. **Choose** a generation method (CTGAN, TVAE, or Gaussian Copula)
        3. **Generate** synthetic data that preserves patterns and relationships
        4. **Download** the synthetic dataset for testing, sharing, or analysis
        
        ### **Best Practices:**
        - **Dataset Size**: Works best with 100-10,000 rows
        - **Column Types**: Handles both numeric and categorical data
        - **Missing Values**: Can handle null values automatically
        - **Privacy**: Generated data is synthetic, protecting original data privacy
        
        **Upload a CSV to get started!**
        """)
    
    # Footer with status
    st.markdown("---")
    if st.session_state.get('sdv_available', False):
        st.success("✅ SDV v1.29.1 is ready")
    else:
        st.error("❌ SDV not available")

if __name__ == "__main__":
    main()
