# pages/5_📁_CSV_SQL_Assistant.py
import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import tempfile
import os
import re
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"

# =============================================================================
# CSV SCHEMA MANAGER
# =============================================================================
class CSVSchemaManager:
    def __init__(self):
        self.uploaded_files = {}
        self.schema_info = {}
        self.connection = None
    
    def detect_data_types(self, series):
        """Automatically detect data types for a pandas series"""
        if series.dtype == 'object':
            # Try to convert to datetime
            try:
                pd.to_datetime(series, errors='raise')
                return 'DATE'
            except:
                pass
            
            # Check if it's boolean
            if series.dropna().isin(['true', 'false', 'True', 'False', '1', '0', 'yes', 'no']).all():
                return 'BOOLEAN'
            
            # Check if it's numeric but stored as string
            try:
                pd.to_numeric(series, errors='raise')
                return 'NUMERIC'
            except:
                return 'TEXT'
        
        elif pd.api.types.is_numeric_dtype(series):
            if series.dropna().apply(float.is_integer).all():
                return 'INTEGER'
            else:
                return 'FLOAT'
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'DATE'
        
        return 'TEXT'
    
    def process_csv_file(self, uploaded_file, file_name):
        """Process uploaded CSV file and extract schema"""
        try:
            # Read CSV with flexible parsing
            df = pd.read_csv(uploaded_file)
            
            # Basic info
            file_info = {
                'name': file_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': {},
                'sample_data': df.head(10).to_dict('records'),
                'dataframe': df
            }
            
            # Analyze each column
            for col in df.columns:
                dtype = self.detect_data_types(df[col])
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                
                file_info['columns'][col] = {
                    'type': dtype,
                    'unique_values': unique_count,
                    'null_count': null_count,
                    'sample_values': df[col].dropna().head(5).tolist()
                }
            
            self.uploaded_files[file_name] = file_info
            self._update_schema_info()
            
            return True, f"✅ Successfully processed {file_name} ({len(df)} rows, {len(df.columns)} columns)"
            
        except Exception as e:
            return False, f"❌ Error processing {file_name}: {str(e)}"
    
    def _update_schema_info(self):
        """Update combined schema information for LLM"""
        self.schema_info = {}
        for file_name, file_info in self.uploaded_files.items():
            self.schema_info[file_name] = {
                'columns': list(file_info['columns'].keys()),
                'column_types': {col: info['type'] for col, info in file_info['columns'].items()},
                'sample_data': file_info['sample_data'],
                'row_count': file_info['row_count']
            }
    
    def initialize_duckdb(self):
        """Initialize DuckDB connection and load data"""
        try:
            self.connection = duckdb.connect(database=':memory:')
            
            # Register all dataframes as DuckDB tables
            for file_name, file_info in self.uploaded_files.items():
                table_name = self._clean_table_name(file_name)
                self.connection.register(table_name, file_info['dataframe'])
            
            return True, "DuckDB initialized successfully"
        except Exception as e:
            return False, f"DuckDB initialization failed: {str(e)}"
    
    def _clean_table_name(self, file_name):
        """Clean filename to create valid table name"""
        name = os.path.splitext(file_name)[0]  # Remove extension
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace special chars with underscore
        name = re.sub(r'_+', '_', name)  # Replace multiple underscores with single
        name = name.strip('_')  # Remove leading/trailing underscores
        
        if not name or name[0].isdigit():
            name = 'table_' + name
        
        return name.lower()
    
    def execute_sql(self, sql_query):
        """Execute SQL query on loaded CSV data"""
        if not self.connection:
            return None, "No data loaded. Please upload CSV files first."
        
        try:
            # Basic safety check - only allow SELECT queries
            clean_query = sql_query.strip().upper()
            if not clean_query.startswith('SELECT'):
                return None, "Only SELECT queries are allowed for safety"
            
            # Execute query
            result = self.connection.execute(sql_query).fetchdf()
            return result, f"Query executed successfully. Returned {len(result)} rows."
            
        except Exception as e:
            return None, f"Query error: {str(e)}"
    
    def get_table_names(self):
        """Get list of available table names"""
        return [self._clean_table_name(name) for name in self.uploaded_files.keys()]
    
    def clear_data(self):
        """Clear all loaded data"""
        self.uploaded_files = {}
        self.schema_info = {}
        if self.connection:
            self.connection.close()
            self.connection = None

# =============================================================================
# SQL QUERY GENERATOR (Reused from SQL Assistant)
# =============================================================================
class SQLQueryGenerator:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None
    
    def generate_sql(self, natural_language_query, schema_info):
        """Convert natural language to SQL using LLM"""
        if not self.client:
            return None, "LLM service not available"
        
        # Prepare schema context for CSV files
        schema_context = self._prepare_csv_schema_context(schema_info)
        
        prompt = f"""
        You are an expert SQL query generator. Convert the natural language question into a valid SQL query.
        
        AVAILABLE TABLES (from CSV files):
        {schema_context}
        
        NATURAL LANGUAGE QUESTION: {natural_language_query}
        
        IMPORTANT RULES:
        1. Return ONLY the SQL query, no explanations
        2. Use proper JOINs if multiple tables are needed
        3. Include WHERE clauses when filtering is implied
        4. Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) when suitable
        5. Use readable column aliases
        6. Only generate SELECT queries (no INSERT, UPDATE, DELETE)
        7. Make the query efficient and accurate
        8. Use table names exactly as provided
        9. Handle different data types appropriately (TEXT, INTEGER, FLOAT, DATE, BOOLEAN)
        
        Return the SQL query only:
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a SQL expert that converts natural language to SQL queries for CSV data."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response (remove markdown code blocks if present)
            sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query).strip()
            
            return sql_query, "SQL generated successfully"
            
        except Exception as e:
            return None, f"LLM error: {str(e)}"
    
    def _prepare_csv_schema_context(self, schema_info):
        """Prepare schema information for CSV files"""
        context = ""
        for table_name, table_info in schema_info.items():
            context += f"Table: {table_name}\n"
            context += f"Columns: {', '.join(table_info['columns'])}\n"
            context += f"Column Types: {table_info['column_types']}\n"
            context += f"Row Count: {table_info['row_count']}\n"
            
            # Add sample data for context
            if table_info['sample_data']:
                context += "Sample data (first few rows):\n"
                for i, row in enumerate(table_info['sample_data'][:2]):  # First 2 rows
                    context += f"  Row {i+1}: {row}\n"
            context += "\n"
        
        return context

# =============================================================================
# CHART GENERATOR (Reused from SQL Assistant)
# =============================================================================
class ChartGenerator:
    def __init__(self):
        pass
    
    def auto_generate_chart(self, df, query_type=""):
        """Automatically generate appropriate charts based on data"""
        if df.empty or len(df) == 0:
            return None, "No data to visualize"
        
        charts = []
        
        try:
            # Analyze dataframe structure
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            # Chart 1: Bar chart for categorical vs numeric
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                if len(df[cat_col].unique()) <= 20:  # Reasonable number of categories
                    fig_bar = px.bar(
                        df, x=cat_col, y=num_col,
                        title=f"{num_col} by {cat_col}",
                        color=cat_col
                    )
                    charts.append(("Bar Chart", fig_bar))
            
            # Chart 2: Line chart for time series
            if date_cols and numeric_cols:
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    fig_line = px.line(
                        df, x=date_col, y=num_col,
                        title=f"{num_col} Over Time"
                    )
                    charts.append(("Line Chart", fig_line))
                except:
                    pass
            
            # Chart 3: Pie chart for categorical distribution
            if categorical_cols and len(df[categorical_cols[0]].unique()) <= 10:
                cat_col = categorical_cols[0]
                if numeric_cols:
                    num_col = numeric_cols[0]
                    fig_pie = px.pie(
                        df, names=cat_col, values=num_col,
                        title=f"Distribution of {num_col} by {cat_col}"
                    )
                else:
                    value_counts = df[cat_col].value_counts()
                    fig_pie = px.pie(
                        names=value_counts.index,
                        values=value_counts.values,
                        title=f"Distribution of {cat_col}"
                    )
                charts.append(("Pie Chart", fig_pie))
            
            # Chart 4: Scatter plot for numeric relationships
            if len(numeric_cols) >= 2:
                fig_scatter = px.scatter(
                    df, x=numeric_cols[0], y=numeric_cols[1],
                    title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
                )
                charts.append(("Scatter Plot", fig_scatter))
            
            # Chart 5: Histogram for numeric distribution
            if numeric_cols:
                fig_hist = px.histogram(
                    df, x=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}"
                )
                charts.append(("Histogram", fig_hist))
            
            return charts, f"Generated {len(charts)} charts"
            
        except Exception as e:
            return None, f"Chart generation error: {str(e)}"

# =============================================================================
# MAIN CSV SQL ASSISTANT APP
# =============================================================================
def main():
    # Page configuration
    st.set_page_config(
        page_title="CSV SQL Assistant - irmc Aura",
        page_icon="📁",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e6f3ff;
            border-left: 4px solid #175CFF;
        }
        .assistant-message {
            background-color: #f0f8ff;
            border-left: 4px solid #00A3FF;
        }
        .sql-box {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            margin: 1rem 0;
        }
        .file-card {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .data-loaded {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📁 CSV SQL Assistant - irmc Aura")
        st.markdown("### Upload CSV files and query with natural language")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize session state
    if 'csv_manager' not in st.session_state:
        st.session_state.csv_manager = CSVSchemaManager()
    if 'sql_generator' not in st.session_state:
        st.session_state.sql_generator = SQLQueryGenerator()
    if 'chart_generator' not in st.session_state:
        st.session_state.chart_generator = ChartGenerator()
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload CSV Files", "💬 Query Assistant", "📈 Results & Analytics"])
    
    with tab1:
        st.header("📤 Upload Your CSV Files")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Files")
            uploaded_files = st.file_uploader(
                "Choose CSV files",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload one or more CSV files to analyze"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.csv_manager.uploaded_files:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            success, message = st.session_state.csv_manager.process_csv_file(
                                uploaded_file, uploaded_file.name
                            )
                            
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
            
            # Initialize DuckDB when files are uploaded
            if (st.session_state.csv_manager.uploaded_files and 
                not st.session_state.data_loaded):
                with st.spinner("Initializing query engine..."):
                    success, message = st.session_state.csv_manager.initialize_duckdb()
                    if success:
                        st.session_state.data_loaded = True
                        st.success("✅ Query engine ready! You can now run SQL queries.")
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            st.subheader("Data Status")
            
            if st.session_state.data_loaded:
                st.markdown('<div class="data-loaded">✅ Data Loaded & Ready</div>', unsafe_allow_html=True)
                
                # Show file statistics
                total_files = len(st.session_state.csv_manager.uploaded_files)
                total_rows = sum(info['row_count'] for info in st.session_state.csv_manager.uploaded_files.values())
                total_columns = sum(info['column_count'] for info in st.session_state.csv_manager.uploaded_files.values())
                
                st.metric("Files Uploaded", total_files)
                st.metric("Total Rows", total_rows)
                st.metric("Total Columns", total_columns)
                
                # Available tables
                st.subheader("📋 Available Tables")
                table_names = st.session_state.csv_manager.get_table_names()
                for table_name in table_names:
                    st.write(f"• `{table_name}`")
                
                if st.button("🗑️ Clear All Data"):
                    st.session_state.csv_manager.clear_data()
                    st.session_state.data_loaded = False
                    st.session_state.query_history = []
                    st.rerun()
            else:
                st.info("📤 Upload CSV files to get started")
        
        # Show file details
        if st.session_state.csv_manager.uploaded_files:
            st.subheader("📊 File Details")
            
            for file_name, file_info in st.session_state.csv_manager.uploaded_files.items():
                with st.expander(f"📄 {file_name} ({file_info['row_count']} rows, {file_info['column_count']} columns)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Columns & Types:**")
                        for col_name, col_info in file_info['columns'].items():
                            st.write(f"- `{col_name}`: {col_info['type']} "
                                   f"(Unique: {col_info['unique_values']}, "
                                   f"Nulls: {col_info['null_count']})")
                    
                    with col2:
                        st.write("**Sample Data:**")
                        if file_info['sample_data']:
                            sample_df = pd.DataFrame(file_info['sample_data'])
                            st.dataframe(sample_df, use_container_width=True, height=200)
    
    with tab2:
        st.header("💬 Natural Language to SQL")
        
        if not st.session_state.data_loaded:
            st.warning("📤 Please upload CSV files first in the 'Upload CSV Files' tab")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Ask Your Question")
                natural_language_query = st.text_area(
                    "Enter your question in natural language:",
                    placeholder="e.g., Show me the top 5 products by sales\nFind customers from New York\nWhat is the average price by category?\nShow monthly sales trends",
                    height=100
                )
                
                if st.button("🔍 Generate SQL Query", type="primary"):
                    if natural_language_query:
                        with st.spinner("🤔 Generating SQL query..."):
                            sql_query, message = st.session_state.sql_generator.generate_sql(
                                natural_language_query,
                                st.session_state.csv_manager.schema_info
                            )
                            
                            if sql_query:
                                st.session_state.generated_sql = sql_query
                                st.session_state.current_nl_query = natural_language_query
                                st.success("✅ SQL query generated successfully!")
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("Please enter a question")
            
            with col2:
                st.subheader("💡 Example Questions")
                examples = [
                    "Show me the top 10 products by sales amount",
                    "Find all customers from California",
                    "What is the average price by product category?",
                    "List orders from the last 30 days",
                    "Which products have less than 10 units in stock?",
                    "Show monthly revenue trends for this year",
                    "Find the most expensive product in each category",
                    "Count customers by city and state"
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{example}", use_container_width=True):
                        st.session_state.current_nl_query = example
                        st.rerun()
            
            # Show generated SQL
            if hasattr(st.session_state, 'generated_sql'):
                st.subheader("Generated SQL Query")
                st.markdown(f'<div class="sql-box">{st.session_state.generated_sql}</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("▶️ Execute Query", type="primary"):
                        with st.spinner("Executing query..."):
                            df, message = st.session_state.csv_manager.execute_sql(st.session_state.generated_sql)
                            
                            if df is not None:
                                st.session_state.current_results = df
                                st.session_state.current_query = st.session_state.generated_sql
                                st.session_state.query_executed = True
                                
                                # Add to history
                                history_item = {
                                    'timestamp': pd.Timestamp.now(),
                                    'nl_query': st.session_state.current_nl_query,
                                    'sql_query': st.session_state.generated_sql,
                                    'row_count': len(df)
                                }
                                st.session_state.query_history.insert(0, history_item)
                                
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                
                with col2:
                    if st.button("📋 Copy SQL"):
                        st.code(st.session_state.generated_sql, language='sql')
                        st.success("SQL copied to clipboard!")
                
                with col3:
                    if st.button("🔄 Regenerate SQL"):
                        del st.session_state.generated_sql
                        st.rerun()
    
    with tab3:
        st.header("📈 Query Results & Analytics")
        
        if not st.session_state.data_loaded:
            st.warning("📤 Please upload CSV files and execute a query first")
        elif not hasattr(st.session_state, 'query_executed'):
            st.info("💬 Generate and execute a SQL query to see results here")
        else:
            # Results table
            st.subheader("📊 Query Results")
            st.dataframe(st.session_state.current_results, use_container_width=True)
            
            st.metric("Total Rows", len(st.session_state.current_results))
            
            # Download results
            csv_data = st.session_state.current_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Auto-generated charts
            if len(st.session_state.current_results) > 0:
                st.subheader("📈 Automatic Visualizations")
                charts, chart_message = st.session_state.chart_generator.auto_generate_chart(
                    st.session_state.current_results,
                    st.session_state.current_nl_query
                )
                
                if charts:
                    st.success(f"✅ {chart_message}")
                    
                    # Display charts in columns
                    cols = st.columns(2)
                    for i, (chart_name, chart_fig) in enumerate(charts):
                        with cols[i % 2]:
                            st.plotly_chart(chart_fig, use_container_width=True)
                else:
                    st.info("No suitable charts could be generated for this data")
            
            # Query History
            st.subheader("📚 Query History")
            if st.session_state.query_history:
                for i, item in enumerate(st.session_state.query_history[:10]):  # Last 10 queries
                    with st.expander(f"Query {i+1}: {item['nl_query'][:50]}..."):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Natural Language:** {item['nl_query']}")
                            st.code(item['sql_query'], language='sql')
                        with col2:
                            st.write(f"**Rows:** {item['row_count']}")
                            st.write(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
                            
                        # Quick re-execute button
                        if st.button(f"🔄 Re-run Query {i+1}", key=f"rerun_{i}"):
                            st.session_state.generated_sql = item['sql_query']
                            st.session_state.current_nl_query = item['nl_query']
                            st.rerun()
            else:
                st.info("No query history yet")

# Run the app
if __name__ == "__main__":
    main()
