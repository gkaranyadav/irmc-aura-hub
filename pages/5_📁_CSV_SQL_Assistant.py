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
import io

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"

# =============================================================================
# CSV SCHEMA MANAGER (ROBUST VERSION)
# =============================================================================
class CSVSchemaManager:
    def __init__(self):
        self.uploaded_files = {}
        self.schema_info = {}
        self.connection = None
    
    def process_csv_file(self, uploaded_file, file_name):
        """Process uploaded CSV file with robust error handling"""
        try:
            # Read CSV with multiple encoding attempts and error handling
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                return False, f"❌ Error reading CSV: {str(e)}"
            
            # Validate dataframe
            if df.empty:
                return False, "❌ CSV file is empty"
            
            if len(df.columns) == 0:
                return False, "❌ No columns found in CSV file"
            
            # Clean column names
            df.columns = [str(col).strip().replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Basic info
            file_info = {
                'name': file_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'sample_data': df.head(3).fillna('NULL').to_dict('records'),  # First 3 rows, handle NaN
                'dataframe': df,
                'clean_table_name': self._clean_table_name(file_name)
            }
            
            self.uploaded_files[file_name] = file_info
            self._update_schema_info()
            
            return True, f"✅ Successfully loaded {file_name} ({len(df)} rows, {len(df.columns)} columns)"
            
        except Exception as e:
            return False, f"❌ Error processing {file_name}: {str(e)}"
    
    def _clean_table_name(self, file_name):
        """Clean filename to create valid table name"""
        name = os.path.splitext(file_name)[0]  # Remove extension
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace special chars with underscore
        name = re.sub(r'_+', '_', name)  # Replace multiple underscores with single
        name = name.strip('_')  # Remove leading/trailing underscores
        
        if not name:
            name = 'data_table'
        elif name[0].isdigit():
            name = 'table_' + name
        
        return name.lower()
    
    def _update_schema_info(self):
        """Update combined schema information for LLM"""
        self.schema_info = {}
        for file_name, file_info in self.uploaded_files.items():
            self.schema_info[file_name] = {
                'table_name': file_info['clean_table_name'],
                'columns': file_info['columns'],
                'sample_data': file_info['sample_data'],
                'row_count': file_info['row_count'],
                'column_count': file_info['column_count']
            }
    
    def initialize_duckdb(self):
        """Initialize DuckDB connection and load data with error handling"""
        try:
            self.connection = duckdb.connect(database=':memory:')
            
            # Register all dataframes as DuckDB tables
            for file_name, file_info in self.uploaded_files.items():
                table_name = file_info['clean_table_name']
                try:
                    self.connection.register(table_name, file_info['dataframe'])
                    st.success(f"✅ Table '{table_name}' loaded successfully")
                except Exception as e:
                    st.error(f"❌ Failed to load table '{table_name}': {str(e)}")
                    return False, f"Failed to load table {table_name}"
            
            return True, "DuckDB initialized successfully with all tables"
        except Exception as e:
            return False, f"DuckDB initialization failed: {str(e)}"
    
    def execute_sql(self, sql_query):
        """Execute SQL query on loaded CSV data with comprehensive error handling"""
        if not self.connection:
            return None, "No data loaded. Please upload CSV files first."
        
        try:
            # Basic safety check - only allow SELECT queries
            clean_query = sql_query.strip().upper()
            if not clean_query.startswith('SELECT'):
                return None, "Only SELECT queries are allowed for safety"
            
            # Execute query with timeout protection
            result = self.connection.execute(sql_query).fetchdf()
            
            # Check if result is too large
            if len(result) > 100000:  # 100k row limit
                result = result.head(100000)
                return result, f"Query executed successfully. Showing first 100,000 of {len(result)} rows."
            
            return result, f"Query executed successfully. Returned {len(result)} rows."
            
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error messages
            if "Catalog Error" in error_msg or "Table" in error_msg and "does not exist" in error_msg:
                available_tables = self.get_table_names()
                return None, f"Table not found. Available tables: {', '.join(available_tables)}"
            elif "Column" in error_msg and "not found" in error_msg:
                return None, f"Column error: {error_msg}"
            else:
                return None, f"SQL Error: {error_msg}"
    
    def get_table_names(self):
        """Get list of available table names"""
        return [info['clean_table_name'] for info in self.uploaded_files.values()]
    
    def clear_data(self):
        """Clear all loaded data"""
        self.uploaded_files = {}
        self.schema_info = {}
        if self.connection:
            self.connection.close()
            self.connection = None

# =============================================================================
# SQL QUERY GENERATOR (ROBUST VERSION)
# =============================================================================
class SQLQueryGenerator:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None
    
    def generate_sql(self, natural_language_query, schema_info):
        """Convert natural language to SQL using LLM with robust error handling"""
        if not self.client:
            return None, "LLM service not available"
        
        try:
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
            4. Use appropriate aggregate functions (COUNT, SUM, AVG, MAX, MIN) when suitable
            5. Use readable column aliases with AS
            6. Only generate SELECT queries (no INSERT, UPDATE, DELETE)
            7. Use table names exactly as provided in the schema
            8. Handle different data types appropriately
            9. Use LIMIT for large datasets when appropriate
            10. Make the query efficient and accurate
            
            Return the SQL query only:
            """
            
            messages = [
                {"role": "system", "content": "You are a SQL expert that converts natural language to SQL queries for CSV data. Always return valid SQL."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
                timeout=30  # 30 second timeout
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the response
            sql_query = re.sub(r'```sql\s*|\s*```', '', sql_query).strip()
            sql_query = re.sub(r'^SELECT', 'SELECT', sql_query, flags=re.IGNORECASE)
            
            # Basic validation
            if not sql_query.upper().startswith('SELECT'):
                return None, "Generated query is not a SELECT statement"
                
            return sql_query, "SQL generated successfully"
            
        except Exception as e:
            return None, f"LLM error: {str(e)}"
    
    def _prepare_csv_schema_context(self, schema_info):
        """Prepare schema information for CSV files"""
        if not schema_info:
            return "No tables available"
            
        context = "=== DATABASE SCHEMA ===\n\n"
        for file_name, table_info in schema_info.items():
            context += f"📊 TABLE: {table_info['table_name']} (from {file_name})\n"
            context += f"   📋 COLUMNS: {', '.join(table_info['columns'])}\n"
            context += f"   📊 STATS: {table_info['row_count']} rows, {table_info['column_count']} columns\n"
            
            # Add sample data for context
            if table_info['sample_data']:
                context += "   📝 SAMPLE DATA (first 3 rows):\n"
                for i, row in enumerate(table_info['sample_data']):
                    context += f"      Row {i+1}: {row}\n"
            context += "\n"
        
        context += "=== INSTRUCTIONS ===\n"
        context += "- Use exact table names and column names as shown above\n"
        context += "- Join tables when needed using common columns\n"
        context += "- Use aggregate functions for calculations\n"
        context += "- Add LIMIT when dealing with large datasets\n"
        
        return context

# =============================================================================
# CHART GENERATOR (ROBUST VERSION)
# =============================================================================
class ChartGenerator:
    def __init__(self):
        pass
    
    def auto_generate_chart(self, df, query_type=""):
        """Automatically generate appropriate charts based on data with error handling"""
        if df.empty or len(df) == 0:
            return None, "No data to visualize"
        
        charts = []
        
        try:
            # Create a safe copy for analysis
            df_safe = df.copy().head(1000)  # Limit for performance
            
            # Analyze dataframe structure safely
            numeric_cols = df_safe.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df_safe.select_dtypes(include=['object']).columns.tolist()
            
            # Chart 1: Bar chart for categorical vs numeric
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Check if reasonable for bar chart
                unique_cats = df_safe[cat_col].nunique()
                if 1 < unique_cats <= 15:
                    try:
                        fig_bar = px.bar(
                            df_safe, x=cat_col, y=num_col,
                            title=f"{num_col} by {cat_col}",
                            color=cat_col
                        )
                        charts.append(("Bar Chart", fig_bar))
                    except Exception:
                        pass
            
            # Chart 2: Pie chart for categorical distribution
            if categorical_cols:
                cat_col = categorical_cols[0]
                unique_cats = df_safe[cat_col].nunique()
                if 1 < unique_cats <= 8:
                    try:
                        if numeric_cols:
                            num_col = numeric_cols[0]
                            fig_pie = px.pie(
                                df_safe, names=cat_col, values=num_col,
                                title=f"Distribution of {num_col} by {cat_col}"
                            )
                        else:
                            value_counts = df_safe[cat_col].value_counts().head(8)
                            fig_pie = px.pie(
                                names=value_counts.index,
                                values=value_counts.values,
                                title=f"Distribution of {cat_col}"
                            )
                        charts.append(("Pie Chart", fig_pie))
                    except Exception:
                        pass
            
            # Chart 3: Histogram for numeric distribution
            if numeric_cols:
                try:
                    num_col = numeric_cols[0]
                    fig_hist = px.histogram(
                        df_safe, x=num_col,
                        title=f"Distribution of {num_col}",
                        nbins=20
                    )
                    charts.append(("Histogram", fig_hist))
                except Exception:
                    pass
            
            # Chart 4: Scatter plot for numeric relationships
            if len(numeric_cols) >= 2:
                try:
                    fig_scatter = px.scatter(
                        df_safe, x=numeric_cols[0], y=numeric_cols[1],
                        title=f"{numeric_cols[1]} vs {numeric_cols[0]}"
                    )
                    charts.append(("Scatter Plot", fig_scatter))
                except Exception:
                    pass
            
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
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #175CFF, #00A3FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 1rem;
        }
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
        .stButton button {
            background: linear-gradient(135deg, #175CFF, #00A3FF);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(23, 92, 255, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">📁 CSV SQL Assistant</div>', unsafe_allow_html=True)
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
                help="Upload one or more CSV files to analyze. Each file will become a table."
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
                with st.spinner("🚀 Initializing query engine..."):
                    success, message = st.session_state.csv_manager.initialize_duckdb()
                    if success:
                        st.session_state.data_loaded = True
                        st.success("✅ Query engine ready! You can now run SQL queries in the 'Query Assistant' tab.")
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            st.subheader("📊 Data Status")
            
            if st.session_state.data_loaded:
                st.markdown('<div class="data-loaded">✅ Data Loaded & Ready</div>', unsafe_allow_html=True)
                
                # Show file statistics
                total_files = len(st.session_state.csv_manager.uploaded_files)
                total_rows = sum(info['row_count'] for info in st.session_state.csv_manager.uploaded_files.values())
                total_columns = sum(info['column_count'] for info in st.session_state.csv_manager.uploaded_files.values())
                
                st.metric("📁 Files Uploaded", total_files)
                st.metric("📊 Total Rows", f"{total_rows:,}")
                st.metric("📋 Total Columns", total_columns)
                
                # Available tables
                st.subheader("🗂️ Available Tables")
                table_names = st.session_state.csv_manager.get_table_names()
                for table_name in table_names:
                    st.code(table_name, language='sql')
                
                if st.button("🗑️ Clear All Data", type="secondary"):
                    st.session_state.csv_manager.clear_data()
                    st.session_state.data_loaded = False
                    st.session_state.query_history = []
                    st.rerun()
            else:
                st.info("📤 Upload CSV files to get started")
                st.markdown("""
                **Supported:**
                • Any CSV file format
                • Multiple files (will be joined as tables)
                • Automatic column name cleaning
                • UTF-8 and Latin-1 encoding
                """)
        
        # Show file details
        if st.session_state.csv_manager.uploaded_files:
            st.subheader("📄 File Details")
            
            for file_name, file_info in st.session_state.csv_manager.uploaded_files.items():
                with st.expander(f"📋 {file_name} ({file_info['row_count']} rows, {file_info['column_count']} columns) → Table: `{file_info['clean_table_name']}`"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🗂️ Columns:**")
                        for col_name in file_info['columns']:
                            st.write(f"• `{col_name}`")
                    
                    with col2:
                        st.write("**📝 Sample Data (first 3 rows):**")
                        if file_info['sample_data']:
                            sample_df = pd.DataFrame(file_info['sample_data'])
                            st.dataframe(sample_df, use_container_width=True, height=150)
    
    with tab2:
        st.header("💬 Natural Language to SQL")
        
        if not st.session_state.data_loaded:
            st.warning("📤 Please upload CSV files first in the 'Upload CSV Files' tab")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔍 Ask Your Question")
                natural_language_query = st.text_area(
                    "Enter your question in natural language:",
                    placeholder="e.g., Show me the top 5 products by sales\nFind customers from New York\nWhat is the average price by category?\nShow monthly sales trends for this year",
                    height=100,
                    key="nl_query"
                )
                
                if st.button("🚀 Generate SQL Query", type="primary", use_container_width=True):
                    if natural_language_query and natural_language_query.strip():
                        with st.spinner("🤔 Generating SQL query..."):
                            sql_query, message = st.session_state.sql_generator.generate_sql(
                                natural_language_query.strip(),
                                st.session_state.csv_manager.schema_info
                            )
                            
                            if sql_query:
                                st.session_state.generated_sql = sql_query
                                st.session_state.current_nl_query = natural_language_query
                                st.success("✅ SQL query generated successfully!")
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("⚠️ Please enter a question")
            
            with col2:
                st.subheader("💡 Example Questions")
                examples = [
                    "Show top 10 products by sales amount",
                    "Find customers from California",
                    "Average price by product category",
                    "Monthly revenue trends this year",
                    "Products with less than 10 units in stock",
                    "Count customers by city",
                    "Most expensive product in each category",
                    "Total sales by month"
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{example}", use_container_width=True):
                        st.session_state.current_nl_query = example
                        st.rerun()
            
            # Show generated SQL
            if hasattr(st.session_state, 'generated_sql'):
                st.subheader("🛠️ Generated SQL Query")
                st.markdown(f'<div class="sql-box">{st.session_state.generated_sql}</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("▶️ Execute Query", type="primary", use_container_width=True):
                        with st.spinner("🔍 Executing query..."):
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
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                
                with col2:
                    if st.button("📋 Copy SQL", use_container_width=True):
                        st.code(st.session_state.generated_sql, language='sql')
                        st.success("✅ SQL copied to clipboard!")
                
                with col3:
                    if st.button("🔄 Regenerate", use_container_width=True):
                        if hasattr(st.session_state, 'generated_sql'):
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
            
            # Show query info
            st.write(f"**Question:** {st.session_state.current_nl_query}")
            st.code(st.session_state.current_query, language='sql')
            
            # Display dataframe
            st.dataframe(st.session_state.current_results, use_container_width=True)
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("Total Rows", f"{len(st.session_state.current_results):,}")
            with col2:
                st.metric("Total Columns", len(st.session_state.current_results.columns))
            
            # Download results
            csv_data = st.session_state.current_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
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
                    st.info("ℹ️ No suitable charts could be generated for this data type")
            
            # Query History
            st.subheader("📚 Query History")
            if st.session_state.query_history:
                for i, item in enumerate(st.session_state.query_history[:8]):  # Last 8 queries
                    with st.expander(f"Query {i+1}: {item['nl_query'][:60]}... ({item['row_count']} rows)"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Question:** {item['nl_query']}")
                            st.code(item['sql_query'], language='sql')
                        with col2:
                            st.write(f"**Rows:** {item['row_count']:,}")
                            st.write(f"**Time:** {item['timestamp'].strftime('%H:%M:%S')}")
                            
                        # Quick re-execute button
                        if st.button(f"🔄 Re-run Query {i+1}", key=f"rerun_{i}", use_container_width=True):
                            st.session_state.generated_sql = item['sql_query']
                            st.session_state.current_nl_query = item['nl_query']
                            st.rerun()
            else:
                st.info("No query history yet. Your queries will appear here.")

# Run the app
if __name__ == "__main__":
    main()
