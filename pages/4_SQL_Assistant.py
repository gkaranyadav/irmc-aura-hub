# pages/4_📊_SQL_Assistant.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
import json
import re
import time
from groq import Groq
from auth import check_session

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"

# =============================================================================
# SMART DATABASE MANAGER
# =============================================================================
class SmartDatabaseManager:
    def __init__(self):
        self.connection = None
        self.schema_info = None
    
    def connect(self, host, port, username, password, database):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
            
            if self.connection.is_connected():
                # Get schema information
                self.schema_info = self._get_schema_info()
                return True, "Connected successfully!"
            else:
                return False, "Connection failed"
                
        except Error as e:
            return False, f"Connection error: {str(e)}"
    
    def _get_schema_info(self):
        """Get database schema information"""
        try:
            cursor = self.connection.cursor()
            
            # Get all tables
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            schema = {}
            for table in tables:
                # Get table structure
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                
                # Get sample data
                cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                sample_data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                
                schema[table] = {
                    'columns': [col[0] for col in columns],
                    'column_types': {col[0]: col[1] for col in columns},
                    'sample_data': sample_data,
                    'column_names': column_names
                }
            
            cursor.close()
            return schema
            
        except Error as e:
            st.error(f"Error getting schema: {str(e)}")
            return {}
    
    def analyze_sql_structure(self, sql_query):
        """Universal SQL analysis - works for ANY dataset"""
        query_upper = sql_query.upper().replace('\n', ' ').replace('\t', ' ')
        
        # FULL SCAN REQUIRED patterns (universal)
        full_scan_indicators = [
            "ROW_NUMBER()", "RANK()", "DENSE_RANK()",  # Window functions
            "SELECT *",  # Full table scans
            "DISTINCT",  # Deduplication needs full data
            "WHERE" in query_upper and "=" in query_upper,  # Specific lookups
            "ORDER BY" in query_upper and "LIMIT" in query_upper,  # Top N analysis
            "GROUP BY" in query_upper and "HAVING COUNT(*)" in query_upper,  # Duplicate finding
            "MIN(" in query_upper, "MAX(" in query_upper,  # Need full scan for min/max
        ]
        
        # Check if any full scan indicator is present
        requires_full_scan = any(indicator in query_upper if isinstance(indicator, str) else indicator for indicator in full_scan_indicators)
        
        return requires_full_scan
    
    def estimate_result_size(self, sql_query):
        """Estimate query result size (universal approach)"""
        query_upper = sql_query.upper()
        
        try:
            # Method 1: Check if it's an aggregate that returns few rows
            aggregate_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
            if any(agg in query_upper for agg in aggregate_functions) and "GROUP BY" in query_upper:
                return "small"  # Aggregates with GROUP BY return limited rows
            
            # Method 2: Check for explicit LIMIT clause
            limit_match = re.search(r"LIMIT\s+(\d+)", query_upper)
            if limit_match:
                limit_value = int(limit_match.group(1))
                if limit_value <= 10000:
                    return "small"
                else:
                    return "large"
            
            # Method 3: Check if it's a simple aggregate without grouping
            if any(agg in query_upper for agg in aggregate_functions) and "GROUP BY" not in query_upper:
                return "very_small"  # Single row result
            
            # Method 4: If we can't determine, assume large for safety
            return "large"
            
        except:
            return "unknown"
    
    def execute_smart_query(self, sql_query):
        """Universal smart query execution for ANY dataset"""
        try:
            # Basic safety check - only allow SELECT queries
            clean_query = sql_query.strip().upper()
            if not clean_query.startswith('SELECT'):
                return None, "Only SELECT queries are allowed for safety"
            
            # Analyze query structure
            requires_full_scan = self.analyze_sql_structure(sql_query)
            estimated_size = self.estimate_result_size(sql_query)
            
            cursor = self.connection.cursor()
            
            # Determine execution strategy
            if requires_full_scan or estimated_size in ["large", "unknown"]:
                # Full scan required for accuracy
                st.info("🔍 **Full dataset scan required for accurate results**")
                
                # Execute with progress indication
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Executing query...")
                cursor.execute(sql_query)
                
                # Stream results in batches for memory efficiency
                batch_size = 10000
                all_results = []
                column_names = [desc[0] for desc in cursor.description]
                
                while True:
                    batch = cursor.fetchmany(batch_size)
                    if not batch:
                        break
                    all_results.extend(batch)
                    
                    # Update progress (approximate)
                    if len(all_results) > 100000:
                        progress = min(0.9, len(all_results) / 500000)  # Cap at 90% for unknown size
                        progress_bar.progress(progress)
                
                progress_bar.progress(1.0)
                status_text.text("Query completed!")
                
                df = pd.DataFrame(all_results, columns=column_names)
                return df, f"Query executed successfully. Returned {len(df)} rows."
                
            else:
                # Safe to execute directly
                cursor.execute(sql_query)
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                
                df = pd.DataFrame(results, columns=column_names)
                return df, f"Query executed successfully. Returned {len(df)} rows."
            
        except Error as e:
            return None, f"Query error: {str(e)}"
        finally:
            if cursor:
                cursor.close()
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()

# =============================================================================
# SQL QUERY GENERATOR
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
        
        # Prepare schema context
        schema_context = self._prepare_schema_context(schema_info)
        
        prompt = f"""
        You are an expert SQL query generator. Convert the natural language question into a valid MySQL SELECT query.
        
        DATABASE SCHEMA:
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
        
        Return the SQL query only:
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a SQL expert that converts natural language to MySQL queries."},
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
    
    def _prepare_schema_context(self, schema_info):
        """Prepare schema information for the LLM"""
        context = ""
        for table_name, table_info in schema_info.items():
            context += f"Table: {table_name}\n"
            context += f"Columns: {', '.join(table_info['columns'])}\n"
            
            # Add sample data for context
            if table_info['sample_data']:
                context += "Sample data:\n"
                for i, row in enumerate(table_info['sample_data'][:3]):  # First 3 rows
                    row_data = dict(zip(table_info['column_names'], row))
                    context += f"  Row {i+1}: {row_data}\n"
            context += "\n"
        
        return context

# =============================================================================
# CHART GENERATOR
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
# MAIN SQL ASSISTANT APP
# =============================================================================
def main():
    # Check authentication first
    if not check_session():
        st.warning("Please login first to access SQL Assistant")
        st.stop()
    
    # Page configuration
    st.set_page_config(
        page_title="SQL Assistant - irmc Aura",
        page_icon="📊",
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
        .connection-status {
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .connected {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .disconnected {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">iRMC SQL_Assistant</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = SmartDatabaseManager()
    if 'sql_generator' not in st.session_state:
        st.session_state.sql_generator = SQLQueryGenerator()
    if 'chart_generator' not in st.session_state:
        st.session_state.chart_generator = ChartGenerator()
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔗 Database Connection", "💬 Query Assistant", "📈 Results & Analytics"])
    
    with tab1:
        st.header("🔗 Connect to Your Database")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Connection Details")
            host = st.text_input("Host", placeholder="e.g., irmc-irmc1.e.aivencloud.com")
            port = st.number_input("Port", value=28701, min_value=1, max_value=65535)
            username = st.text_input("Username", value="avnadmin")
            password = st.text_input("Password", type="password")
            database = st.text_input("Database", value="defaultdb")
            
            if st.button("🚀 Connect to Database", type="primary"):
                with st.spinner("Connecting to database..."):
                    success, message = st.session_state.db_manager.connect(
                        host, port, username, password, database
                    )
                    
                    if success:
                        st.session_state.db_connected = True
                        st.success(f"✅ {message}")
                        
                        # Show schema information
                        if st.session_state.db_manager.schema_info:
                            st.subheader("📋 Database Schema")
                            for table_name, table_info in st.session_state.db_manager.schema_info.items():
                                with st.expander(f"📊 Table: {table_name}"):
                                    st.write(f"**Columns:** {', '.join(table_info['columns'])}")
                                    
                                    if table_info['sample_data']:
                                        st.write("**Sample Data:**")
                                        sample_df = pd.DataFrame(
                                            table_info['sample_data'], 
                                            columns=table_info['column_names']
                                        )
                                        st.dataframe(sample_df, use_container_width=True)
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            st.subheader("Connection Status")
            if st.session_state.db_connected:
                st.markdown('<div class="connection-status connected">✅ Connected to Database</div>', unsafe_allow_html=True)
                
                # Show quick stats
                if st.session_state.db_manager.schema_info:
                    table_count = len(st.session_state.db_manager.schema_info)
                    total_columns = sum(len(info['columns']) for info in st.session_state.db_manager.schema_info.values())
                    
                    st.metric("Tables Available", table_count)
                    st.metric("Total Columns", total_columns)
                    
                    # Table list
                    st.subheader("📋 Available Tables")
                    for table_name in st.session_state.db_manager.schema_info.keys():
                        st.write(f"• {table_name}")
            else:
                st.markdown('<div class="connection-status disconnected">❌ Not Connected</div>', unsafe_allow_html=True)
                st.info("Please enter your database connection details and click 'Connect to Database'")
                
                # Aiven quick help
                with st.expander("ℹ️ Aiven MySQL Connection Help"):
                    st.write("""
                    **For Aiven MySQL:**
                    - **Host:** Your service host (e.g., `irmc-irmc1.e.aivencloud.com`)
                    - **Port:** Usually 28701 or similar
                    - **Username:** `avnadmin`
                    - **Password:** Your Aiven password
                    - **Database:** `defaultdb`
                    """)
            
            if st.session_state.db_connected and st.button("🔌 Disconnect"):
                st.session_state.db_manager.disconnect()
                st.session_state.db_connected = False
                st.rerun()
    
    with tab2:
        st.header("💬 Natural Language to SQL")
        
        if not st.session_state.db_connected:
            st.warning("🔗 Please connect to a database first in the 'Database Connection' tab")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Ask Your Question")
                natural_language_query = st.text_area(
                    "Enter your question in natural language:",
                    placeholder="e.g., Show me the top 5 customers by total spending\nFind customer John Smith's phone number\nWhich products are low in stock?\nWhat is the monthly revenue trend?",
                    height=100
                )
                
                if st.button("🔍 Generate SQL Query", type="primary"):
                    if natural_language_query:
                        with st.spinner("🤔 Generating SQL query..."):
                            sql_query, message = st.session_state.sql_generator.generate_sql(
                                natural_language_query,
                                st.session_state.db_manager.schema_info
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
                st.subheader("Example Questions")
                examples = [
                    "Show me the top 10 customers by total orders",
                    "Find customer John Smith's contact details",
                    "What is the average price of products by category?",
                    "List all orders from the last 30 days",
                    "Which products have less than 10 items in stock?",
                    "Show monthly revenue trends for this year"
                ]
                
                for example in examples:
                    if st.button(example, key=f"example_{example}", use_container_width=True):
                        st.session_state.current_nl_query = example
                        st.rerun()
            
            # Show generated SQL
            if hasattr(st.session_state, 'generated_sql'):
                st.subheader("Generated SQL Query")
                st.markdown(f'<div class="sql-box">{st.session_state.generated_sql}</div>', unsafe_allow_html=True)
                
                # Analyze the SQL structure
                requires_full_scan = st.session_state.db_manager.analyze_sql_structure(st.session_state.generated_sql)
                estimated_size = st.session_state.db_manager.estimate_result_size(st.session_state.generated_sql)
                
                if requires_full_scan or estimated_size == "large":
                    st.info("""
                    🔍 **Full Dataset Scan Required**
                    *This query will scan the entire dataset for accurate results*
                    • Specific record lookups
                    • Top N rankings  
                    • Data validation tasks
                    • Accurate aggregations
                    """)
                else:
                    st.info("""
                    ⚡ **Optimized Query**
                    *This query can execute efficiently*
                    • Aggregates and summaries
                    • Limited result sets
                    """)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("▶️ Execute Query", type="primary"):
                        with st.spinner("Executing query..."):
                            df, message = st.session_state.db_manager.execute_smart_query(st.session_state.generated_sql)
                            
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
        
        if not st.session_state.db_connected:
            st.warning("🔗 Please connect to a database and execute a query first")
        elif not hasattr(st.session_state, 'query_executed'):
            st.info("💬 Generate and execute a SQL query to see results here")
        else:
            # Results table
            st.subheader("📊 Query Results")
            st.dataframe(st.session_state.current_results, use_container_width=True)
            
            st.metric("Total Rows", len(st.session_state.current_results))
            
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
