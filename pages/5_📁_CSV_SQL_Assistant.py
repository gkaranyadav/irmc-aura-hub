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

    def generate_suggested_questions(self, schema_info):
        """Generate relevant questions based on the actual CSV data"""
        if not self.client:
            return self._get_fallback_questions()
        
        try:
            schema_context = self._prepare_csv_schema_context(schema_info)
            
            prompt = f"""
            Based on the following CSV data schema, generate 5-6 specific, relevant questions that would help someone understand and analyze this data. 
            Make the questions directly relevant to the actual columns and data structure shown below.
            
            DATA SCHEMA:
            {schema_context}
            
            Generate 5-6 specific, relevant questions that:
            - Are directly related to the columns and data available
            - Cover different aspects of analysis (trends, comparisons, insights)
            - Are practical and answerable with SQL queries
            - Are specific to this dataset's structure
            
            Return only the questions as a numbered list:
            """
            
            messages = [
                {"role": "system", "content": "You are a data analyst that generates insightful questions based on dataset schemas."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            questions_text = response.choices[0].message.content
            
            # Extract questions from the response
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                # Remove numbering and bullets
                clean_line = re.sub(r'^[\d\-•\.\s]+', '', line).strip()
                if clean_line and len(clean_line) > 10 and ('?' in clean_line or 'show' in clean_line.lower() or 'find' in clean_line.lower()):
                    questions.append(clean_line)
                    
            return questions[:6] if questions else self._get_fallback_questions()
            
        except Exception as e:
            return self._get_fallback_questions()

    def _get_fallback_questions(self):
        """Fallback questions if LLM fails"""
        return [
            "What are the top 10 records by the first numeric column?",
            "Show the distribution of values in the first categorical column",
            "What is the average of numeric columns?",
            "How many unique values are in each column?",
            "Show trends over time if date columns exist",
            "What are the most frequent values in the dataset?"
        ]

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
# MAIN CSV SQL ASSISTANT APP - CONVERSATIONAL VERSION
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
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e6f3ff;
            border-left: 5px solid #175CFF;
        }
        .assistant-message {
            background-color: #f0f8ff;
            border-left: 5px solid #00A3FF;
        }
        .sql-box {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Courier New', monospace;
            margin: 1rem 0;
            font-size: 0.9em;
        }
        .suggestion-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 12px 20px;
            font-weight: 600;
            margin: 5px;
            width: 100%;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .suggestion-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
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
        .big-suggest-btn {
            background: linear-gradient(135deg, #175CFF, #00A3FF);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 15px 25px;
            font-weight: 600;
            font-size: 16px;
            box-shadow: 0 4px 12px rgba(23, 92, 255, 0.3);
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin: 10px 0;
        }
        .big-suggest-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(23, 92, 255, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">📁 CSV SQL Assistant</div>', unsafe_allow_html=True)
        st.markdown("### Chat with your CSV data using natural language")
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
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'generating_questions' not in st.session_state:
        st.session_state.generating_questions = False
    
    # Sidebar for file upload
    with st.sidebar:
        st.title("📁 File Management")
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "Upload CSV Files",
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
                            st.success(f"✅ {uploaded_file.name}")
                        else:
                            st.error(message)
        
        # Initialize DuckDB when files are uploaded
        if (st.session_state.csv_manager.uploaded_files and 
            not st.session_state.data_loaded):
            with st.spinner("🚀 Initializing query engine..."):
                success, message = st.session_state.csv_manager.initialize_duckdb()
                if success:
                    st.session_state.data_loaded = True
                    st.success("✅ Query engine ready!")
                else:
                    st.error(f"❌ {message}")
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.subheader("📊 Data Status")
            total_files = len(st.session_state.csv_manager.uploaded_files)
            total_rows = sum(info['row_count'] for info in st.session_state.csv_manager.uploaded_files.values())
            
            st.metric("Files Uploaded", total_files)
            st.metric("Total Rows", f"{total_rows:,}")
            
            st.subheader("🗂️ Available Tables")
            table_names = st.session_state.csv_manager.get_table_names()
            for table_name in table_names:
                st.code(table_name, language='sql')
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                st.session_state.csv_manager.clear_data()
                st.session_state.data_loaded = False
                st.session_state.messages = []
                st.session_state.suggested_questions = []
                st.session_state.show_suggestions = False
                st.rerun()
    
    # Main chat area
    if not st.session_state.data_loaded:
        st.info("📤 Please upload CSV files using the sidebar to get started")
    else:
        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Assistant message with answer
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Aura:</strong> {msg["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show SQL query in expandable section
                if "sql_query" in msg:
                    with st.expander("🔍 View SQL Query", expanded=False):
                        st.markdown(f'<div class="sql-box">{msg["sql_query"]}</div>', unsafe_allow_html=True)
                
                # Show analytics in separate expandable section
                if "charts" in msg and msg["charts"]:
                    with st.expander("📈 View Analytics & Visualizations", expanded=False):
                        charts = msg["charts"]
                        cols = st.columns(2)
                        for i, (chart_name, chart_fig) in enumerate(charts):
                            with cols[i % 2]:
                                st.plotly_chart(chart_fig, use_container_width=True)
                
                # Show data preview in separate expandable section
                if "data_preview" in msg:
                    with st.expander("📊 View Data Results", expanded=False):
                        st.dataframe(msg["data_preview"], use_container_width=True)
                        st.metric("Total Rows", len(msg["data_preview"]))
        
        # BIG SUGGEST BUTTON - Show only if no messages yet
        if (st.session_state.data_loaded and 
            not st.session_state.show_suggestions and 
            not st.session_state.generating_questions and
            len(st.session_state.messages) == 0):
            
            st.markdown("---")
            st.subheader("💡 Get started with suggested questions")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("💡 Let Aura suggest questions based on your data", 
                            key="big_suggest_btn", 
                            use_container_width=True,
                            type="primary"):
                    st.session_state.generating_questions = True
                    st.rerun()
        
        # Generate questions when button is clicked
        if st.session_state.generating_questions:
            with st.spinner("🤔 Analyzing your data and generating relevant questions..."):
                # Generate questions based on actual CSV data
                st.session_state.suggested_questions = st.session_state.sql_generator.generate_suggested_questions(
                    st.session_state.csv_manager.schema_info
                )
                
                st.session_state.generating_questions = False
                st.session_state.show_suggestions = True
                st.rerun()
        
        # Show suggested questions if generated
        if st.session_state.show_suggestions and st.session_state.suggested_questions:
            st.markdown("---")
            st.subheader("💡 Suggested questions based on your data")
            st.write("Click on any question to ask Aura:")
            
            # Display questions as clickable buttons
            for i, question in enumerate(st.session_state.suggested_questions):
                col1, col2, col1 = st.columns([1, 3, 1])
                with col2:
                    if st.button(question, key=f"suggested_{i}", use_container_width=True):
                        # Auto-execute the selected question
                        st.session_state.selected_question = question
                        st.session_state.show_suggestions = False
                        st.rerun()
        
        # Handle selected question from suggestions
        if 'selected_question' in st.session_state:
            question = st.session_state.selected_question
            del st.session_state.selected_question
            
            # Add user question to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Generate and execute SQL immediately
            with st.spinner("🔍 Aura is thinking..."):
                # Generate SQL
                sql_query, sql_message = st.session_state.sql_generator.generate_sql(
                    question,
                    st.session_state.csv_manager.schema_info
                )
                
                if sql_query:
                    # Execute query
                    df, exec_message = st.session_state.csv_manager.execute_sql(sql_query)
                    
                    if df is not None:
                        # Generate answer
                        answer = f"I found {len(df)} records for your question. "
                        if len(df) > 0:
                            numeric_cols = df.select_dtypes(include=np.number).columns
                            if len(numeric_cols) > 0:
                                answer += f"The data includes numeric columns like {', '.join(numeric_cols[:2])}."
                            else:
                                answer += "Here's what I discovered in the data."
                        else:
                            answer = "I couldn't find any records matching your query."
                        
                        # Generate charts
                        charts, chart_message = st.session_state.chart_generator.auto_generate_chart(df, question)
                        
                        # Add assistant response to chat history
                        assistant_msg = {
                            "role": "assistant", 
                            "content": answer,
                            "sql_query": sql_query,
                            "data_preview": df,
                            "row_count": len(df)
                        }
                        
                        if charts:
                            assistant_msg["charts"] = charts
                        
                        st.session_state.messages.append(assistant_msg)
                    else:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"❌ Sorry, I encountered an error: {exec_message}"
                        })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ Sorry, I couldn't generate a query for that question: {sql_message}"
                    })
            
            st.rerun()
        
        # Regular chat input
        question = st.chat_input("Ask Aura about your data...")
        
        if question:
            # Add user question to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Generate and execute SQL immediately
            with st.spinner("🔍 Aura is thinking..."):
                # Generate SQL
                sql_query, sql_message = st.session_state.sql_generator.generate_sql(
                    question,
                    st.session_state.csv_manager.schema_info
                )
                
                if sql_query:
                    # Execute query
                    df, exec_message = st.session_state.csv_manager.execute_sql(sql_query)
                    
                    if df is not None:
                        # Generate answer
                        answer = f"I found {len(df)} records for your question. "
                        if len(df) > 0:
                            numeric_cols = df.select_dtypes(include=np.number).columns
                            if len(numeric_cols) > 0:
                                answer += f"The data includes numeric columns like {', '.join(numeric_cols[:2])}."
                            else:
                                answer += "Here's what I discovered in the data."
                        else:
                            answer = "I couldn't find any records matching your query."
                        
                        # Generate charts
                        charts, chart_message = st.session_state.chart_generator.auto_generate_chart(df, question)
                        
                        # Add assistant response to chat history
                        assistant_msg = {
                            "role": "assistant", 
                            "content": answer,
                            "sql_query": sql_query,
                            "data_preview": df,
                            "row_count": len(df)
                        }
                        
                        if charts:
                            assistant_msg["charts"] = charts
                        
                        st.session_state.messages.append(assistant_msg)
                    else:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"❌ Sorry, I encountered an error: {exec_message}"
                        })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"❌ Sorry, I couldn't generate a query for that question: {sql_message}"
                    })
            
            st.rerun()

# Run the app
if __name__ == "__main__":
    main()
