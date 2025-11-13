# pages/3_🌐_Graph_RAG.py
import streamlit as st
import tempfile
import os
import re
import io
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
from neo4j import GraphDatabase
import groq
import speech_recognition as sr
from gtts import gTTS
import networkx as nx
from pyvis.network import Network

# --------------------------------------------------------------------------------------
# CONFIG
class Config:
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "")

class GroqConfig:
    API_KEY = st.secrets.get("GROQ_API_KEY", "")
    MODEL = "llama-3.3-70b-versatile"

# --------------------------------------------------------------------------------------
# NEO4J SERVICE WITH GRAPH VISUALIZATION
class Neo4jService:
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        # Check if credentials are provided
        if not Neo4jConfig.URI or not Neo4jConfig.USERNAME or not Neo4jConfig.PASSWORD:
            st.sidebar.error("❌ Neo4j credentials not configured. Please check your Streamlit secrets.")
            return
            
        try:
            uri = Neo4jConfig.URI
            
            # Handle different URI formats for Neo4j Aura/Cloud
            if uri.startswith('neo4j+s://'):
                uri = uri.replace('neo4j+s://', 'neo4j+ssc://')
            elif uri.startswith('https://'):
                uri = uri.replace('https://', 'bolt://')
            elif uri.startswith('http://'):
                uri = uri.replace('http://', 'bolt://')
            
            self.driver = GraphDatabase.driver(
                uri,
                auth=(Neo4jConfig.USERNAME, Neo4jConfig.PASSWORD),
                connection_timeout=30,
                max_connection_lifetime=3600,
                keep_alive=True
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value == 1:
                    st.sidebar.success("✅ Connected to Neo4j Aura Cloud")
                else:
                    st.sidebar.error("❌ Neo4j connection test failed")
                    
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j Connection Failed: {str(e)}")
            if "DNS resolve" in str(e):
                st.sidebar.info("💡 Check your Neo4j Aura URI format. Try 'neo4j+ssc://' instead of 'neo4j+s://'")
            self.driver = None
    
    def create_document_graph(self, document_name, pages_data):
        if not self.driver:
            st.error("❌ No Neo4j connection available")
            return False
            
        try:
            with self.driver.session() as session:
                # Clear existing data for this document
                session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)
                
                # Create Document node
                session.run("CREATE (d:Document {name: $name, processed_at: datetime()})", name=document_name)
                
                # Create Pages and Paragraphs
                for page_num, page_data in pages_data.items():
                    session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (p:Page {number: $page_num, total_paragraphs: $para_count})
                    CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num, para_count=len(page_data['paragraphs']))
                    
                    # Create Paragraph nodes
                    for para_idx, paragraph in enumerate(page_data['paragraphs']):
                        entities = self._extract_simple_entities(paragraph)
                        
                        session.run("""
                        MATCH (p:Page {number: $page_num})<-[:HAS_PAGE]-(d:Document {name: $doc_name})
                        CREATE (para:Paragraph {
                            content: $content, 
                            chunk_id: $chunk_id,
                            paragraph_index: $para_idx,
                            entity_count: $entity_count
                        })
                        CREATE (p)-[:HAS_PARAGRAPH]->(para)
                        """, doc_name=document_name, page_num=page_num, 
                           content=paragraph, chunk_id=f"page_{page_num}_para_{para_idx}",
                           para_idx=para_idx, entity_count=len(entities))
                        
                        # Create entity relationships
                        for entity in entities:
                            session.run("""
                            MERGE (e:Entity {name: $name, type: $type})
                            WITH e
                            MATCH (para:Paragraph {chunk_id: $chunk_id})
                            MERGE (para)-[:MENTIONS {strength: 1.0}]->(e)
                            """, name=entity['name'], type=entity['type'], chunk_id=f"page_{page_num}_para_{para_idx}")
                
                st.sidebar.success(f"✅ Graph created for {document_name}")
                return True
                
        except Exception as e:
            st.error(f"❌ Graph creation failed: {str(e)}")
            return False
    
    def _extract_simple_entities(self, text):
        entities = []
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Company|Ltd|LLC|Corporation)\b',
            'TECH': r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Neural Network|Deep Learning|Blockchain|IoT|Cloud Computing)\b',
            'DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',
            'CONCEPT': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3 and match not in [entity['name'] for entity in entities]:
                    entities.append({"name": match, "type": entity_type})
        
        return entities[:10]
    
    def search_relationships(self, query_terms, document_name):
        if not self.driver:
            return []
            
        try:
            with self.driver.session() as session:
                results = []
                
                for term in query_terms:
                    # Find paragraphs mentioning the concept
                    paragraphs_result = session.run("""
                    MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
                    WHERE toLower(para.content) CONTAINS toLower($term)
                    RETURN para.content AS content, p.number AS page
                    ORDER BY length(para.content) ASC
                    LIMIT 3
                    """, doc_name=document_name, term=term)
                    
                    for record in paragraphs_result:
                        results.append({
                            "content": record["content"],
                            "page": record["page"],
                            "concept": term,
                            "type": "DIRECT_MENTION"
                        })
                
                # Find relationships between concepts
                if len(query_terms) >= 2:
                    for i, term1 in enumerate(query_terms):
                        for term2 in query_terms[i+1:]:
                            relationship_result = session.run("""
                            MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
                            WHERE toLower(para.content) CONTAINS toLower($term1) AND toLower(para.content) CONTAINS toLower($term2)
                            RETURN para.content AS content, p.number AS page
                            LIMIT 2
                            """, doc_name=document_name, term1=term1, term2=term2)
                            
                            for record in relationship_result:
                                results.append({
                                    "content": record["content"],
                                    "page": record["page"],
                                    "concept": f"{term1} & {term2}",
                                    "type": "RELATIONSHIP"
                                })
                
                return results
                
        except Exception as e:
            st.error(f"❌ Relationship search failed: {str(e)}")
            return []
    
    def get_graph_statistics(self, document_name):
        if not self.driver:
            return {}
            
        try:
            with self.driver.session() as session:
                stats = session.run("""
                MATCH (d:Document {name: $doc_name})
                OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:HAS_PARAGRAPH]->(para:Paragraph)
                OPTIONAL MATCH (para)-[:MENTIONS]->(e:Entity)
                RETURN 
                    count(DISTINCT p) AS page_count,
                    count(DISTINCT para) AS paragraph_count,
                    count(DISTINCT e) AS entity_count
                """, doc_name=document_name).single()
                
                return {
                    "pages": stats["page_count"] or 0,
                    "paragraphs": stats["paragraph_count"] or 0,
                    "entities": stats["entity_count"] or 0
                }
        except Exception as e:
            return {"pages": 0, "paragraphs": 0, "entities": 0}

    def create_interactive_network(self, document_name):
        """Create interactive network visualization"""
        if not self.driver:
            st.error("❌ No Neo4j connection for graph visualization")
            return None
            
        try:
            with self.driver.session() as session:
                # Get all graph data
                result = session.run("""
                MATCH (d:Document {name: $doc_name})
                OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:HAS_PARAGRAPH]->(para:Paragraph)
                OPTIONAL MATCH (para)-[:MENTIONS]->(e:Entity)
                RETURN d, p, para, e
                LIMIT 100
                """, doc_name=document_name)
                
                # Create network
                net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
                
                # Add document node
                net.add_node("document", label="Document", color="#FF6B6B", size=30, title=document_name)
                
                pages = set()
                paragraphs = set()
                entities = set()
                
                for record in result:
                    # Add pages
                    if record["p"]:
                        page_id = f"page_{record['p']['number']}"
                        if page_id not in pages:
                            net.add_node(page_id, label=f"Page {record['p']['number']}", color="#4ECDC4", size=25)
                            net.add_edge("document", page_id, label="HAS_PAGE", color="#45B7D1")
                            pages.add(page_id)
                    
                    # Add paragraphs
                    if record["para"]:
                        para_id = record['para']['chunk_id']
                        if para_id not in paragraphs:
                            net.add_node(para_id, label="Paragraph", color="#96CEB4", size=15, 
                                       title=record['para']['content'][:100] + "...")
                            page_num = para_id.split('_')[1]
                            net.add_edge(f"page_{page_num}", para_id, label="HAS_PARAGRAPH", color="#FECA57")
                            paragraphs.add(para_id)
                    
                    # Add entities
                    if record["e"]:
                        entity_id = f"entity_{record['e']['name']}"
                        if entity_id not in entities:
                            entity_type = record['e'].get('type', 'ENTITY')
                            net.add_node(entity_id, label=record['e']['name'], 
                                       color=self._get_entity_color(entity_type), size=20,
                                       title=f"{record['e']['name']} ({entity_type})")
                            entities.add(entity_id)
                        
                        # Connect entity to paragraph
                        if record["para"]:
                            para_id = record['para']['chunk_id']
                            net.add_edge(para_id, entity_id, label="MENTIONS", color="#FF9FF3")
                
                # Set physics options for better layout
                net.set_options("""
                {
                    "physics": {
                        "enabled": true,
                        "stabilization": {"iterations": 100},
                        "barnesHut": {
                            "gravitationalConstant": -8000,
                            "springLength": 95,
                            "springConstant": 0.04
                        }
                    },
                    "interaction": {
                        "hover": true,
                        "tooltipDelay": 200
                    }
                }
                """)
                
                # Generate HTML
                html_path = "temp_graph.html"
                net.save_graph(html_path)
                
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Clean up
                if os.path.exists(html_path):
                    os.remove(html_path)
                    
                return html_content
                
        except Exception as e:
            st.error(f"❌ Graph visualization failed: {str(e)}")
            return None
    
    def _get_entity_color(self, entity_type):
        color_map = {
            'PERSON': '#FF9FF3',
            'ORG': '#F368E0',
            'TECH': '#54A0FF',
            'DATE': '#FF9F43',
            'CONCEPT': '#5F27CD',
            'ENTITY': '#00D2D3'
        }
        return color_map.get(entity_type, '#00D2D3')

# --------------------------------------------------------------------------------------
# GROQ LLM SERVICE
class GroqService:
    def __init__(self):
        self.client = None
        self.neo4j = Neo4jService()
        
        # Check if API key is provided
        if not GroqConfig.API_KEY:
            st.sidebar.warning("⚠️ Groq API key not configured")
            return
            
        try:
            self.client = groq.Groq(api_key=GroqConfig.API_KEY)
            st.sidebar.success("✅ Groq Connected")
        except Exception as e:
            st.sidebar.error(f"❌ Groq Initialization Failed: {str(e)}")

    def generate_answer(self, question, document_name):
        if not document_name:
            return "Please upload and process a document first.", 0.0
            
        query_terms = self._extract_query_terms(question)
        relationship_data = self.neo4j.search_relationships(query_terms, document_name)
        
        if not relationship_data:
            return f"No relationship information found for '{question}'. Try asking about specific concepts, people, or organizations mentioned in the document.", 0.0
        
        confidence = min(len(relationship_data) / 5.0, 1.0)
        
        if self.client:
            return self._groq_analysis(question, relationship_data, confidence)
        else:
            return self._fallback_analysis(relationship_data, confidence)

    def _extract_query_terms(self, question):
        words = question.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'this', 'that', 'these', 'those'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:4]

    def _groq_analysis(self, question, relationship_data, confidence):
        context_text = "\n\n".join([
            f"Page {item['page']} ({item['type']}): {item['content'][:300]}..."
            for item in relationship_data
        ])
        
        user_prompt = f"QUESTION: {question}\n\nRELATIONSHIP DATA:\n{context_text}\n\nProvide a detailed answer explaining relationships and connections found in the document:"
        
        try:
            response = self.client.chat.completions.create(
                model=GroqConfig.MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional document analyst. Use the relationship data to provide insightful answers about connections between concepts, people, and organizations. Be specific about the relationships found."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            return answer + self._format_graph_sources(relationship_data), confidence
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return self._fallback_analysis(relationship_data, confidence)

    def _fallback_analysis(self, relationship_data, confidence):
        answer = "**Based on document relationships:**\n\n"
        for i, item in enumerate(relationship_data[:3], 1):
            answer += f"{i}. **Page {item['page']}** ({item['type']}): {item['content'][:200]}...\n\n"
        return answer + self._format_graph_sources(relationship_data), confidence

    def _format_graph_sources(self, relationship_data):
        if not relationship_data:
            return ""
        sources = "\n\n**🔗 Graph Sources:**\n"
        unique_pages = set(item['page'] for item in relationship_data)
        for page in sorted(unique_pages):
            sources += f"• Page {page}\n"
        return sources

    def analyze_document_overview(self, document_name):
        stats = self.neo4j.get_graph_statistics(document_name)
        
        overview = f"""
**📊 Knowledge Graph Overview - {document_name}**

- **📄 Pages Processed**: {stats.get('pages', 0)}
- **📝 Paragraphs Extracted**: {stats.get('paragraphs', 0)}  
- **🔗 Entities Identified**: {stats.get('entities', 0)}
- **🕸️ Total Graph Nodes**: {stats.get('pages', 0) + stats.get('paragraphs', 0) + stats.get('entities', 0)}

**💡 Ask about relationships between:**
- People and organizations
- Concepts and technologies  
- Dates and events
- Any specific terms from your document
        """
        return overview

# --------------------------------------------------------------------------------------
# DOCUMENT PROCESSOR
class DocumentProcessor:
    def __init__(self):
        self.neo4j = Neo4jService()

    def process_pdf(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            pages_data = self._extract_pdf_text(pdf_path)
            if not pages_data:
                st.error("❌ No text extracted from PDF.")
                return 0
                
            success = self.neo4j.create_document_graph(uploaded_file.name, pages_data)
            return len(pages_data) if success else 0
                
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            return 0
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_pdf_text(self, pdf_path):
        pdf_type = self._analyze_pdf_type(pdf_path)
        if pdf_type == "text_based":
            return self._extract_text_direct(pdf_path)
        else:
            return self._extract_text_ocr(pdf_path)

    def _analyze_pdf_type(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
            total_pages = len(reader.pages)
            return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"
        except:
            return "scanned"

    def _extract_text_direct(self, pdf_path):
        reader = PdfReader(pdf_path)
        pages_data = {}
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                paragraphs = self._split_into_paragraphs(text)
                pages_data[i] = {'paragraphs': paragraphs}
        return pages_data

    def _extract_text_ocr(self, pdf_path):
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=200)
            pages_data = {}
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text.strip():
                    paragraphs = self._split_into_paragraphs(text)
                    pages_data[i+1] = {'paragraphs': paragraphs}
            return pages_data
        except Exception as e:
            st.error(f"❌ OCR processing failed: {str(e)}")
            return {}

    def _split_into_paragraphs(self, text):
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > Config.CHUNK_SIZE:
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < Config.CHUNK_SIZE:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            final_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk.strip():
                    final_paragraphs.append(current_chunk.strip())
            else:
                final_paragraphs.append(para)
        return final_paragraphs

# --------------------------------------------------------------------------------------
# GRAPH RAG PAGE FUNCTION
def graph_rag_page():
    # Page header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🌐 Graph RAG - Knowledge Graph Analysis")
        st.markdown("### Explore document relationships with Neo4j Graph Database")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize services
    doc_processor = DocumentProcessor()
    llm_service = GroqService()
    
    # Initialize session state
    if 'graph_rag_processed' not in st.session_state:
        st.session_state.graph_rag_processed = False
    if 'graph_rag_messages' not in st.session_state:
        st.session_state.graph_rag_messages = []
    if 'graph_rag_document' not in st.session_state:
        st.session_state.graph_rag_document = None
    if 'show_graph_viz' not in st.session_state:
        st.session_state.show_graph_viz = False
    
    # Sidebar for document upload and controls
    with st.sidebar:
        st.header("📁 Document Controls")
        
        # Connection status
        if doc_processor.neo4j.driver:
            st.success("✅ Neo4j Connected")
        else:
            st.error("❌ Neo4j Disconnected")
            
        if llm_service.client:
            st.success("✅ Groq Connected")
        else:
            st.warning("⚠️ Groq Disconnected")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        if uploaded_file and not st.session_state.graph_rag_processed:
            if st.button("🚀 Process to Knowledge Graph", use_container_width=True):
                with st.spinner("🔄 Processing document and building knowledge graph..."):
                    page_count = doc_processor.process_pdf(uploaded_file)
                    if page_count > 0:
                        st.session_state.graph_rag_processed = True
                        st.session_state.graph_rag_document = uploaded_file.name
                        st.success(f"✅ Processed {page_count} pages!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")
        
        # Graph controls
        if st.session_state.graph_rag_processed:
            st.markdown("---")
            st.header("🕸️ Graph Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Show Graph", use_container_width=True):
                    st.session_state.show_graph_viz = True
                    st.rerun()
            with col2:
                if st.button("📈 Statistics", use_container_width=True):
                    stats = doc_processor.neo4j.get_graph_statistics(st.session_state.graph_rag_document)
                    st.session_state.graph_rag_messages.append({
                        "role": "assistant", 
                        "content": llm_service.analyze_document_overview(st.session_state.graph_rag_document)
                    })
                    st.rerun()
            
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.graph_rag_messages = []
                st.rerun()
    
    # Main content area
    if not st.session_state.graph_rag_processed:
        st.info("👆 Upload a PDF document from the sidebar to build a knowledge graph and explore relationships!")
        
        # Feature explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #175CFF;'>
                <h4>🔗 Relationship Discovery</h4>
                <p>Find connections between concepts, people, and organizations in your documents.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00A3FF;'>
                <h4>🕸️ Visual Knowledge Graph</h4>
                <p>See your document's structure as an interactive graph with entities and relationships.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4ECDC4;'>
                <h4>🤖 AI-Powered Analysis</h4>
                <p>Get intelligent answers about relationships using Groq's fast LLM.</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Show graph visualization if enabled
    if st.session_state.show_graph_viz:
        st.markdown("### 🕸️ Knowledge Graph Visualization")
        with st.spinner("Generating interactive graph..."):
            html_content = doc_processor.neo4j.create_interactive_network(st.session_state.graph_rag_document)
            if html_content:
                st.components.v1.html(html_content, height=600, scrolling=True)
                
                # Graph statistics
                stats = doc_processor.neo4j.get_graph_statistics(st.session_state.graph_rag_document)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Pages", stats.get('pages', 0))
                with col2:
                    st.metric("📝 Paragraphs", stats.get('paragraphs', 0))
                with col3:
                    st.metric("🔗 Entities", stats.get('entities', 0))
                with col4:
                    total_nodes = stats.get('pages', 0) + stats.get('paragraphs', 0) + stats.get('entities', 0)
                    st.metric("🕸️ Total Nodes", total_nodes)
            else:
                st.error("Failed to generate graph visualization")
        
        st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Ask About Document Relationships")
    
    # Display chat messages
    for msg in st.session_state.graph_rag_messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style='background: #e6f3ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #175CFF; margin-bottom: 1rem;'>
                <strong>You:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #00A3FF; margin-bottom: 1rem;'>
                <strong>Aura:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    question = st.chat_input("Ask about relationships in your document...")
    
    if question:
        # Add user question
        st.session_state.graph_rag_messages.append({"role": "user", "content": question})
        
        # Generate answer
        with st.spinner("🔍 Exploring graph relationships..."):
            answer, confidence = llm_service.generate_answer(question, st.session_state.graph_rag_document)
            st.session_state.graph_rag_messages.append({"role": "assistant", "content": answer})
        
        st.rerun()

# Run the page
if __name__ == "__main__":
    graph_rag_page()
