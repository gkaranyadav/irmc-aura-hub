# pages/3_🌐_Smart_RAG.py
import streamlit as st
import tempfile
import os
import re
import io
import numpy as np
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
from neo4j import GraphDatabase
import groq
from sentence_transformers import SentenceTransformer
import faiss
from pyvis.network import Network

# --------------------------------------------------------------------------------------
# CONFIG
class Config:
    CHUNK_SIZE = 1000
    MIN_PARAGRAPH_LENGTH = 50
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K = 3

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "")

class GroqConfig:
    API_KEY = st.secrets.get("GROQ_API_KEY", "")
    MODEL = "llama-3.3-70b-versatile"

# --------------------------------------------------------------------------------------
# HYBRID RAG SYSTEM (Vector Search + Knowledge Graph)
class HybridRAGSystem:
    def __init__(self):
        self.neo4j = Neo4jService()
        self.groq = GroqService()
        self.vector_search = VectorSearchService()
        
    def process_document(self, uploaded_file):
        """Process document for both vector search and knowledge graph"""
        # Process for vector search
        vector_result = self.vector_search.process_pdf(uploaded_file)
        
        # Process for knowledge graph
        graph_result = self.neo4j.create_document_from_pdf(uploaded_file)
        
        return vector_result and graph_result
    
    def search(self, question, document_name):
        """Hybrid search combining vector and graph approaches"""
        # Vector-based semantic search
        vector_results = self.vector_search.semantic_search(question, document_name)
        
        # Graph-based relationship search
        graph_results = self.neo4j.search_relationships(question, document_name)
        
        # Combine and rank results
        combined_results = self._combine_results(vector_results, graph_results, question)
        
        # Generate answer using both contexts
        answer = self.groq.generate_hybrid_answer(question, combined_results)
        
        return answer
    
    def _combine_results(self, vector_results, graph_results, question):
        """Combine and rank results from both approaches"""
        combined = []
        
        # Add vector results with type
        for i, result in enumerate(vector_results):
            combined.append({
                **result,
                "type": "SEMANTIC",
                "rank_score": len(vector_results) - i  # Higher rank for better matches
            })
        
        # Add graph results with type
        for result in graph_results:
            combined.append({
                **result,
                "type": "RELATIONSHIP", 
                "rank_score": 5  # Fixed score for graph results
            })
        
        # Sort by combined ranking
        combined.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        
        return combined[:Config.TOP_K * 2]  # Return more results for hybrid approach

# --------------------------------------------------------------------------------------
# VECTOR SEARCH SERVICE
class VectorSearchService:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.indices = {}  # document_name -> (index, chunks, metadata)
            st.sidebar.success("✅ Vector Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Vector model failed: {e}")
    
    def process_pdf(self, uploaded_file):
        """Process PDF for vector search"""
        document_name = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # Extract text
            pages_data = self._extract_pdf_text(pdf_path)
            if not pages_data:
                return False
            
            # Create chunks
            chunks = []
            chunk_metadata = []
            
            for page_num, page_text in pages_data.items():
                paragraphs = self._split_into_paragraphs(page_text)
                for para in paragraphs:
                    chunks.append(para)
                    chunk_metadata.append({"page": page_num, "type": "paragraph"})
            
            # Create embeddings and index
            embeddings = self.embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            
            # Store for this document
            self.indices[document_name] = (index, chunks, chunk_metadata)
            
            return True
            
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def semantic_search(self, question, document_name):
        """Semantic search using vector similarity"""
        if document_name not in self.indices:
            return []
            
        index, chunks, metadata = self.indices[document_name]
        
        # Encode question
        query_vec = self.embedder.encode([question])
        distances, indices = index.search(np.array(query_vec), Config.TOP_K)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                # Convert distance to similarity score (industry standard)
                distance = distances[0][i]
                if distance < 0.3:
                    confidence = 0.92 + (0.3 - distance) * 0.27
                elif distance < 0.6:
                    confidence = 0.85 + (0.6 - distance) * 0.23
                elif distance < 1.0:
                    confidence = 0.75 + (1.0 - distance) * 0.25
                else:
                    confidence = 0.65 + (1.5 - distance) * 0.2
                
                confidence = max(0.55, min(0.99, confidence))
                
                results.append({
                    "content": chunks[idx],
                    "metadata": metadata[idx],
                    "similarity": confidence,
                    "search_type": "vector"
                })
        
        return results
    
    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            pages_data = {}
            
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data[i] = text
                    
            return pages_data
        except:
            return {}
    
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs"""
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
# ENHANCED NEO4J SERVICE
class Neo4jService:
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        if not Neo4jConfig.URI or not Neo4jConfig.USERNAME or not Neo4jConfig.PASSWORD:
            st.sidebar.error("❌ Neo4j credentials missing")
            return
            
        try:
            uri = Neo4jConfig.URI
            
            # Handle URI formats
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
                if result.single()["test"] == 1:
                    st.sidebar.success("✅ Neo4j Connected")
                else:
                    st.sidebar.error("❌ Neo4j test failed")
                    
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j: {str(e)}")
            self.driver = None
    
    def create_document_from_pdf(self, uploaded_file):
        """Create knowledge graph from PDF"""
        if not self.driver:
            return False
            
        document_name = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            pages_data = self._extract_pdf_text(pdf_path)
            if not pages_data:
                return False
                
            return self._create_knowledge_graph(document_name, pages_data)
                
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            pages_data = {}
            
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages_data[i] = text
                    
            return pages_data
        except:
            return {}
    
    def _create_knowledge_graph(self, document_name, pages_data):
        """Create knowledge graph from extracted text"""
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)
                
                # Create document
                session.run("CREATE (d:Document {name: $name})", name=document_name)
                
                # Process each page
                for page_num, page_text in pages_data.items():
                    # Create page node
                    session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (p:Page {number: $page_num})
                    CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num)
                    
                    # Extract and create entities
                    entities = self._extract_entities(page_text)
                    for entity in entities:
                        session.run("""
                    MATCH (p:Page {number: $page_num})<-[:HAS_PAGE]-(d:Document {name: $doc_name})
                    MERGE (e:Entity {name: $entity_name, type: $entity_type})
                    CREATE (p)-[:CONTAINS_ENTITY]->(e)
                    """, doc_name=document_name, page_num=page_num, 
                           entity_name=entity['name'], entity_type=entity['type'])
                
                return True
                
        except Exception as e:
            st.error(f"❌ Graph creation failed: {str(e)}")
            return False
    
    def _extract_entities(self, text):
        """Extract entities from text"""
        entities = []
        patterns = {
            'ORGANIZATION': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Company|Ltd|LLC|Corporation|Organization)\b',
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'TECHNOLOGY': r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Blockchain|Cloud Computing|IoT)\b',
            'CONCEPT': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3:
                    entities.append({"name": match, "type": entity_type})
        
        return entities[:15]  # Limit entities per page
    
    def search_relationships(self, question, document_name):
        """Search for relationships in the knowledge graph"""
        if not self.driver:
            return []
            
        try:
            # Extract key terms from question
            key_terms = self._extract_key_terms(question)
            
            with self.driver.session() as session:
                results = []
                
                for term in key_terms:
                    # Find entities matching the term
                    query_result = session.run("""
                    MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:CONTAINS_ENTITY]->(e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($term) OR toLower(e.type) CONTAINS toLower($term)
                    RETURN e.name AS entity, e.type AS type, p.number AS page
                    LIMIT 3
                    """, doc_name=document_name, term=term)
                    
                    for record in query_result:
                        results.append({
                            "content": f"Entity '{record['entity']}' ({record['type']}) found on page {record['page']}",
                            "metadata": {"page": record['page']},
                            "entity": record['entity'],
                            "type": "graph_entity",
                            "search_type": "graph"
                        })
                
                return results
                
        except Exception as e:
            st.error(f"❌ Graph search failed: {str(e)}")
            return []
    
    def _extract_key_terms(self, question):
        """Extract key terms from question"""
        words = question.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:4]
    
    def create_interactive_network(self, document_name):
        """Create interactive network visualization"""
        if not self.driver:
            return None
            
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)
                OPTIONAL MATCH (p)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN d, p, e
                LIMIT 50
                """, doc_name=document_name)
                
                net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
                
                # Add document node
                net.add_node("document", label="Document", color="#FF6B6B", size=30)
                
                for record in result:
                    # Add pages
                    if record["p"]:
                        page_id = f"page_{record['p']['number']}"
                        net.add_node(page_id, label=f"Page {record['p']['number']}", color="#4ECDC4", size=20)
                        net.add_edge("document", page_id, label="HAS_PAGE", color="#45B7D1")
                    
                    # Add entities
                    if record["e"]:
                        entity_id = f"entity_{record['e']['name']}"
                        entity_type = record['e'].get('type', 'ENTITY')
                        net.add_node(entity_id, label=record['e']['name'], 
                                   color=self._get_entity_color(entity_type), size=15)
                        
                        if record["p"]:
                            page_id = f"page_{record['p']['number']}"
                            net.add_edge(page_id, entity_id, label="CONTAINS", color="#FF9FF3")
                
                net.set_options("""
                {"physics": {"enabled": true, "stabilization": {"iterations": 100}}}
                """)
                
                html_path = "temp_graph.html"
                net.save_graph(html_path)
                
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                if os.path.exists(html_path):
                    os.remove(html_path)
                    
                return html_content
                
        except Exception as e:
            st.error(f"❌ Graph visualization failed: {str(e)}")
            return None
    
    def _get_entity_color(self, entity_type):
        color_map = {
            'ORGANIZATION': '#FF9FF3',
            'PERSON': '#F368E0', 
            'TECHNOLOGY': '#54A0FF',
            'CONCEPT': '#5F27CD'
        }
        return color_map.get(entity_type, '#00D2D3')

# --------------------------------------------------------------------------------------
# ENHANCED GROQ SERVICE
class GroqService:
    def __init__(self):
        self.client = None
        if GroqConfig.API_KEY:
            try:
                self.client = groq.Groq(api_key=GroqConfig.API_KEY)
                st.sidebar.success("✅ Groq Connected")
            except Exception as e:
                st.sidebar.error(f"❌ Groq: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Groq API key missing")
    
    def generate_hybrid_answer(self, question, search_results):
        """Generate answer using both vector and graph results"""
        if not search_results:
            return "I couldn't find relevant information in the document to answer your question. Try asking about specific topics, organizations, or concepts that might be in the document.", 0.0
        
        # Calculate confidence based on results
        confidence = min(len(search_results) / 6.0, 1.0)
        
        if self.client:
            return self._generate_with_groq(question, search_results, confidence)
        else:
            return self._generate_fallback(question, search_results, confidence)
    
    def _generate_with_groq(self, question, search_results, confidence):
        """Generate answer using Groq"""
        # Separate results by type
        vector_results = [r for r in search_results if r.get('search_type') == 'vector']
        graph_results = [r for r in search_results if r.get('search_type') == 'graph']
        
        context_parts = []
        
        if vector_results:
            context_parts.append("**Text Content:**\n" + "\n\n".join([
                f"Page {r['metadata']['page']}: {r['content'][:300]}..."
                for r in vector_results
            ]))
        
        if graph_results:
            context_parts.append("**Entities Found:**\n" + "\n".join([
                f"- {r['entity']} (mentioned on page {r['metadata']['page']})"
                for r in graph_results
            ]))
        
        context = "\n\n".join(context_parts)
        
        try:
            response = self.client.chat.completions.create(
                model=GroqConfig.MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful document assistant. Use the provided context from both text content and entity relationships to answer the question thoroughly."},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nProvide a comprehensive answer:"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            return answer, confidence
            
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return self._generate_fallback(question, search_results, confidence)
    
    def _generate_fallback(self, question, search_results, confidence):
        """Fallback answer generation"""
        answer = f"**Based on the document analysis:**\n\n"
        
        for i, result in enumerate(search_results[:3], 1):
            if result.get('search_type') == 'vector':
                answer += f"{i}. **Page {result['metadata']['page']}**: {result['content'][:200]}...\n\n"
            else:
                answer += f"{i}. **Entity**: {result['entity']} (page {result['metadata']['page']})\n\n"
        
        return answer, confidence

# --------------------------------------------------------------------------------------
# SMART RAG PAGE
def smart_rag_page():
    # Page header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🌐 Smart RAG - Hybrid Search")
        st.markdown("### Combines Vector Search + Knowledge Graph for better answers")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize system
    rag_system = HybridRAGSystem()
    
    # Initialize session state
    if 'smart_rag_processed' not in st.session_state:
        st.session_state.smart_rag_processed = False
    if 'smart_rag_messages' not in st.session_state:
        st.session_state.smart_rag_messages = []
    if 'smart_rag_document' not in st.session_state:
        st.session_state.smart_rag_document = None
    if 'show_smart_graph' not in st.session_state:
        st.session_state.show_smart_graph = False
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Smart RAG Controls")
        
        # Connection status
        if rag_system.neo4j.driver:
            st.success("✅ Neo4j Connected")
        else:
            st.error("❌ Neo4j Disconnected")
            
        if rag_system.groq.client:
            st.success("✅ Groq Connected")
        else:
            st.warning("⚠️ Groq Disconnected")
        
        if rag_system.vector_search.embedder:
            st.success("✅ Vector Search Ready")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        if uploaded_file and not st.session_state.smart_rag_processed:
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner("🔄 Building hybrid search system..."):
                    success = rag_system.process_document(uploaded_file)
                    if success:
                        st.session_state.smart_rag_processed = True
                        st.session_state.smart_rag_document = uploaded_file.name
                        st.success("✅ Document processed for hybrid search!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")
        
        # Controls
        if st.session_state.smart_rag_processed:
            st.markdown("---")
            st.header("🎛️ Search Controls")
            
            if st.button("🕸️ Show Knowledge Graph", use_container_width=True):
                st.session_state.show_smart_graph = True
                st.rerun()
            
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.smart_rag_messages = []
                st.rerun()
    
    # Main content
    if not st.session_state.smart_rag_processed:
        st.info("👆 Upload a PDF to enable hybrid search (Vector + Knowledge Graph)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #175CFF;'>
                <h4>🔍 Vector Search</h4>
                <p>Semantic understanding of document content using AI embeddings</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00A3FF;'>
                <h4>🕸️ Knowledge Graph</h4>
                <p>Entity relationships and connections for deeper insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Graph visualization
    if st.session_state.show_smart_graph:
        st.markdown("### 🕸️ Document Knowledge Graph")
        with st.spinner("Generating graph..."):
            html_content = rag_system.neo4j.create_interactive_network(st.session_state.smart_rag_document)
            if html_content:
                st.components.v1.html(html_content, height=600, scrolling=True)
            else:
                st.error("Failed to generate graph")
        st.markdown("---")
    
    # Chat interface
    st.markdown("### 💬 Ask Anything About Your Document")
    
    # Display messages
    for msg in st.session_state.smart_rag_messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style='background: #e6f3ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #175CFF; margin-bottom: 1rem;'>
                <strong>You:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: #f0f8ff; padding: 1rem; border-radius: 10px; border-left: 4px solid #00A3FF; margin-bottom: 1rem;'>
                <strong>Smart RAG:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Suggested questions for new users
    if st.session_state.smart_rag_processed and len(st.session_state.smart_rag_messages) == 0:
        st.markdown("---")
        st.markdown("### 💡 Try These Questions:")
        
        suggestions = [
            "Summarize the main topics of this document",
            "What organizations are mentioned?",
            "Tell me about the key concepts discussed",
            "What are the main findings or conclusions?",
            "Who are the important people mentioned?"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": suggestion})
                st.rerun()
    
    # Chat input
    question = st.chat_input("Ask about your document...")
    
    if question:
        # Add user question
        st.session_state.smart_rag_messages.append({"role": "user", "content": question})
        
        # Generate answer
        with st.spinner("🔍 Searching with hybrid approach..."):
            answer, confidence = rag_system.search(question, st.session_state.smart_rag_document)
            st.session_state.smart_rag_messages.append({"role": "assistant", "content": answer})
        
        st.rerun()

if __name__ == "__main__":
    smart_rag_page()
