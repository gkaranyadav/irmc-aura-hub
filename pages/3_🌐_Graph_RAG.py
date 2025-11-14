# pages/3_🌐_Smart_RAG.py - FIXED OCR VERSION
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
# FIXED DOCUMENT PROCESSOR WITH PROPER OCR
class DocumentProcessor:
    def __init__(self):
        self.neo4j = Neo4jService()
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.indices = {}
            st.sidebar.success("✅ Vector Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Vector model failed: {e}")

    def process_pdf(self, uploaded_file):
        """Process PDF with proper OCR support"""
        document_name = uploaded_file.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # Extract text using your original method
            pages_data = self._extract_pdf_text(pdf_path)
            
            if not pages_data:
                st.error("❌ No text extracted from PDF.")
                return False
            
            # Create vector index
            chunks = []
            chunk_metadata = []
            
            for page_num, page_data in pages_data.items():
                for paragraph in page_data['paragraphs']:
                    chunks.append(paragraph)
                    chunk_metadata.append({
                        "page": page_num,
                        "method": page_data['method']
                    })
            
            if not chunks:
                st.error("❌ No text chunks created")
                return False
            
            st.success(f"✅ Extracted {len(chunks)} text chunks using {pages_data[1]['method'].upper()}")
            
            # Create embeddings
            embeddings = self.embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            
            self.indices[document_name] = (index, chunks, chunk_metadata)
            
            # Create knowledge graph if Neo4j is available
            if self.neo4j.driver:
                graph_success = self.neo4j.create_document_graph(document_name, pages_data)
                if graph_success:
                    st.sidebar.success("✅ Knowledge graph created")
            
            return True
            
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            return False
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_pdf_text(self, pdf_path):
        """Your original PDF extraction method with OCR"""
        # Determine PDF type
        pdf_type = self._analyze_pdf_type(pdf_path)
        
        st.sidebar.info(f"📄 PDF type: {pdf_type}")
        
        if pdf_type == "text_based":
            data = self._extract_text_direct(pdf_path)
            method = "direct"
        else:
            data = self._extract_text_ocr(pdf_path)
            method = "ocr"
        
        # Add method info to all pages
        for page_num in data:
            data[page_num]['method'] = method
            
        return data

    def _analyze_pdf_type(self, pdf_path):
        """Analyze if PDF is text-based or scanned"""
        try:
            reader = PdfReader(pdf_path)
            text_pages = 0
            total_pages = len(reader.pages)
            
            for page in reader.pages:
                text = page.extract_text() or ""
                if len(text.strip()) > 100:  # More lenient threshold
                    text_pages += 1
            
            # If more than 30% pages have decent text, consider it text-based
            if total_pages == 0 or text_pages / total_pages > 0.3:
                return "text_based"
            else:
                return "scanned"
                
        except Exception as e:
            st.sidebar.warning(f"PDF analysis warning: {e}")
            return "scanned"  # Fallback to OCR

    def _extract_text_direct(self, pdf_path):
        """Extract text directly from PDF"""
        reader = PdfReader(pdf_path)
        pages_data = {}
        
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                paragraphs = self._split_into_paragraphs(text)
                pages_data[i] = {
                    'preview': text[:500],
                    'paragraphs': paragraphs
                }
                st.sidebar.success(f"📝 Page {i}: {len(paragraphs)} paragraphs (direct)")
        
        return pages_data

    def _extract_text_ocr(self, pdf_path):
        """Extract text using OCR - YOUR ORIGINAL METHOD"""
        st.sidebar.info("🔍 Using OCR for text extraction...")
        images = pdf2image.convert_from_path(pdf_path, dpi=200)
        pages_data = {}
        
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                paragraphs = self._split_into_paragraphs(text)
                pages_data[i+1] = {
                    'preview': text[:500],
                    'paragraphs': paragraphs
                }
                st.sidebar.success(f"📝 Page {i+1}: {len(paragraphs)} paragraphs (OCR)")
            else:
                st.sidebar.warning(f"Page {i+1}: No text found with OCR")
        
        return pages_data

    def _split_into_paragraphs(self, text):
        """Your original paragraph splitting logic"""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Further split large paragraphs
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > Config.CHUNK_SIZE:
                # Split by sentences for large paragraphs
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
                # Convert distance to similarity score
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

# --------------------------------------------------------------------------------------
# NEO4J SERVICE (Your original)
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
            if uri.startswith('https://'):
                uri = uri.replace('https://', 'bolt://')
            elif uri.startswith('http://'):
                uri = uri.replace('http://', 'bolt://')
            
            self.driver = GraphDatabase.driver(
                uri,
                auth=(Neo4jConfig.USERNAME, Neo4jConfig.PASSWORD)
            )
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
            st.sidebar.success("✅ Connected to Neo4j")
        except Exception as e:
            st.sidebar.error(f"❌ Neo4j: {str(e)}")
            self.driver = None
    
    def create_document_graph(self, document_name, pages_data):
        if not self.driver:
            return False
            
        try:
            with self.driver.session() as session:
                # Clear existing data
                session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)
                
                # Create Document node
                session.run("CREATE (d:Document {name: $name})", name=document_name)
                
                # Create Pages and Paragraphs
                for page_num, page_data in pages_data.items():
                    session.run("""
                    MATCH (d:Document {name: $doc_name})
                    CREATE (p:Page {number: $page_num})
                    CREATE (d)-[:HAS_PAGE]->(p)
                    """, doc_name=document_name, page_num=page_num)
                    
                    # Create Paragraph nodes
                    for para_idx, paragraph in enumerate(page_data['paragraphs']):
                        # Extract simple entities using regex
                        entities = self._extract_simple_entities(paragraph)
                        
                        session.run("""
                        MATCH (p:Page {number: $page_num})<-[:HAS_PAGE]-(d:Document {name: $doc_name})
                        CREATE (para:Paragraph {content: $content, chunk_id: $chunk_id})
                        CREATE (p)-[:HAS_PARAGRAPH]->(para)
                        """, doc_name=document_name, page_num=page_num, 
                           content=paragraph, chunk_id=f"page_{page_num}_para_{para_idx}")
                        
                        # Create entity relationships
                        for entity in entities:
                            session.run("""
                            MERGE (e:Entity {name: $name, type: $type})
                            WITH e
                            MATCH (para:Paragraph {chunk_id: $chunk_id})
                            MERGE (para)-[:MENTIONS]->(e)
                            """, name=entity['name'], type=entity['type'], chunk_id=f"page_{page_num}_para_{para_idx}")
                
                st.sidebar.success(f"✅ Graph created for {document_name}")
                return True
                
        except Exception as e:
            st.error(f"❌ Graph creation failed: {str(e)}")
            return False
    
    def _extract_simple_entities(self, text):
        """Simple entity extraction using regex patterns"""
        entities = []
        
        # Patterns for common entities
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Company|Ltd)\b',
            'TECH': r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Neural Network|Deep Learning)\b',
            'CONCEPT': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3:
                    entities.append({"name": match, "type": entity_type})
        
        return entities[:8]

# --------------------------------------------------------------------------------------
# GROQ SERVICE
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
    
    def generate_answer(self, question, search_results):
        """Generate answer from search results"""
        if not search_results:
            return "I couldn't find relevant information in the document to answer your question. Try asking about specific topics that might be in the document.", 0.0
        
        confidence = min(len(search_results) / 3.0, 1.0)
        
        if self.client:
            return self._generate_with_groq(question, search_results, confidence)
        else:
            return self._generate_fallback(question, search_results, confidence)
    
    def _generate_with_groq(self, question, search_results, confidence):
        """Generate answer using Groq"""
        context = "\n\n".join([
            f"From page {r['metadata']['page']}: {r['content']}"
            for r in search_results
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=GroqConfig.MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful document assistant. Answer the question based ONLY on the provided context from the document. Be concise and informative."},
                    {"role": "user", "content": f"Question: {question}\n\nDocument Context:\n{context}\n\nAnswer based on the document:"}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content
            return answer, confidence
            
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return self._generate_fallback(question, search_results, confidence)
    
    def _generate_fallback(self, question, search_results, confidence):
        """Simple fallback answer"""
        answer = f"**Based on the document:**\n\n"
        for i, result in enumerate(search_results, 1):
            answer += f"{i}. {result['content'][:300]}...\n\n"
        return answer, confidence

# --------------------------------------------------------------------------------------
# HYBRID RAG SYSTEM
class HybridRAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.groq = GroqService()
        
    def process_document(self, uploaded_file):
        """Process document using your original OCR method"""
        return self.doc_processor.process_pdf(uploaded_file)
    
    def search(self, question, document_name):
        """Search using vector similarity"""
        # Vector-based semantic search
        vector_results = self.doc_processor.semantic_search(question, document_name)
        
        # Generate answer using vector results
        answer, confidence = self.groq.generate_answer(question, vector_results)
        return answer, confidence

# --------------------------------------------------------------------------------------
# SMART RAG PAGE
def smart_rag_page():
    # Page header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🌐 Smart RAG - Hybrid Search")
        st.markdown("### Vector Search + Knowledge Graph with OCR Support")
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
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Controls")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        if uploaded_file and not st.session_state.smart_rag_processed:
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner("🔄 Processing document with OCR support..."):
                    success = rag_system.process_document(uploaded_file)
                    if success:
                        st.session_state.smart_rag_processed = True
                        st.session_state.smart_rag_document = uploaded_file.name
                        st.success("✅ Document processed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process document")
        
        if st.session_state.smart_rag_processed:
            st.markdown("---")
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.smart_rag_messages = []
                st.rerun()
    
    # Main content
    if not st.session_state.smart_rag_processed:
        st.info("👆 Upload a PDF document to start asking questions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #175CFF;'>
                <h4>🔍 Smart OCR</h4>
                <p>Automatically detects text-based vs scanned PDFs and uses appropriate extraction method</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00A3FF;'>
                <h4>🤖 AI-Powered Q&A</h4>
                <p>Get intelligent answers about your document using Groq's fast LLM</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
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
    
    # Chat input
    question = st.chat_input("Ask about your document...")
    
    if question:
        # Add user question
        st.session_state.smart_rag_messages.append({"role": "user", "content": question})
        
        # Generate answer
        with st.spinner("🔍 Searching document..."):
            answer, confidence = rag_system.search(question, st.session_state.smart_rag_document)
            st.session_state.smart_rag_messages.append({"role": "assistant", "content": answer})
        
        st.rerun()

if __name__ == "__main__":
    smart_rag_page()
