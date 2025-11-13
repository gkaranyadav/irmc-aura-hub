# pages/3_🌐_Smart_RAG.py - DEBUG VERSION
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
    MIN_PARAGRAPH_LENGTH = 10  # Reduced to catch more text
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
# DEBUG VECTOR SEARCH SERVICE
class VectorSearchService:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.indices = {}
            st.sidebar.success("✅ Vector Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Vector model failed: {e}")
    
    def process_pdf(self, uploaded_file):
        """Process PDF for vector search - DEBUG VERSION"""
        document_name = uploaded_file.name
        
        st.info(f"📄 Processing: {document_name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # DEBUG: Show file info
            file_size = len(uploaded_file.getvalue())
            st.sidebar.info(f"📁 File size: {file_size} bytes")
            
            # Extract text with detailed debugging
            pages_data = self._extract_pdf_text_debug(pdf_path)
            
            if not pages_data:
                st.error("❌ CRITICAL: No text extracted from PDF at all!")
                st.info("💡 The PDF might be: 1) Empty 2) Image-only 3) Password protected 4) Corrupted")
                return False
            
            # Create chunks
            chunks = []
            chunk_metadata = []
            
            for page_num, page_data in pages_data.items():
                st.sidebar.info(f"📝 Page {page_num}: {len(page_data['paragraphs'])} paragraphs")
                for paragraph in page_data['paragraphs']:
                    chunks.append(paragraph)
                    chunk_metadata.append({
                        "page": page_num, 
                        "type": "paragraph",
                        "method": page_data['method']
                    })
            
            if not chunks:
                st.error("❌ No text chunks created from extracted text")
                # Show sample of what WAS extracted
                for page_num, page_data in pages_data.items():
                    st.write(f"Page {page_num} sample: {str(page_data)[:500]}...")
                return False
            
            st.success(f"✅ Created {len(chunks)} text chunks from {len(pages_data)} pages")
            
            # Show sample chunks
            with st.expander("🔍 View Sample Text Chunks"):
                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"Chunk {i+1}: {chunk[:200]}...")
            
            # Create embeddings and index
            embeddings = self.embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            
            # Store for this document
            self.indices[document_name] = (index, chunks, chunk_metadata)
            
            return True
            
        except Exception as e:
            st.error(f"❌ PDF processing failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def _extract_pdf_text_debug(self, pdf_path):
        """DEBUG PDF text extraction"""
        pages_data = {}
        
        st.sidebar.info("🔄 Starting text extraction...")
        
        # METHOD 1: Direct text extraction
        st.sidebar.info("🔧 Trying direct text extraction...")
        direct_data = self._extract_text_direct_debug(pdf_path)
        if direct_data:
            st.sidebar.success(f"✅ Direct extraction: {len(direct_data)} pages")
            for page_num, text in direct_data.items():
                paragraphs = self._split_into_paragraphs(text)
                if paragraphs:
                    pages_data[page_num] = {
                        'paragraphs': paragraphs,
                        'method': 'direct',
                        'raw_text_sample': text[:100] + "..." if text else "EMPTY"
                    }
                    st.sidebar.info(f"   Page {page_num}: {len(paragraphs)} paragraphs")
        else:
            st.sidebar.warning("❌ Direct extraction got NO text")
        
        # METHOD 2: OCR as fallback
        if not pages_data:
            st.sidebar.info("🔧 Trying OCR extraction...")
            ocr_data = self._extract_text_ocr_debug(pdf_path)
            if ocr_data:
                st.sidebar.success(f"✅ OCR extraction: {len(ocr_data)} pages")
                for page_num, text in ocr_data.items():
                    paragraphs = self._split_into_paragraphs(text)
                    if paragraphs:
                        pages_data[page_num] = {
                            'paragraphs': paragraphs,
                            'method': 'ocr',
                            'raw_text_sample': text[:100] + "..." if text else "EMPTY"
                        }
                        st.sidebar.info(f"   Page {page_num}: {len(paragraphs)} paragraphs")
            else:
                st.sidebar.error("❌ OCR extraction also failed")
        
        # DEBUG: Show what we got
        if pages_data:
            with st.expander("🔍 Extraction Debug Info"):
                for page_num, data in pages_data.items():
                    st.write(f"Page {page_num} ({data['method']}): {len(data['paragraphs'])} paragraphs")
                    st.write(f"Sample: {data['raw_text_sample']}")
        else:
            st.error("❌ ALL extraction methods failed!")
            
            # Try to at least read PDF metadata
            try:
                reader = PdfReader(pdf_path)
                st.info(f"PDF has {len(reader.pages)} pages")
                for i, page in enumerate(reader.pages[:3]):
                    raw_text = page.extract_text() or "NO TEXT"
                    st.write(f"Page {i+1} raw: '{raw_text[:100]}'")
            except Exception as e:
                st.error(f"Even PDF metadata reading failed: {e}")
        
        return pages_data
    
    def _extract_text_direct_debug(self, pdf_path):
        """Debug version of direct text extraction"""
        try:
            reader = PdfReader(pdf_path)
            pages_data = {}
            total_pages = len(reader.pages)
            
            st.sidebar.info(f"📖 PDF has {total_pages} pages")
            
            for i, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_data[i] = text
                        st.sidebar.success(f"   Page {i}: {len(text)} characters")
                    else:
                        st.sidebar.warning(f"   Page {i}: NO TEXT")
                except Exception as e:
                    st.sidebar.error(f"   Page {i} error: {e}")
                    
            return pages_data
        except Exception as e:
            st.sidebar.error(f"❌ Direct extraction crashed: {e}")
            return {}
    
    def _extract_text_ocr_debug(self, pdf_path):
        """Debug version of OCR extraction"""
        try:
            st.sidebar.info("🖼️ Converting PDF to images...")
            images = pdf2image.convert_from_path(pdf_path, dpi=150)  # Lower DPI for speed
            pages_data = {}
            
            st.sidebar.info(f"🖼️ Converted to {len(images)} images")
            
            for i, image in enumerate(images):
                try:
                    st.sidebar.info(f"   OCR page {i+1}...")
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        pages_data[i+1] = text
                        st.sidebar.success(f"   Page {i+1}: {len(text)} characters")
                    else:
                        st.sidebar.warning(f"   Page {i+1}: NO TEXT from OCR")
                except Exception as e:
                    st.sidebar.error(f"   Page {i+1} OCR error: {e}")
                    
            return pages_data
        except Exception as e:
            st.sidebar.error(f"❌ OCR extraction crashed: {e}")
            return {}
    
    def semantic_search(self, question, document_name):
        """Semantic search using vector similarity"""
        if document_name not in self.indices:
            st.error(f"❌ No vector index found for {document_name}")
            return []
            
        index, chunks, metadata = self.indices[document_name]
        
        st.sidebar.info(f"🔍 Searching {len(chunks)} chunks for: '{question}'")
        
        if not chunks:
            st.error("❌ No text chunks available for search")
            return []
        
        # Encode question
        query_vec = self.embedder.encode([question])
        distances, indices = index.search(np.array(query_vec), Config.TOP_K)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
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
                
                st.sidebar.success(f"✅ Found match: {confidence:.2f} confidence")
        
        if not results:
            st.sidebar.warning("❌ No vector matches found")
        
        return results
    
    def _split_into_paragraphs(self, text):
        """Split text into paragraphs - ULTRA PERMISSIVE VERSION"""
        # Try multiple splitting strategies
        
        # Strategy 1: Double newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Strategy 2: Single newlines (for poorly formatted PDFs)
        if len(paragraphs) < 2:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Strategy 3: Sentences (as last resort)
        if len(paragraphs) < 2:
            sentences = re.split(r'[.!?]+', text)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Strategy 4: Just split by length if nothing else works
        if len(paragraphs) < 2 and len(text) > 50:
            # Split into chunks of ~500 characters
            chunk_size = 500
            paragraphs = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            paragraphs = [p.strip() for p in paragraphs if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        return paragraphs

# --------------------------------------------------------------------------------------
# SIMPLIFIED NEO4J SERVICE (Focus on vector search first)
class Neo4jService:
    def __init__(self):
        self.driver = None
        # Don't auto-connect - let it be optional
        if Neo4jConfig.URI and Neo4jConfig.USERNAME and Neo4jConfig.PASSWORD:
            self.connect()
    
    def connect(self):
        try:
            uri = Neo4jConfig.URI
            if uri.startswith('neo4j+s://'):
                uri = uri.replace('neo4j+s://', 'neo4j+ssc://')
            
            self.driver = GraphDatabase.driver(
                uri,
                auth=(Neo4jConfig.USERNAME, Neo4jConfig.PASSWORD),
                connection_timeout=15
            )
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                if result.single()["test"] == 1:
                    st.sidebar.success("✅ Neo4j Connected")
                else:
                    st.sidebar.error("❌ Neo4j test failed")
                    
        except Exception as e:
            st.sidebar.warning(f"⚠️ Neo4j: {str(e)}")
            self.driver = None

# --------------------------------------------------------------------------------------
# SIMPLIFIED HYBRID RAG SYSTEM
class HybridRAGSystem:
    def __init__(self):
        self.neo4j = Neo4jService()
        self.groq = GroqService()
        self.vector_search = VectorSearchService()
        
    def process_document(self, uploaded_file):
        """Process document - FOCUS ON VECTOR SEARCH FIRST"""
        st.info("🔄 Processing document for vector search...")
        vector_result = self.vector_search.process_pdf(uploaded_file)
        
        # Neo4j is optional for now
        neo4j_result = True
        if self.neo4j.driver:
            st.info("🔄 (Optional) Building knowledge graph...")
            # We'll add Neo4j later once vector works
        
        return vector_result and neo4j_result
    
    def search(self, question, document_name):
        """Search - FOCUS ON VECTOR SEARCH"""
        # First try vector search
        vector_results = self.vector_search.semantic_search(question, document_name)
        
        # If vector search fails completely, provide a helpful fallback
        if not vector_results:
            return self._generate_fallback_answer(question), 0.0
        
        # Generate answer using vector results
        answer, confidence = self.groq.generate_answer(question, vector_results)
        return answer, confidence
    
    def _generate_fallback_answer(self, question):
        """Helpful fallback when no results found"""
        fallbacks = [
            "I apologize, but I couldn't extract any readable text from your PDF document. ",
            "The document appears to be empty, image-based, or in a format I can't process. ",
            "Please try uploading a different PDF file that contains selectable text.",
            "\n\n💡 **Tips for better results:**",
            "- Upload PDFs with selectable text (not scanned images)",
            "- Try documents with clear English content",
            "- Ensure the file isn't password protected",
            "- For scanned documents, try OCR-enabled PDF processors first"
        ]
        return "".join(fallbacks)

# --------------------------------------------------------------------------------------
# SIMPLIFIED GROQ SERVICE
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
            return "No relevant information found.", 0.0
        
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
                    {"role": "system", "content": "You are a helpful document assistant. Answer the question based ONLY on the provided context from the document."},
                    {"role": "user", "content": f"Question: {question}\n\nDocument Context:\n{context}\n\nAnswer:"}
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
# DEBUG SMART RAG PAGE
def smart_rag_page():
    # Page header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🔧 DEBUG: Smart RAG - Vector Search")
        st.markdown("### **DEBUG MODE**: Testing PDF text extraction")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Warning about debug mode
    st.warning("🔧 **DEBUG MODE ACTIVE** - This version focuses on fixing text extraction")
    
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
        st.header("🔧 DEBUG Controls")
        
        # File upload
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        if uploaded_file and not st.session_state.smart_rag_processed:
            if st.button("🚀 DEBUG Process Document", use_container_width=True):
                with st.spinner("🔄 DEBUG: Processing document..."):
                    success = rag_system.process_document(uploaded_file)
                    if success:
                        st.session_state.smart_rag_processed = True
                        st.session_state.smart_rag_document = uploaded_file.name
                        st.success("✅ Document processed!")
                        st.rerun()
                    else:
                        st.error("❌ Processing failed - check debug info above")
        
        if st.session_state.smart_rag_processed:
            st.markdown("---")
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.smart_rag_messages = []
                st.rerun()
    
    # Main content
    if not st.session_state.smart_rag_processed:
        st.info("👆 Upload a PDF to test text extraction")
        
        # Test with sample questions
        st.markdown("### 💡 Once processed, try these questions:")
        test_questions = [
            "What is this document about?",
            "Summarize the main topics",
            "What are the key points?",
            "Tell me about this document"
        ]
        
        for q in test_questions:
            st.write(f"- `{q}`")
        
        return
    
    # Chat interface
    st.markdown("### 💬 Test Questions (DEBUG MODE)")
    
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
                <strong>DEBUG RAG:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Quick test buttons
    if st.session_state.smart_rag_processed and len(st.session_state.smart_rag_messages) == 0:
        st.markdown("### 🧪 Quick Tests:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test: 'What is this about?'", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What is this document about?"})
                st.rerun()
            if st.button("🧪 Test: 'Summarize'", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "Summarize this document"})
                st.rerun()
        
        with col2:
            if st.button("🧪 Test: 'Main topics'", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What are the main topics?"})
                st.rerun()
            if st.button("🧪 Test: 'Key points'", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What are the key points?"})
                st.rerun()
    
    # Chat input
    question = st.chat_input("Ask about your document...")
    
    if question:
        # Add user question
        st.session_state.smart_rag_messages.append({"role": "user", "content": question})
        
        # Generate answer with detailed feedback
        with st.spinner("🔍 DEBUG: Searching..."):
            answer, confidence = rag_system.search(question, st.session_state.smart_rag_document)
            st.session_state.smart_rag_messages.append({"role": "assistant", "content": answer})
            
            # Show search confidence
            st.sidebar.info(f"🎯 Search confidence: {confidence:.2f}")
        
        st.rerun()

if __name__ == "__main__":
    smart_rag_page()
