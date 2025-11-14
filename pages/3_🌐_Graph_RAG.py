# pages/3_🌐_Smart_RAG.py - OCR DEBUG VERSION
import streamlit as st
import tempfile
import os
import re
import numpy as np
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
from neo4j import GraphDatabase
import groq
from sentence_transformers import SentenceTransformer
import faiss

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
# OCR DEBUG PROCESSOR
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
        """Process PDF with detailed OCR debugging"""
        document_name = uploaded_file.name
        
        st.info(f"🔍 DEBUG: Processing {document_name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # DEBUG: Show file info
            file_size = len(uploaded_file.getvalue())
            st.sidebar.info(f"📁 File: {file_size} bytes")
            
            # Extract text with detailed debugging
            pages_data = self._extract_pdf_text_debug(pdf_path)
            
            if not pages_data:
                st.error("❌ CRITICAL: No pages data extracted!")
                return False
            
            # Create chunks
            chunks = []
            chunk_metadata = []
            
            total_paragraphs = 0
            for page_num, page_data in pages_data.items():
                paragraphs = page_data['paragraphs']
                total_paragraphs += len(paragraphs)
                for paragraph in paragraphs:
                    chunks.append(paragraph)
                    chunk_metadata.append({
                        "page": page_num,
                        "method": page_data['method']
                    })
            
            st.sidebar.info(f"📝 Total paragraphs: {total_paragraphs}")
            
            if not chunks:
                st.error("❌ No text chunks created!")
                # Show what we got
                with st.expander("🔍 Raw Pages Data"):
                    st.write(pages_data)
                return False
            
            st.success(f"✅ Created {len(chunks)} text chunks")
            
            # Show sample chunks
            with st.expander("🔍 Sample Extracted Text"):
                for i, chunk in enumerate(chunks[:5]):
                    st.write(f"**Chunk {i+1}:** {chunk[:200]}...")
            
            # Create embeddings and index
            embeddings = self.embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            
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
        """DEBUG PDF text extraction with detailed logging"""
        pages_data = {}
        
        # METHOD 1: Try direct extraction first
        st.sidebar.info("🔄 Trying direct text extraction...")
        direct_data = self._extract_text_direct_debug(pdf_path)
        
        if direct_data:
            st.sidebar.success(f"✅ Direct: {len(direct_data)} pages")
            for page_num, page_info in direct_data.items():
                pages_data[page_num] = {
                    'paragraphs': page_info['paragraphs'],
                    'method': 'direct',
                    'raw_sample': page_info['raw_text'][:100] + "..." if page_info['raw_text'] else "EMPTY"
                }
        else:
            st.sidebar.warning("❌ Direct extraction got NO pages")
        
        # METHOD 2: If direct failed or got little text, try OCR
        if not pages_data:
            st.sidebar.info("🔄 Trying OCR extraction...")
            ocr_data = self._extract_text_ocr_debug(pdf_path)
            
            if ocr_data:
                st.sidebar.success(f"✅ OCR: {len(ocr_data)} pages")
                for page_num, page_info in ocr_data.items():
                    pages_data[page_num] = {
                        'paragraphs': page_info['paragraphs'],
                        'method': 'ocr',
                        'raw_sample': page_info['raw_text'][:100] + "..." if page_info['raw_text'] else "EMPTY"
                    }
            else:
                st.sidebar.error("❌ OCR extraction also failed")
        
        # DEBUG: Show what we got
        if pages_data:
            with st.expander("🔍 Extraction Results"):
                for page_num, data in pages_data.items():
                    st.write(f"**Page {page_num}** ({data['method']}): {len(data['paragraphs'])} paragraphs")
                    st.write(f"Sample: {data['raw_sample']}")
                    if data['paragraphs']:
                        st.write(f"First paragraph: {data['paragraphs'][0][:100]}...")
        else:
            st.error("❌ ALL extraction methods failed completely!")
            
            # Try basic PDF reading as last resort
            try:
                reader = PdfReader(pdf_path)
                st.info(f"📄 PDF has {len(reader.pages)} pages")
                for i, page in enumerate(reader.pages[:2]):
                    raw_text = page.extract_text() or "NO TEXT"
                    st.write(f"Page {i+1} raw extract: '{raw_text[:200]}'")
            except Exception as e:
                st.error(f"Even basic PDF reading failed: {e}")
        
        return pages_data

    def _extract_text_direct_debug(self, pdf_path):
        """Debug direct text extraction"""
        try:
            reader = PdfReader(pdf_path)
            pages_data = {}
            total_pages = len(reader.pages)
            
            st.sidebar.info(f"📖 PDF has {total_pages} pages")
            
            for i, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                    if text.strip():
                        paragraphs = self._split_into_paragraphs(text)
                        pages_data[i] = {
                            'raw_text': text,
                            'paragraphs': paragraphs
                        }
                        st.sidebar.success(f"   Page {i}: {len(paragraphs)} paragraphs, {len(text)} chars")
                    else:
                        st.sidebar.warning(f"   Page {i}: NO TEXT")
                        pages_data[i] = {
                            'raw_text': "",
                            'paragraphs': []
                        }
                except Exception as e:
                    st.sidebar.error(f"   Page {i} error: {e}")
                    pages_data[i] = {
                        'raw_text': "",
                        'paragraphs': []
                    }
                    
            return pages_data
        except Exception as e:
            st.sidebar.error(f"❌ Direct extraction crashed: {e}")
            return {}

    def _extract_text_ocr_debug(self, pdf_path):
        """Debug OCR extraction with detailed logging"""
        try:
            st.sidebar.info("🖼️ Converting PDF to images for OCR...")
            
            # Try different DPI settings
            for dpi in [200, 150, 100]:
                try:
                    images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
                    st.sidebar.success(f"✅ Converted to {len(images)} images at {dpi} DPI")
                    break
                except Exception as e:
                    st.sidebar.warning(f"❌ DPI {dpi} failed: {e}")
                    continue
            else:
                st.sidebar.error("❌ All DPI conversions failed")
                return {}
            
            pages_data = {}
            
            for i, image in enumerate(images):
                try:
                    st.sidebar.info(f"   🔍 OCR processing page {i+1}...")
                    
                    # Try different OCR configurations
                    text = pytesseract.image_to_string(image)
                    
                    if text.strip():
                        paragraphs = self._split_into_paragraphs(text)
                        pages_data[i+1] = {
                            'raw_text': text,
                            'paragraphs': paragraphs
                        }
                        st.sidebar.success(f"   Page {i+1}: {len(paragraphs)} paragraphs, {len(text)} chars")
                        
                        # Show OCR sample
                        if i == 0:  # Only show first page sample
                            with st.expander(f"🔍 OCR Sample - Page {i+1}"):
                                st.write(f"**Raw OCR text:** {text[:500]}...")
                    else:
                        st.sidebar.warning(f"   Page {i+1}: NO TEXT from OCR")
                        pages_data[i+1] = {
                            'raw_text': "",
                            'paragraphs': []
                        }
                        
                except Exception as e:
                    st.sidebar.error(f"   Page {i+1} OCR error: {e}")
                    pages_data[i+1] = {
                        'raw_text': "",
                        'paragraphs': []
                    }
                    
            return pages_data
        except Exception as e:
            st.sidebar.error(f"❌ OCR extraction crashed: {e}")
            return {}

    def _split_into_paragraphs(self, text):
        """Ultra-permissive paragraph splitting"""
        if not text.strip():
            return []
        
        # Strategy 1: Double newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Strategy 2: Single newlines
        if len(paragraphs) < 2:
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Strategy 3: Sentences
        if len(paragraphs) < 2:
            sentences = re.split(r'[.!?]+', text)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
        
        # Strategy 4: Fixed chunks as last resort
        if len(paragraphs) < 2 and len(text) > 50:
            chunk_size = 300
            paragraphs = [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)]
            paragraphs = [p for p in paragraphs if len(p) >= Config.MIN_PARAGRAPH_LENGTH]
        
        return paragraphs

    def semantic_search(self, question, document_name):
        """Semantic search with debugging"""
        if document_name not in self.indices:
            st.error(f"❌ No index for {document_name}")
            return []
            
        index, chunks, metadata = self.indices[document_name]
        
        st.sidebar.info(f"🔍 Searching {len(chunks)} chunks...")
        
        if not chunks:
            st.error("❌ No chunks available")
            return []
        
        # Encode question
        query_vec = self.embedder.encode([question])
        distances, indices = index.search(np.array(query_vec), Config.TOP_K)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                distance = distances[0][i]
                # Simple confidence calculation
                confidence = max(0.6, 1.0 - distance)
                
                results.append({
                    "content": chunks[idx],
                    "metadata": metadata[idx],
                    "similarity": confidence,
                    "search_type": "vector"
                })
                
                st.sidebar.success(f"✅ Match {i+1}: {confidence:.2f} confidence")
        
        if not results:
            st.sidebar.warning("❌ No matches found for query")
            
        return results

# --------------------------------------------------------------------------------------
# SIMPLIFIED SERVICES
class Neo4jService:
    def __init__(self):
        self.driver = None
        # Neo4j is optional for now

class GroqService:
    def __init__(self):
        self.client = None
        if GroqConfig.API_KEY:
            try:
                self.client = groq.Groq(api_key=GroqConfig.API_KEY)
                st.sidebar.success("✅ Groq Connected")
            except Exception as e:
                st.sidebar.error(f"❌ Groq: {str(e)}")

    def generate_answer(self, question, search_results):
        """Generate answer with fallback"""
        if not search_results:
            return "I couldn't find any text content in the document to answer your question. The document might be image-based, corrupted, or in an unsupported format.", 0.0
        
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
                    {"role": "system", "content": "You are a helpful document assistant. Answer the question based ONLY on the provided context."},
                    {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"}
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
        answer = f"**Based on the document content:**\n\n"
        for i, result in enumerate(search_results, 1):
            answer += f"{i}. {result['content'][:400]}...\n\n"
        return answer, confidence

# --------------------------------------------------------------------------------------
# MAIN RAG SYSTEM
class HybridRAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.groq = GroqService()
        
    def process_document(self, uploaded_file):
        """Process document with debugging"""
        return self.doc_processor.process_pdf(uploaded_file)
    
    def search(self, question, document_name):
        """Search with debugging"""
        vector_results = self.doc_processor.semantic_search(question, document_name)
        answer, confidence = self.groq.generate_answer(question, vector_results)
        return answer, confidence

# --------------------------------------------------------------------------------------
# DEBUG PAGE
def smart_rag_page():
    st.title("🔧 OCR DEBUG - Smart RAG")
    st.markdown("### **DEBUG MODE**: Testing OCR Extraction")
    
    st.warning("🔧 **DEBUG MODE ACTIVE** - Detailed OCR logging enabled")
    
    # Initialize system
    rag_system = HybridRAGSystem()
    
    # Session state
    if 'smart_rag_processed' not in st.session_state:
        st.session_state.smart_rag_processed = False
    if 'smart_rag_messages' not in st.session_state:
        st.session_state.smart_rag_messages = []
    if 'smart_rag_document' not in st.session_state:
        st.session_state.smart_rag_document = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 DEBUG Controls")
        
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file and not st.session_state.smart_rag_processed:
            if st.button("🚀 DEBUG Process", use_container_width=True):
                with st.spinner("🔄 DEBUG: Processing with OCR..."):
                    success = rag_system.process_document(uploaded_file)
                    if success:
                        st.session_state.smart_rag_processed = True
                        st.session_state.smart_rag_document = uploaded_file.name
                        st.success("✅ Processing complete!")
                        st.rerun()
                    else:
                        st.error("❌ Processing failed - check debug info")
        
        if st.session_state.smart_rag_processed:
            st.markdown("---")
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.smart_rag_messages = []
                st.rerun()
    
    # Main content
    if not st.session_state.smart_rag_processed:
        st.info("👆 Upload a PDF to test OCR extraction")
        return
    
    # Chat interface
    st.markdown("### 💬 Test OCR Results")
    
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
                <strong>DEBUG:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Quick test buttons
    if st.session_state.smart_rag_processed and len(st.session_state.smart_rag_messages) == 0:
        st.markdown("### 🧪 Quick Tests:")
        test_questions = [
            "What is this document about?",
            "What is the name of the person?",
            "What skills are mentioned?",
            "What experience is listed?",
            "Summarize this document"
        ]
        
        for q in test_questions:
            if st.button(f"🧪 {q}", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": q})
                st.rerun()
    
    # Chat input
    question = st.chat_input("Ask about the document...")
    
    if question:
        st.session_state.smart_rag_messages.append({"role": "user", "content": question})
        
        with st.spinner("🔍 DEBUG: Searching..."):
            answer, confidence = rag_system.search(question, st.session_state.smart_rag_document)
            st.session_state.smart_rag_messages.append({"role": "assistant", "content": answer})
            
            st.sidebar.info(f"🎯 Confidence: {confidence:.2f}")
        
        st.rerun()

if __name__ == "__main__":
    smart_rag_page()
