# pages/3_🌐_Smart_RAG.py - FIXED OCR VERSION
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
    CHUNK_SIZE = 500  # Reduced for better handling
    MIN_PARAGRAPH_LENGTH = 5  # Very permissive
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
# ROBUST DOCUMENT PROCESSOR
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.indices = {}
            st.sidebar.success("✅ Vector Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Vector model failed: {e}")
            self.embedder = None

    def process_pdf(self, uploaded_file):
        """Robust PDF processing with multiple fallbacks"""
        document_name = uploaded_file.name
        
        st.info(f"📄 Processing: {document_name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # Try multiple extraction methods
            all_text = self._extract_text_all_methods(pdf_path)
            
            if not all_text:
                st.error("❌ ALL text extraction methods failed!")
                st.info("""
                💡 **Possible issues:**
                - PDF is image-only with no text layer
                - PDF is password protected
                - PDF is corrupted
                - OCR dependencies missing
                """)
                return False
            
            # Create chunks from extracted text
            chunks = self._create_chunks(all_text)
            
            if not chunks:
                st.error("❌ No text chunks could be created")
                return False
            
            st.success(f"✅ Created {len(chunks)} text chunks")
            
            # Show what was extracted
            with st.expander("🔍 View Extracted Text"):
                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"**Chunk {i+1}:** {chunk}")
            
            # Create vector index
            if self.embedder:
                embeddings = self.embedder.encode(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))
                
                self.indices[document_name] = (index, chunks, [{"page": 1, "method": "combined"}] * len(chunks))
            else:
                # Fallback: store chunks without embeddings
                self.indices[document_name] = (None, chunks, [{"page": 1, "method": "combined"}] * len(chunks))
            
            return True
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return False
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_text_all_methods(self, pdf_path):
        """Try multiple text extraction methods"""
        all_text = ""
        
        # METHOD 1: Direct PDF text extraction
        st.sidebar.info("🔧 Method 1: Direct extraction...")
        direct_text = self._extract_direct_text(pdf_path)
        if direct_text:
            st.sidebar.success(f"✅ Direct: {len(direct_text)} chars")
            all_text += direct_text + "\n\n"
        else:
            st.sidebar.warning("❌ Direct: No text")
        
        # METHOD 2: OCR with pytesseract
        st.sidebar.info("🔧 Method 2: OCR extraction...")
        ocr_text = self._extract_ocr_text(pdf_path)
        if ocr_text:
            st.sidebar.success(f"✅ OCR: {len(ocr_text)} chars")
            all_text += ocr_text + "\n\n"
        else:
            st.sidebar.warning("❌ OCR: No text")
        
        # METHOD 3: Fallback - try basic PDF reading
        if not all_text.strip():
            st.sidebar.info("🔧 Method 3: Fallback extraction...")
            fallback_text = self._extract_fallback_text(pdf_path)
            if fallback_text:
                st.sidebar.success(f"✅ Fallback: {len(fallback_text)} chars")
                all_text += fallback_text
            else:
                st.sidebar.error("❌ Fallback: No text")
        
        return all_text.strip()

    def _extract_direct_text(self, pdf_path):
        """Direct text extraction from PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            st.sidebar.error(f"Direct extraction error: {e}")
            return ""

    def _extract_ocr_text(self, pdf_path):
        """OCR text extraction with error handling"""
        try:
            # Check if pytesseract is available
            pytesseract.get_tesseract_version()
        except Exception:
            st.sidebar.error("❌ pytesseract not installed")
            return ""
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=150)
            text = ""
            
            for i, image in enumerate(images):
                try:
                    # Use PIL to preprocess image for better OCR
                    # Convert to grayscale
                    if image.mode != 'L':
                        image = image.convert('L')
                    
                    # OCR with configuration for better accuracy
                    page_text = pytesseract.image_to_string(
                        image, 
                        config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%&*()_+-=[]{}|;:,<>?/ '
                    )
                    
                    if page_text.strip():
                        text += f"Page {i+1}:\n{page_text}\n\n"
                        st.sidebar.success(f"   Page {i+1}: {len(page_text)} chars")
                    else:
                        st.sidebar.warning(f"   Page {i+1}: No OCR text")
                        
                except Exception as e:
                    st.sidebar.error(f"   Page {i+1} OCR failed: {e}")
            
            return text.strip()
        except Exception as e:
            st.sidebar.error(f"OCR extraction error: {e}")
            return ""

    def _extract_fallback_text(self, pdf_path):
        """Fallback text extraction"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for i, page in enumerate(reader.pages):
                # Try different extraction methods
                try:
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        # Try alternative method
                        page_text = str(page) if hasattr(page, '__str__') else ""
                    
                    if page_text.strip():
                        text += f"Page {i+1}:\n{page_text}\n\n"
                except:
                    continue
            return text.strip()
        except Exception as e:
            return f"Fallback error: {e}"

    def _create_chunks(self, text):
        """Create text chunks from extracted text"""
        if not text.strip():
            return []
        
        # Clean text
        text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        
        # Split into paragraphs/sentences
        chunks = []
        
        # Method 1: Split by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Method 2: Split by single newlines if no paragraphs found
        if not paragraphs:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Method 3: Split by sentences
        if not paragraphs:
            sentences = re.split(r'[.!?]+', text)
            paragraphs = [s.strip() for s in sentences if s.strip()]
        
        # Further split large paragraphs
        for para in paragraphs:
            if len(para) > Config.CHUNK_SIZE:
                # Split into smaller chunks
                words = para.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= Config.CHUNK_SIZE:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                chunks.append(para)
        
        # Filter by minimum length
        chunks = [chunk for chunk in chunks if len(chunk) >= Config.MIN_PARAGRAPH_LENGTH]
        
        return chunks

    def semantic_search(self, question, document_name):
        """Semantic search with fallbacks"""
        if document_name not in self.indices:
            st.error(f"❌ Document not processed: {document_name}")
            return []
            
        index, chunks, metadata = self.indices[document_name]
        
        if not chunks:
            return []
        
        # If we have embeddings, use semantic search
        if index is not None and self.embedder:
            query_vec = self.embedder.encode([question])
            distances, indices = index.search(np.array(query_vec), Config.TOP_K)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):
                    distance = distances[0][i]
                    confidence = max(0.6, 1.0 - distance)
                    
                    results.append({
                        "content": chunks[idx],
                        "metadata": metadata[idx],
                        "similarity": confidence,
                        "search_type": "vector"
                    })
            
            return results
        else:
            # Fallback: return first few chunks
            return [{
                "content": chunk,
                "metadata": meta,
                "similarity": 0.7,
                "search_type": "fallback"
            } for chunk, meta in zip(chunks[:Config.TOP_K], metadata[:Config.TOP_K])]

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

    def generate_answer(self, question, search_results):
        """Generate answer with better fallbacks"""
        if not search_results:
            fallback_answers = [
                "I couldn't extract any readable text from your PDF document. ",
                "This might be because the PDF is image-based, scanned, or in an unsupported format. ",
                "Please try uploading a PDF that contains selectable text, or try using a different PDF file.",
                "\n\n💡 **Tips:**",
                "- Upload PDFs with text content (not just images)",
                "- Try converting scanned documents to searchable PDF first",
                "- Ensure the file isn't password protected"
            ]
            return "".join(fallback_answers), 0.0
        
        confidence = min(len(search_results) / 3.0, 1.0)
        
        if self.client:
            return self._generate_with_groq(question, search_results, confidence)
        else:
            return self._generate_fallback(question, search_results, confidence)
    
    def _generate_with_groq(self, question, search_results, confidence):
        """Generate answer using Groq"""
        context = "\n\n".join([
            f"{r['content']}"
            for r in search_results
        ])
        
        try:
            response = self.client.chat.completions.create(
                model=GroqConfig.MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context from a document. If the context seems like random text or OCR errors, do your best to interpret it."},
                    {"role": "user", "content": f"Question: {question}\n\nDocument Context:\n{context}\n\nAnswer based on the document context:"}
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
        answer = f"**Question:** {question}\n\n**Document Content:**\n\n"
        for i, result in enumerate(search_results, 1):
            answer += f"{i}. {result['content']}\n\n"
        return answer, confidence

# --------------------------------------------------------------------------------------
# MAIN RAG SYSTEM
class HybridRAGSystem:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.groq = GroqService()
        
    def process_document(self, uploaded_file):
        """Process document"""
        return self.doc_processor.process_pdf(uploaded_file)
    
    def search(self, question, document_name):
        """Search document"""
        vector_results = self.doc_processor.semantic_search(question, document_name)
        answer, confidence = self.groq.generate_answer(question, vector_results)
        return answer, confidence

# --------------------------------------------------------------------------------------
# MAIN PAGE
def smart_rag_page():
    st.title("🌐 Smart RAG - Document Q&A")
    st.markdown("### Upload a PDF and ask questions about its content")
    
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
        st.header("📁 Document Controls")
        
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
        
        if uploaded_file and not st.session_state.smart_rag_processed:
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner("🔄 Processing document..."):
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
        
        st.markdown("""
        ### Supported PDF Types:
        - **Text-based PDFs** (direct text extraction)
        - **Scanned PDFs** (OCR text recognition)  
        - **Image-heavy PDFs** (fallback methods)
        
        ### Tips for Best Results:
        - Use PDFs with readable text content
        - For scanned documents, ensure good image quality
        - Avoid password-protected PDFs
        """)
        
        return
    
    # Chat interface
    st.markdown("### 💬 Ask Questions About Your Document")
    
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
                <strong>Assistant:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Quick questions for CVs
    if st.session_state.smart_rag_processed and len(st.session_state.smart_rag_messages) == 0:
        st.markdown("### 💡 Try These Questions:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👤 What is the name?", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What is the name of the person?"})
                st.rerun()
            if st.button("🎓 What education?", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What education background is mentioned?"})
                st.rerun()
        
        with col2:
            if st.button("💼 What experience?", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What work experience is listed?"})
                st.rerun()
            if st.button("🛠️ What skills?", use_container_width=True):
                st.session_state.smart_rag_messages.append({"role": "user", "content": "What skills are mentioned?"})
                st.rerun()
    
    # Chat input
    question = st.chat_input("Ask about your document...")
    
    if question:
        st.session_state.smart_rag_messages.append({"role": "user", "content": question})
        
        with st.spinner("🔍 Searching document..."):
            answer, confidence = rag_system.search(question, st.session_state.smart_rag_document)
            st.session_state.smart_rag_messages.append({"role": "assistant", "content": answer})
        
        st.rerun()

if __name__ == "__main__":
    smart_rag_page()
