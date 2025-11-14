# pages/3_🌐_Smart_RAG.py - PROPER OCR VERSION
import streamlit as st
import tempfile
import os
import re
import numpy as np
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
import pdf2image
import groq
from sentence_transformers import SentenceTransformer
import faiss

# --------------------------------------------------------------------------------------
# CONFIG
class Config:
    CHUNK_SIZE = 500
    MIN_PARAGRAPH_LENGTH = 10
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    TOP_K = 3

class GroqConfig:
    API_KEY = st.secrets.get("GROQ_API_KEY", "")
    MODEL = "llama-3.3-70b-versatile"

# --------------------------------------------------------------------------------------
# PROPER OCR PROCESSOR
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.indices = {}
            st.sidebar.success("✅ Vector Model Loaded")
            
            # Test if pytesseract is working
            try:
                pytesseract.get_tesseract_version()
                st.sidebar.success("✅ Tesseract OCR Available")
            except:
                st.sidebar.error("❌ Tesseract not found")
                
        except Exception as e:
            st.sidebar.error(f"❌ Vector model failed: {e}")
            self.embedder = None

    def process_pdf(self, uploaded_file):
        """Process PDF with proper OCR"""
        document_name = uploaded_file.name
        
        st.info(f"📄 Processing: {document_name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
            
        try:
            # First try direct extraction
            st.sidebar.info("🔧 Step 1: Testing direct text extraction...")
            direct_text = self._extract_direct_text(pdf_path)
            
            if direct_text and len(direct_text) > 100:
                st.sidebar.success(f"✅ Direct text: {len(direct_text)} chars")
                text = direct_text
                method = "direct"
            else:
                st.sidebar.warning("❌ Little/no direct text, using OCR...")
                # Use OCR
                ocr_text = self._extract_ocr_text_proper(pdf_path)
                if ocr_text:
                    st.sidebar.success(f"✅ OCR text: {len(ocr_text)} chars")
                    text = ocr_text
                    method = "ocr"
                else:
                    st.error("❌ Both direct and OCR extraction failed!")
                    return False
            
            # Create chunks
            chunks = self._create_chunks(text)
            
            if not chunks:
                st.error("❌ No text chunks could be created")
                # Show raw text for debugging
                with st.expander("🔍 Raw Extracted Text"):
                    st.text(text[:3000] if text else "NO TEXT")
                return False
            
            st.success(f"✅ Created {len(chunks)} text chunks using {method.upper()}")
            
            # Show sample
            with st.expander("🔍 Sample Extracted Content"):
                st.write(f"**Extraction method:** {method}")
                st.write(f"**Total characters:** {len(text)}")
                st.write(f"**Text chunks:** {len(chunks)}")
                for i, chunk in enumerate(chunks[:3]):
                    st.write(f"**Chunk {i+1}:** {chunk}")
            
            # Create vector index
            if self.embedder:
                embeddings = self.embedder.encode(chunks)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(np.array(embeddings))
                self.indices[document_name] = (index, chunks, [{"page": 1, "method": method}] * len(chunks))
            else:
                self.indices[document_name] = (None, chunks, [{"page": 1, "method": method}] * len(chunks))
            
            return True
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _extract_direct_text(self, pdf_path):
        """Direct text extraction"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += f"PAGE {i+1}:\n{page_text}\n\n"
            
            return text.strip()
        except Exception as e:
            st.sidebar.error(f"Direct extraction error: {e}")
            return ""

    def _extract_ocr_text_proper(self, pdf_path):
        """Proper OCR extraction with better configuration"""
        try:
            st.sidebar.info("🖼️ Converting PDF to images...")
            
            # Convert PDF to images with better settings
            try:
                images = pdf2image.convert_from_path(
                    pdf_path, 
                    dpi=300,  # Higher DPI for better OCR
                    poppler_path=None  # Let it use system poppler
                )
                st.sidebar.success(f"✅ Converted to {len(images)} images")
            except Exception as e:
                st.sidebar.error(f"PDF to image conversion failed: {e}")
                return ""
            
            all_text = ""
            
            for i, image in enumerate(images):
                try:
                    st.sidebar.info(f"   🔍 OCR page {i+1}...")
                    
                    # Preprocess image for better OCR
                    # Convert to grayscale
                    if image.mode != 'L':
                        image = image.convert('L')
                    
                    # Enhance contrast
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2.0)  # Increase contrast
                    
                    enhancer = ImageEnhance.Sharpness(image)
                    image = enhancer.enhance(2.0)  # Increase sharpness
                    
                    # Try different OCR configurations
                    configs = [
                        '--psm 6',  # Uniform block of text
                        '--psm 4',  # Single column of text
                        '--psm 3',  # Fully automatic page segmentation
                        '--psm 1',  # Automatic page segmentation with OSD
                    ]
                    
                    page_text = ""
                    for config in configs:
                        try:
                            text = pytesseract.image_to_string(image, config=config)
                            if len(text.strip()) > len(page_text.strip()):
                                page_text = text
                        except:
                            continue
                    
                    if page_text.strip():
                        all_text += f"PAGE {i+1}:\n{page_text}\n\n"
                        st.sidebar.success(f"   Page {i+1}: {len(page_text)} chars")
                        
                        # Show OCR sample for first page
                        if i == 0:
                            with st.expander("🔍 First Page OCR Sample"):
                                st.text(f"OCR Output:\n{page_text[:1000]}")
                    else:
                        st.sidebar.warning(f"   Page {i+1}: No OCR text")
                        
                except Exception as e:
                    st.sidebar.error(f"   Page {i+1} OCR failed: {e}")
            
            return all_text.strip()
            
        except Exception as e:
            st.sidebar.error(f"OCR extraction failed: {e}")
            return ""

    def _create_chunks(self, text):
        """Create text chunks"""
        if not text.strip():
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        
        # Split by pages first
        page_sections = re.split(r'PAGE \d+:', text)
        chunks = []
        
        for section in page_sections:
            section = section.strip()
            if not section:
                continue
                
            # Split into paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            # If no paragraphs, split by lines
            if not paragraphs:
                paragraphs = [p.strip() for p in section.split('\n') if p.strip()]
            
            # Further split large paragraphs
            for para in paragraphs:
                if len(para) > Config.CHUNK_SIZE:
                    # Split by sentences
                    sentences = re.split(r'[.!?]+', para)
                    current_chunk = ""
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if len(current_chunk + sentence) < Config.CHUNK_SIZE:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                else:
                    chunks.append(para)
        
        # Filter chunks
        chunks = [chunk for chunk in chunks if len(chunk) >= Config.MIN_PARAGRAPH_LENGTH]
        
        return chunks

    def semantic_search(self, question, document_name):
        """Semantic search"""
        if document_name not in self.indices:
            return []
            
        index, chunks, metadata = self.indices[document_name]
        
        if not chunks:
            return []
        
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
            # Fallback: return all chunks for simple matching
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
        """Generate answer"""
        if not search_results:
            return """
❌ **No text could be extracted from your PDF**

🔍 **What happened:**
- Direct text extraction found no selectable text
- OCR (image-to-text) also found no readable text

💡 **Solutions:**
1. **Check if your PDF has selectable text** - try highlighting text with your cursor
2. **Convert scanned PDFs to searchable PDF** using:
   - Adobe Acrobat
   - Online tools like SmallPDF, iLovePDF
   - Google Drive (upload PDF → Open with Google Docs)
3. **Ensure good image quality** for scanned documents

📝 **For best results, use PDFs where you can select and copy text!**
            """, 0.0
        
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
                    {"role": "system", "content": "You are a helpful assistant analyzing a document. Answer the question based ONLY on the provided context. Be concise and accurate."},
                    {"role": "user", "content": f"Question: {question}\n\nDocument Content:\n{context}\n\nAnswer:"}
                ],
                temperature=0.1,
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
    st.markdown("### PDF Text Extraction + AI Analysis")
    
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
        ### 🔧 How It Works:
        1. **Direct Text Extraction** - For PDFs with selectable text
        2. **OCR (Optical Character Recognition)** - For scanned/image PDFs
        3. **AI Analysis** - Answer questions about the content
        
        ### 📝 For Best Results:
        - Use **text-based PDFs** (you can highlight text)
        - For **scanned PDFs**, ensure good image quality
        - Avoid password-protected files
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
