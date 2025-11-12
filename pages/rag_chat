import streamlit as st
import tempfile, os, time, base64, re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pytesseract
import pdf2image
from gtts import gTTS
from groq import Groq

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    CHUNK_SIZE = 500
    MIN_PARAGRAPH_LENGTH = 20

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================
class DocumentProcessor:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
        except Exception as e:
            st.error(f"❌ Document processor failed: {e}")

    def extract_text_direct(self, pdf_path):
        reader = PdfReader(pdf_path)
        extracted = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                extracted.append({"page": i, "text": text.strip()})
        return extracted

    def extract_text_ocr(self, pdf_path):
        images = pdf2image.convert_from_path(pdf_path, dpi=200)
        extracted = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, image in enumerate(images):
            status_text.text(f"Processing page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i + 1, "text": text.strip()})
            progress_bar.progress((i + 1) / len(images))
        status_text.text("✅ PDF processed")
        return extracted

    def analyze_pdf_type(self, pdf_path):
        reader = PdfReader(pdf_path)
        text_pages = sum(1 for page in reader.pages if len((page.extract_text() or "").strip()) > 50)
        total_pages = len(reader.pages)
        return "text_based" if total_pages == 0 or text_pages / total_pages > 0.5 else "scanned"

    def process_pdf(self, uploaded_file):
        self.chunks, self.chunk_metadata, self.index = [], [], None
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        pdf_path = tmp_file.name

        try:
            st.info("Processing your document...")
            pdf_type = self.analyze_pdf_type(pdf_path)
            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
                method = "text"
            else:
                extracted = self.extract_text_ocr(pdf_path)
                method = "ocr"

            for item in extracted:
                page = item["page"]
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                for para in paragraphs:
                    if len(para) > Config.CHUNK_SIZE:
                        sentences = para.split(". ")
                        chunk = ""
                        for s in sentences:
                            if len(chunk + s) < Config.CHUNK_SIZE:
                                chunk += s + ". "
                            else:
                                if chunk.strip():
                                    self.chunks.append(chunk.strip())
                                    self.chunk_metadata.append({"page": page, "method": method})
                                chunk = s + ". "
                        if chunk.strip():
                            self.chunks.append(chunk.strip())
                            self.chunk_metadata.append({"page": page, "method": method})
                    else:
                        if para.strip():
                            self.chunks.append(para)
                            self.chunk_metadata.append({"page": page, "method": method})

            embeddings = self.embedder.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            st.success("✅ PDF processed successfully!")
            return len(self.chunks)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:
            st.warning("⚠️ Please process a document first")
            return []
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                sim = 1 / (1 + distances[0][i])
                results.append({"content": self.chunks[idx], "metadata": self.chunk_metadata[idx], "similarity": round(sim, 3)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

# =============================================================================
# LLM SERVICE
# =============================================================================
class LLMService:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None

    def generate_answer(self, question, chunks):
        if not chunks:
            return f"❌ No relevant info found for '{question}'", 0.0
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)
        if self.client:
            return self._generate_llm_answer(question, chunks, avg_conf)
        else:
            return self._simple_answer(question, chunks, avg_conf)

    def _generate_llm_answer(self, question, chunks, avg_conf):
        try:
            context = "\n\n".join([f"Page {c['metadata']['page']}: {c['content']}" for c in chunks])
            messages = [
                {"role": "system", "content": "You are an expert document analyst. Only use the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            ai_answer = response.choices[0].message.content
            return ai_answer, avg_conf
        except:
            return self._simple_answer(question, chunks, avg_conf)

    def _simple_answer(self, question, chunks, avg_conf):
        key_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]
            key_sentences.extend(sentences[:2])
        summary = ' '.join(key_sentences[:6])
        return summary, avg_conf

# =============================================================================
# VOICE SERVICE
# =============================================================================
class VoiceService:
    def clean_text_for_tts(self, text):
        text = re.sub(r'[^\w\s.,?-]', '', text)
        replacements = {"%": " percent", "$": " dollars", "°": " degrees", "&": " and "}
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def speak_text(self, text):
        text = self.clean_text_for_tts(text)
        tts = gTTS(text)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        with open(tmp_file.name, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        html = f"""<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>"""
        st.components.v1.html(html, height=50)

# =============================================================================
# MAIN RAG CHAT APP
# =============================================================================
def main():
    # Page configuration for RAG Chat
    st.set_page_config(
        page_title="Doc RAG Chat - IRMC aura",
        page_icon="📄",
        layout="wide"
    )
    
    # Add Back to Home button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📄 Doc RAG Chat")
        st.markdown("### Chat with your documents using AI")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize session state for RAG app
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    # Initialize services
    llm_service = LLMService()
    voice_service = VoiceService()
    
    # Sidebar
    st.sidebar.title("📁 Document Controls")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ PDF ready for queries")
    else:
        st.sidebar.info("📤 Upload a PDF to get started")
    
    if uploaded_file:
        st.sidebar.write(f"**File:** {uploaded_file.name}")
        if st.sidebar.button("🚀 Process Document"):
            with st.spinner("Processing document..."):
                count = st.session_state.doc_processor.process_pdf(uploaded_file)
                if count > 0:
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.sidebar.success(f"✅ Processed {count} text chunks")
    
    top_k = st.sidebar.slider("Sources to retrieve", 1, 5, 3)
    enable_voice = st.sidebar.checkbox("Enable Voice Output", True)
    
    # Main chat interface
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload and process a PDF document to start chatting")
        return
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = st.session_state.doc_processor.search_similar(question, top_k)
                answer, conf = llm_service.generate_answer(question, chunks)
                st.markdown(answer)
                if enable_voice:
                    voice_service.speak_text(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
