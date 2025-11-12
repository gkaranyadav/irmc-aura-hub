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
            st.error(f"❌ document processor failed: {e}")

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
            status_text.text(f"processing page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i + 1, "text": text.strip()})
            progress_bar.progress((i + 1) / len(images))
        status_text.text("✅ pdf processed")
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
            st.info("processing your document...")
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
            return len(self.chunks)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def search_similar(self, query, top_k=3):
        if not self.chunks or self.index is None:
            st.warning("⚠️ please process a document first")
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
            st.error(f"❌ groq initialization failed: {e}")
            self.client = None

    def generate_answer(self, question, chunks):
        if not chunks:
            return f"❌ no relevant info found for '{question}'", 0.0, []
        avg_conf = sum(c["similarity"] for c in chunks)/len(chunks)
        if self.client:
            return self._generate_llm_answer(question, chunks, avg_conf)
        else:
            return self._simple_answer(question, chunks, avg_conf)

    def _generate_llm_answer(self, question, chunks, avg_conf):
        try:
            context = "\n\n".join([f"page {c['metadata']['page']}: {c['content']}" for c in chunks])
            messages = [
                {"role": "system", "content": "you are an expert document analyst. only use the provided context. always mention the page numbers where you found the information."},
                {"role": "user", "content": f"context:\n{context}\n\nquestion: {question}"}
            ]
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            ai_answer = response.choices[0].message.content
            return ai_answer, avg_conf, chunks
        except:
            return self._simple_answer(question, chunks, avg_conf)

    def _simple_answer(self, question, chunks, avg_conf):
        key_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]
            key_sentences.extend(sentences[:2])
        summary = ' '.join(key_sentences[:6])
        return summary, avg_conf, chunks

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
# SUGGESTED QUESTIONS
# =============================================================================
def get_suggested_questions():
    """Return suggested questions based on document content"""
    return [
        "what is this document about?",
        "can you summarize the main points?",
        "what are the key findings or conclusions?",
        "who is the target audience for this document?",
        "what methodology was used in this document?"
    ]

# =============================================================================
# MAIN RAG CHAT APP
# =============================================================================
def main():
    # page configuration for rag chat
    st.set_page_config(
        page_title="doc rag chat - irmc aura",
        page_icon="📄",
        layout="wide"
    )
    
    # custom css for better chat display
    st.markdown("""
    <style>
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
        .confidence-badge {
            background-color: #e8f5e8;
            color: #2e7d32;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }
        .page-badge {
            background-color: #fff3e0;
            color: #ef6c00;
            padding: 0.2rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # add back to home button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📄 doc rag chat")
        st.markdown("### chat with your documents using ai")
    with col2:
        if st.button("🏠 back to home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # initialize session state for rag app
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    # initialize services
    llm_service = LLMService()
    voice_service = VoiceService()
    
    # sidebar - simplified without process button
    st.sidebar.title("📁 document controls")
    uploaded_file = st.sidebar.file_uploader("upload pdf", type="pdf", key="pdf_uploader")
    
    # auto-process when file is uploaded
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("🔄 processing your document automatically..."):
            count = st.session_state.doc_processor.process_pdf(uploaded_file)
            if count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully")
                st.sidebar.info(f"📊 processed {count} text chunks")
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ document ready for queries")
    
    top_k = st.sidebar.slider("sources to retrieve", 1, 5, 3)
    enable_voice = st.sidebar.checkbox("enable voice output", True)
    
    # main chat interface
    if not st.session_state.pdf_processed:
        st.info("👆 please upload a pdf document to get started")
        
        # show suggested questions section
        st.markdown("---")
        st.subheader("💡 let aura suggest you some questions")
        st.write("once you upload a document, you can ask questions like:")
        
        suggested_questions = get_suggested_questions()
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    # store the suggested question for when document is processed
                    st.session_state.suggested_question = question
        return
    
    # display document info
    st.success(f"📄 **document:** {st.session_state.pdf_name}")
    
    # display chat messages with both questions and answers
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>you:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            # extract confidence and pages from the message data
            confidence = msg.get("confidence", 0)
            pages = msg.get("pages", [])
            
            confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>' if confidence > 0 else ""
            pages_html = f'<span class="page-badge">pages: {", ".join(map(str, pages))}</span>' if pages else ""
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>aura:</strong> {msg["content"]} {confidence_html} {pages_html}
            </div>
            """, unsafe_allow_html=True)
    
    # chat input with placeholder
    question = st.chat_input("ask a question about your document... or type 'suggest' for question ideas")
    
    if question:
        # handle "suggest" command
        if question.lower() == "suggest":
            st.info("💡 **suggested questions:** " + " | ".join(get_suggested_questions()))
            return
            
        # add user question to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # display user question immediately
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>you:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
        
        # generate and display answer
        with st.spinner("🔍 searching document..."):
            chunks = st.session_state.doc_processor.search_similar(question, top_k)
            answer, confidence, source_chunks = llm_service.generate_answer(question, chunks)
            
            # extract page numbers from source chunks
            source_pages = list(set([chunk["metadata"]["page"] for chunk in source_chunks]))
            source_pages.sort()
            
            # add assistant answer to chat history with metadata
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "confidence": confidence,
                "pages": source_pages
            })
            
            # display assistant answer with confidence and pages
            confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>'
            pages_html = f'<span class="page-badge">pages: {", ".join(map(str, source_pages))}</span>'
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>aura:</strong> {answer} {confidence_html} {pages_html}
            </div>
            """, unsafe_allow_html=True)
            
            if enable_voice:
                voice_service.speak_text(answer)

if __name__ == "__main__":
    main()
