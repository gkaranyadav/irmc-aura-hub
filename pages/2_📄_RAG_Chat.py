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
    TOP_K = 3

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
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                extracted.append({"page": i + 1, "text": text.strip()})
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

    def search_similar(self, query, top_k=Config.TOP_K):
        if not self.chunks or self.index is None:
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

    def get_document_sample(self, num_chunks=5):
        """Get sample content from document for question generation"""
        if not self.chunks:
            return ""
        sample_chunks = self.chunks[:num_chunks]
        return " ".join(sample_chunks)

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
        
        # IMPROVED CONFIDENCE SCORE CALCULATION
        # Give more weight to higher similarity scores
        similarities = [c["similarity"] for c in chunks]
        avg_conf = sum(similarities) / len(similarities)
        
        # Boost confidence for high-quality matches
        max_sim = max(similarities)
        if max_sim > 0.8:  # Very good match
            avg_conf = min(1.0, avg_conf * 1.2)
        elif max_sim > 0.6:  # Good match
            avg_conf = min(1.0, avg_conf * 1.1)
        
        if self.client:
            return self._generate_llm_answer(question, chunks, avg_conf)
        else:
            return self._simple_answer(question, chunks, avg_conf)

    def _generate_llm_answer(self, question, chunks, avg_conf):
        try:
            context = "\n\n".join([f"{c['content']}" for c in chunks])  # REMOVED PAGE NUMBERS FROM CONTEXT
            
            # IMPROVED PROMPT - NO PAGE NUMBERS IN ANSWER
            messages = [
                {"role": "system", "content": "you are an expert document analyst. provide clear, concise answers based only on the provided context. do not mention page numbers or sources in your answer - they will be displayed separately. if the context doesn't contain the answer, say so clearly."},
                {"role": "user", "content": f"context:\n{context}\n\nquestion: {question}\n\nprovide a direct answer without mentioning page numbers or sources."}
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

    def generate_suggested_questions_from_pdf(self, document_sample):
        """Generate relevant questions based on actual PDF content using LLM"""
        if not self.client:
            return self._get_fallback_questions()
            
        try:
            messages = [
                {"role": "system", "content": "you are an expert at analyzing documents and generating relevant questions. generate 5-6 specific questions that would help someone understand the key points of this specific document. make the questions directly relevant to the content provided."},
                {"role": "user", "content": f"based on this document content, suggest 5-6 specific relevant questions:\n\n{document_sample}\n\nprovide the questions as a simple list, one per line."}
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
                if clean_line and len(clean_line) > 10 and '?' in clean_line:
                    questions.append(clean_line)
                    
            return questions[:6] if questions else self._get_fallback_questions()
        except Exception as e:
            st.error(f"❌ failed to generate questions: {e}")
            return self._get_fallback_questions()

    def _get_fallback_questions(self):
        """Fallback questions if LLM fails"""
        return [
            "what is the main purpose of this document?",
            "what are the key findings or conclusions?",
            "who is the intended audience for this content?",
            "what methodology or approach was used?",
            "what are the main recommendations?",
            "what problems or challenges does this address?"
        ]

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
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'generating_questions' not in st.session_state:
        st.session_state.generating_questions = False
    if 'suggest_button_used' not in st.session_state:  # NEW: Track if button was used
        st.session_state.suggest_button_used = False
    
    # initialize services
    llm_service = LLMService()
    voice_service = VoiceService()
    
    # sidebar - simplified
    st.sidebar.title("📁 document controls")
    uploaded_file = st.sidebar.file_uploader("upload pdf", type="pdf", key="pdf_uploader")
    
    # auto-process when file is uploaded
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner(""):
            count = st.session_state.doc_processor.process_pdf(uploaded_file)
            if count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ document ready")
    
    enable_voice = st.sidebar.checkbox("enable voice output", True)
    
    # main chat interface
    if not st.session_state.pdf_processed:
        st.info("👆 please upload a pdf document to get started")
        return
    
    # display chat messages with both questions and answers
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>you:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = msg.get("confidence", 0)
            pages = msg.get("pages", [])
            
            confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>' if confidence > 0 else ""
            pages_html = f'<span class="page-badge">pages: {", ".join(map(str, pages))}</span>' if pages else ""
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>aura:</strong> {msg["content"]} {confidence_html} {pages_html}
            </div>
            """, unsafe_allow_html=True)
    
    # BIG SUGGEST BUTTON - Show only once at the beginning if no messages yet
    if (st.session_state.pdf_processed and 
        not st.session_state.show_suggestions and 
        not st.session_state.generating_questions and
        not st.session_state.suggest_button_used and  # Only show if not used before
        len(st.session_state.messages) == 0):  # Only show at the beginning
        
        st.markdown("---")
        st.subheader("💡 get started with suggested questions")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💡 let aura suggest questions based on your document", 
                        key="big_suggest_btn", 
                        use_container_width=True,
                        type="primary"):
                st.session_state.generating_questions = True
                st.session_state.suggest_button_used = True  # Mark as used
                st.rerun()
    
    # Generate questions when button is clicked
    if st.session_state.generating_questions:
        with st.spinner("🤔 analyzing your document and generating relevant questions..."):
            # Get sample content from the processed document
            document_sample = st.session_state.doc_processor.get_document_sample()
            
            # Generate questions based on actual PDF content
            st.session_state.suggested_questions = llm_service.generate_suggested_questions_from_pdf(document_sample)
            
            st.session_state.generating_questions = False
            st.session_state.show_suggestions = True
            st.rerun()
    
    # show suggested questions if generated
    if st.session_state.show_suggestions and st.session_state.suggested_questions:
        st.markdown("---")
        st.subheader("💡 suggested questions based on your document")
        st.write("click on any question to ask it:")
        
        # display questions as clickable buttons
        for i, question in enumerate(st.session_state.suggested_questions):
            col1, col2, col1 = st.columns([1, 3, 1])
            with col2:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    # set the question in session state to be processed
                    st.session_state.selected_question = question
                    st.session_state.show_suggestions = False
                    st.rerun()
    
    # handle selected question from suggestions
    if 'selected_question' in st.session_state:
        question = st.session_state.selected_question
        del st.session_state.selected_question
        
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
            chunks = st.session_state.doc_processor.search_similar(question)
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
    
    # regular chat input
    question = st.chat_input("ask a question about your document...")
    
    if question:
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
            chunks = st.session_state.doc_processor.search_similar(question)
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
