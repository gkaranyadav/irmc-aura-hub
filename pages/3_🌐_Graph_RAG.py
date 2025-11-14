# pages/3_🌐_Graph_RAG.py
import streamlit as st
import tempfile, os, time, base64, re
from PyPDF2 import PdfReader
import pytesseract
import pdf2image
from neo4j import GraphDatabase
from groq import Groq
from gtts import gTTS

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    CHUNK_SIZE = 500
    MIN_PARAGRAPH_LENGTH = 20

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")

# =============================================================================
# NEO4J GRAPH PROCESSOR
# =============================================================================
class GraphProcessor:
    def __init__(self):
        self.driver = None
        self.connect()
        self.current_document = None

    def connect(self):
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
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1 AS test")
        except Exception as e:
            st.error(f"❌ Neo4j connection failed: {e}")
            self.driver = None

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
        if not self.driver:
            st.error("❌ No Neo4j connection available")
            return 0

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

            # Clear previous data for this document
            self.current_document = uploaded_file.name
            self._clear_document_data(self.current_document)

            # Create document node
            self._create_document_node(self.current_document)

            total_chunks = 0
            for item in extracted:
                page = item["page"]
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                
                for para_idx, para in enumerate(paragraphs):
                    if para.strip():
                        # Create page and paragraph nodes
                        self._create_page_paragraph(page, para_idx, para, method)
                        total_chunks += 1

            return total_chunks

        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _clear_document_data(self, document_name):
        """Clear existing data for this document"""
        with self.driver.session() as session:
            session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)

    def _create_document_node(self, document_name):
        """Create document node"""
        with self.driver.session() as session:
            session.run("CREATE (d:Document {name: $name})", name=document_name)

    def _create_page_paragraph(self, page_num, para_idx, paragraph, method):
        """Create page and paragraph nodes with relationships"""
        with self.driver.session() as session:
            # Create page node and connect to document
            session.run("""
            MATCH (d:Document {name: $doc_name})
            MERGE (p:Page {number: $page_num, document: $doc_name})
            MERGE (d)-[:HAS_PAGE]->(p)
            """, doc_name=self.current_document, page_num=page_num)

            # Create paragraph node and connect to page
            session.run("""
            MATCH (p:Page {number: $page_num, document: $doc_name})
            CREATE (para:Paragraph {
                content: $content, 
                chunk_id: $chunk_id,
                method: $method
            })
            CREATE (p)-[:HAS_PARAGRAPH]->(para)
            """, 
            doc_name=self.current_document, 
            page_num=page_num, 
            content=paragraph,
            chunk_id=f"page_{page_num}_para_{para_idx}",
            method=method)

            # Extract and create entities
            entities = self._extract_entities(paragraph)
            for entity in entities:
                session.run("""
                MATCH (para:Paragraph {chunk_id: $chunk_id})
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (para)-[:MENTIONS]->(e)
                """, 
                chunk_id=f"page_{page_num}_para_{para_idx}",
                name=entity['name'], 
                type=entity['type'])

    def _extract_entities(self, text):
        """Extract entities from text using regex patterns"""
        entities = []
        
        patterns = {
            'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ORG': r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Company|Ltd|LLC|Corporation)\b',
            'TECH': r'\b(?:AI|ML|Machine Learning|Artificial Intelligence|Neural Network|Deep Learning)\b',
            'DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',
            'CONCEPT': r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 3:
                    entities.append({"name": match, "type": entity_type})
        
        return entities[:5]  # Limit to 5 entities per paragraph

    def search_graph(self, query, top_k=5):
        """Search the graph database for relevant content - FIXED CYPHER"""
        if not self.driver or not self.current_document:
            return []

        # Extract key terms from query
        query_terms = self._extract_query_terms(query)
        
        results = []
        
        with self.driver.session() as session:
            # Search for paragraphs containing query terms - FIXED CYPHER
            for term in query_terms:
                paragraphs_result = session.run("""
                MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
                WHERE para.content CONTAINS $term
                RETURN para.content AS content, p.number AS page, para.method AS method
                ORDER BY length(para.content) ASC
                LIMIT 3
                """, doc_name=self.current_document, term=term)
                
                for record in paragraphs_result:
                    results.append({
                        "content": record["content"],
                        "metadata": {"page": record["page"], "method": record["method"]},
                        "similarity": 0.8,  # Base confidence for direct matches
                        "match_type": "DIRECT"
                    })

            # Search for entities related to query terms - FIXED CYPHER
            for term in query_terms:
                entities_result = session.run("""
                MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)-[:MENTIONS]->(e:Entity)
                WHERE e.name CONTAINS $term OR e.type CONTAINS $term
                RETURN para.content AS content, p.number AS page, e.name AS entity, e.type AS entity_type
                LIMIT 3
                """, doc_name=self.current_document, term=term)
                
                for record in entities_result:
                    results.append({
                        "content": record["content"],
                        "metadata": {"page": record["page"], "entity": record["entity"], "entity_type": record["entity_type"]},
                        "similarity": 0.9,  # Higher confidence for entity matches
                        "match_type": "ENTITY"
                    })

            # Remove duplicates and sort by confidence
            unique_results = []
            seen_content = set()
            for result in results:
                if result["content"] not in seen_content:
                    unique_results.append(result)
                    seen_content.add(result["content"])

            return unique_results[:top_k]

    def _extract_query_terms(self, query):
        """Extract key terms from query"""
        words = query.lower().split()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:4]

    def get_document_sample(self, num_chunks=5):
        """Get sample content from document for question generation"""
        if not self.driver or not self.current_document:
            return ""

        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document {name: $doc_name})-[:HAS_PAGE]->(p:Page)-[:HAS_PARAGRAPH]->(para:Paragraph)
            RETURN para.content AS content
            LIMIT $limit
            """, doc_name=self.current_document, limit=num_chunks)
            
            sample_chunks = [record["content"] for record in result]
            return " ".join(sample_chunks)

    def get_graph_statistics(self):
        """Get statistics about the graph"""
        if not self.driver or not self.current_document:
            return {}
            
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document {name: $doc_name})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
            OPTIONAL MATCH (p)-[:HAS_PARAGRAPH]->(para:Paragraph)
            OPTIONAL MATCH (para)-[:MENTIONS]->(e:Entity)
            RETURN 
                count(DISTINCT p) AS page_count,
                count(DISTINCT para) AS paragraph_count,
                count(DISTINCT e) AS entity_count
            """, doc_name=self.current_document)
            
            stats = result.single()
            return {
                "pages": stats["page_count"] or 0,
                "paragraphs": stats["paragraph_count"] or 0,
                "entities": stats["entity_count"] or 0
            }

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
        
        # Calculate confidence based on graph matches
        confidences = [c["similarity"] for c in chunks]
        if len(confidences) == 1:
            final_confidence = confidences[0]
        elif len(confidences) == 2:
            final_confidence = (confidences[0] * 0.7 + confidences[1] * 0.3)
        else:
            final_confidence = (confidences[0] * 0.5 + confidences[1] * 0.3 + confidences[2] * 0.2)
        
        final_confidence = max(0.6, min(0.98, final_confidence))
        
        if self.client:
            return self._generate_llm_answer(question, chunks, final_confidence)
        else:
            return self._simple_answer(question, chunks, final_confidence)

    def _generate_llm_answer(self, question, chunks, confidence):
        try:
            context = "\n\n".join([f"{c['content']}" for c in chunks])
            
            messages = [
                {"role": "system", "content": "You are an expert document analyst. Provide clear, concise answers based only on the provided context. Do not mention page numbers, sources, or say 'based on the context' in your answer. Just provide the direct answer. If the context doesn't contain the answer, say 'I cannot find this information in the document.'"},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a direct answer without mentioning sources."}
            ]
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            ai_answer = response.choices[0].message.content
            return ai_answer, confidence, chunks
        except:
            return self._simple_answer(question, chunks, confidence)

    def _simple_answer(self, question, chunks, confidence):
        key_sentences = []
        for chunk in chunks:
            sentences = [s.strip() for s in chunk['content'].split('.') if s.strip()]
            key_sentences.extend(sentences[:2])
        summary = ' '.join(key_sentences[:6])
        return summary, confidence, chunks

    def generate_suggested_questions_from_pdf(self, document_sample):
        """Generate relevant questions based on actual PDF content using LLM"""
        if not self.client:
            return self._get_fallback_questions()
            
        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing documents and generating relevant questions. Generate 5-6 specific questions that would help someone understand the key points of this specific document. Make the questions directly relevant to the content provided."},
                {"role": "user", "content": f"Based on this document content, suggest 5-6 specific relevant questions:\n\n{document_sample}\n\nProvide the questions as a simple list, one per line."}
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
            "What is the main purpose of this document?",
            "What are the key findings or conclusions?",
            "Who is the intended audience for this content?",
            "What methodology or approach was used?",
            "What are the main recommendations?",
            "What problems or challenges does this address?"
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
# MAIN GRAPH RAG APP
# =============================================================================
def main():
    # page configuration
    st.set_page_config(
        page_title="Graph RAG - irmc Aura",
        page_icon="🌐",
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
        .graph-badge {
            background-color: #e3f2fd;
            color: #1565c0;
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
        .stats-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #4ECDC4;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # add back to home button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🌐 Graph RAG - irmc Aura")
        st.markdown("### Chat with your documents using Knowledge Graph")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'graph_processor' not in st.session_state:
        st.session_state.graph_processor = GraphProcessor()
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'generating_questions' not in st.session_state:
        st.session_state.generating_questions = False
    if 'suggest_button_used' not in st.session_state:
        st.session_state.suggest_button_used = False
    
    # initialize services
    llm_service = LLMService()
    voice_service = VoiceService()
    
    # sidebar
    st.sidebar.title("📁 Document Controls")
    
    # Show Neo4j connection status
    if st.session_state.graph_processor.driver:
        st.sidebar.success("✅ Neo4j Connected")
        
        # Show graph statistics if document is processed
        if st.session_state.pdf_processed:
            stats = st.session_state.graph_processor.get_graph_statistics()
            st.sidebar.markdown("### 📊 Graph Statistics")
            st.sidebar.write(f"📄 Pages: {stats['pages']}")
            st.sidebar.write(f"📝 Paragraphs: {stats['paragraphs']}")
            st.sidebar.write(f"🔗 Entities: {stats['entities']}")
    else:
        st.sidebar.error("❌ Neo4j Disconnected")
    
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    
    # auto-process when file is uploaded
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("🔄 Processing document and building knowledge graph..."):
            count = st.session_state.graph_processor.process_pdf(uploaded_file)
            if count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.sidebar.success(f"✅ Processed {count} chunks into graph")
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ Document processed to graph")
    
    enable_voice = st.sidebar.checkbox("Enable Voice", True)
    
    # main chat interface
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload a PDF document to build a knowledge graph and get started")
        return
    
    # display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = msg.get("confidence", 0)
            pages = msg.get("pages", [])
            
            confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>' if confidence > 0 else ""
            pages_html = f'<span class="page-badge">pages: {", ".join(map(str, pages))}</span>' if pages else ""
            graph_html = '<span class="graph-badge">🌐 Graph Search</span>'
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Aura:</strong> {msg["content"]} {confidence_html} {pages_html} {graph_html}
            </div>
            """, unsafe_allow_html=True)
    
    # BIG SUGGEST BUTTON
    if (st.session_state.pdf_processed and 
        not st.session_state.show_suggestions and 
        not st.session_state.generating_questions and
        not st.session_state.suggest_button_used and
        len(st.session_state.messages) == 0):
        
        st.markdown("---")
        st.subheader("💡 Get Started with Suggested Questions")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💡 Let Aura suggest questions based on your document", 
                        key="big_suggest_btn", 
                        use_container_width=True,
                        type="primary"):
                st.session_state.generating_questions = True
                st.session_state.suggest_button_used = True
                st.rerun()
    
    # Generate questions when button is clicked
    if st.session_state.generating_questions:
        with st.spinner("🤔 Analyzing your document and generating relevant questions..."):
            document_sample = st.session_state.graph_processor.get_document_sample()
            st.session_state.suggested_questions = llm_service.generate_suggested_questions_from_pdf(document_sample)
            st.session_state.generating_questions = False
            st.session_state.show_suggestions = True
            st.rerun()
    
    # Show suggested questions
    if st.session_state.show_suggestions and st.session_state.suggested_questions:
        st.markdown("---")
        st.subheader("💡 Suggested Questions Based on Your Document")
        st.write("Click on any question to ask it:")
        
        for i, question in enumerate(st.session_state.suggested_questions):
            col1, col2, col1 = st.columns([1, 3, 1])
            with col2:
                if st.button(question, key=f"suggested_{i}", use_container_width=True):
                    st.session_state.selected_question = question
                    st.session_state.show_suggestions = False
                    st.rerun()
    
    # Handle selected question
    if 'selected_question' in st.session_state:
        question = st.session_state.selected_question
        del st.session_state.selected_question
        
        st.session_state.messages.append({"role": "user", "content": question})
        
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching knowledge graph..."):
            chunks = st.session_state.graph_processor.search_graph(question)
            answer, confidence, source_chunks = llm_service.generate_answer(question, chunks)
            
            source_pages = list(set([chunk["metadata"]["page"] for chunk in source_chunks]))
            source_pages.sort()
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "confidence": confidence,
                "pages": source_pages
            })
            
            confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>'
            pages_html = f'<span class="page-badge">pages: {", ".join(map(str, source_pages))}</span>'
            graph_html = '<span class="graph-badge">🌐 Graph Search</span>'
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Aura:</strong> {answer} {confidence_html} {pages_html} {graph_html}
            </div>
            """, unsafe_allow_html=True)
            
            if enable_voice:
                voice_service.speak_text(answer)
    
    # Regular chat input
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {question}
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching knowledge graph..."):
            chunks = st.session_state.graph_processor.search_graph(question)
            answer, confidence, source_chunks = llm_service.generate_answer(question, chunks)
            
            source_pages = list(set([chunk["metadata"]["page"] for chunk in source_chunks]))
            source_pages.sort()
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "confidence": confidence,
                "pages": source_pages
            })
            
            confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>'
            pages_html = f'<span class="page-badge">pages: {", ".join(map(str, source_pages))}</span>'
            graph_html = '<span class="graph-badge">🌐 Graph Search</span>'
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Aura:</strong> {answer} {confidence_html} {pages_html} {graph_html}
            </div>
            """, unsafe_allow_html=True)
            
            if enable_voice:
                voice_service.speak_text(answer)

if __name__ == "__main__":
    main()
