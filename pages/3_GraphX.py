# pages/3_🌐_Graph_RAG.py
import streamlit as st
import tempfile, os, time, base64, re, json
from PyPDF2 import PdfReader
import pytesseract
import pdf2image
from neo4j import GraphDatabase
from groq import Groq
from gtts import gTTS
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    CHUNK_SIZE = 500
    MIN_PARAGRAPH_LENGTH = 20
    TOP_K = 3

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")

# =============================================================================
# VECTOR STORE FOR SIMILARITY SEARCH
# =============================================================================
class VectorStore:
    def __init__(self):
        try:
            self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
        except Exception as e:
            st.error(f"❌ Vector store initialization failed: {e}")

    def add_documents(self, chunks, metadata):
        self.chunks = chunks
        self.chunk_metadata = metadata
        
        if chunks:
            embeddings = self.embedder.encode(chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))

    def similarity_search(self, query, k=Config.TOP_K):
        if not self.chunks or self.index is None:
            return []
            
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                distance = distances[0][i]
                
                # Confidence scoring (same as RAG chat)
                if distance < 0.3:
                    confidence = 0.92 + (0.3 - distance) * 0.27
                elif distance < 0.6:
                    confidence = 0.85 + (0.6 - distance) * 0.23
                elif distance < 1.0:
                    confidence = 0.75 + (1.0 - distance) * 0.25
                elif distance < 1.5:
                    confidence = 0.65 + (1.5 - distance) * 0.2
                else:
                    confidence = 0.55 + (2.0 - distance) * 0.2
                    
                confidence = max(0.55, min(0.99, confidence))
                
                results.append({
                    "content": self.chunks[idx], 
                    "metadata": self.chunk_metadata[idx], 
                    "similarity": round(confidence, 3)
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def get_document_sample(self, num_chunks=5):
        if not self.chunks:
            return ""
        sample_chunks = self.chunks[:num_chunks]
        return " ".join(sample_chunks)

# =============================================================================
# LLM RELATIONSHIP EXTRACTOR
# =============================================================================
class LLMRelationshipExtractor:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None

    def extract_relationships_from_chunks(self, chunks):
        if not self.client or not chunks:
            return [], []
        
        combined_text = "\n\n".join([chunk["content"] for chunk in chunks])
        
        prompt = f"""
        Extract a KNOWLEDGE GRAPH from this relevant text:
        
        {combined_text}
        
        Return JSON:
        {{
            "entities": [
                {{
                    "name": "entity_name",
                    "type": "PERSON|ORGANIZATION|CONCEPT|PRODUCT|EVENT|TECHNOLOGY"
                }}
            ],
            "relationships": [
                {{
                    "source": "source_entity",
                    "target": "target_entity", 
                    "type": "IMPACTS|CAUSES|INVESTS_IN|DEVELOPS|COMPETES_WITH|PARTNERS_WITH"
                }}
            ]
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "Extract knowledge graphs from text."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("entities", []), result.get("relationships", [])
                
        except Exception as e:
            st.warning(f"⚠️ LLM extraction failed: {str(e)}")
        
        return [], []

# =============================================================================
# DYNAMIC GRAPH PROCESSOR
# =============================================================================
class DynamicGraphProcessor:
    def __init__(self):
        self.driver = None
        self.vector_store = VectorStore()
        self.llm_extractor = LLMRelationshipExtractor()
        self.current_document = None
        self.connect()

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

    def process_pdf_for_search(self, uploaded_file):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.close()
        pdf_path = tmp_file.name

        try:
            pdf_type = self.analyze_pdf_type(pdf_path)
            if pdf_type == "text_based":
                extracted = self.extract_text_direct(pdf_path)
            else:
                extracted = self.extract_text_ocr(pdf_path)

            self.current_document = uploaded_file.name
            self._clear_document_data(self.current_document)
            self._create_document_node(self.current_document)

            chunks = []
            metadata = []
            
            for item in extracted:
                page = item["page"]
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                
                for para in paragraphs:
                    if para.strip():
                        chunks.append(para)
                        metadata.append({"page": page, "document": self.current_document})

            self.vector_store.add_documents(chunks, metadata)
            return len(chunks)

        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def process_question_dynamically(self, question):
        if not self.driver or not self.current_document:
            return [], "No document processed"

        relevant_chunks = self.vector_store.similarity_search(question, k=Config.TOP_K)
        
        if not relevant_chunks:
            return [], "No relevant content found"

        entities, relationships = self.llm_extractor.extract_relationships_from_chunks(relevant_chunks)

        if entities or relationships:
            self._store_dynamic_knowledge(entities, relationships, relevant_chunks)
        
        return relevant_chunks, f"Built knowledge graph with {len(entities)} entities and {len(relationships)} relationships"

    def _store_dynamic_knowledge(self, entities, relationships, source_chunks):
        with self.driver.session() as session:
            source_pages = list(set([chunk["metadata"]["page"] for chunk in source_chunks]))
            
            for entity in entities:
                session.run("""
                MERGE (e:Entity {name: $name, document: $doc_name})
                SET e.type = $type, e.source = $source
                """, 
                doc_name=self.current_document,
                name=entity.get('name', ''),
                type=entity.get('type', 'CONCEPT'),
                source='dynamic_extraction'
                )

            for rel in relationships:
                session.run("""
                MATCH (e1:Entity {name: $source, document: $doc_name})
                MATCH (e2:Entity {name: $target, document: $doc_name})
                WHERE e1 <> e2
                MERGE (e1)-[r:RELATED_TO {type: $rel_type}]->(e2)
                SET r.source_pages = $source_pages
                """, 
                doc_name=self.current_document,
                source=rel.get('source', ''),
                target=rel.get('target', ''),
                rel_type=rel.get('type', 'related_to'),
                source_pages=source_pages
                )

    def search_graph(self, query, top_k=5):
        if not self.driver or not self.current_document:
            return []

        results = []
        
        with self.driver.session() as session:
            entity_results = session.run("""
            MATCH (e:Entity {document: $doc_name})
            WHERE e.name CONTAINS $query
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
            RETURN e.name AS entity, e.type AS type,
                   r.type AS relation_type, other.name AS related_entity,
                   'GRAPH_ENTITY' AS match_type
            LIMIT 10
            """, doc_name=self.current_document, query=query)
            
            for record in entity_results:
                content_parts = []
                if record["entity"]:
                    content_parts.append(f"Entity: {record['entity']} ({record['type']})")
                if record["related_entity"] and record["relation_type"]:
                    content_parts.append(f"Relationship: {record['relation_type']} -> {record['related_entity']}")
                
                if content_parts:
                    results.append({
                        "content": ". ".join(content_parts),
                        "metadata": {"match_type": record["match_type"]},
                        "similarity": 0.9,
                        "match_type": "GRAPH_ENTITY"
                    })

        return results[:top_k]

    def get_graph_statistics(self):
        if not self.driver or not self.current_document:
            return {}
            
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document {name: $doc_name})
            OPTIONAL MATCH (e:Entity {document: $doc_name})
            OPTIONAL MATCH (e)-[r:RELATED_TO]->(other:Entity)
            RETURN 
                count(DISTINCT e) AS entity_count,
                count(DISTINCT r) AS relationship_count
            """, doc_name=self.current_document)
            
            stats = result.single()
            return {
                "entities": stats["entity_count"] or 0,
                "relationships": stats["relationship_count"] or 0
            }

    def get_knowledge_graph_data(self, limit=100):
        if not self.driver or not self.current_document:
            return None
            
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e:Entity {document: $doc_name})
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
            RETURN e, r, other
            LIMIT $limit
            """, doc_name=self.current_document, limit=limit)
            
            nodes = []
            edges = []
            node_ids = set()
            
            for record in result:
                entity = record['e']
                if entity and entity.id not in node_ids:
                    nodes.append({
                        'id': entity.id,
                        'label': entity.get('name', 'Entity'),
                        'type': entity.get('type', 'entity'),
                        'size': 20
                    })
                    node_ids.add(entity.id)
                
                other_entity = record['other']
                if other_entity and other_entity.id not in node_ids:
                    nodes.append({
                        'id': other_entity.id,
                        'label': other_entity.get('name', 'Entity'),
                        'type': other_entity.get('type', 'entity'),
                        'size': 20
                    })
                    node_ids.add(other_entity.id)
                
                relationship = record['r']
                if relationship and entity and other_entity:
                    edges.append({
                        'source': entity.id,
                        'target': other_entity.id,
                        'type': relationship.get('type', 'related_to')
                    })
            
            return {'nodes': nodes, 'edges': edges}

    def _clear_document_data(self, document_name):
        with self.driver.session() as session:
            session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)

    def _create_document_node(self, document_name):
        with self.driver.session() as session:
            session.run("CREATE (d:Document {name: $name})", name=document_name)

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
                {"role": "system", "content": "You are an expert document analyst. Provide clear, concise answers based only on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a direct answer."}
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
        if not self.client:
            return self._get_fallback_questions()
            
        try:
            messages = [
                {"role": "system", "content": "You are an expert at analyzing documents and generating relevant questions. Generate 5-6 specific questions that would help someone understand the key points of this specific document."},
                {"role": "user", "content": f"Based on this document content, suggest 5-6 specific relevant questions:\n\n{document_sample}\n\nProvide the questions as a simple list, one per line."}
            ]
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            questions_text = response.choices[0].message.content
            
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
        try:
            text = self.clean_text_for_tts(text)
            if len(text) > 500:
                text = text[:500] + "..."
                
            tts = gTTS(text=text, lang='en', slow=False)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp_file.name)
            
            with open(tmp_file.name, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            html = f"""<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>"""
            st.components.v1.html(html, height=50)
            
            os.unlink(tmp_file.name)
            
        except Exception as e:
            st.warning(f"⚠️ Voice feature unavailable: {str(e)}")

# =============================================================================
# KNOWLEDGE GRAPH VISUALIZER
# =============================================================================
class KnowledgeGraphVisualizer:
    def __init__(self):
        pass
    
    def create_interactive_graph(self, graph_data):
        if not graph_data or not graph_data['nodes']:
            st.warning("No graph data available for visualization")
            return None
            
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.barnes_hut()
        
        type_colors = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4', 
            'CONCEPT': '#45B7D1',
            'PRODUCT': '#96CEB4',
            'TECHNOLOGY': '#FFE66D',
            'EVENT': '#FF9F1C'
        }
        
        for node in graph_data['nodes']:
            node_type = node.get('type', 'CONCEPT')
            color = type_colors.get(node_type, '#777777')
            
            net.add_node(
                node['id'],
                label=node['label'],
                color=color,
                size=node.get('size', 15),
                title=node['label']
            )
        
        for edge in graph_data['edges']:
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge.get('type', ''),
                color='#cccccc'
            )
        
        try:
            net.save_graph("knowledge_graph.html")
            with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_content
        except Exception as e:
            st.error(f"Error creating graph: {e}")
            return None

# =============================================================================
# MAIN GRAPH RAG APP
# =============================================================================
def main():
    # page configuration
    st.set_page_config(
        page_title="iRMC GraphX",
        page_icon="",
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
        st.title("iRMC GraphX")
    with col2:
        if st.button("🏠 back to home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'graph_processor' not in st.session_state:
        st.session_state.graph_processor = DynamicGraphProcessor()
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'generating_questions' not in st.session_state:
        st.session_state.generating_questions = False
    if 'suggest_button_used' not in st.session_state:
        st.session_state.suggest_button_used = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Chat"
    
    # initialize services
    llm_service = LLMService()
    voice_service = VoiceService()
    graph_visualizer = KnowledgeGraphVisualizer()
    
    # sidebar - simplified (same as RAG chat)
    st.sidebar.title("📁 Document controls")
    uploaded_file = st.sidebar.file_uploader("   upload pdf", type="pdf", key="pdf_uploader")
    
    # auto-process when file is uploaded
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner(""):
            count = st.session_state.graph_processor.process_pdf_for_search(uploaded_file)
            if count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ document is uploaded")
    
    enable_voice = st.sidebar.checkbox("Enable voice", True)
    
    # Show Neo4j connection status
    if st.session_state.graph_processor.driver:
        st.sidebar.success("✅ Neo4j Connected")
    else:
        st.sidebar.error("❌ Neo4j Disconnected")
    
    # Main tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🕸️ Knowledge Graph", "📊 Analytics"])
    
    with tab1:
        # main chat interface
        if not st.session_state.pdf_processed:
            st.info("👆 please upload a pdf document to get started")
        else:
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
                    graph_info = msg.get("graph_info", "")
                    
                    confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>' if confidence > 0 else ""
                    pages_html = f'<span class="page-badge">pages: {", ".join(map(str, pages))}</span>' if pages else ""
                    graph_html = f'<span class="page-badge">🌐 {graph_info}</span>' if graph_info else ""
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Aura:</strong> {msg["content"]} {confidence_html} {pages_html} {graph_html}
                    </div>
                    """, unsafe_allow_html=True)
            
            # BIG SUGGEST BUTTON - Show only once at the beginning if no messages yet
            if (st.session_state.pdf_processed and 
                not st.session_state.show_suggestions and 
                not st.session_state.generating_questions and
                not st.session_state.suggest_button_used and
                len(st.session_state.messages) == 0):
                
                st.markdown("---")
                st.subheader("💡 Get started with Aura suggested questions")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("💡 let Aura suggest questions based on your document", 
                                key="big_suggest_btn", 
                                use_container_width=True,
                                type="primary"):
                        st.session_state.generating_questions = True
                        st.session_state.suggest_button_used = True
                        st.rerun()
            
            # Generate questions when button is clicked
            if st.session_state.generating_questions:
                with st.spinner("🤔 analyzing your document and generating relevant questions..."):
                    # Get sample content from the processed document
                    document_sample = st.session_state.graph_processor.vector_store.get_document_sample()
                    
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
                with st.spinner("🔍 building knowledge graph and searching..."):
                    # Your dynamic approach: Similarity search → LLM → GraphDB
                    relevant_chunks, graph_info = st.session_state.graph_processor.process_question_dynamically(question)
                    answer, confidence, source_chunks = llm_service.generate_answer(question, relevant_chunks)
                    
                    # extract page numbers from source chunks
                    source_pages = list(set([chunk["metadata"]["page"] for chunk in relevant_chunks]))
                    source_pages.sort()
                    
                    # add assistant answer to chat history with metadata
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "confidence": confidence,
                        "pages": source_pages,
                        "graph_info": graph_info
                    })
                    
                    # display assistant answer with confidence and pages
                    confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>'
                    pages_html = f'<span class="page-badge">pages: {", ".join(map(str, source_pages))}</span>'
                    graph_html = f'<span class="page-badge">🌐 {graph_info}</span>'
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Aura:</strong> {answer} {confidence_html} {pages_html} {graph_html}
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
                with st.spinner("🔍 building knowledge graph and searching..."):
                    # Your dynamic approach: Similarity search → LLM → GraphDB
                    relevant_chunks, graph_info = st.session_state.graph_processor.process_question_dynamically(question)
                    answer, confidence, source_chunks = llm_service.generate_answer(question, relevant_chunks)
                    
                    # extract page numbers from source chunks
                    source_pages = list(set([chunk["metadata"]["page"] for chunk in relevant_chunks]))
                    source_pages.sort()
                    
                    # add assistant answer to chat history with metadata
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "confidence": confidence,
                        "pages": source_pages,
                        "graph_info": graph_info
                    })
                    
                    # display assistant answer with confidence and pages
                    confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>'
                    pages_html = f'<span class="page-badge">pages: {", ".join(map(str, source_pages))}</span>'
                    graph_html = f'<span class="page-badge">🌐 {graph_info}</span>'
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Aura:</strong> {answer} {confidence_html} {pages_html} {graph_html}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if enable_voice:
                        voice_service.speak_text(answer)
    
    with tab2:
        st.header("🕸️ Knowledge Graph Visualization")
        
        if not st.session_state.pdf_processed:
            st.info("👆 Please upload a PDF document to visualize the knowledge graph")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Interactive Knowledge Graph")
                
                # Get graph data
                graph_data = st.session_state.graph_processor.get_knowledge_graph_data()
                
                if graph_data and graph_data['nodes']:
                    # Create interactive graph
                    html_content = graph_visualizer.create_interactive_graph(graph_data)
                    
                    if html_content:
                        st.components.v1.html(html_content, height=600, scrolling=True)
                    else:
                        st.warning("Could not generate graph visualization")
                        
                    st.info(f"📊 Showing {len(graph_data['nodes'])} entities and {len(graph_data['edges'])} relationships")
                else:
                    st.info("Ask questions to build the knowledge graph dynamically!")
            
            with col2:
                st.subheader("Graph Legend")
                st.markdown("""
                <div style="background: #FF6B6B; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                    <strong>🔴 PERSON</strong>
                </div>
                <div style="background: #4ECDC4; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                    <strong>🟢 ORGANIZATION</strong>
                </div>
                <div style="background: #45B7D1; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                    <strong>🔵 CONCEPT</strong>
                </div>
                <div style="background: #96CEB4; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                    <strong>🟢 PRODUCT</strong>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.header("📊 Document Analytics")
        
        if not st.session_state.pdf_processed:
            st.info("👆 Please upload a PDF document to view analytics")
        else:
            stats = st.session_state.graph_processor.get_graph_statistics()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("AI Entities", stats['entities'])
            with col2:
                st.metric("Relationships", stats['relationships'])
            
            # Entity type distribution
            st.subheader("Entity Type Distribution")
            
            if st.session_state.graph_processor.driver:
                with st.session_state.graph_processor.driver.session() as session:
                    result = session.run("""
                    MATCH (e:Entity {document: $doc_name})
                    RETURN e.type AS type, count(*) AS count
                    ORDER BY count DESC
                    LIMIT 10
                    """, doc_name=st.session_state.graph_processor.current_document)
                    
                    entity_types = []
                    counts = []
                    
                    for record in result:
                        entity_types.append(record['type'] or 'Unknown')
                        counts.append(record['count'])
                    
                    if entity_types:
                        fig = go.Figure(data=[go.Bar(x=entity_types, y=counts, marker_color='#175CFF')])
                        fig.update_layout(
                            title="Entities by Type",
                            xaxis_title="Entity Type",
                            yaxis_title="Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No entities found for analytics")

if __name__ == "__main__":
    main()
