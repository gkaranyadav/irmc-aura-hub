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
    GROQ_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
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
        """Add document chunks to vector store"""
        self.chunks = chunks
        self.chunk_metadata = metadata
        
        if chunks:
            embeddings = self.embedder.encode(chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))

    def similarity_search(self, query, k=Config.TOP_K):
        """Find similar chunks to query"""
        if not self.chunks or self.index is None:
            return []
            
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "similarity": 1 - distances[0][i]  # Convert distance to similarity
                })
        
        return results

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
        """Extract entities and relationships from relevant chunks using LLM"""
        if not self.client or not chunks:
            return [], []
        
        # Combine chunks for context
        combined_text = "\n\n".join([chunk["content"] for chunk in chunks])
        
        prompt = f"""
        Extract a KNOWLEDGE GRAPH from this relevant text. Focus on meaningful entities and relationships.
        
        RELEVANT TEXT:
        {combined_text}
        
        Extract:
        1. Key ENTITIES (important concepts, people, organizations, products)
        2. Semantic RELATIONSHIPS between these entities
        3. Focus on what matters in this specific context
        
        Return JSON:
        {{
            "entities": [
                {{
                    "name": "entity_name",
                    "type": "PERSON|ORGANIZATION|CONCEPT|PRODUCT|EVENT|TECHNOLOGY",
                    "description": "brief description"
                }}
            ],
            "relationships": [
                {{
                    "source": "source_entity",
                    "target": "target_entity", 
                    "type": "IMPACTS|CAUSES|INVESTS_IN|DEVELOPS|COMPETES_WITH|PARTNERS_WITH",
                    "evidence": "text evidence",
                    "confidence": 0.85
                }}
            ]
        }}
        """
        
        try:
            messages = [
                {"role": "system", "content": "You extract precise knowledge graphs from relevant text snippets. Focus on meaningful relationships."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
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
        """Process PDF for vector search (fast, no LLM)"""
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

            # Clear previous data
            self.current_document = uploaded_file.name
            self._clear_document_data(self.current_document)

            # Create document node
            self._create_document_node(self.current_document)

            # Prepare chunks for vector store
            chunks = []
            metadata = []
            
            for item in extracted:
                page = item["page"]
                paragraphs = [p.strip() for p in item["text"].split("\n\n") if len(p.strip()) >= Config.MIN_PARAGRAPH_LENGTH]
                
                for para in paragraphs:
                    if para.strip():
                        chunks.append(para)
                        metadata.append({"page": page, "document": self.current_document})

            # Initialize vector store
            self.vector_store.add_documents(chunks, metadata)
            
            return len(chunks)

        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def process_question_dynamically(self, question):
        """Your brilliant approach: Similarity search → LLM → GraphDB"""
        if not self.driver or not self.current_document:
            return [], "No document processed"

        # Step 1: Similarity search (like RAG)
        with st.spinner("🔍 Finding relevant content..."):
            relevant_chunks = self.vector_store.similarity_search(question, k=Config.TOP_K)
        
        if not relevant_chunks:
            return [], "No relevant content found"

        # Step 2: LLM extract relationships from relevant chunks only
        with st.spinner("🧠 Extracting knowledge relationships..."):
            entities, relationships = self.llm_extractor.extract_relationships_from_chunks(relevant_chunks)

        # Step 3: Store in GraphDB
        if entities or relationships:
            self._store_dynamic_knowledge(entities, relationships, relevant_chunks)
        
        return relevant_chunks, f"Extracted {len(entities)} entities and {len(relationships)} relationships"

    def _store_dynamic_knowledge(self, entities, relationships, source_chunks):
        """Store dynamically extracted knowledge in Neo4j"""
        with self.driver.session() as session:
            # Get source pages from chunks
            source_pages = list(set([chunk["metadata"]["page"] for chunk in source_chunks]))
            
            # Store entities
            for entity in entities:
                session.run("""
                MERGE (e:Entity {name: $name, document: $doc_name})
                SET e.type = $type, 
                    e.description = $description,
                    e.source = $source,
                    e.last_updated = timestamp()
                """, 
                doc_name=self.current_document,
                name=entity.get('name', ''),
                type=entity.get('type', 'CONCEPT'),
                description=entity.get('description', ''),
                source='dynamic_extraction'
                )

            # Store relationships
            for rel in relationships:
                session.run("""
                MATCH (e1:Entity {name: $source, document: $doc_name})
                MATCH (e2:Entity {name: $target, document: $doc_name})
                WHERE e1 <> e2
                MERGE (e1)-[r:RELATED_TO {type: $rel_type}]->(e2)
                SET r.evidence = $evidence,
                    r.confidence = $confidence,
                    r.source_pages = $source_pages,
                    r.last_updated = timestamp(),
                    r.query_based = true
                """, 
                doc_name=self.current_document,
                source=rel.get('source', ''),
                target=rel.get('target', ''),
                rel_type=rel.get('type', 'related_to'),
                evidence=rel.get('evidence', ''),
                confidence=rel.get('confidence', 0.8),
                source_pages=source_pages
                )

    def search_graph(self, query, top_k=5):
        """Search existing knowledge graph"""
        if not self.driver or not self.current_document:
            return []

        results = []
        
        with self.driver.session() as session:
            # Search by entity matching
            entity_results = session.run("""
            MATCH (e:Entity {document: $doc_name})
            WHERE e.name CONTAINS $query OR e.description CONTAINS $query
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
            RETURN e.name AS entity, e.type AS type, e.description AS description,
                   r.type AS relation_type, other.name AS related_entity,
                   r.source_pages AS pages,
                   'GRAPH_ENTITY' AS match_type
            LIMIT 10
            """, doc_name=self.current_document, query=query)
            
            for record in entity_results:
                content_parts = []
                if record["entity"]:
                    content_parts.append(f"Entity: {record['entity']} ({record['type']})")
                if record["description"]:
                    content_parts.append(f"Description: {record['description']}")
                if record["related_entity"] and record["relation_type"]:
                    content_parts.append(f"Relationship: {record['relation_type']} -> {record['related_entity']}")
                
                if content_parts:
                    results.append({
                        "content": ". ".join(content_parts),
                        "metadata": {"page": record["pages"] or [1], "match_type": record["match_type"]},
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
        """Get dynamically built knowledge graph data"""
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
                        'description': entity.get('description', ''),
                        'size': 20
                    })
                    node_ids.add(entity.id)
                
                other_entity = record['other']
                if other_entity and other_entity.id not in node_ids:
                    nodes.append({
                        'id': other_entity.id,
                        'label': other_entity.get('name', 'Entity'),
                        'type': other_entity.get('type', 'entity'),
                        'description': other_entity.get('description', ''),
                        'size': 20
                    })
                    node_ids.add(other_entity.id)
                
                relationship = record['r']
                if relationship and entity and other_entity:
                    edges.append({
                        'source': entity.id,
                        'target': other_entity.id,
                        'type': relationship.get('type', 'related_to'),
                        'label': relationship.get('type', 'related_to')
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
        
        # Calculate confidence based on similarity scores
        confidences = [c.get("similarity", 0.7) for c in chunks]
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
            
            html = f"""
            <audio autoplay controls style="width: 100%; margin: 10px 0;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.components.v1.html(html, height=80)
            
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
                title=f"{node['label']}\nType: {node_type}"
            )
        
        for edge in graph_data['edges']:
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge.get('type', ''),
                label=edge.get('label', '')[:20],
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
    st.set_page_config(
        page_title="Dynamic Graph RAG - irmc Aura",
        page_icon="🌐",
        layout="wide"
    )
    
    st.markdown("""
    <style>
        .chat-message { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; }
        .user-message { background-color: #e6f3ff; border-left: 4px solid #175CFF; }
        .assistant-message { background-color: #f0f8ff; border-left: 4px solid #00A3FF; }
        .confidence-badge { background-color: #e8f5e8; color: #2e7d32; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem; }
        .page-badge { background-color: #fff3e0; color: #ef6c00; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem; }
        .graph-badge { background-color: #e3f2fd; color: #1565c0; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🌐 Dynamic Graph RAG - irmc Aura")
        st.markdown("### Smart Knowledge Graph built from your questions")
    with col2:
        if st.button("🏠 Back to Home"):
            st.switch_page("app.py")
    
    st.markdown("---")
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'graph_processor' not in st.session_state:
        st.session_state.graph_processor = DynamicGraphProcessor()
    if 'dynamic_graph_built' not in st.session_state:
        st.session_state.dynamic_graph_built = False
    
    # Initialize services
    llm_service = LLMService()
    voice_service = VoiceService()
    graph_visualizer = KnowledgeGraphVisualizer()
    
    # Sidebar
    st.sidebar.title("📁 Document Controls")
    
    if st.session_state.graph_processor.driver:
        st.sidebar.success("✅ Neo4j Connected")
        
        if st.session_state.pdf_processed:
            stats = st.session_state.graph_processor.get_graph_statistics()
            st.sidebar.markdown("### 📊 Dynamic Graph Stats")
            st.sidebar.write(f"🔗 Entities: {stats['entities']}")
            st.sidebar.write(f"🔄 Relationships: {stats['relationships']}")
    else:
        st.sidebar.error("❌ Neo4j Disconnected")
    
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    
    # Fast PDF processing (no LLM)
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("📄 Processing document for fast search..."):
            chunk_count = st.session_state.graph_processor.process_pdf_for_search(uploaded_file)
            if chunk_count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.sidebar.success(f"✅ Ready for questions! ({chunk_count} chunks)")
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ Document ready for dynamic graph building")
    
    enable_voice = st.sidebar.checkbox("Enable Voice", True)
    
    # Main chat interface
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload a PDF document to start building dynamic knowledge graphs")
    else:
        # Display chat messages
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
                graph_info = msg.get("graph_info", "")
                
                confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>' if confidence > 0 else ""
                pages_html = f'<span class="page-badge">pages: {", ".join(map(str, pages))}</span>' if pages else ""
                graph_html = f'<span class="graph-badge">🌐 {graph_info}</span>' if graph_info else ""
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Aura:</strong> {msg["content"]} {confidence_html} {pages_html} {graph_html}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        question = st.chat_input("Ask a question to build knowledge graph dynamically...")
        
        if question:
            # Add user question
            st.session_state.messages.append({"role": "user", "content": question})
            
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            # Your brilliant dynamic approach!
            with st.spinner("🚀 Building knowledge graph dynamically..."):
                # Step 1-3: Similarity search → LLM → GraphDB
                relevant_chunks, graph_info = st.session_state.graph_processor.process_question_dynamically(question)
                
                # Step 4: Generate answer
                answer, confidence, source_chunks = llm_service.generate_answer(question, relevant_chunks)
                
                # Extract pages
                source_pages = list(set([chunk["metadata"]["page"] for chunk in relevant_chunks]))
                source_pages.sort()
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "confidence": confidence,
                    "pages": source_pages,
                    "graph_info": graph_info
                })
                
                # Display response
                confidence_html = f'<span class="confidence-badge">confidence: {confidence*100:.1f}%</span>'
                pages_html = f'<span class="page-badge">pages: {", ".join(map(str, source_pages))}</span>'
                graph_html = f'<span class="graph-badge">🌐 {graph_info}</span>'
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Aura:</strong> {answer} {confidence_html} {pages_html} {graph_html}
                </div>
                """, unsafe_allow_html=True)
                
                if enable_voice:
                    voice_service.speak_text(answer)
                
                # Mark that we've built some graph
                st.session_state.dynamic_graph_built = True
    
    # Show graph visualization if we have data
    if st.session_state.dynamic_graph_built:
        st.markdown("---")
        st.subheader("🕸️ Dynamically Built Knowledge Graph")
        
        graph_data = st.session_state.graph_processor.get_knowledge_graph_data()
        if graph_data and graph_data['nodes']:
            html_content = graph_visualizer.create_interactive_graph(graph_data)
            if html_content:
                st.components.v1.html(html_content, height=600, scrolling=True)
                st.info(f"📊 Dynamic Graph: {len(graph_data['nodes'])} entities, {len(graph_data['edges'])} relationships")
        else:
            st.info("Ask questions to build the knowledge graph dynamically!")

if __name__ == "__main__":
    main()
