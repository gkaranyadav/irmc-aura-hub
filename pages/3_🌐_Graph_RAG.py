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

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    GROQ_MODEL = "llama-3.3-70b-versatile"
    CHUNK_SIZE = 1500
    MIN_PARAGRAPH_LENGTH = 50

class Neo4jConfig:
    URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
    USERNAME = st.secrets.get("NEO4J_USERNAME", "neo4j")
    PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")

# =============================================================================
# LLM KNOWLEDGE EXTRACTOR
# =============================================================================
class LLMKnowledgeExtractor:
    def __init__(self):
        try:
            self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        except Exception as e:
            st.error(f"❌ Groq initialization failed: {e}")
            self.client = None

    def extract_knowledge_graph(self, text_chunk, page_num):
        """Universal LLM extraction for ANY domain"""
        if not self.client or not text_chunk.strip():
            return [], []
        
        # Dynamic prompt that works for ANY domain
        prompt = f"""
        Analyze this text and extract a knowledge graph. Work for ANY domain - medical, technical, business, educational, etc.
        
        TEXT:
        {text_chunk}
        
        Extract:
        1. IMPORTANT entities (names, concepts, things that matter)
        2. REAL relationships between these entities
        3. Focus on meaningful connections, not just word co-occurrence
        
        Return JSON:
        {{
            "entities": [
                {{
                    "name": "entity_name",
                    "type": "appropriate_type_based_on_context",
                    "description": "what this entity represents"
                }}
            ],
            "relationships": [
                {{
                    "source": "source_entity",
                    "target": "target_entity", 
                    "relationship": "meaningful_relationship_type",
                    "evidence": "why this relationship exists"
                }}
            ]
        }}
        
        Be domain-agnostic and extract what actually matters in the text.
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a universal knowledge extraction expert. Work across all domains and extract meaningful entities and relationships from any text."},
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
            else:
                # Fallback: try to parse as plain text
                return self._fallback_extraction(response_text)
                
        except Exception as e:
            st.warning(f"⚠️ LLM extraction failed for page {page_num}: {str(e)}")
            return [], []

    def _fallback_extraction(self, text):
        """Fallback if JSON parsing fails"""
        entities = []
        relationships = []
        
        # Simple pattern matching as fallback
        lines = text.split('\n')
        for line in lines:
            if '->' in line or '→' in line:
                parts = re.split(r'->|→', line)
                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    if source and target:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "relationship": "related_to",
                            "evidence": "extracted from text"
                        })
        
        return entities, relationships

# =============================================================================
# ENHANCED GRAPH PROCESSOR WITH LLM
# =============================================================================
class GraphProcessor:
    def __init__(self):
        self.driver = None
        self.llm_extractor = LLMKnowledgeExtractor()
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
            return 0, 0

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

            # Clear previous data
            self.current_document = uploaded_file.name
            self._clear_document_data(self.current_document)

            # Create document node
            self._create_document_node(self.current_document)

            total_entities = 0
            total_relationships = 0
            
            # Process with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, item in enumerate(extracted):
                page = item["page"]
                text = item["text"]
                
                status_text.text(f"📄 Processing page {page}/{len(extracted)} with LLM...")
                progress_bar.progress((i + 1) / len(extracted))
                
                # Use LLM to extract knowledge from this page
                entities, relationships = self.llm_extractor.extract_knowledge_graph(text, page)
                
                # Store in Neo4j
                if entities or relationships:
                    self._store_llm_knowledge(page, entities, relationships, method)
                    total_entities += len(entities)
                    total_relationships += len(relationships)
                
                # Small delay to avoid rate limits
                time.sleep(0.5)

            status_text.text("✅ Document processing complete!")
            return total_entities, total_relationships

        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def _store_llm_knowledge(self, page_num, entities, relationships, method):
        """Store LLM-extracted knowledge in Neo4j"""
        with self.driver.session() as session:
            # Ensure page exists
            session.run("""
            MATCH (d:Document {name: $doc_name})
            MERGE (p:Page {number: $page_num, document: $doc_name})
            MERGE (d)-[:HAS_PAGE]->(p)
            """, doc_name=self.current_document, page_num=page_num)

            # Store entities
            for entity in entities:
                session.run("""
                MATCH (p:Page {number: $page_num, document: $doc_name})
                MERGE (e:Entity {name: $name, document: $doc_name})
                SET e.type = $type, 
                    e.description = $description,
                    e.source = $source,
                    e.page = $page_num
                MERGE (p)-[:CONTAINS_ENTITY]->(e)
                """, 
                doc_name=self.current_document,
                page_num=page_num,
                name=entity.get('name', ''),
                type=entity.get('type', 'CONCEPT'),
                description=entity.get('description', ''),
                source='llm_extraction'
                )

            # Store relationships
            for rel in relationships:
                session.run("""
                MATCH (e1:Entity {name: $source, document: $doc_name})
                MATCH (e2:Entity {name: $target, document: $doc_name})
                WHERE e1 <> e2
                MERGE (e1)-[r:RELATED_TO {type: $rel_type}]->(e2)
                SET r.evidence = $evidence,
                    r.source = $source_method,
                    r.page = $page_num,
                    r.confidence = $confidence
                """, 
                doc_name=self.current_document,
                source=rel.get('source', ''),
                target=rel.get('target', ''),
                rel_type=rel.get('relationship', 'related_to'),
                evidence=rel.get('evidence', ''),
                source_method=method,
                page_num=page_num,
                confidence=0.8
                )

    def _clear_document_data(self, document_name):
        with self.driver.session() as session:
            session.run("MATCH (d:Document {name: $name}) DETACH DELETE d", name=document_name)

    def _create_document_node(self, document_name):
        with self.driver.session() as session:
            session.run("CREATE (d:Document {name: $name})", name=document_name)

    def search_graph(self, query, top_k=5):
        """Enhanced graph search using LLM-extracted relationships"""
        if not self.driver or not self.current_document:
            return []

        results = []
        
        with self.driver.session() as session:
            # Search by entity matching
            entity_results = session.run("""
            MATCH (e:Entity {document: $doc_name})
            WHERE e.name CONTAINS $query OR e.description CONTAINS $query
            MATCH (e)<-[:CONTAINS_ENTITY]-(p:Page)
            OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
            RETURN e.name AS entity, e.type AS type, e.description AS description,
                   p.number AS page, r.type AS relation_type, other.name AS related_entity,
                   'ENTITY_MATCH' AS match_type
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
                        "metadata": {"page": record["page"], "match_type": record["match_type"]},
                        "similarity": 0.9,
                        "match_type": "ENTITY_RELATION"
                    })

            # Search by relationship patterns
            rel_results = session.run("""
            MATCH (e1:Entity {document: $doc_name})-[r:RELATED_TO]-(e2:Entity {document: $doc_name})
            WHERE r.type CONTAINS $query OR r.evidence CONTAINS $query
            MATCH (e1)<-[:CONTAINS_ENTITY]-(p:Page)
            RETURN e1.name AS source, e2.name AS target, r.type AS relation,
                   r.evidence AS evidence, p.number AS page,
                   'RELATIONSHIP_MATCH' AS match_type
            LIMIT 10
            """, doc_name=self.current_document, query=query)
            
            for record in rel_results:
                content = f"{record['source']} --[{record['relation']}]--> {record['target']}. Evidence: {record['evidence']}"
                results.append({
                    "content": content,
                    "metadata": {"page": record["page"], "match_type": record["match_type"]},
                    "similarity": 0.85,
                    "match_type": "RELATIONSHIP"
                })

        return results[:top_k]

    def get_graph_statistics(self):
        if not self.driver or not self.current_document:
            return {}
            
        with self.driver.session() as session:
            result = session.run("""
            MATCH (d:Document {name: $doc_name})
            OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
            OPTIONAL MATCH (p)-[:CONTAINS_ENTITY]->(e:Entity)
            OPTIONAL MATCH (e)-[r:RELATED_TO]->(other:Entity)
            RETURN 
                count(DISTINCT p) AS page_count,
                count(DISTINCT e) AS entity_count,
                count(DISTINCT r) AS relationship_count
            """, doc_name=self.current_document)
            
            stats = result.single()
            return {
                "pages": stats["page_count"] or 0,
                "entities": stats["entity_count"] or 0,
                "relationships": stats["relationship_count"] or 0
            }

    def get_knowledge_graph_data(self, limit=100):
        """Get LLM-extracted knowledge graph data"""
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

    def get_entity_relationships(self):
        """Get meaningful entity relationships"""
        if not self.driver or not self.current_document:
            return None
            
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e1:Entity {document: $doc_name})-[r:RELATED_TO]-(e2:Entity {document: $doc_name})
            WHERE e1 <> e2
            RETURN e1.name AS entity1, e2.name AS entity2, r.type AS relationship_type,
                   r.evidence AS evidence, count(*) AS strength
            ORDER BY strength DESC
            LIMIT 20
            """, doc_name=self.current_document)
            
            relationships = []
            for record in result:
                relationships.append({
                    'source': record['entity1'],
                    'target': record['entity2'],
                    'relationship': record['relationship_type'],
                    'evidence': record['evidence'],
                    'strength': record['strength']
                })
            
            return relationships

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
        try:
            text = self.clean_text_for_tts(text)
            
            # Limit text length to avoid TTS issues
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
                Your browser does not support the audio element.
            </audio>
            """
            st.components.v1.html(html, height=80)
            
            # Clean up temporary file
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
        """Create interactive knowledge graph using PyVis"""
        if not graph_data or not graph_data['nodes']:
            st.warning("No graph data available for visualization")
            return None
            
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.barnes_hut()
        
        # Add nodes with different colors based on type
        type_colors = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4', 
            'CONCEPT': '#45B7D1',
            'PRODUCT': '#96CEB4',
            'TECHNOLOGY': '#FFE66D',
            'EVENT': '#FF9F1C',
            'LOCATION': '#6A0572',
            'METRIC': '#06D6A0'
        }
        
        for node in graph_data['nodes']:
            node_type = node.get('type', 'CONCEPT')
            color = type_colors.get(node_type, '#777777')
            
            net.add_node(
                node['id'],
                label=node['label'],
                color=color,
                size=node.get('size', 15),
                title=f"{node['label']}\nType: {node_type}\n{node.get('description', '')}"
            )
        
        # Add edges
        for edge in graph_data['edges']:
            net.add_edge(
                edge['source'],
                edge['target'],
                title=edge.get('type', ''),
                label=edge.get('label', '')[:20],
                color='#cccccc'
            )
        
        # Save and return HTML
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
        .tab-content {
            padding: 1rem 0;
        }
        .entity-type-PERSON { background-color: #FF6B6B; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
        .entity-type-ORGANIZATION { background-color: #4ECDC4; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
        .entity-type-CONCEPT { background-color: #45B7D1; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
        .entity-type-PRODUCT { background-color: #96CEB4; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    </style>
    """, unsafe_allow_html=True)
    
    # add back to home button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🌐 Graph RAG - irmc Aura")
        st.markdown("### Chat with your documents using AI-Powered Knowledge Graph")
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
    graph_visualizer = KnowledgeGraphVisualizer()
    
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
            st.sidebar.write(f"🔗 Entities: {stats['entities']}")
            st.sidebar.write(f"🔄 Relationships: {stats['relationships']}")
            
            # Knowledge Graph Options
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🕸️ Knowledge Graph")
            if st.sidebar.button("🔄 Refresh Graph Visualization"):
                st.rerun()
                
    else:
        st.sidebar.error("❌ Neo4j Disconnected")
    
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    
    # auto-process when file is uploaded
    if uploaded_file and not st.session_state.pdf_processed:
        with st.spinner("🔄 Processing document with AI..."):
            entity_count, relationship_count = st.session_state.graph_processor.process_pdf(uploaded_file)
            if entity_count > 0:
                st.session_state.pdf_processed = True
                st.session_state.pdf_name = uploaded_file.name
                st.sidebar.success(f"✅ Extracted {entity_count} entities and {relationship_count} relationships")
    
    if st.session_state.pdf_processed:
        st.sidebar.success("✅ Document processed with AI")
    
    enable_voice = st.sidebar.checkbox("Enable Voice", True)
    
    # Main tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "🕸️ Knowledge Graph", "📊 Analytics"])
    
    with tab1:
        # main chat interface
        if not st.session_state.pdf_processed:
            st.info("👆 Please upload a PDF document to build an AI knowledge graph")
        else:
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
                    graph_html = '<span class="graph-badge">🌐 AI Knowledge Graph</span>'
                    
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
                st.subheader("💡 Get Started with AI-Suggested Questions")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("💡 Let AI suggest questions based on your document", 
                                key="big_suggest_btn", 
                                use_container_width=True,
                                type="primary"):
                        st.session_state.generating_questions = True
                        st.session_state.suggest_button_used = True
                        st.rerun()
            
            # Generate questions when button is clicked
            if st.session_state.generating_questions:
                with st.spinner("🤔 AI is analyzing your document and generating relevant questions..."):
                    document_sample = st.session_state.graph_processor.get_document_sample()
                    st.session_state.suggested_questions = llm_service.generate_suggested_questions_from_pdf(document_sample)
                    st.session_state.generating_questions = False
                    st.session_state.show_suggestions = True
                    st.rerun()
            
            # Show suggested questions
            if st.session_state.show_suggestions and st.session_state.suggested_questions:
                st.markdown("---")
                st.subheader("💡 AI-Suggested Questions")
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
                
                with st.spinner("🔍 AI is searching knowledge graph..."):
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
                    graph_html = '<span class="graph-badge">🌐 AI Knowledge Graph</span>'
                    
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
                
                with st.spinner("🔍 AI is searching knowledge graph..."):
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
                    graph_html = '<span class="graph-badge">🌐 AI Knowledge Graph</span>'
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Aura:</strong> {answer} {confidence_html} {pages_html} {graph_html}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if enable_voice:
                        voice_service.speak_text(answer)
    
    with tab2:
        st.header("🕸️ AI Knowledge Graph Visualization")
        
        if not st.session_state.pdf_processed:
            st.info("👆 Please upload a PDF document to visualize the AI knowledge graph")
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
                    st.warning("No graph data available for visualization")
            
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
                <div style="background: #FFE66D; color: black; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                    <strong>🟡 TECHNOLOGY</strong>
                </div>
                <div style="background: #FF9F1C; color: white; padding: 0.5rem; border-radius: 5px; margin: 0.2rem 0;">
                    <strong>🟠 EVENT</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Entity relationships
                st.subheader("Key Relationships")
                relationships = st.session_state.graph_processor.get_entity_relationships()
                
                if relationships:
                    for rel in relationships[:8]:
                        st.write(f"**{rel['source']}** → **{rel['target']}**")
                        st.caption(f"Relationship: {rel['relationship']}")
                        if rel.get('evidence'):
                            st.caption(f"Evidence: {rel['evidence'][:100]}...")
                        st.markdown("---")
                else:
                    st.info("No relationships found")
    
    with tab3:
        st.header("📊 Document Analytics")
        
        if not st.session_state.pdf_processed:
            st.info("👆 Please upload a PDF document to view analytics")
        else:
            stats = st.session_state.graph_processor.get_graph_statistics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Pages", stats['pages'])
            with col2:
                st.metric("AI Entities", stats['entities'])
            with col3:
                st.metric("Relationships", stats['relationships'])
            
            # Entity type distribution
            st.subheader("Entity Type Distribution")
            
            # Get entity types count
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
                            title="Entities by Type (AI-Extracted)",
                            xaxis_title="Entity Type",
                            yaxis_title="Count",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No entities found for analytics")

if __name__ == "__main__":
    main()
