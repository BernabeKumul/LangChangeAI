"""
LangChain Streamlit Demo Application
A web interface to demonstrate LangChain capabilities
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import tempfile
import json
from datetime import datetime
from config import Config

# Page configuration
st.set_page_config(
    page_title="LangChain Demo",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Custom tools for the agent
@tool
def calculator(operation: str) -> str:
    """Perform basic mathematical operations"""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in operation):
            return "Error: Invalid characters in operation"
        result = eval(operation)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_time() -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def text_analyzer(text: str) -> str:
    """Analyze text statistics"""
    words = text.split()
    analysis = {
        "characters": len(text),
        "words": len(words),
        "sentences": len(text.split('.')),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    return json.dumps(analysis, indent=2)

# Pydantic model for IPM Audit response
class IPMAuditResponse(BaseModel):
    """Response model for IPM Compliance Audit"""
    ComplianceLevel: int = Field(description="Always returns the value 2", default=2)
    Comments: str = Field(description="Multi-paragraph compliance analysis")
    FilesSearch: list = Field(description="List of files with FileName and DocumentID", default=[])

def check_api_key():
    """Check if API key is configured"""
    try:
        Config.validate_config()
        return True
    except:
        return False

def init_llm():
    """Initialize LLM with error handling"""
    try:
        return ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.DEFAULT_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def ipm_audit_chain(llm):
    """Create IPM Audit chain using LangChain structured prompts"""
    
    # Define the structured prompt template
    system_prompt = """You are an IPM Compliance Auditor evaluating compliance with PrimusGFS Module 9 – Integrated Pest Management (IPM) Practices.

AUDIT RULES:
- Evaluate strictly based on provided documents
- Do not assume compliance where documentation is missing or unclear
- Do not offer suggestions or improvements
- Focus only on determining if documents meet compliance expectations
- Use all documents provided, even duplicates or scans with limited content

EVALUATION CRITERIA for Question 9.01.01:
The operation should have a documented IPM plan that:
• Describes pest monitoring and identification practices
• Explains use of action/economic thresholds for treatment decisions
• Includes at least two pest prevention practices
• Addresses pesticide resistance management strategies
• Includes measures to protect pollinators

RESPONSE REQUIREMENTS:
- Write in audit-style format with professional tone
- Reference exact document names (with file extensions)
- Include document sections, dates, and personnel when identified
- Break response into multiple paragraphs by topic
- Maximum 2,000 characters
- State clearly if documentation is missing or insufficient

CRITICAL: You MUST respond with a valid JSON object with exactly these keys:
- ComplianceLevel: integer (always 2)
- Comments: string (detailed multi-paragraph audit analysis)
- FilesSearch: array of objects with FileName and DocumentID (can be empty array if no files)

Example JSON format:
{
  "ComplianceLevel": 2,
  "Comments": "The operation has submitted an IPM plan titled...",
  "FilesSearch": [{"FileName": "document.pdf", "DocumentID": "DOC_001"}]
}"""

    human_prompt = """Evaluate the IPM compliance for {operation_name} with product: {product}

QUESTION: Does the operation have a documented Integrated Pest Management (IPM) plan?

DOCUMENTS PROVIDED:
{documents}

Provide your audit evaluation focusing on:
1. Pest monitoring and identification practices
2. Action/economic thresholds
3. Prevention practices (minimum 2)
4. Resistance management strategies
5. Pollinator protection measures

Respond in {language}.

IMPORTANT: Your response must be a valid JSON object only, no additional text before or after."""

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # Set up JSON output parser
    parser = JsonOutputParser(pydantic_object=IPMAuditResponse)
    
    # Create the chain
    chain = prompt | llm | parser
    
    return chain

def ipm_audit_fallback(llm, operation_name, product, documents, language, file_name, document_id):
    """Fallback IPM audit method when JSON parsing fails"""
    try:
        # Simple prompt for fallback
        fallback_prompt = ChatPromptTemplate.from_template("""
Eres un auditor IPM. Evalúa la operación {operation_name} para el producto {product}.

Documentos disponibles: {documents}

INSTRUCCIONES:
1. Analiza el cumplimiento del plan IPM
2. Responde ÚNICAMENTE en formato JSON válido
3. Usa exactamente estas claves: ComplianceLevel, Comments, FilesSearch

Formato requerido:
{{
  "ComplianceLevel": 2,
  "Comments": "Análisis detallado del cumplimiento IPM...",
  "FilesSearch": [{{"FileName": "{file_name}", "DocumentID": "{document_id}"}}]
}}

Idioma: {language}
Respuesta en JSON:""")
        
        chain = fallback_prompt | llm | StrOutputParser()
        raw_response = chain.invoke({
            "operation_name": operation_name,
            "product": product,
            "documents": documents,
            "language": language,
            "file_name": file_name,
            "document_id": document_id
        })
        
        # Clean response and try to parse
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        cleaned = cleaned.strip()
        
        return json.loads(cleaned)
        
    except Exception as e:
        # Ultimate fallback
        return {
            "ComplianceLevel": 2,
            "Comments": f"Error en el procesamiento JSON. Respuesta original: {raw_response[:500]}...",
            "FilesSearch": [{"FileName": file_name, "DocumentID": document_id}]
        }

def create_ipm_knowledge_base():
    """Create a vector store with IPM regulations and best practices"""
    
    # Sample IPM knowledge base documents
    ipm_knowledge = [
        Document(
            page_content="""PrimusGFS Module 9.01.01 - IPM Plan Requirements:
            The operation must have a documented IPM plan that includes:
            1. Pest monitoring and identification practices (visual scouting, traps, threshold levels)
            2. Action thresholds for treatment decisions (economic thresholds, damage levels)
            3. At least two pest prevention practices (crop rotation, sanitation, beneficial habitats)
            4. Pesticide resistance management strategies (rotation of chemical classes, IRAC guidelines)
            5. Pollinator protection measures (application timing, habitat preservation)
            
            Documentation must include revision dates, responsible personnel, and specific procedures.""",
            metadata={"source": "PrimusGFS_Module9", "section": "9.01.01", "type": "regulation"}
        ),
        Document(
            page_content="""IPM Monitoring Best Practices:
            Effective pest monitoring includes:
            - Weekly visual inspections during growing season
            - Strategic placement of monitoring traps (yellow sticky traps for aphids, pheromone traps for moths)
            - Recording pest counts and damage levels
            - Weather monitoring for pest development models
            - Use of degree-day calculations for pest lifecycle prediction
            
            Common monitoring tools: sticky traps, pheromone traps, beat sheets, visual inspection forms.""",
            metadata={"source": "IPM_Best_Practices", "topic": "monitoring", "type": "guidance"}
        ),
        Document(
            page_content="""Economic Thresholds for Common Pests:
            Strawberry Production:
            - Spider mites: 5-8 mites per leaflet before treatment
            - Aphids: 10-15 aphids per plant or 5% infested plants
            - Thrips: 20+ per sticky trap per week
            - Spotted wing drosophila: 1 adult per trap per week during fruit development
            
            Thresholds may vary based on crop stage, market value, and weather conditions.""",
            metadata={"source": "Economic_Thresholds", "crop": "strawberry", "type": "reference"}
        ),
        Document(
            page_content="""Pollinator Protection Measures:
            Required practices for IPM compliance:
            - Avoid applications during peak pollinator activity (10 AM - 4 PM)
            - Maintain flowering habitat strips or borders
            - Use selective pesticides when possible (avoid broad-spectrum insecticides)
            - Provide advance notice to beekeepers within 1 mile
            - Consider application timing relative to bloom periods
            - Implement buffer zones around sensitive habitats""",
            metadata={"source": "Pollinator_Protection", "type": "requirement"}
        ),
        Document(
            page_content="""Common IPM Compliance Failures:
            Frequent audit findings include:
            - Missing or outdated IPM plans (not revised annually)
            - Lack of documented monitoring procedures
            - Undefined action thresholds
            - Insufficient prevention practices (less than 2 documented)
            - No pesticide resistance management strategy
            - Missing pollinator protection measures
            - Incomplete record keeping of pest monitoring activities""",
            metadata={"source": "Audit_Findings", "type": "compliance_issues"}
        )
    ]
    
    return ipm_knowledge

def ipm_audit_with_rag(llm, embeddings, operation_name, product, documents, language, file_name, document_id):
    """Enhanced IPM audit using RAG for better accuracy"""
    
    try:
        # Create knowledge base
        ipm_knowledge = create_ipm_knowledge_base()
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=ipm_knowledge,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Get relevant regulations and best practices
        query = f"IPM plan requirements compliance audit {product}"
        relevant_knowledge = retriever.get_relevant_documents(query)
        
        # Format knowledge base context
        knowledge_context = "\n\n".join([
            f"REGULATION/GUIDANCE: {doc.page_content}"
            for doc in relevant_knowledge
        ])
        
        # Enhanced prompt with RAG context
        rag_prompt = ChatPromptTemplate.from_template("""
Eres un auditor IPM experto con acceso a la base de conocimientos de regulaciones y mejores prácticas.

CONTEXTO DE CONOCIMIENTOS IPM:
{knowledge_context}

OPERACIÓN A AUDITAR:
Operación: {operation_name}
Producto: {product}
Documentos Proporcionados: {documents}

PREGUNTA DE AUDITORÍA: ¿Tiene la operación un plan IPM documentado que cumpla con PrimusGFS Module 9.01.01?

INSTRUCCIONES:
1. Usa el contexto de conocimientos para evaluar el cumplimiento
2. Compara los documentos proporcionados con los requisitos regulatorios
3. Identifica brechas específicas de cumplimiento
4. Proporciona recomendaciones basadas en mejores prácticas
5. Responde en formato JSON con las claves exactas: ComplianceLevel, Comments, FilesSearch

Formato JSON requerido:
{{
  "ComplianceLevel": 2,
  "Comments": "Análisis detallado basado en regulaciones específicas...",
  "FilesSearch": [{{"FileName": "{file_name}", "DocumentID": "{document_id}"}}]
}}

Idioma: {language}
""")
        
        # Create RAG chain
        chain = rag_prompt | llm | StrOutputParser()
        
        # Run enhanced audit
        raw_response = chain.invoke({
            "knowledge_context": knowledge_context,
            "operation_name": operation_name,
            "product": product,
            "documents": documents,
            "language": language,
            "file_name": file_name,
            "document_id": document_id
        })
        
        # Clean and parse response
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        cleaned = cleaned.strip()
        
        result = json.loads(cleaned)
        
        # Add knowledge sources used
        result["KnowledgeSources"] = [doc.metadata.get("source", "Unknown") for doc in relevant_knowledge]
        
        return result
        
    except Exception as e:
        # Fallback to standard method
        return ipm_audit_fallback(llm, operation_name, product, documents, language, file_name, document_id)

def ipm_audit_tab():
    """IPM Compliance Audit functionality using LangChain"""
    st.header("🔍 IPM Compliance Audit")
    st.markdown("*Evaluación de cumplimiento IPM usando LangChain estructurado*")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
    if not llm:
        return
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        operation_name = st.text_input("Nombre de la Operación", value="Operación Ejemplo", key="operation_name")
        product = st.text_input("Producto", value="Fresas", key="product")
    
    with col2:
        language = st.selectbox("Idioma", ["Spanish", "English"], key="language")
    
    # Initialize session state for IPM audit
    if "ipm_file_name" not in st.session_state:
        st.session_state.ipm_file_name = "IPM_Plan_2024.pdf"
    if "ipm_doc_id" not in st.session_state:
        st.session_state.ipm_doc_id = "DOC_001"
    if "ipm_doc_content" not in st.session_state:
        st.session_state.ipm_doc_content = "Este es un plan IPM que incluye monitoreo semanal de plagas mediante trampas pegajosas. Se establecen umbrales de acción de 5 ácaros por hoja. Incluye rotación de cultivos y plantas de borde para atraer beneficiosos."
    
    # Document input
    st.subheader("📄 Documentos para Auditoría")
    
    # Option 2: Example documents (moved up to avoid conflicts)
    st.markdown("**Opción 1: Usar documento de ejemplo**")
    if st.button("📋 Cargar Ejemplo de Documento IPM", key="load_example"):
        st.session_state.ipm_file_name = "IPM_Strawberry2024.pdf"
        st.session_state.ipm_doc_id = "DOC_IPM_001"
        st.session_state.ipm_doc_content = """Integrated Pest Management Plan for Strawberry Operation
        
Revision Date: March 12, 2024
Prepared by: Juan Pérez, Farm Manager

Section 3 - Pest Monitoring:
Weekly visual scouting conducted on Mondays and Thursdays. Sticky traps placed every 50 meters to monitor flying insects. Monitoring forms document pest counts and locations.

Section 4 - Action Thresholds:
- Spider mites: Treatment when 5 or more mites per leaflet
- Aphids: Treatment when 10 or more per plant
- Thrips: Treatment when trap counts exceed 20 per week

Section 5 - Prevention Methods:
1. Crop rotation with lettuce every 2 years
2. Border plantings of alyssum to attract beneficial insects
3. Sanitation practices including removal of plant debris

Section 6 - Resistance Management:
Rotation of chemical classes following IRAC guidelines. Maximum 2 applications per season of same mode of action.

Note: Pollinator protection measures not documented in this plan."""
        st.rerun()
    
    # Option 1: Text input
    st.markdown("**Opción 2: Editar contenido del documento**")
    col1, col2 = st.columns(2)
    with col1:
        file_name = st.text_input("Nombre del archivo", value=st.session_state.ipm_file_name, key="file_name")
        if file_name != st.session_state.ipm_file_name:
            st.session_state.ipm_file_name = file_name
    with col2:
        document_id = st.text_input("ID del documento", value=st.session_state.ipm_doc_id, key="doc_id")
        if document_id != st.session_state.ipm_doc_id:
            st.session_state.ipm_doc_id = document_id
    
    document_content = st.text_area(
        "Contenido del documento:", 
        value=st.session_state.ipm_doc_content,
        height=150,
        key="doc_content"
    )
    if document_content != st.session_state.ipm_doc_content:
        st.session_state.ipm_doc_content = document_content
    
    # RAG Enhancement Option
    st.subheader("🚀 Método de Auditoría")
    use_rag = st.checkbox("🧠 Usar RAG (Conocimiento Mejorado)", value=True, key="use_rag",
                          help="Activa RAG para usar base de conocimientos de regulaciones IPM")
    
    if use_rag:
        st.info("✅ RAG Activado: La auditoría usará base de conocimientos de PrimusGFS y mejores prácticas")
    else:
        st.info("ℹ️ RAG Desactivado: Auditoría estándar sin base de conocimientos adicional")
    
    # Process audit
    if st.button("🔍 Realizar Auditoría IPM", key="process_audit", type="primary"):
        if not document_content.strip():
            st.error("Por favor proporciona contenido del documento para la auditoría.")
            return
        
        with st.spinner("Procesando auditoría IPM..."):
            try:
                # Prepare documents in the expected format
                documents = f"FileName: {file_name}\nDocumentID: {document_id}\nContent: {document_content}"
                
                # Choose audit method based on RAG setting
                if use_rag:
                    st.info("🧠 Ejecutando auditoría con RAG...")
                    try:
                        # Initialize embeddings for RAG
                        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
                        
                        # Run RAG-enhanced audit
                        result = ipm_audit_with_rag(
                            llm, embeddings, operation_name, product, documents,
                            language, file_name, document_id
                        )
                    except Exception as e:
                        st.warning("⚠️ Error en RAG. Usando método estándar...")
                        audit_chain = ipm_audit_chain(llm)
                        result = audit_chain.invoke({
                            "operation_name": operation_name,
                            "product": product,
                            "documents": documents,
                            "language": language
                        })
                else:
                    st.info("⚙️ Ejecutando auditoría estándar...")
                    # Run standard audit
                    try:
                        audit_chain = ipm_audit_chain(llm)
                        result = audit_chain.invoke({
                            "operation_name": operation_name,
                            "product": product,
                            "documents": documents,
                            "language": language
                        })
                    except Exception as e:
                        st.warning("⚠️ Error en método principal. Intentando método alternativo...")
                        result = ipm_audit_fallback(
                            llm, operation_name, product, documents, 
                            language, file_name, document_id
                        )
                
                # Display results
                st.success("✅ Auditoría IPM Completada")
                
                # Show structured results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Resultado de la Auditoría")
                    st.markdown(f"**Nivel de Cumplimiento:** {result['ComplianceLevel']}")
                    
                    # Show RAG indicator if used
                    if use_rag and "KnowledgeSources" in result:
                        st.success("🧠 **Auditoría mejorada con RAG**")
                    
                    st.markdown("**Comentarios:**")
                    st.write(result['Comments'])
                
                with col2:
                    st.subheader("📄 Archivos Analizados")
                    if result.get('FilesSearch'):
                        for file_info in result['FilesSearch']:
                            st.write(f"• **{file_info.get('FileName', 'N/A')}**")
                            st.write(f"  ID: {file_info.get('DocumentID', 'N/A')}")
                    else:
                        st.write("• " + file_name)
                        st.write(f"  ID: {document_id}")
                    
                    # Show knowledge sources if RAG was used
                    if use_rag and "KnowledgeSources" in result:
                        st.subheader("🧠 Fuentes de Conocimiento")
                        st.markdown("*Regulaciones y guías consultadas:*")
                        for source in result['KnowledgeSources']:
                            st.write(f"📚 {source}")
                
                # Show raw JSON response
                with st.expander("🔧 Respuesta JSON Completa"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"Error en la auditoría: {e}")
                st.error("Verifica que el modelo pueda generar respuestas JSON válidas.")
    
    # Information about the improvement
    st.markdown("---")
    st.subheader("💡 Mejoras con LangChain")
    
    improvements = [
        "**Prompt Estructurado**: Separación clara entre instrucciones del sistema y entrada del usuario",
        "**Output Parser**: Validación automática de la respuesta JSON con Pydantic",
        "**Reutilizable**: Función que puede ser llamada desde otras partes del código",
        "**Mantenible**: Fácil modificación de reglas y criterios de auditoría",
        "**Type Safety**: Validación de tipos con Pydantic models",
        "**RAG Integration**: Base de conocimientos con regulaciones PrimusGFS y mejores prácticas",
        "**Contexto Enriquecido**: Auditorías más precisas basadas en normativas oficiales",
        "**Transparencia**: Muestra las fuentes de conocimiento utilizadas en cada auditoría"
    ]
    
    for improvement in improvements:
        st.markdown(f"• {improvement}")
    
    # RAG Benefits
    st.subheader("🧠 Beneficios del RAG para Auditorías IPM")
    
    rag_benefits = [
        "**📚 Base de Conocimiento**: Acceso a regulaciones PrimusGFS actualizadas",
        "**🎯 Precisión Mejorada**: Comparación directa con requisitos oficiales",
        "**🔍 Detección de Brechas**: Identifica faltantes específicos de cumplimiento",
        "**📊 Mejores Prácticas**: Incorpora experiencia de auditorías previas",
        "**🌱 Específico por Cultivo**: Umbrales y prácticas adaptadas al producto",
        "**🔗 Trazabilidad**: Referencias claras a fuentes regulatorias"
    ]
    
    for benefit in rag_benefits:
        st.markdown(f"• {benefit}")

def basic_chat_tab():
    """Basic chat functionality"""
    st.header("💬 Chat Básico")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
    if not llm:
        return
    
    # Chat interface
    if prompt := st.chat_input("Escribe tu mensaje aquí...", key="basic_chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Generando respuesta..."):
            try:
                response = llm.invoke(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Clear chat button
    if st.button("🗑️ Limpiar Chat", key="clear_basic_chat"):
        st.session_state.messages = []
        st.rerun()

def memory_chat_tab():
    """Chat with memory functionality"""
    st.header("🧠 Chat con Memoria")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
    if not llm:
        return
    
    # Create prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil que mantiene contexto de la conversación."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Chat interface
    if user_input := st.chat_input("Escribe tu mensaje aquí...", key="memory_chat_input"):
        # Add to memory
        st.session_state.memory.chat_memory.add_user_message(user_input)
        
        # Generate response
        with st.spinner("Generando respuesta..."):
            try:
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.memory.chat_memory.messages
                })
                
                # Add response to memory
                st.session_state.memory.chat_memory.add_ai_message(response)
                
            except Exception as e:
                st.error(f"Error: {e}")
                response = "Lo siento, ocurrió un error."
    
    # Display conversation
    for message in st.session_state.memory.chat_memory.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(message.content)
    
    # Clear memory button
    if st.button("🗑️ Limpiar Memoria", key="clear_memory_chat"):
        st.session_state.memory.clear()
        st.rerun()

def agents_tab():
    """Agent with tools functionality"""
    st.header("🤖 Agentes con Herramientas")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
    if not llm:
        return
    
    # Tool selection
    st.subheader("Herramientas Disponibles")
    tools = [calculator, get_current_time, text_analyzer]
    
    for tool in tools:
        st.write(f"• **{tool.name}**: {tool.description}")
    
    # Create agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente útil que puede usar herramientas para ayudar al usuario.
        Tienes acceso a calculadora, información de tiempo y análisis de texto.
        Usa las herramientas cuando sea apropiado."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Agent interface
    if user_input := st.chat_input("Prueba el agente con herramientas...", key="agent_chat_input"):
        with st.spinner("El agente está trabajando..."):
            try:
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": []
                })
                
                st.success("✅ Respuesta del agente:")
                st.write(response['output'])
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Example queries
    st.subheader("Ejemplos de Consultas")
    examples = [
        "¿Cuál es el resultado de 25 * 34 + 12?",
        "¿Qué hora es?",
        "Analiza este texto: 'LangChain es una herramienta poderosa para desarrolladores.'",
        "Calcula 2^10 y luego dime qué hora es"
    ]
    
    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
            with st.spinner("Procesando..."):
                try:
                    response = agent_executor.invoke({
                        "input": example,
                        "chat_history": []
                    })
                    st.success("✅ Respuesta:")
                    st.write(response['output'])
                except Exception as e:
                    st.error(f"Error: {e}")

def rag_tab():
    """RAG (Retrieval Augmented Generation) functionality"""
    st.header("📚 RAG - Preguntas sobre Documentos")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize components
    llm = init_llm()
    if not llm:
        return
    
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    
    # Sample documents
    sample_docs = [
        Document(
            page_content="LangChain es un framework para desarrollar aplicaciones con LLMs. Fue creado por Harrison Chase.",
            metadata={"source": "intro.txt"}
        ),
        Document(
            page_content="Los componentes principales de LangChain incluyen LLMs, prompts, chains, agents, memory y retrievers.",
            metadata={"source": "components.txt"}
        ),
        Document(
            page_content="RAG combina recuperación de información con generación de texto para responder preguntas basadas en documentos.",
            metadata={"source": "rag.txt"}
        )
    ]
    
    # Create or load vector store
    if st.session_state.vectorstore is None:
        with st.spinner("Creando base de datos vectorial..."):
            try:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=sample_docs,
                    embedding=embeddings
                )
                st.success("✅ Base de datos vectorial creada")
            except Exception as e:
                st.error(f"Error creando vector store: {e}")
                return
    
    # Document upload
    st.subheader("📄 Documentos en la Base de Datos")
    with st.expander("Ver documentos disponibles"):
        for i, doc in enumerate(sample_docs, 1):
            st.write(f"**Documento {i}**: {doc.metadata['source']}")
            st.write(doc.page_content[:200] + "...")
    
    # RAG interface
    st.subheader("❓ Hacer Pregunta")
    
    if question := st.text_input("Escribe tu pregunta sobre los documentos:", key="rag_question_input"):
        if st.button("🔍 Buscar Respuesta", key="search_answer_button"):
            with st.spinner("Buscando respuesta..."):
                try:
                    # Create retriever
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                    
                    # Get relevant documents
                    docs = retriever.get_relevant_documents(question)
                    
                    # Create RAG chain
                    template = """
Eres un asistente que SOLO puede responder basándose en el contexto de documentos proporcionado. 

REGLAS ESTRICTAS:
- ÚNICAMENTE usa la información que aparece en el contexto de documentos
- Si la pregunta no se puede responder con el contexto proporcionado, di claramente: "No puedo responder esa pregunta basándome en los documentos proporcionados"
- NO uses tu conocimiento general o información externa
- Sintetiza y explica la información del contexto de manera natural y conversacional
- No copies fragmentos literales, pero mantente fiel al contenido

Contexto de documentos disponible:
{context}

Pregunta: {question}

Respuesta (solo basada en el contexto):"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Generate answer
                    answer = rag_chain.invoke(question)
                    
                    # Display results
                    st.success("✅ Respuesta encontrada:")
                    st.write(answer)
                    
                    # Show relevant sentences
                    st.subheader("🎯 Oraciones Más Relevantes")
                    for i, doc in enumerate(docs, 1):
                        # Split document into sentences
                        sentences = [s.strip() for s in doc.page_content.split('.') if s.strip()]
                        
                        st.markdown(f"**Del documento {doc.metadata['source']}:**")
                        for j, sentence in enumerate(sentences, 1):
                            if sentence:  # Only show non-empty sentences
                                st.markdown(f"• {sentence}.")
                        st.markdown("---")
                    
                    st.subheader("📖 Documentos Relevantes Completos")
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Documento {i}: {doc.metadata['source']}"):
                            st.write(doc.page_content)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Example questions
    st.subheader("💡 Preguntas de Ejemplo")
    example_questions = [
        "¿Qué es LangChain?",
        "¿Quién creó LangChain?",
        "¿Cuáles son los componentes principales?",
        "¿Cómo funciona RAG?"
    ]
    
    for question in example_questions:
        if st.button(f"❓ {question}", key=f"rag_example_{hash(question)}"):
            with st.spinner("Buscando respuesta..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                    docs = retriever.get_relevant_documents(question)
                    
                    template = """Usa el siguiente contexto para responder la pregunta. Si no sabes la respuesta, di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    answer = rag_chain.invoke(question)
                    
                    st.success("✅ Respuesta:")
                    st.write(answer)
                    
                    # Show relevant sentences for example questions
                    st.subheader("🎯 Oraciones Más Relevantes")
                    for i, doc in enumerate(docs, 1):
                        # Split document into sentences
                        sentences = [s.strip() for s in doc.page_content.split('.') if s.strip()]
                        
                        st.markdown(f"**Del documento {doc.metadata['source']}:**")
                        for j, sentence in enumerate(sentences, 1):
                            if sentence:  # Only show non-empty sentences
                                st.markdown(f"• {sentence}.")
                        st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

def main():
    """Main application"""
    st.title("🦜 LangChain Demo Application")
    st.markdown("Explora las capacidades de LangChain a través de esta interfaz interactiva")
    
    # Sidebar
    st.sidebar.title("📋 Configuración")
    
    # API Key status
    if check_api_key():
        st.sidebar.success("✅ API Key configurada correctamente")
    else:
        st.sidebar.error("❌ API Key no configurada")
        st.sidebar.info("Configura tu API key en el archivo .env")
    
    # Model settings
    st.sidebar.subheader("🔧 Configuración del Modelo")
    st.sidebar.write(f"**Modelo**: {Config.DEFAULT_MODEL}")
    st.sidebar.write(f"**Temperatura**: {Config.DEFAULT_TEMPERATURE}")
    st.sidebar.write(f"**Max Tokens**: {Config.MAX_TOKENS}")
    
    # Navigation
    st.sidebar.subheader("🚀 Funcionalidades")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat Básico",
        "🧠 Chat con Memoria", 
        "🤖 Agentes",
        "📚 RAG",
        "🔍 IPM Audit"
    ])
    
    with tab1:
        basic_chat_tab()
    
    with tab2:
        memory_chat_tab()
    
    with tab3:
        agents_tab()
    
    with tab4:
        rag_tab()
    
    with tab5:
        ipm_audit_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Asegúrate de tener configurada tu API key de OpenAI para usar todas las funcionalidades.")

if __name__ == "__main__":
    main() 