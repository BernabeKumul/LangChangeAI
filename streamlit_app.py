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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
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
                    
                    # Generate answer
                    answer = rag_chain.invoke(question)
                    
                    # Display results
                    st.success("✅ Respuesta encontrada:")
                    st.write(answer)
                    
                    st.subheader("📖 Documentos Relevantes")
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Básico",
        "🧠 Chat con Memoria", 
        "🤖 Agentes",
        "📚 RAG"
    ])
    
    with tab1:
        basic_chat_tab()
    
    with tab2:
        memory_chat_tab()
    
    with tab3:
        agents_tab()
    
    with tab4:
        rag_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Asegúrate de tener configurada tu API key de OpenAI para usar todas las funcionalidades.")

if __name__ == "__main__":
    main() 