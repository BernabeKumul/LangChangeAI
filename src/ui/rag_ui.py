"""
Componente de UI para RAG (Retrieval Augmented Generation)
"""

import streamlit as st
from langchain.schema import Document
from src.services import LLMService, EmbeddingService, ChainService


class RAGUI:
    """Componente de UI para RAG"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de sesión para RAG"""
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = None
    
    def render(self):
        """Renderiza la interfaz de RAG"""
        st.header("📚 RAG - Preguntas sobre Documentos")
        
        if not LLMService.check_api_key():
            st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
            return
        
        # Get services
        llm = LLMService.get_llm()
        embeddings = EmbeddingService.get_embeddings()
        
        if not llm or not embeddings:
            return
        
        self._create_vector_store(embeddings)
        self._render_documents_info()
        self._render_rag_interface(llm, embeddings)
        self._render_example_questions(llm, embeddings)
    
    def _create_vector_store(self, embeddings):
        """Crea o carga el vector store"""
        if st.session_state.vectorstore is None:
            sample_docs = self._get_sample_documents()
            
            with st.spinner("Creando base de datos vectorial..."):
                try:
                    from langchain_community.vectorstores import Chroma
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=sample_docs,
                        embedding=embeddings
                    )
                    st.success("✅ Base de datos vectorial creada")
                except Exception as e:
                    st.error(f"Error creando vector store: {e}")
                    return
    
    def _get_sample_documents(self):
        """Obtiene documentos de ejemplo"""
        return [
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
    
    def _render_documents_info(self):
        """Renderiza información sobre documentos"""
        st.subheader("📄 Documentos en la Base de Datos")
        with st.expander("Ver documentos disponibles"):
            sample_docs = self._get_sample_documents()
            for i, doc in enumerate(sample_docs, 1):
                st.write(f"**Documento {i}**: {doc.metadata['source']}")
                st.write(doc.page_content[:200] + "...")
    
    def _render_rag_interface(self, llm, embeddings):
        """Renderiza interfaz principal de RAG"""
        st.subheader("❓ Hacer Pregunta")
        
        if question := st.text_input("Escribe tu pregunta sobre los documentos:", key="rag_question_input"):
            if st.button("🔍 Buscar Respuesta", key="search_answer_button"):
                self._process_rag_question(question, llm)
    
    def _process_rag_question(self, question, llm):
        """Procesa pregunta usando RAG"""
        with st.spinner("Buscando respuesta..."):
            try:
                sample_docs = self._get_sample_documents()
                rag_chain, retriever = ChainService.create_rag_chain(
                    llm, EmbeddingService.get_embeddings(), sample_docs
                )
                
                # Generate answer
                answer = rag_chain.invoke(question)
                
                # Get relevant documents
                docs = retriever.get_relevant_documents(question)
                
                # Display results
                st.success("✅ Respuesta encontrada:")
                st.write(answer)
                
                # Show relevant documents
                st.subheader("📖 Documentos Relevantes")
                for i, doc in enumerate(docs, 1):
                    with st.expander(f"Documento {i}: {doc.metadata['source']}"):
                        st.write(doc.page_content)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    def _render_example_questions(self, llm, embeddings):
        """Renderiza preguntas de ejemplo"""
        st.subheader("💡 Preguntas de Ejemplo")
        example_questions = [
            "¿Qué es LangChain?",
            "¿Quién creó LangChain?",
            "¿Cuáles son los componentes principales?",
            "¿Cómo funciona RAG?"
        ]
        
        for question in example_questions:
            if st.button(f"❓ {question}", key=f"rag_example_{hash(question)}"):
                self._process_rag_question(question, llm) 