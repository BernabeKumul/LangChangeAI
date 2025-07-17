"""
RAG (Retrieval Augmented Generation) Example
This example demonstrates document retrieval and question answering:
- Document loading and processing
- Vector embeddings
- Similarity search
- Question answering with context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
import tempfile
import shutil
from config import Config

def create_sample_documents():
    """Create sample documents for RAG demonstration"""
    documents = [
        Document(
            page_content="""
            LangChain es un framework de desarrollo que permite crear aplicaciones potenciadas por modelos de lenguaje grandes (LLM). 
            Fue desarrollado por Harrison Chase y se ha convertido en una herramienta fundamental para los desarrolladores que trabajan con IA.
            
            Las principales características de LangChain incluyen:
            - Integración con múltiples LLMs (OpenAI, Anthropic, Google, etc.)
            - Gestión de prompts y plantillas
            - Cadenas de procesamiento
            - Agentes con herramientas
            - Memoria conversacional
            - Capacidades de recuperación de información
            """,
            metadata={"source": "langchain_intro.txt", "topic": "introduccion"}
        ),
        Document(
            page_content="""
            Los componentes principales de LangChain son:
            
            1. LLMs y Chat Models: Interfaces para interactuar con modelos de lenguaje
            2. Prompt Templates: Plantillas para formatear inputs a los LLMs
            3. Chains: Secuencias de llamadas a LLMs y otras herramientas
            4. Agents: Sistemas que pueden decidir qué acciones tomar
            5. Memory: Sistemas para mantener estado entre llamadas
            6. Retrievers: Interfaces para recuperar información de fuentes de datos
            7. Document Loaders: Herramientas para cargar documentos de diversas fuentes
            """,
            metadata={"source": "langchain_components.txt", "topic": "componentes"}
        ),
        Document(
            page_content="""
            RAG (Retrieval Augmented Generation) es una técnica que combina la recuperación de información con la generación de texto.
            
            El proceso RAG típico incluye:
            1. Indexación: Los documentos se procesan y almacenan en un vector store
            2. Recuperación: Se buscan documentos relevantes basados en la consulta
            3. Generación: Se usa el contexto recuperado para generar una respuesta
            
            LangChain facilita la implementación de RAG proporcionando:
            - Document loaders para diversos formatos
            - Text splitters para dividir documentos
            - Vector stores para almacenamiento eficiente
            - Retrieval chains para QA
            """,
            metadata={"source": "rag_explanation.txt", "topic": "rag"}
        ),
        Document(
            page_content="""
            Los agentes en LangChain son sistemas que pueden razonar sobre qué acciones tomar para resolver una tarea.
            
            Componentes de un agente:
            1. LLM: El modelo de lenguaje que actúa como el "cerebro"
            2. Tools: Funciones que el agente puede usar
            3. Agent: La lógica que decide qué herramienta usar
            4. Agent Executor: El sistema que ejecuta las acciones
            
            Tipos de agentes:
            - Zero-shot React: Agente que puede usar herramientas sin ejemplos previos
            - Conversational React: Agente optimizado para conversaciones
            - OpenAI Functions: Agente que usa function calling de OpenAI
            """,
            metadata={"source": "agents_guide.txt", "topic": "agentes"}
        ),
        Document(
            page_content="""
            Vector stores son bases de datos especializadas en almacenar y buscar vectores de alta dimensión.
            
            En LangChain, los vector stores más populares incluyen:
            - Chroma: Base de datos vectorial ligera y fácil de usar
            - Pinecone: Servicio de vector database en la nube
            - Weaviate: Base de datos vectorial open-source
            - FAISS: Biblioteca de Facebook para búsqueda de similitud
            
            Los embeddings son representaciones numéricas de texto que capturan el significado semántico.
            LangChain soporta varios proveedores de embeddings:
            - OpenAI Embeddings
            - HuggingFace Embeddings
            - Cohere Embeddings
            """,
            metadata={"source": "vectorstores_guide.txt", "topic": "vectorstores"}
        )
    ]
    
    return documents

def basic_rag_example():
    """Basic RAG example with in-memory vector store"""
    print("=== Basic RAG Example ===")
    
    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Document split into {len(splits)} chunks")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
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
    
    # Test questions
    questions = [
        "¿Qué es LangChain?",
        "¿Cuáles son los componentes principales de LangChain?",
        "¿Cómo funciona RAG?",
        "¿Qué tipos de agentes existen en LangChain?",
        "¿Qué son los vector stores?",
        "¿Cómo se puede usar Python con LangChain?"  # Esta pregunta no tiene respuesta en el contexto
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)
        print(f"Retrieved {len(docs)} documents")
        
        # Generate answer
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")
        print("-" * 30)
    
    print("-" * 50)

def persistent_rag_example():
    """RAG example with persistent vector store"""
    print("=== Persistent RAG Example ===")
    
    # Create temporary directory for persistent storage
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Initialize embeddings and LLM
        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.DEFAULT_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Create sample documents
        documents = create_sample_documents()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Create persistent vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Persist the vector store
        vectorstore.persist()
        print(f"Vector store persisted to: {persist_directory}")
        
        # Load from persistent storage
        vectorstore_loaded = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_loaded.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )
        
        # Test questions
        questions = [
            "¿Quién desarrolló LangChain?",
            "¿Cuáles son las ventajas de usar RAG?",
            "¿Qué es un agente en LangChain?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}: {question}")
            
            result = qa_chain({"query": question})
            print(f"Answer: {result['result']}")
            
            print("Source documents:")
            for j, doc in enumerate(result['source_documents'], 1):
                print(f"  {j}. {doc.metadata['source']} - {doc.page_content[:100]}...")
            print("-" * 30)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(persist_directory)
    
    print("-" * 50)

def similarity_search_example():
    """Example of similarity search with metadata filtering"""
    print("=== Similarity Search Example ===")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Test similarity search
    queries = [
        "¿Qué es RAG?",
        "agentes y herramientas",
        "bases de datos vectoriales"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Basic similarity search
        docs = vectorstore.similarity_search(query, k=2)
        print(f"Found {len(docs)} similar documents:")
        
        for j, doc in enumerate(docs, 1):
            print(f"  {j}. Topic: {doc.metadata.get('topic', 'unknown')}")
            print(f"     Content: {doc.page_content[:150]}...")
            print(f"     Source: {doc.metadata.get('source', 'unknown')}")
        
        # Similarity search with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
        print(f"\nSimilarity scores:")
        for doc, score in docs_with_scores:
            print(f"  Score: {score:.4f} - {doc.metadata.get('topic', 'unknown')}")
        
        print("-" * 30)
    
    print("-" * 50)

def metadata_filtering_example():
    """Example of metadata filtering in retrieval"""
    print("=== Metadata Filtering Example ===")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Search with metadata filtering
    print("Search for 'componentes' in documents about 'componentes' topic:")
    docs = vectorstore.similarity_search(
        "componentes principales",
        k=3,
        filter={"topic": "componentes"}
    )
    
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc.metadata}")
        print(f"     {doc.page_content[:100]}...")
    
    print(f"\nSearch for 'agentes' in all documents:")
    docs = vectorstore.similarity_search("agentes", k=3)
    
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. Topic: {doc.metadata.get('topic')}")
        print(f"     {doc.page_content[:100]}...")
    
    print("-" * 50)

def custom_retrieval_example():
    """Example of custom retrieval with different strategies"""
    print("=== Custom Retrieval Example ===")
    
    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    # Different retrieval strategies
    retrievers = {
        "similarity": vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
        "mmr": vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}),
        "similarity_score": vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.5}
        )
    }
    
    query = "¿Cómo funcionan los agentes en LangChain?"
    
    for strategy, retriever in retrievers.items():
        print(f"\nRetrieval strategy: {strategy}")
        try:
            docs = retriever.get_relevant_documents(query)
            print(f"Retrieved {len(docs)} documents:")
            
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc.metadata.get('topic', 'unknown')}")
                print(f"     {doc.page_content[:100]}...")
        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")
        
        print("-" * 20)
    
    print("-" * 50)

def main():
    """Main function to run all RAG examples"""
    try:
        print("📚 LangChain RAG Examples")
        print("=" * 50)
        
        Config.validate_config()
        
        basic_rag_example()
        persistent_rag_example()
        similarity_search_example()
        metadata_filtering_example()
        custom_retrieval_example()
        
        print("✅ All RAG examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create .env file with your OPENAI_API_KEY")

if __name__ == "__main__":
    main() 