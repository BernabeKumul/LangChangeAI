"""
Utilidades para procesamiento y análisis de documentos
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from typing import List, Dict, Tuple, Any
from src.models.ipm_models import DocumentInfo


class DocumentProcessor:
    """Clase para procesamiento y análisis de documentos"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estima el número de tokens (aproximación: 1 token ≈ 4 caracteres)"""
        return len(text) // 4
    
    @staticmethod
    def extract_ipm_relevant_content(text: str, max_tokens: int = 1000) -> Tuple[str, int]:
        """Extrae secciones IPM-relevantes del texto del documento"""
        
        # IPM-related keywords to search for
        ipm_keywords = [
            "pest", "ipm", "integrated pest management", "monitoring", "threshold", 
            "pesticide", "biological control", "prevention", "scouting", "trap",
            "beneficial", "pollinator", "resistance management", "economic threshold",
            "action threshold", "crop rotation", "sanitation", "habitat"
        ]
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        relevant_paragraphs = []
        total_tokens = 0
        
        # Score and select most relevant paragraphs
        scored_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
                
            # Count IPM-related keywords
            score = sum(1 for keyword in ipm_keywords if keyword.lower() in para.lower())
            if score > 0:
                scored_paragraphs.append((score, para))
        
        # Sort by relevance score (descending)
        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
        
        # Select paragraphs until token limit
        for score, para in scored_paragraphs:
            para_tokens = DocumentProcessor.estimate_tokens(para)
            if total_tokens + para_tokens <= max_tokens:
                relevant_paragraphs.append(para)
                total_tokens += para_tokens
            else:
                break
        
        return '\n\n'.join(relevant_paragraphs), total_tokens
    
    @staticmethod
    def summarize_document(text: str, llm, max_length: int = 500) -> str:
        """Resume un documento enfocándose en contenido IPM-relevante"""
        
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            summarize_prompt = ChatPromptTemplate.from_template("""
            Resume el siguiente documento enfocándote únicamente en aspectos relacionados con IPM (Manejo Integrado de Plagas):
            
            - Monitoreo de plagas
            - Umbrales de acción
            - Prácticas de prevención
            - Manejo de resistencia
            - Protección de polinizadores
            - Cualquier plan o procedimiento IPM
            
            Documento:
            {document}
            
            Resumen IPM (máximo {max_length} caracteres):
            """)
            
            chain = summarize_prompt | llm | StrOutputParser()
            summary = chain.invoke({
                "document": text[:8000],  # Limit input to avoid token overflow
                "max_length": max_length
            })
            
            return summary[:max_length]  # Ensure length limit
            
        except Exception as e:
            # Fallback: simple truncation
            return text[:max_length] + "..."
    
    @staticmethod
    def process_multiple_documents(
        documents_list: List[Dict[str, Any]], 
        llm, 
        strategy: str = "smart_extraction",
        max_total_tokens: int = 3000
    ) -> Tuple[List[DocumentInfo], int]:
        """
        Procesa múltiples documentos con optimización de tokens
        
        Strategies:
        - smart_extraction: Extract only IPM-relevant content
        - summarization: Summarize each document
        - chunking: Use semantic chunking and similarity search
        """
        
        processed_docs = []
        total_tokens = 0
        
        for doc_info in documents_list:
            file_name = doc_info.get('filename', 'unknown.pdf')
            doc_id = doc_info.get('doc_id', 'DOC_UNKNOWN')
            content = doc_info.get('content', '')
            
            if not content.strip():
                continue
                
            content_tokens = DocumentProcessor.estimate_tokens(content)
            
            if strategy == "smart_extraction":
                # Extract only IPM-relevant content
                if content_tokens > 1000:  # Only process if document is large
                    extracted_content, used_tokens = DocumentProcessor.extract_ipm_relevant_content(
                        content, max_tokens=800
                    )
                    processed_content = extracted_content
                    st.info(f"📄 {file_name}: Extraídas secciones relevantes ({used_tokens} tokens de {content_tokens})")
                else:
                    processed_content = content
                    used_tokens = content_tokens
                    
            elif strategy == "summarization":
                # Summarize the document
                if content_tokens > 800:
                    processed_content = DocumentProcessor.summarize_document(content, llm, max_length=600)
                    used_tokens = DocumentProcessor.estimate_tokens(processed_content)
                    st.info(f"📄 {file_name}: Documento resumido ({used_tokens} tokens de {content_tokens})")
                else:
                    processed_content = content
                    used_tokens = content_tokens
                    
            elif strategy == "chunking":
                # Use text splitting and keep most relevant chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    separators=["\n\n", "\n", ". ", " "]
                )
                chunks = text_splitter.split_text(content)
                
                # Score chunks for IPM relevance
                relevant_chunks = []
                for chunk in chunks[:6]:  # Limit to first 6 chunks
                    chunk_tokens = DocumentProcessor.estimate_tokens(chunk)
                    if total_tokens + chunk_tokens <= max_total_tokens:
                        relevant_chunks.append(chunk)
                        total_tokens += chunk_tokens
                        
                processed_content = '\n\n'.join(relevant_chunks)
                used_tokens = DocumentProcessor.estimate_tokens(processed_content)
                st.info(f"📄 {file_name}: Procesados {len(relevant_chunks)} chunks ({used_tokens} tokens)")
            
            # Check total token limit
            if total_tokens + used_tokens > max_total_tokens:
                st.warning(f"⚠️ Límite de tokens alcanzado. Omitiendo documento: {file_name}")
                break
                
            processed_docs.append(DocumentInfo(
                filename=file_name,
                doc_id=doc_id,
                content=processed_content,
                original_tokens=content_tokens,
                processed_tokens=used_tokens
            ))
            
            total_tokens += used_tokens
        
        return processed_docs, total_tokens
    
    @staticmethod
    def chunk_and_search_documents(
        documents_list: List[Dict[str, Any]], 
        embeddings: OpenAIEmbeddings,
        query: str = "IPM plan compliance monitoring", 
        max_chunks: int = 8
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Usa búsqueda semántica para encontrar chunks más relevantes del documento"""
        
        try:
            # Create documents for chunking
            all_docs = []
            for doc_info in documents_list:
                content = doc_info.get('content', '')
                if content.strip():
                    # Split into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=100,
                        separators=["\n\n", "\n", ". ", " "]
                    )
                    chunks = text_splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        all_docs.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": doc_info.get('filename', 'unknown'),
                                "doc_id": doc_info.get('doc_id', 'unknown'),
                                "chunk_id": i
                            }
                        ))
            
            if not all_docs:
                return [], 0
                
            # Create vector store
            vectorstore = Chroma.from_documents(all_docs, embeddings)
            
            # Search for relevant chunks
            retriever = vectorstore.as_retriever(search_kwargs={"k": max_chunks})
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Format results
            relevant_content = []
            total_tokens = 0
            
            for doc in relevant_docs:
                chunk_tokens = DocumentProcessor.estimate_tokens(doc.page_content)
                total_tokens += chunk_tokens
                
                relevant_content.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "doc_id": doc.metadata.get("doc_id", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "tokens": chunk_tokens
                })
            
            return relevant_content, total_tokens
            
        except Exception as e:
            st.error(f"Error en búsqueda semántica: {e}")
            return [], 0 