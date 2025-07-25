"""
Servicio para gestión de embeddings
"""

from langchain_openai import OpenAIEmbeddings
from config import Config
from typing import Optional

try:
    import streamlit as st
except ImportError:
    st = None


class EmbeddingService:
    """Servicio centralizado para gestión de embeddings"""
    
    _instance: Optional[OpenAIEmbeddings] = None
    
    @classmethod
    def get_embeddings(cls) -> Optional[OpenAIEmbeddings]:
        """
        Obtiene una instancia de embeddings con configuración cached
        
        Returns:
            OpenAIEmbeddings instance o None si hay error
        """
        if cls._instance is None:
            cls._instance = cls._initialize_embeddings()
        return cls._instance
    
    @classmethod
    def _initialize_embeddings(cls) -> Optional[OpenAIEmbeddings]:
        """Inicializa embeddings con manejo de errores"""
        try:
            Config.validate_config()
            return OpenAIEmbeddings(
                openai_api_key=Config.OPENAI_API_KEY
            )
        except ValueError as e:
            if st:
                st.error(f"Error de configuración para embeddings: {e}")
            print(f"Error de configuración para embeddings: {e}")
            return None
        except Exception as e:
            if st:
                st.error(f"Error inicializando embeddings: {e}")
            print(f"Error inicializando embeddings: {e}")
            return None
    
    @classmethod
    def reset_instance(cls):
        """Resetea la instancia cached (útil para testing)"""
        cls._instance = None 