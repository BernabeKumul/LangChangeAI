"""
Servicio para gestión de LLM (Large Language Models)
"""

from langchain_openai import ChatOpenAI
from config import Config
from typing import Optional

try:
    import streamlit as st
except ImportError:
    st = None


class LLMService:
    """Servicio centralizado para gestión de LLM"""
    
    _instance: Optional[ChatOpenAI] = None
    
    @classmethod
    def get_llm(cls) -> Optional[ChatOpenAI]:
        """
        Obtiene una instancia del LLM con configuración cached
        
        Returns:
            ChatOpenAI instance o None si hay error
        """
        if cls._instance is None:
            cls._instance = cls._initialize_llm()
        return cls._instance
    
    @classmethod
    def _initialize_llm(cls) -> Optional[ChatOpenAI]:
        """Inicializa el LLM con manejo de errores"""
        try:
            Config.validate_config()
            return ChatOpenAI(
                model=Config.DEFAULT_MODEL,
                temperature=Config.DEFAULT_TEMPERATURE,
                openai_api_key=Config.OPENAI_API_KEY,
                max_tokens=Config.MAX_TOKENS
            )
        except ValueError as e:
            if st:
                st.error(f"Error de configuración: {e}")
            print(f"Error de configuración: {e}")
            return None
        except Exception as e:
            if st:
                st.error(f"Error inicializando LLM: {e}")
            print(f"Error inicializando LLM: {e}")
            return None
    
    @classmethod
    def check_api_key(cls) -> bool:
        """Verifica si la API key está configurada correctamente"""
        try:
            Config.validate_config()
            return True
        except:
            return False
    
    @classmethod
    def reset_instance(cls):
        """Resetea la instancia cached (útil para testing)"""
        cls._instance = None 