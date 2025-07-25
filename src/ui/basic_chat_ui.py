"""
Componente de UI para chat básico
"""

import streamlit as st
from src.services import LLMService


class BasicChatUI:
    """Componente de UI para chat básico"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de sesión para chat básico"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    def render(self):
        """Renderiza la interfaz de chat básico"""
        st.header("💬 Chat Básico")
        
        if not LLMService.check_api_key():
            st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
            return
        
        # Get LLM instance
        llm = LLMService.get_llm()
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