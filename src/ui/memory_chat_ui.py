"""
Componente de UI para chat con memoria
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from src.services import LLMService


class MemoryChatUI:
    """Componente de UI para chat con memoria"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Inicializa el estado de sesión para chat con memoria"""
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    def render(self):
        """Renderiza la interfaz de chat con memoria"""
        st.header("🧠 Chat con Memoria")
        
        if not LLMService.check_api_key():
            st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
            return
        
        # Get LLM instance
        llm = LLMService.get_llm()
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