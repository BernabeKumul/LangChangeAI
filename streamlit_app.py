"""
LangChain Streamlit Demo Application - Modular Version
A web interface to demonstrate LangChain capabilities using modular architecture
"""

import streamlit as st
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from src.services import LLMService
from src.ui import IMPAuditUI, BasicChatUI, MemoryChatUI, AgentsUI, RAGUI, DemoUI


def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="LangChain Demo",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_sidebar():
    """Render sidebar with configuration and status"""
    st.sidebar.title("📋 Configuración")
    
    # API Key status
    if LLMService.check_api_key():
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
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Asegúrate de tener configurada tu API key de OpenAI para usar todas las funcionalidades.")


def main():
    """Main application entry point"""
    # Configure page
    configure_page()
    
    # Initialize session state for memory and vector store
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
    # Main title
    st.title("🦜 LangChain Demo Application")
    st.markdown("Explora las capacidades de LangChain a través de esta interfaz interactiva")
    
    # Render sidebar
    render_sidebar()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat Básico",
        "🧠 Chat con Memoria", 
        "🤖 Agentes",
        "📚 RAG",
        "🔍 IPM Audit",
        "📝 Demo Prompts"
    ])
    
    # Render each tab using modular UI components
    with tab1:
        BasicChatUI().render()
    
    with tab2:
        MemoryChatUI().render()
    
    with tab3:
        AgentsUI().render()
    
    with tab4:
        RAGUI().render()
    
    with tab5:
        IMPAuditUI().render()
    
    with tab6:
        DemoUI().render()


if __name__ == "__main__":
    main() 