"""
Componente de UI para agentes con herramientas
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from src.services import LLMService
from src.tools import (
    calculator, get_current_time, text_analyzer,
    analyze_pest_monitoring, check_action_thresholds, evaluate_prevention_practices,
    assess_resistance_management, verify_pollinator_protection, format_audit_response
)


class AgentsUI:
    """Componente de UI para agentes con herramientas"""
    
    def render(self):
        """Renderiza la interfaz de agentes"""
        st.header("🤖 Agentes con Herramientas")
        
        if not LLMService.check_api_key():
            st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
            return
        
        # Get LLM instance
        llm = LLMService.get_llm()
        if not llm:
            return
        
        self._render_tools_info()
        
        # Create agent with tools
        all_tools = self._get_all_tools()
        agent_executor = self._create_agent(llm, all_tools)
        
        self._render_agent_interface(agent_executor)
        self._render_example_queries(agent_executor)
    
    def _render_tools_info(self):
        """Renderiza información sobre herramientas disponibles"""
        st.subheader("Herramientas Disponibles")
        
        general_tools = [calculator, get_current_time, text_analyzer]
        ipm_tools = [analyze_pest_monitoring, check_action_thresholds, evaluate_prevention_practices, 
                     assess_resistance_management, verify_pollinator_protection, format_audit_response]
        
        st.markdown("**📊 Herramientas Generales:**")
        for tool in general_tools:
            st.write(f"• **{tool.name}**: {tool.description}")
        
        st.markdown("**🔍 Herramientas IPM Especializadas:**")
        for tool in ipm_tools:
            st.write(f"• **{tool.name}**: {tool.description}")
    
    def _get_all_tools(self):
        """Obtiene todas las herramientas disponibles"""
        general_tools = [calculator, get_current_time, text_analyzer]
        ipm_tools = [analyze_pest_monitoring, check_action_thresholds, evaluate_prevention_practices, 
                     assess_resistance_management, verify_pollinator_protection, format_audit_response]
        return general_tools + ipm_tools
    
    def _create_agent(self, llm, tools):
        """Crea el agente con herramientas"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente útil que puede usar herramientas para ayudar al usuario.
            Tienes acceso a herramientas generales (calculadora, tiempo, análisis de texto) y 
            herramientas especializadas para análisis de documentos IPM.
            Usa las herramientas cuando sea apropiado para la consulta del usuario."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    def _render_agent_interface(self, agent_executor):
        """Renderiza interfaz del agente"""
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
    
    def _render_example_queries(self, agent_executor):
        """Renderiza consultas de ejemplo"""
        st.subheader("Ejemplos de Consultas")
        
        examples = [
            "¿Cuál es el resultado de 25 * 34 + 12?",
            "¿Qué hora es?",
            "Analiza este texto: 'LangChain permite crear aplicaciones con LLMs'",
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