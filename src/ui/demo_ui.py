"""
Componente de UI para demostración de simplificación de prompts
"""

import streamlit as st
import json
import pandas as pd
from src.services import LLMService
from src.tools import (
    analyze_pest_monitoring, check_action_thresholds, evaluate_prevention_practices,
    assess_resistance_management, verify_pollinator_protection, format_audit_response
)


class DemoUI:
    """Componente de UI para demostración de simplificación de prompts"""
    
    def render(self):
        """Renderiza la demostración"""
        st.header("📝 Comparación: Prompt Complejo vs. Agente")
        st.markdown("*Demostración práctica de cómo los agentes simplifican prompts complejos*")
        
        self._render_comparison()
        self._render_interactive_demo()
        self._render_code_comparison()
        self._render_benefits_comparison()
    
    def _render_comparison(self):
        """Renderiza comparación de métodos"""
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_traditional_method()
        
        with col2:
            self._render_agent_method()
    
    def _render_traditional_method(self):
        """Renderiza método tradicional"""
        st.subheader("❌ Método Tradicional (Prompt Complejo)")
        st.markdown("**Tamaño:** 2000+ caracteres")
        
        traditional_prompt = """Act as an IPM Compliance Auditor. Evaluate compliance with PrimusGFS Module 9 – Integrated Pest Management (IPM) Practices strictly based on the uploaded documents.
- Do not assume compliance where documentation is missing or unclear
- Do not offer suggestions or improvements
- Focus only on determining if the documents meet compliance expectations
- Use all documents provided, even duplicates or scans with limited content

Write detailed, structured compliance summaries in a professional, audit-style format. Your summaries should reference:
- Document names (including file extensions)
- Document sections or content descriptions
- Relevant dates and timeframes
- Personnel or titles when identified

If documentation is missing or insufficient, clearly state this and explain why the operation is considered non-compliant.

For the purpose of this exercise, consider [product] as the product of the audit and [operationName] as the audited operation.

General Response Format for Each Question
Each response must include:
1. Summary of key compliance findings, citing document names and details
2. Clear statement of any missing or insufficient elements
3. Explanation if submitted documents were not used
4. Multiple paragraphs, each focused on a specific theme
5. Response must not exceed 2,000 characters
6. Always use the exact document file name
7. Write in the requested [Language]

Question 9.01.01 – Does the operation have a documented Integrated Pest Management (IPM) plan?
The operation should provide a written IPM plan that outlines how it identifies and manages pests while minimizing environmental risk. The plan should:
• Describe pest monitoring and identification practices
• Explain the use of action or economic thresholds to guide treatment decisions
• Include at least two pest prevention practices
• Address pesticide resistance management strategies
• Include measures to protect pollinators

Response Format:
• Reference document title(s), sections, and last revision dates
• Identify if pest monitoring is described
• Explain whether thresholds are defined and used
• Note if prevention and resistance strategies are described
• State if pollinator protection is included or missing
• Break into multiple paragraphs by topic

Provides a json response with the following keys:
1. ComplianceLevel: Always returns the value 2.
2. Comments: Break the response into multiple paragraphs.
3. FilesSearch: return a JSON with the FileName and DocumentID"""
        
        st.code(traditional_prompt, language="text")
        
        st.markdown("**❌ Problemas:**")
        problems = [
            "Prompt extremadamente largo (2000+ caracteres)",
            "Difícil de mantener y modificar",
            "Lógica monolítica",
            "Difícil debugging si algo falla",
            "Costoso en tokens"
        ]
        for problem in problems:
            st.markdown(f"• {problem}")
    
    def _render_agent_method(self):
        """Renderiza método con agentes"""
        st.subheader("✅ Método con Agentes (Simplificado)")
        st.markdown("**Tamaño:** 500 caracteres + herramientas modulares")
        
        agent_prompt = """You are an IPM Compliance Auditor agent with access to specialized analysis tools.

YOUR MISSION: Evaluate IPM compliance for PrimusGFS Module 9.01.01 using your available tools.

PROCESS:
1. Use each specialized tool to analyze different aspects of the documents
2. Analyze: pest monitoring, action thresholds, prevention practices, resistance management, pollinator protection
3. Format the final response using the format_audit_response tool

TOOLS AVAILABLE:
- analyze_pest_monitoring: Check for monitoring practices
- check_action_thresholds: Look for action/economic thresholds  
- evaluate_prevention_practices: Find prevention methods (need minimum 2)
- assess_resistance_management: Check resistance strategies
- verify_pollinator_protection: Look for pollinator measures
- format_audit_response: Create final JSON response

RULES:
- Use ALL analysis tools for each document
- Base conclusions ONLY on provided documents
- Be precise and reference specific document content
- Use format_audit_response to create the final JSON output

Remember: You have tools to do the heavy lifting - use them systematically!"""
        
        st.code(agent_prompt, language="text")
        
        st.markdown("**✅ Ventajas:**")
        advantages = [
            "Prompt principal muy corto (500 caracteres)",
            "Lógica distribuida en herramientas especializadas",
            "Fácil mantener y extender",
            "Debug granular por herramienta",
            "Herramientas reutilizables"
        ]
        for advantage in advantages:
            st.markdown(f"• {advantage}")
    
    def _render_interactive_demo(self):
        """Renderiza demostración interactiva"""
        st.markdown("---")
        st.subheader("🎮 Demostración Interactiva")
        
        demo_doc = st.text_area(
            "Ingresa contenido de documento IPM para analizar:",
            value="Plan IPM para Fresas 2024. Monitoreo semanal con trampas pegajosas. Umbrales: 5 ácaros por hoja. Prevención: rotación de cultivos y plantas refugio. Resistencia: rotación IRAC.",
            height=100
        )
        
        if st.button("🔍 Analizar con Agente", type="primary"):
            self._run_demo_analysis(demo_doc)
    
    def _run_demo_analysis(self, demo_doc):
        """Ejecuta análisis de demostración"""
        if not LLMService.check_api_key():
            st.error("⚠️ Configura tu API key para usar la demostración")
            return
        
        with st.spinner("Analizando documento con agente..."):
            try:
                st.subheader("🔧 Análisis Paso a Paso")
                
                # Tool 1: Monitoring
                with st.expander("1️⃣ analyze_pest_monitoring", expanded=True):
                    monitoring_result = analyze_pest_monitoring.invoke({
                        "document_content": demo_doc,
                        "file_name": "demo.pdf"
                    })
                    st.write(monitoring_result)
                
                # Tool 2: Thresholds  
                with st.expander("2️⃣ check_action_thresholds"):
                    threshold_result = check_action_thresholds.invoke({
                        "document_content": demo_doc,
                        "file_name": "demo.pdf"
                    })
                    st.write(threshold_result)
                
                # Tool 3: Prevention
                with st.expander("3️⃣ evaluate_prevention_practices"):
                    prevention_result = evaluate_prevention_practices.invoke({
                        "document_content": demo_doc,
                        "file_name": "demo.pdf"
                    })
                    st.write(prevention_result)
                
                # Tool 4: Resistance
                with st.expander("4️⃣ assess_resistance_management"):
                    resistance_result = assess_resistance_management.invoke({
                        "document_content": demo_doc,
                        "file_name": "demo.pdf"
                    })
                    st.write(resistance_result)
                
                # Tool 5: Pollinator
                with st.expander("5️⃣ verify_pollinator_protection"):
                    pollinator_result = verify_pollinator_protection.invoke({
                        "document_content": demo_doc,
                        "file_name": "demo.pdf"
                    })
                    st.write(pollinator_result)
                
                # Tool 6: Format response
                with st.expander("6️⃣ format_audit_response", expanded=True):
                    final_result = format_audit_response.invoke({
                        "monitoring_analysis": monitoring_result,
                        "threshold_analysis": threshold_result,
                        "prevention_analysis": prevention_result,
                        "resistance_analysis": resistance_result,
                        "pollinator_analysis": pollinator_result,
                        "operation_name": "Demo Operation",
                        "product": "Fresas",
                        "file_name": "demo.pdf",
                        "document_id": "DEMO_001",
                        "language": "Spanish"
                    })
                    st.json(json.loads(final_result))
                
                st.success("✅ Análisis completado usando 6 herramientas especializadas")
                
            except Exception as e:
                st.error(f"Error en demostración: {e}")
    
    def _render_code_comparison(self):
        """Renderiza comparación de código"""
        st.markdown("---")
        st.subheader("💻 Comparación de Implementación")
        
        tab1, tab2 = st.tabs(["Método Tradicional", "Método con Agente"])
        
        with tab1:
            st.markdown("**Implementación tradicional:**")
            traditional_code = """
# Prompt monolítico de 2000+ caracteres
def traditional_audit(llm, documents, operation, product):
    huge_prompt = '''[2000+ caracteres de instrucciones complejas]'''
    
    chain = ChatPromptTemplate.from_template(huge_prompt) | llm | JsonOutputParser()
    result = chain.invoke({
        "documents": documents,
        "operation": operation, 
        "product": product
    })
    return result

# Problemas:
# - Difícil de debuggear si algo falla
# - Cambiar un criterio requiere modificar todo el prompt
# - No hay reutilización de lógica
# - Prompt costoso en tokens
            """
            st.code(traditional_code, language="python")
        
        with tab2:
            st.markdown("**Implementación con agente:**")
            agent_code = """
# Herramientas especializadas y modulares
@tool
def analyze_pest_monitoring(document_content: str, file_name: str) -> str:
    # Lógica específica para monitoreo (50 líneas)
    pass

@tool 
def check_action_thresholds(document_content: str, file_name: str) -> str:
    # Lógica específica para umbrales (50 líneas)
    pass

# ... más herramientas ...

# Agente simple que usa las herramientas
def agent_audit(llm, document, operation, product):
    tools = [analyze_pest_monitoring, check_action_thresholds, ...]
    agent = create_openai_functions_agent(llm, tools, simple_prompt)
    
    result = agent.invoke({
        "document_content": document,
        "operation_name": operation,
        "product": product
    })
    return result

# Ventajas:
# - Cada herramienta se puede debuggear independientemente
# - Agregar criterios = agregar herramienta
# - Herramientas reutilizables en otros contextos
# - Prompt principal muy simple
            """
            st.code(agent_code, language="python")
    
    def _render_benefits_comparison(self):
        """Renderiza tabla de comparación de beneficios"""
        st.subheader("⚖️ Comparación de Enfoques")
        
        comparison_data = {
            "Característica": [
                "Prompt Size",
                "Mantenibilidad", 
                "Modularidad",
                "Tokens Utilizados",
                "Precisión",
                "Extensibilidad",
                "Debugging"
            ],
            "Método Tradicional": [
                "2000+ caracteres",
                "Difícil",
                "Monolítico", 
                "Alto",
                "Buena",
                "Difícil",
                "Complejo"
            ],
            "RAG Enhancement": [
                "1500+ caracteres",
                "Moderada",
                "Semi-modular",
                "Alto",
                "Muy buena",
                "Moderada", 
                "Moderado"
            ],
            "Agente con Herramientas": [
                "500 caracteres",
                "Muy fácil",
                "Totalmente modular",
                "Optimizado",
                "Muy buena",
                "Muy fácil",
                "Muy fácil"
            ]
        }
        
        df = pd.DataFrame(comparison_data)
        st.table(df)
        
        st.success("💡 **Resultado**: Los agentes con herramientas ofrecen la mejor combinación de simplicidad, mantenibilidad y eficiencia para auditorías IPM complejas.") 