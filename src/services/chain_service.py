"""
Servicio para gestión de chains y agentes de LangChain
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.models.ipm_models import IPMAuditResponse
from src.tools import (
    analyze_pest_monitoring, check_action_thresholds, evaluate_prevention_practices,
    assess_resistance_management, verify_pollinator_protection, format_audit_response
)
from typing import List, Optional
import json


class ChainService:
    """Servicio para crear y gestionar chains y agentes"""
    
    @staticmethod
    def create_ipm_audit_chain(llm: ChatOpenAI):
        """Crea chain para auditoría IPM con prompt estructurado"""
        
        system_prompt = """You are an IPM Compliance Auditor evaluating compliance with PrimusGFS Module 9 – Integrated Pest Management (IPM) Practices.

AUDIT RULES:
- Evaluate strictly based on provided documents
- Do not assume compliance where documentation is missing or unclear
- Do not offer suggestions or improvements
- Focus only on determining if documents meet compliance expectations
- Use all documents provided, even duplicates or scans with limited content

EVALUATION CRITERIA for Question 9.01.01:
The operation should have a documented IPM plan that:
• Describes pest monitoring and identification practices
• Explains use of action/economic thresholds for treatment decisions
• Includes at least two pest prevention practices
• Addresses pesticide resistance management strategies
• Includes measures to protect pollinators

RESPONSE REQUIREMENTS:
- Write in audit-style format with professional tone
- Reference exact document names (with file extensions)
- Include document sections, dates, and personnel when identified
- Break response into multiple paragraphs by topic
- Maximum 2,000 characters
- State clearly if documentation is missing or insufficient

CRITICAL: You MUST respond with a valid JSON object with exactly these keys:
- ComplianceLevel: integer (always 2)
- Comments: string (detailed multi-paragraph audit analysis)
- FilesSearch: array of objects with FileName and DocumentID (can be empty array if no files)

Example JSON format:
{
  "ComplianceLevel": 2,
  "Comments": "The operation has submitted an IPM plan titled...",
  "FilesSearch": [{"FileName": "document.pdf", "DocumentID": "DOC_001"}]
}"""

        human_prompt = """Evaluate the IPM compliance for {operation_name} with product: {product}

QUESTION: Does the operation have a documented Integrated Pest Management (IPM) plan?

DOCUMENTS PROVIDED:
{documents}

Provide your audit evaluation focusing on:
1. Pest monitoring and identification practices
2. Action/economic thresholds
3. Prevention practices (minimum 2)
4. Resistance management strategies
5. Pollinator protection measures

Respond in {language}.

IMPORTANT: Your response must be a valid JSON object only, no additional text before or after."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # Set up JSON output parser
        parser = JsonOutputParser(pydantic_object=IPMAuditResponse)
        
        # Create the chain
        chain = prompt | llm | parser
        
        return chain
    
    @staticmethod
    def create_imp_audit_agent(llm: ChatOpenAI):
        """Crea agente IPM con herramientas especializadas"""
        
        # IPM-specific tools for the agent
        imp_tools = [
            analyze_pest_monitoring,
            check_action_thresholds, 
            evaluate_prevention_practices,
            assess_resistance_management,
            verify_pollinator_protection,
            format_audit_response
        ]
        
        # Simplified system prompt for the agent
        system_prompt = """You are an IPM Compliance Auditor agent with access to specialized analysis tools.

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

        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Please audit the IPM compliance for:
Operation: {operation_name}
Product: {product}  
Language: {language}

Document to analyze:
FileName: {file_name}
DocumentID: {document_id}
Content: {document_content}

Use all your tools to analyze this document systematically, then format the final response."""),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(llm, imp_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=imp_tools, 
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
        return agent_executor
    
    @staticmethod
    def create_imp_knowledge_base() -> List[Document]:
        """Crea base de conocimientos IPM para RAG"""
        
        imp_knowledge = [
            Document(
                page_content="""PrimusGFS Module 9.01.01 - IPM Plan Requirements:
                The operation must have a documented IPM plan that includes:
                1. Pest monitoring and identification practices (visual scouting, traps, threshold levels)
                2. Action thresholds for treatment decisions (economic thresholds, damage levels)
                3. At least two pest prevention practices (crop rotation, sanitation, beneficial habitats)
                4. Pesticide resistance management strategies (rotation of chemical classes, IRAC guidelines)
                5. Pollinator protection measures (application timing, habitat preservation)
                
                Documentation must include revision dates, responsible personnel, and specific procedures.""",
                metadata={"source": "PrimusGFS_Module9", "section": "9.01.01", "type": "regulation"}
            ),
            Document(
                page_content="""IPM Monitoring Best Practices:
                Effective pest monitoring includes:
                - Weekly visual inspections during growing season
                - Strategic placement of monitoring traps (yellow sticky traps for aphids, pheromone traps for moths)
                - Recording pest counts and damage levels
                - Weather monitoring for pest development models
                - Use of degree-day calculations for pest lifecycle prediction
                
                Common monitoring tools: sticky traps, pheromone traps, beat sheets, visual inspection forms.""",
                metadata={"source": "IPM_Best_Practices", "topic": "monitoring", "type": "guidance"}
            ),
            Document(
                page_content="""Economic Thresholds for Common Pests:
                Strawberry Production:
                - Spider mites: 5-8 mites per leaflet before treatment
                - Aphids: 10-15 aphids per plant or 5% infested plants
                - Thrips: 20+ per sticky trap per week
                - Spotted wing drosophila: 1 adult per trap per week during fruit development
                
                Thresholds may vary based on crop stage, market value, and weather conditions.""",
                metadata={"source": "Economic_Thresholds", "crop": "strawberry", "type": "reference"}
            ),
            Document(
                page_content="""Pollinator Protection Measures:
                Required practices for IPM compliance:
                - Avoid applications during peak pollinator activity (10 AM - 4 PM)
                - Maintain flowering habitat strips or borders
                - Use selective pesticides when possible (avoid broad-spectrum insecticides)
                - Provide advance notice to beekeepers within 1 mile
                - Consider application timing relative to bloom periods
                - Implement buffer zones around sensitive habitats""",
                metadata={"source": "Pollinator_Protection", "type": "requirement"}
            ),
            Document(
                page_content="""Common IPM Compliance Failures:
                Frequent audit findings include:
                - Missing or outdated IPM plans (not revised annually)
                - Lack of documented monitoring procedures
                - Undefined action thresholds
                - Insufficient prevention practices (less than 2 documented)
                - No pesticide resistance management strategy
                - Missing pollinator protection measures
                - Incomplete record keeping of pest monitoring activities""",
                metadata={"source": "Audit_Findings", "type": "compliance_issues"}
            )
        ]
        
        return imp_knowledge
    
    @staticmethod
    def create_rag_chain(llm: ChatOpenAI, embeddings: OpenAIEmbeddings, documents: List[Document]):
        """Crea chain RAG para preguntas sobre documentos"""
        
        # Create vector store
        vectorstore = Chroma.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        template = """Usa el siguiente contexto para responder la pregunta. Si no sabes la respuesta, di que no lo sabes.

Contexto: {context}

Pregunta: {question}

Respuesta:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        from langchain_core.runnables import RunnablePassthrough
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever 