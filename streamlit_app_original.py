"""
LangChain Streamlit Demo Application
A web interface to demonstrate LangChain capabilities
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import tempfile
import json
import pandas as pd
from datetime import datetime
from config import Config

# Page configuration
st.set_page_config(
    page_title="LangChain Demo",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Custom tools for the agent
# IPM Audit Specialized Tools for Agents
@tool  
def analyze_pest_monitoring(document_content: str, file_name: str) -> str:
    """
    Analyze if the document contains pest monitoring and identification practices.
    Returns findings about monitoring methods, frequency, and documentation.
    """
    try:
        monitoring_keywords = [
            "monitoring", "scouting", "visual inspection", "trap", "sticky trap", 
            "pheromone trap", "weekly", "surveillance", "field inspection",
            "pest count", "monitoring log", "inspection form"
        ]
        
        content_lower = document_content.lower()
        findings = []
        
        # Check for monitoring practices
        for keyword in monitoring_keywords:
            if keyword in content_lower:
                # Find sentences containing the keyword
                sentences = document_content.split('.')
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        findings.append(f"Found: {sentence.strip()}")
                        break
        
        if findings:
            result = f"PEST MONITORING ANALYSIS for {file_name}:\n"
            result += "✅ Monitoring practices found:\n"
            for finding in findings[:3]:  # Limit to top 3 findings
                result += f"• {finding}\n"
        else:
            result = f"PEST MONITORING ANALYSIS for {file_name}:\n"
            result += "❌ No clear pest monitoring practices documented\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing pest monitoring in {file_name}: {str(e)}"

@tool
def check_action_thresholds(document_content: str, file_name: str) -> str:
    """
    Check if the document defines action or economic thresholds for pest treatment decisions.
    Returns findings about threshold levels and decision criteria.
    """
    try:
        threshold_keywords = [
            "threshold", "economic threshold", "action threshold", "treatment threshold",
            "mites per leaflet", "aphids per plant", "per trap", "treatment level",
            "intervention level", "damage level", "5 mites", "10 aphids"
        ]
        
        content_lower = document_content.lower()
        findings = []
        
        # Check for threshold definitions
        for keyword in threshold_keywords:
            if keyword in content_lower:
                sentences = document_content.split('.')
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        findings.append(f"Found: {sentence.strip()}")
                        break
        
        if findings:
            result = f"ACTION THRESHOLDS ANALYSIS for {file_name}:\n"
            result += "✅ Thresholds documented:\n"
            for finding in findings[:3]:
                result += f"• {finding}\n"
        else:
            result = f"ACTION THRESHOLDS ANALYSIS for {file_name}:\n"
            result += "❌ No clear action/economic thresholds defined\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing thresholds in {file_name}: {str(e)}"

@tool
def evaluate_prevention_practices(document_content: str, file_name: str) -> str:
    """
    Evaluate if the document includes at least two pest prevention practices.
    Returns findings about prevention methods and strategies.
    """
    try:
        prevention_keywords = [
            "crop rotation", "rotation", "sanitation", "beneficial", "border planting",
            "habitat", "cover crop", "companion planting", "soil health",
            "cultural control", "biological control", "prevention", "barrier",
            "resistant varieties", "clean cultivation"
        ]
        
        content_lower = document_content.lower()
        findings = []
        
        # Check for prevention practices
        for keyword in prevention_keywords:
            if keyword in content_lower:
                sentences = document_content.split('.')
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        findings.append(f"Found: {sentence.strip()}")
                        break
        
        prevention_count = len(set(findings))  # Remove duplicates
        
        if prevention_count >= 2:
            result = f"PREVENTION PRACTICES ANALYSIS for {file_name}:\n"
            result += f"✅ {prevention_count} prevention practices documented:\n"
            for finding in findings[:4]:
                result += f"• {finding}\n"
        elif prevention_count == 1:
            result = f"PREVENTION PRACTICES ANALYSIS for {file_name}:\n"
            result += "⚠️ Only 1 prevention practice found (minimum 2 required):\n"
            result += f"• {findings[0]}\n"
        else:
            result = f"PREVENTION PRACTICES ANALYSIS for {file_name}:\n"
            result += "❌ No clear prevention practices documented\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing prevention practices in {file_name}: {str(e)}"

@tool
def assess_resistance_management(document_content: str, file_name: str) -> str:
    """
    Assess if the document addresses pesticide resistance management strategies.
    Returns findings about resistance management approaches.
    """
    try:
        resistance_keywords = [
            "resistance management", "resistance", "IRAC", "rotation", "chemical class",
            "mode of action", "MOA", "insecticide resistance", "fungicide resistance",
            "alternate", "chemical rotation", "maximum applications"
        ]
        
        content_lower = document_content.lower()
        findings = []
        
        # Check for resistance management
        for keyword in resistance_keywords:
            if keyword in content_lower:
                sentences = document_content.split('.')
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        findings.append(f"Found: {sentence.strip()}")
                        break
        
        if findings:
            result = f"RESISTANCE MANAGEMENT ANALYSIS for {file_name}:\n"
            result += "✅ Resistance management strategies found:\n"
            for finding in findings[:3]:
                result += f"• {finding}\n"
        else:
            result = f"RESISTANCE MANAGEMENT ANALYSIS for {file_name}:\n"
            result += "❌ No pesticide resistance management strategies documented\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing resistance management in {file_name}: {str(e)}"

@tool
def verify_pollinator_protection(document_content: str, file_name: str) -> str:
    """
    Verify if the document includes measures to protect pollinators.
    Returns findings about pollinator protection practices.
    """
    try:
        pollinator_keywords = [
            "pollinator", "bee", "beekeeper", "flowering", "bloom", "habitat strip",
            "beneficial insect", "10 AM", "4 PM", "peak activity", "buffer zone",
            "application timing", "pollinator protection", "flowering border"
        ]
        
        content_lower = document_content.lower()
        findings = []
        
        # Check for pollinator protection
        for keyword in pollinator_keywords:
            if keyword in content_lower:
                sentences = document_content.split('.')
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        findings.append(f"Found: {sentence.strip()}")
                        break
        
        if findings:
            result = f"POLLINATOR PROTECTION ANALYSIS for {file_name}:\n"
            result += "✅ Pollinator protection measures found:\n"
            for finding in findings[:3]:
                result += f"• {finding}\n"
        else:
            result = f"POLLINATOR PROTECTION ANALYSIS for {file_name}:\n"
            result += "❌ No pollinator protection measures documented\n"
        
        return result
        
    except Exception as e:
        return f"Error analyzing pollinator protection in {file_name}: {str(e)}"

@tool
def format_audit_response(
    monitoring_analysis: str,
    threshold_analysis: str, 
    prevention_analysis: str,
    resistance_analysis: str,
    pollinator_analysis: str,
    operation_name: str,
    product: str,
    file_name: str,
    document_id: str,
    language: str = "Spanish"
) -> str:
    """
    Format the final IPM audit response in JSON format based on all analyses.
    Combines findings from all tools into a structured compliance summary.
    """
    try:
        # Combine all analyses
        all_analyses = [
            monitoring_analysis,
            threshold_analysis,
            prevention_analysis,
            resistance_analysis,
            pollinator_analysis
        ]
        
        # Count compliant areas (those with ✅)
        compliant_count = sum(1 for analysis in all_analyses if "✅" in analysis)
        warning_count = sum(1 for analysis in all_analyses if "⚠️" in analysis)
        non_compliant_count = sum(1 for analysis in all_analyses if "❌" in analysis)
        
        # Generate compliance summary
        if language.lower() == "english":
            if compliant_count >= 4:
                compliance_summary = f"The operation '{operation_name}' demonstrates good IPM compliance for {product} production."
            elif compliant_count >= 2:
                compliance_summary = f"The operation '{operation_name}' shows partial IPM compliance for {product} production with areas for improvement."
            else:
                compliance_summary = f"The operation '{operation_name}' has significant IPM compliance gaps for {product} production."
                
            # Create detailed comments
            comments = compliance_summary + f" Document {file_name} was analyzed for PrimusGFS Module 9.01.01 compliance.\n\n"
        else:
            if compliant_count >= 4:
                compliance_summary = f"La operación '{operation_name}' demuestra buen cumplimiento IPM para la producción de {product}."
            elif compliant_count >= 2:
                compliance_summary = f"La operación '{operation_name}' muestra cumplimiento parcial IPM para la producción de {product} con áreas de mejora."
            else:
                compliance_summary = f"La operación '{operation_name}' tiene brechas significativas de cumplimiento IPM para la producción de {product}."
                
            # Create detailed comments
            comments = compliance_summary + f" Se analizó el documento {file_name} para cumplimiento con PrimusGFS Módulo 9.01.01.\n\n"
        
        # Add detailed findings
        for analysis in all_analyses:
            if analysis.strip():
                comments += analysis + "\n\n"
        
        # Ensure comments don't exceed 2000 characters
        if len(comments) > 2000:
            comments = comments[:1997] + "..."
        
        # Create JSON response
        response = {
            "ComplianceLevel": 2,
            "Comments": comments.strip(),
            "FilesSearch": [{"FileName": file_name, "DocumentID": document_id}]
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        # Fallback response
        fallback_response = {
            "ComplianceLevel": 2,
            "Comments": f"Error al formatear respuesta de auditoría: {str(e)}",
            "FilesSearch": [{"FileName": file_name, "DocumentID": document_id}]
        }
        return json.dumps(fallback_response, indent=2, ensure_ascii=False)

# Original tools (keeping for general agent functionality)
@tool
def calculator(operation: str) -> str:
    """Perform basic mathematical operations"""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in operation):
            return "Error: Invalid characters in operation"
        result = eval(operation)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_time() -> str:
    """Get current date and time"""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def text_analyzer(text: str) -> str:
    """Analyze text statistics"""
    words = text.split()
    analysis = {
        "characters": len(text),
        "words": len(words),
        "sentences": len(text.split('.')),
        "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }
    return json.dumps(analysis, indent=2)

# Pydantic model for IPM Audit response
class IPMAuditResponse(BaseModel):
    """Response model for IPM Compliance Audit"""
    ComplianceLevel: int = Field(description="Always returns the value 2", default=2)
    Comments: str = Field(description="Multi-paragraph compliance analysis")
    FilesSearch: list = Field(description="List of files with FileName and DocumentID", default=[])

def check_api_key():
    """Check if API key is configured"""
    try:
        Config.validate_config()
        return True
    except:
        return False

def init_llm():
    """Initialize LLM with error handling"""
    try:
        return ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.DEFAULT_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def ipm_audit_chain(llm):
    """Create IPM Audit chain using LangChain structured prompts"""
    
    # Define the structured prompt template
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

def ipm_audit_fallback(llm, operation_name, product, documents, language, file_name, document_id):
    """Fallback IPM audit method when JSON parsing fails"""
    try:
        # Simple prompt for fallback
        fallback_prompt = ChatPromptTemplate.from_template("""
Eres un auditor IPM. Evalúa la operación {operation_name} para el producto {product}.

Documentos disponibles: {documents}

INSTRUCCIONES:
1. Analiza el cumplimiento del plan IPM
2. Responde ÚNICAMENTE en formato JSON válido
3. Usa exactamente estas claves: ComplianceLevel, Comments, FilesSearch

Formato requerido:
{{
  "ComplianceLevel": 2,
  "Comments": "Análisis detallado del cumplimiento IPM...",
  "FilesSearch": [{{"FileName": "{file_name}", "DocumentID": "{document_id}"}}]
}}

Idioma: {language}
Respuesta en JSON:""")
        
        chain = fallback_prompt | llm | StrOutputParser()
        raw_response = chain.invoke({
            "operation_name": operation_name,
            "product": product,
            "documents": documents,
            "language": language,
            "file_name": file_name,
            "document_id": document_id
        })
        
        # Clean response and try to parse
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        cleaned = cleaned.strip()
        
        return json.loads(cleaned)
        
    except Exception as e:
        # Ultimate fallback
        return {
            "ComplianceLevel": 2,
            "Comments": f"Error en el procesamiento JSON. Respuesta original: {raw_response[:500]}...",
            "FilesSearch": [{"FileName": file_name, "DocumentID": document_id}]
        }

def create_ipm_knowledge_base():
    """Create a vector store with IPM regulations and best practices"""
    
    # Sample IPM knowledge base documents
    ipm_knowledge = [
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
    
    return ipm_knowledge

def estimate_tokens(text):
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
    return len(text) // 4

def extract_ipm_relevant_content(text, max_tokens=1000):
    """Extract IPM-relevant sections from document text"""
    
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
        para_tokens = estimate_tokens(para)
        if total_tokens + para_tokens <= max_tokens:
            relevant_paragraphs.append(para)
            total_tokens += para_tokens
        else:
            break
    
    return '\n\n'.join(relevant_paragraphs), total_tokens

def summarize_document(text, llm, max_length=500):
    """Summarize a document focusing on IPM-relevant content"""
    
    try:
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

def process_multiple_documents(documents_list, llm, strategy="smart_extraction"):
    """
    Process multiple documents with token optimization
    
    Strategies:
    - smart_extraction: Extract only IPM-relevant content
    - summarization: Summarize each document
    - chunking: Use semantic chunking and similarity search
    """
    
    processed_docs = []
    total_tokens = 0
    max_total_tokens = 3000  # Maximum tokens for all documents combined
    
    for doc_info in documents_list:
        file_name = doc_info.get('filename', 'unknown.pdf')
        doc_id = doc_info.get('doc_id', 'DOC_UNKNOWN')
        content = doc_info.get('content', '')
        
        if not content.strip():
            continue
            
        content_tokens = estimate_tokens(content)
        
        if strategy == "smart_extraction":
            # Extract only IPM-relevant content
            if content_tokens > 1000:  # Only process if document is large
                extracted_content, used_tokens = extract_ipm_relevant_content(content, max_tokens=800)
                processed_content = extracted_content
                st.info(f"📄 {file_name}: Extraídas secciones relevantes ({used_tokens} tokens de {content_tokens})")
            else:
                processed_content = content
                used_tokens = content_tokens
                
        elif strategy == "summarization":
            # Summarize the document
            if content_tokens > 800:
                processed_content = summarize_document(content, llm, max_length=600)
                used_tokens = estimate_tokens(processed_content)
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
                chunk_tokens = estimate_tokens(chunk)
                if total_tokens + chunk_tokens <= max_total_tokens:
                    relevant_chunks.append(chunk)
                    total_tokens += chunk_tokens
                    
            processed_content = '\n\n'.join(relevant_chunks)
            used_tokens = estimate_tokens(processed_content)
            st.info(f"📄 {file_name}: Procesados {len(relevant_chunks)} chunks ({used_tokens} tokens)")
        
        # Check total token limit
        if total_tokens + used_tokens > max_total_tokens:
            st.warning(f"⚠️ Límite de tokens alcanzado. Omitiendo documento: {file_name}")
            break
            
        processed_docs.append({
            'filename': file_name,
            'doc_id': doc_id,
            'content': processed_content,
            'original_tokens': content_tokens,
            'processed_tokens': used_tokens
        })
        
        total_tokens += used_tokens
    
    return processed_docs, total_tokens

def chunk_and_search_documents(documents_list, query="IPM plan compliance monitoring", max_chunks=8):
    """Use semantic search to find most relevant document chunks"""
    
    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        
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
            chunk_tokens = estimate_tokens(doc.page_content)
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

def ipm_audit_with_rag(llm, embeddings, operation_name, product, documents, language, file_name, document_id):
    """Enhanced IPM audit using RAG for better accuracy"""
    
    try:
        # Create knowledge base
        ipm_knowledge = create_ipm_knowledge_base()
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=ipm_knowledge,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Get relevant regulations and best practices
        query = f"IPM plan requirements compliance audit {product}"
        relevant_knowledge = retriever.get_relevant_documents(query)
        
        # Format knowledge base context
        knowledge_context = "\n\n".join([
            f"REGULATION/GUIDANCE: {doc.page_content}"
            for doc in relevant_knowledge
        ])
        
        # Enhanced prompt with RAG context
        rag_prompt = ChatPromptTemplate.from_template("""
Eres un auditor IPM experto con acceso a la base de conocimientos de regulaciones y mejores prácticas.

CONTEXTO DE CONOCIMIENTOS IPM:
{knowledge_context}

OPERACIÓN A AUDITAR:
Operación: {operation_name}
Producto: {product}
Documentos Proporcionados: {documents}

PREGUNTA DE AUDITORÍA: ¿Tiene la operación un plan IPM documentado que cumpla con PrimusGFS Module 9.01.01?

INSTRUCCIONES:
1. Usa el contexto de conocimientos para evaluar el cumplimiento
2. Compara los documentos proporcionados con los requisitos regulatorios
3. Identifica brechas específicas de cumplimiento
4. Proporciona recomendaciones basadas en mejores prácticas
5. Responde en formato JSON con las claves exactas: ComplianceLevel, Comments, FilesSearch

Formato JSON requerido:
{{
  "ComplianceLevel": 2,
  "Comments": "Análisis detallado basado en regulaciones específicas...",
  "FilesSearch": [{{"FileName": "{file_name}", "DocumentID": "{document_id}"}}]
}}

Idioma: {language}
""")
        
        # Create RAG chain
        chain = rag_prompt | llm | StrOutputParser()
        
        # Run enhanced audit
        raw_response = chain.invoke({
            "knowledge_context": knowledge_context,
            "operation_name": operation_name,
            "product": product,
            "documents": documents,
            "language": language,
            "file_name": file_name,
            "document_id": document_id
        })
        
        # Clean and parse response
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        cleaned = cleaned.strip()
        
        result = json.loads(cleaned)
        
        # Add knowledge sources used
        result["KnowledgeSources"] = [doc.metadata.get("source", "Unknown") for doc in relevant_knowledge]
        
        return result
        
    except Exception as e:
        # Fallback to standard method
        return ipm_audit_fallback(llm, operation_name, product, documents, language, file_name, document_id)

def create_ipm_audit_agent(llm):
    """
    Create an IPM audit agent that uses specialized tools instead of monolithic prompts.
    This agent intelligently uses specific tools to analyze different aspects of IPM compliance.
    """
    
    # IPM-specific tools for the agent
    ipm_tools = [
        analyze_pest_monitoring,
        check_action_thresholds, 
        evaluate_prevention_practices,
        assess_resistance_management,
        verify_pollinator_protection,
        format_audit_response
    ]
    
    # Simplified system prompt for the agent - much shorter than the original!
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
    agent = create_openai_functions_agent(llm, ipm_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=ipm_tools, 
        verbose=True,
        max_iterations=10,  # Allow multiple tool calls
        early_stopping_method="generate"
    )
    
    return agent_executor

def run_ipm_agent_audit(llm, operation_name, product, document_content, file_name, document_id, language="Spanish"):
    """
    Run IPM audit using the agent-based approach instead of monolithic prompts.
    This demonstrates how agents with tools can replace complex single prompts.
    """
    try:
        # Create the agent
        agent_executor = create_ipm_audit_agent(llm)
        
        # Run the agent
        result = agent_executor.invoke({
            "operation_name": operation_name,
            "product": product,
            "language": language,
            "file_name": file_name,
            "document_id": document_id,
            "document_content": document_content
        })
        
        # Extract the output - should be JSON from format_audit_response tool
        output = result.get('output', '')
        
        # Try to parse as JSON
        try:
            json_result = json.loads(output)
            return json_result
        except json.JSONDecodeError:
            # If not JSON, create fallback response
            return {
                "ComplianceLevel": 2,
                "Comments": f"Agent analysis completed: {output}",
                "FilesSearch": [{"FileName": file_name, "DocumentID": document_id}],
                "Method": "Agent-based analysis"
            }
            
    except Exception as e:
        # Fallback error response
        return {
            "ComplianceLevel": 2,
            "Comments": f"Error in agent-based audit: {str(e)}",
            "FilesSearch": [{"FileName": file_name, "DocumentID": document_id}],
            "Method": "Agent-based analysis (error)"
        }

def ipm_audit_tab():
    """IPM Compliance Audit functionality using LangChain"""
    st.header("🔍 IPM Compliance Audit")
    st.markdown("*Evaluación de cumplimiento IPM usando LangChain estructurado*")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
    if not llm:
        return
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        operation_name = st.text_input("Nombre de la Operación", value="Operación Ejemplo", key="operation_name")
        product = st.text_input("Producto", value="Fresas", key="product")
    
    with col2:
        language = st.selectbox("Idioma", ["Spanish", "English"], key="language")
    
    # Initialize session state for IPM audit
    if "ipm_file_name" not in st.session_state:
        st.session_state.ipm_file_name = "IPM_Plan_2024.pdf"
    if "ipm_doc_id" not in st.session_state:
        st.session_state.ipm_doc_id = "DOC_001"
    if "ipm_doc_content" not in st.session_state:
        st.session_state.ipm_doc_content = "Este es un plan IPM que incluye monitoreo semanal de plagas mediante trampas pegajosas. Se establecen umbrales de acción de 5 ácaros por hoja. Incluye rotación de cultivos y plantas de borde para atraer beneficiosos."
    
    # Initialize documents list for multiple document support
    if "ipm_documents" not in st.session_state:
        st.session_state.ipm_documents = []
    
    # Document input
    st.subheader("📄 Documentos para Auditoría con Optimización de Tokens")
    
    # Token optimization strategy
    col1, col2 = st.columns(2)
    with col1:
        optimization_strategy = st.selectbox(
            "💰 Estrategia de Optimización:",
            ["smart_extraction", "summarization", "chunking"],
            format_func=lambda x: {
                "smart_extraction": "🎯 Extracción Inteligente",
                "summarization": "📝 Resumen Automático",
                "chunking": "🔍 Búsqueda Semántica"
            }[x],
            key="optimization_strategy"
        )
    
    with col2:
        max_tokens = st.number_input(
            "Límite máximo de tokens:", 
            min_value=500, 
            max_value=8000, 
            value=3000,
            step=500,
            key="max_tokens"
        )
    
    # Strategy explanations
    strategy_info = {
        "smart_extraction": "Extrae solo párrafos que contienen palabras clave IPM relevantes",
        "summarization": "Resume cada documento enfocándose en aspectos IPM",
        "chunking": "Divide documentos y usa búsqueda semántica para encontrar chunks relevantes"
    }
    st.info(f"ℹ️ **{optimization_strategy.replace('_', ' ').title()}**: {strategy_info[optimization_strategy]}")
    
    # Single document interface (legacy support)
    st.markdown("**📄 Documento Individual (Método Tradicional)**")
    
    # Option 2: Example documents (moved up to avoid conflicts)
    st.markdown("**Opción 1: Usar documento de ejemplo**")
    if st.button("📋 Cargar Ejemplo de Documento IPM", key="load_example"):
        st.session_state.ipm_file_name = "IPM_Strawberry2024.pdf"
        st.session_state.ipm_doc_id = "DOC_IPM_001"
        st.session_state.ipm_doc_content = """Integrated Pest Management Plan for Strawberry Operation
        
Revision Date: March 12, 2024
Prepared by: Juan Pérez, Farm Manager

Section 3 - Pest Monitoring:
Weekly visual scouting conducted on Mondays and Thursdays. Sticky traps placed every 50 meters to monitor flying insects. Monitoring forms document pest counts and locations.

Section 4 - Action Thresholds:
- Spider mites: Treatment when 5 or more mites per leaflet
- Aphids: Treatment when 10 or more per plant
- Thrips: Treatment when trap counts exceed 20 per week

Section 5 - Prevention Methods:
1. Crop rotation with lettuce every 2 years
2. Border plantings of alyssum to attract beneficial insects
3. Sanitation practices including removal of plant debris

Section 6 - Resistance Management:
Rotation of chemical classes following IRAC guidelines. Maximum 2 applications per season of same mode of action.

Note: Pollinator protection measures not documented in this plan."""
        st.rerun()
    
    # Option 1: Text input
    st.markdown("**Opción 2: Editar contenido del documento**")
    col1, col2 = st.columns(2)
    with col1:
        file_name = st.text_input("Nombre del archivo", value=st.session_state.ipm_file_name, key="file_name")
        if file_name != st.session_state.ipm_file_name:
            st.session_state.ipm_file_name = file_name
    with col2:
        document_id = st.text_input("ID del documento", value=st.session_state.ipm_doc_id, key="doc_id")
        if document_id != st.session_state.ipm_doc_id:
            st.session_state.ipm_doc_id = document_id
    
    document_content = st.text_area(
        "Contenido del documento:", 
        value=st.session_state.ipm_doc_content,
        height=150,
        key="doc_content"
    )
    if document_content != st.session_state.ipm_doc_content:
        st.session_state.ipm_doc_content = document_content
    
    # Multiple documents interface
    st.markdown("---")
    st.markdown("**📚 Múltiples Documentos (Método Avanzado con Optimización)**")
    
    # Document management
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("*Agregar documentos para procesamiento optimizado:*")
    with col2:
        if st.button("📋 Ejemplo Multi-Doc", key="load_multi_example"):
            example_docs = [
                {
                    'filename': "IPM_Plan_Main.pdf",
                    'doc_id': "DOC_001",
                    'content': """IPM Plan for Agricultural Operation - Main Document
                    
This comprehensive IPM plan outlines our integrated approach to pest management. Our monitoring program includes weekly field scouting for key pests including spider mites, aphids, and thrips. We maintain detailed monitoring logs and use economic thresholds to guide treatment decisions.

Prevention strategies include crop rotation, beneficial habitat maintenance, and sanitation practices. We rotate between lettuce and strawberry crops every two years and maintain flowering borders to attract natural enemies.

Pesticide resistance management follows IRAC guidelines with rotation of different modes of action. No more than two applications of the same chemical class per season."""
                },
                {
                    'filename': "Monitoring_Records.pdf", 
                    'doc_id': "DOC_002",
                    'content': """Weekly Monitoring Records - June 2024
                    
Week 1: Spider mites detected at 3 mites/leaflet in Block A. Below treatment threshold.
Week 2: Aphid populations increasing, 8 per plant average. Approaching threshold.
Week 3: Beneficial insects observed - lacewings and ladybugs present.
Week 4: Thrips trap counts: 15 per trap, below threshold of 20.

Temperature monitoring shows optimal conditions for mite development. Humidity levels appropriate for beneficial insect activity."""
                },
                {
                    'filename': "Pollinator_Protocol.pdf",
                    'doc_id': "DOC_003", 
                    'content': """Pollinator Protection Protocol
                    
All pesticide applications must avoid peak pollinator activity hours (10 AM - 4 PM). Advance notification provided to three local beekeepers within 1-mile radius.

Flowering habitat strips maintained along field borders with native plants including alyssum, fennel, and buckwheat. These areas provide alternative forage and nesting sites for beneficial insects.

Buffer zones of 50 feet maintained around sensitive pollinator habitats during any pesticide applications."""
                }
            ]
            st.session_state.ipm_documents = example_docs
            st.success("✅ Cargados 3 documentos de ejemplo")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Limpiar Todo", key="clear_docs"):
            st.session_state.ipm_documents = []
            st.rerun()
    
    # Input fields for new document
    st.markdown("**Agregar Nuevo Documento:**")
    col1, col2 = st.columns(2)
    with col1:
        new_filename = st.text_input("Nombre del archivo:", key="new_filename")
    with col2:
        new_doc_id = st.text_input("ID del documento:", key="new_doc_id")
    
    new_content = st.text_area("Contenido del documento:", height=120, key="new_content")
    
    if st.button("➕ Agregar Documento", key="add_document"):
        if new_filename and new_doc_id and new_content.strip():
            new_doc = {
                'filename': new_filename,
                'doc_id': new_doc_id,
                'content': new_content
            }
            st.session_state.ipm_documents.append(new_doc)
            st.success(f"✅ Documento agregado: {new_filename}")
            st.rerun()
        else:
            st.error("❌ Por favor completa todos los campos")
    
    # Display current documents with token analysis
    if st.session_state.ipm_documents:
        st.markdown("**📊 Análisis de Documentos:**")
        
        total_original_tokens = 0
        for i, doc in enumerate(st.session_state.ipm_documents):
            tokens = estimate_tokens(doc['content'])
            total_original_tokens += tokens
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 **{doc['filename']}** (ID: {doc['doc_id']})")
                st.write(f"   📊 {tokens:,} tokens estimados")
            with col2:
                if st.button("👁️", key=f"view_{i}", help="Ver contenido"):
                    with st.expander(f"Contenido de {doc['filename']}", expanded=True):
                        st.text_area("", value=doc['content'], height=200, disabled=True, key=f"view_content_{i}")
            with col3:
                if st.button("🗑️", key=f"delete_{i}", help="Eliminar"):
                    st.session_state.ipm_documents.pop(i)
                    st.rerun()
        
        # Token analysis
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Documentos", len(st.session_state.ipm_documents))
        with col2:
            st.metric("📊 Tokens Originales", f"{total_original_tokens:,}")
        with col3:
            reduction = max(0, total_original_tokens - max_tokens)
            st.metric("💰 Reducción Esperada", f"{reduction:,}")
        
        if total_original_tokens > max_tokens:
            st.warning(f"⚠️ Optimización necesaria: {total_original_tokens:,} → ~{max_tokens:,} tokens")
            
            # Show what will happen
            if optimization_strategy == "smart_extraction":
                st.info("🎯 Se extraerán solo las secciones con contenido IPM relevante")
            elif optimization_strategy == "summarization":
                st.info("📝 Cada documento será resumido manteniendo información IPM clave")
            else:  # chunking
                st.info("🔍 Se usará búsqueda semántica para encontrar los chunks más relevantes")
        else:
            st.success(f"✅ Documentos dentro del límite de tokens ({total_original_tokens:,}/{max_tokens:,})")

    # RAG Enhancement Option
    st.subheader("🚀 Método de Auditoría")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_rag = st.checkbox("🧠 Usar RAG (Conocimiento Mejorado)", value=True, key="use_rag",
                              help="Activa RAG para usar base de conocimientos de regulaciones IPM")
    
    with col2:
        use_agent = st.checkbox("🤖 Usar Agente con Herramientas", value=False, key="use_agent",
                                help="Usa agente que analiza sistemáticamente con herramientas especializadas")
    
    # Information about selected methods
    if use_agent:
        st.success("🤖 **Agente Activado**: Análisis sistemático con herramientas especializadas IPM")
        st.info("🔧 **Ventajas del Agente**: Análisis modular, prompts simplificados, mejor mantenibilidad")
    elif use_rag:
        st.info("✅ **RAG Activado**: La auditoría usará base de conocimientos de PrimusGFS y mejores prácticas")
    else:
        st.info("ℹ️ **Método Estándar**: Auditoría con prompt tradicional sin enhancements")
    
    # Process audit
    if st.button("🔍 Realizar Auditoría IPM", key="process_audit", type="primary"):
        # Determine which documents to use
        documents_to_process = []
        
        if st.session_state.ipm_documents:
            # Use multiple documents
            documents_to_process = st.session_state.ipm_documents
            st.info(f"📚 Procesando {len(documents_to_process)} documentos con optimización")
        elif document_content.strip():
            # Use single document (legacy mode)
            documents_to_process = [{
                'filename': file_name,
                'doc_id': document_id,
                'content': document_content
            }]
            st.info("📄 Procesando documento individual")
        else:
            st.error("❌ Por favor proporciona al menos un documento para la auditoría.")
            return
        
        with st.spinner("Procesando auditoría IPM con optimización de tokens..."):
            try:
                # Process documents with optimization
                if len(documents_to_process) > 1 or estimate_tokens(documents_to_process[0]['content']) > max_tokens:
                    st.info(f"🔧 Aplicando optimización: {optimization_strategy}")
                    
                    # Apply document optimization
                    if optimization_strategy == "chunking":
                        # Use semantic search for chunks
                        processed_content, processed_tokens = chunk_and_search_documents(
                            documents_to_process, 
                            query=f"IPM plan compliance audit {product}",
                            max_chunks=8
                        )
                        
                        if processed_content:
                            documents_formatted = "\n\n".join([
                                f"FileName: {chunk['source']}\nDocumentID: {chunk['doc_id']}\nChunk: {chunk['chunk_id']}\nContent: {chunk['content']}"
                                for chunk in processed_content
                            ])
                            st.success(f"✅ Procesados {len(processed_content)} chunks relevantes ({processed_tokens} tokens)")
                        else:
                            st.error("❌ No se encontraron chunks relevantes")
                            return
                    else:
                        # Use traditional processing methods
                        processed_docs, processed_tokens = process_multiple_documents(
                            documents_to_process, 
                            llm, 
                            strategy=optimization_strategy
                        )
                        
                        if processed_docs:
                            documents_formatted = "\n\n".join([
                                f"FileName: {doc['filename']}\nDocumentID: {doc['doc_id']}\nContent: {doc['content']}"
                                for doc in processed_docs
                            ])
                            st.success(f"✅ Optimización completada: {processed_tokens} tokens finales")
                        else:
                            st.error("❌ No se pudieron procesar los documentos")
                            return
                else:
                    # No optimization needed
                    documents_formatted = "\n\n".join([
                        f"FileName: {doc['filename']}\nDocumentID: {doc['doc_id']}\nContent: {doc['content']}"
                        for doc in documents_to_process
                    ])
                    total_tokens = sum(estimate_tokens(doc['content']) for doc in documents_to_process)
                    st.info(f"ℹ️ Sin optimización necesaria ({total_tokens} tokens)")
                
                # Choose audit method based on selected approach
                if use_agent:
                    st.info("🤖 Ejecutando auditoría con Agente especializado...")
                    try:
                        # For agent, we process each document separately (agents work better with focused content)
                        if len(documents_to_process) > 1:
                            st.warning("⚠️ Agente procesa solo el primer documento. Para múltiples documentos considera RAG o método estándar.")
                        
                        # Use the first document for agent analysis
                        doc = documents_to_process[0]
                        result = run_ipm_agent_audit(
                            llm, 
                            operation_name, 
                            product, 
                            doc['content'], 
                            doc['filename'], 
                            doc['doc_id'], 
                            language
                        )
                        
                        # Add agent indicator
                        result["Method"] = "Agent-based analysis"
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error en Agente: {e}. Usando método estándar...")
                        audit_chain = ipm_audit_chain(llm)
                        result = audit_chain.invoke({
                            "operation_name": operation_name,
                            "product": product,
                            "documents": documents_formatted,
                            "language": language
                        })
                        
                elif use_rag:
                    st.info("🧠 Ejecutando auditoría con RAG...")
                    try:
                        # Initialize embeddings for RAG
                        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
                        
                        # Run RAG-enhanced audit
                        result = ipm_audit_with_rag(
                            llm, embeddings, operation_name, product, documents_formatted,
                            language, documents_to_process[0]['filename'], documents_to_process[0]['doc_id']
                        )
                    except Exception as e:
                        st.warning("⚠️ Error en RAG. Usando método estándar...")
                        audit_chain = ipm_audit_chain(llm)
                        result = audit_chain.invoke({
                            "operation_name": operation_name,
                            "product": product,
                            "documents": documents_formatted,
                            "language": language
                        })
                else:
                    st.info("⚙️ Ejecutando auditoría estándar...")
                    # Run standard audit
                    try:
                        audit_chain = ipm_audit_chain(llm)
                        result = audit_chain.invoke({
                            "operation_name": operation_name,
                            "product": product,
                            "documents": documents_formatted,
                            "language": language
                        })
                    except Exception as e:
                        st.warning("⚠️ Error en método principal. Intentando método alternativo...")
                        result = ipm_audit_fallback(
                            llm, operation_name, product, documents_formatted, 
                            language, documents_to_process[0]['filename'], documents_to_process[0]['doc_id']
                        )
                
                # Display results
                st.success("✅ Auditoría IPM Completada")
                
                # Show structured results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📋 Resultado de la Auditoría")
                    st.markdown(f"**Nivel de Cumplimiento:** {result['ComplianceLevel']}")
                    
                    # Show method indicator if enhanced methods were used
                    if use_agent and result.get("Method") == "Agent-based analysis":
                        st.success("🤖 **Auditoría realizada con Agente especializado**")
                    elif use_rag and "KnowledgeSources" in result:
                        st.success("🧠 **Auditoría mejorada con RAG**")
                    
                    st.markdown("**Comentarios:**")
                    st.write(result['Comments'])
                
                with col2:
                    st.subheader("📄 Archivos Analizados")
                    if result.get('FilesSearch'):
                        for file_info in result['FilesSearch']:
                            st.write(f"• **{file_info.get('FileName', 'N/A')}**")
                            st.write(f"  ID: {file_info.get('DocumentID', 'N/A')}")
                    else:
                        st.write("• " + file_name)
                        st.write(f"  ID: {document_id}")
                    
                    # Show knowledge sources if RAG was used
                    if use_rag and "KnowledgeSources" in result:
                        st.subheader("🧠 Fuentes de Conocimiento")
                        st.markdown("*Regulaciones y guías consultadas:*")
                        for source in result['KnowledgeSources']:
                            st.write(f"📚 {source}")
                    
                    # Show optimization results
                    if len(documents_to_process) > 1:
                        st.subheader("💰 Optimización de Tokens")
                        original_total = sum(estimate_tokens(doc['content']) for doc in documents_to_process)
                        st.write(f"📊 **Documentos procesados**: {len(documents_to_process)}")
                        st.write(f"📈 **Tokens originales**: {original_total:,}")
                        
                        if 'processed_tokens' in locals():
                            reduction_pct = ((original_total - processed_tokens) / original_total) * 100 if original_total > 0 else 0
                            st.write(f"📉 **Tokens finales**: {processed_tokens:,}")
                            st.write(f"💡 **Reducción**: {reduction_pct:.1f}% ({original_total - processed_tokens:,} tokens ahorrados)")
                        
                        st.write(f"⚙️ **Estrategia usada**: {optimization_strategy.replace('_', ' ').title()}")
                
                # Show raw JSON response
                with st.expander("🔧 Respuesta JSON Completa"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"Error en la auditoría: {e}")
                st.error("Verifica que el modelo pueda generar respuestas JSON válidas.")
    
    # Information about the improvement
    st.markdown("---")
    st.subheader("💡 Mejoras con LangChain")
    
    improvements = [
        "**Prompt Estructurado**: Separación clara entre instrucciones del sistema y entrada del usuario",
        "**Output Parser**: Validación automática de la respuesta JSON con Pydantic",
        "**Reutilizable**: Función que puede ser llamada desde otras partes del código",
        "**Mantenible**: Fácil modificación de reglas y criterios de auditoría",
        "**Type Safety**: Validación de tipos con Pydantic models",
        "**Agent Architecture**: Análisis modular con herramientas especializadas para cada aspecto IPM",
        "**RAG Integration**: Base de conocimientos con regulaciones PrimusGFS y mejores prácticas",
        "**Contexto Enriquecido**: Auditorías más precisas basadas en normativas oficiales",
        "**Transparencia**: Muestra las fuentes de conocimiento utilizadas en cada auditoría"
    ]
    
    for improvement in improvements:
        st.markdown(f"• {improvement}")
    
    # Agent Benefits  
    st.subheader("🤖 Beneficios de los Agentes para Auditorías IPM")
    
    agent_benefits = [
        "**🔧 Modularidad**: Cada herramienta analiza un aspecto específico (monitoreo, umbrales, prevención, etc.)",
        "**📝 Prompts Simplificados**: Elimina prompts monolíticos de +2000 caracteres por herramientas específicas",
        "**🧩 Mantenibilidad**: Fácil agregar/modificar criterios sin reescribir todo el prompt",
        "**🎯 Análisis Sistemático**: El agente usa todas las herramientas automáticamente",
        "**🔍 Trazabilidad**: Cada herramienta reporta hallazgos específicos",
        "**⚡ Reutilización**: Las herramientas pueden usarse independientemente",
        "**🛠️ Extensibilidad**: Agregar nuevos criterios es solo crear una nueva herramienta"
    ]
    
    for benefit in agent_benefits:
        st.markdown(f"• {benefit}")
    
    # RAG Benefits
    st.subheader("🧠 Beneficios del RAG para Auditorías IPM")
    
    rag_benefits = [
        "**📚 Base de Conocimiento**: Acceso a regulaciones PrimusGFS actualizadas",
        "**🎯 Precisión Mejorada**: Comparación directa con requisitos oficiales",
        "**🔍 Detección de Brechas**: Identifica faltantes específicos de cumplimiento",
        "**📊 Mejores Prácticas**: Incorpora experiencia de auditorías previas",
        "**🌱 Específico por Cultivo**: Umbrales y prácticas adaptadas al producto",
        "**🔗 Trazabilidad**: Referencias claras a fuentes regulatorias"
    ]
    
    for benefit in rag_benefits:
        st.markdown(f"• {benefit}")
    
    # Token Optimization Benefits
    st.subheader("💰 Optimización de Tokens - Estrategias Disponibles")
    
    optimization_benefits = [
        "**🎯 Extracción Inteligente**: Solo envía párrafos con contenido IPM relevante (ahorro: 60-80%)",
        "**📝 Resumen Automático**: Resume documentos largos manteniendo información clave (ahorro: 70-85%)", 
        "**🔍 Búsqueda Semántica**: Encuentra y envía solo los chunks más relevantes (ahorro: 50-70%)",
        "**📊 Análisis en Tiempo Real**: Muestra tokens antes y después de optimización",
        "**💡 Límites Configurables**: Control total sobre el uso máximo de tokens",
        "**🔄 Procesamiento Múltiple**: Maneja varios documentos simultáneamente"
    ]
    
    for benefit in optimization_benefits:
        st.markdown(f"• {benefit}")
    
    # Comparison of approaches
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

def basic_chat_tab():
    """Basic chat functionality"""
    st.header("💬 Chat Básico")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
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

def memory_chat_tab():
    """Chat with memory functionality"""
    st.header("🧠 Chat con Memoria")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
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

def agents_tab():
    """Agent with tools functionality"""
    st.header("🤖 Agentes con Herramientas")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize LLM
    llm = init_llm()
    if not llm:
        return
    
    # Tool selection
    st.subheader("Herramientas Disponibles")
    
    # General tools and IPM tools
    general_tools = [calculator, get_current_time, text_analyzer]
    ipm_tools = [analyze_pest_monitoring, check_action_thresholds, evaluate_prevention_practices, 
                 assess_resistance_management, verify_pollinator_protection, format_audit_response]
    
    st.markdown("**📊 Herramientas Generales:**")
    for tool in general_tools:
        st.write(f"• **{tool.name}**: {tool.description}")
    
    st.markdown("**🔍 Herramientas IPM Especializadas:**")
    for tool in ipm_tools:
        st.write(f"• **{tool.name}**: {tool.description}")
    
    # Create agent with both tool sets
    all_tools = general_tools + ipm_tools
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente útil que puede usar herramientas para ayudar al usuario.
        Tienes acceso a herramientas generales (calculadora, tiempo, análisis de texto) y 
        herramientas especializadas para análisis de documentos IPM.
        Usa las herramientas cuando sea apropiado para la consulta del usuario."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
    
    # Agent interface
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
    
    # Example queries
    st.subheader("Ejemplos de Consultas")
    
    st.markdown("**📊 Ejemplos Generales:**")
    general_examples = [
        "¿Cuál es el resultado de 25 * 34 + 12?",
        "¿Qué hora es?",
        "Analiza este texto: 'LangChain permite crear aplicaciones con LLMs'",
        "Calcula 2^10 y luego dime qué hora es"
    ]
    
    st.markdown("**🔍 Ejemplos IPM (necesitas proporcionar contenido de documento):**")
    ipm_examples = [
        "Analiza el monitoreo de plagas en: 'Monitoreo semanal con trampas pegajosas para detectar ácaros...'",
        "Verifica umbrales en: 'Tratamiento cuando hay 5 ácaros por hoja...'",
        "Evalúa prevención en: 'Rotación de cultivos cada 2 años y plantas refugio...'",
        "Revisa protección de polinizadores en: 'Aplicaciones fuera de horario 10 AM - 4 PM...'"
    ]
    
    examples = general_examples + ipm_examples
    
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

def rag_tab():
    """RAG (Retrieval Augmented Generation) functionality"""
    st.header("📚 RAG - Preguntas sobre Documentos")
    
    if not check_api_key():
        st.error("⚠️ Por favor configura tu API key de OpenAI en el archivo .env")
        return
    
    # Initialize components
    llm = init_llm()
    if not llm:
        return
    
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    
    # Sample documents
    sample_docs = [
        Document(
            page_content="LangChain es un framework para desarrollar aplicaciones con LLMs. Fue creado por Harrison Chase.",
            metadata={"source": "intro.txt"}
        ),
        Document(
            page_content="Los componentes principales de LangChain incluyen LLMs, prompts, chains, agents, memory y retrievers.",
            metadata={"source": "components.txt"}
        ),
        Document(
            page_content="RAG combina recuperación de información con generación de texto para responder preguntas basadas en documentos.",
            metadata={"source": "rag.txt"}
        )
    ]
    
    # Create or load vector store
    if st.session_state.vectorstore is None:
        with st.spinner("Creando base de datos vectorial..."):
            try:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=sample_docs,
                    embedding=embeddings
                )
                st.success("✅ Base de datos vectorial creada")
            except Exception as e:
                st.error(f"Error creando vector store: {e}")
                return
    
    # Document upload
    st.subheader("📄 Documentos en la Base de Datos")
    with st.expander("Ver documentos disponibles"):
        for i, doc in enumerate(sample_docs, 1):
            st.write(f"**Documento {i}**: {doc.metadata['source']}")
            st.write(doc.page_content[:200] + "...")
    
    # RAG interface
    st.subheader("❓ Hacer Pregunta")
    
    if question := st.text_input("Escribe tu pregunta sobre los documentos:", key="rag_question_input"):
        if st.button("🔍 Buscar Respuesta", key="search_answer_button"):
            with st.spinner("Buscando respuesta..."):
                try:
                    # Create retriever
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                    
                    # Get relevant documents
                    docs = retriever.get_relevant_documents(question)
                    
                    # Create RAG chain
                    template = """
Eres un asistente que SOLO puede responder basándose en el contexto de documentos proporcionado. 

REGLAS ESTRICTAS:
- ÚNICAMENTE usa la información que aparece en el contexto de documentos
- Si la pregunta no se puede responder con el contexto proporcionado, di claramente: "No puedo responder esa pregunta basándome en los documentos proporcionados"
- NO uses tu conocimiento general o información externa
- Sintetiza y explica la información del contexto de manera natural y conversacional
- No copies fragmentos literales, pero mantente fiel al contenido

Contexto de documentos disponible:
{context}

Pregunta: {question}

Respuesta (solo basada en el contexto):"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Generate answer
                    answer = rag_chain.invoke(question)
                    
                    # Display results
                    st.success("✅ Respuesta encontrada:")
                    st.write(answer)
                    
                    # Show relevant sentences
                    st.subheader("🎯 Oraciones Más Relevantes")
                    for i, doc in enumerate(docs, 1):
                        # Split document into sentences
                        sentences = [s.strip() for s in doc.page_content.split('.') if s.strip()]
                        
                        st.markdown(f"**Del documento {doc.metadata['source']}:**")
                        for j, sentence in enumerate(sentences, 1):
                            if sentence:  # Only show non-empty sentences
                                st.markdown(f"• {sentence}.")
                        st.markdown("---")
                    
                    st.subheader("📖 Documentos Relevantes Completos")
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Documento {i}: {doc.metadata['source']}"):
                            st.write(doc.page_content)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Example questions
    st.subheader("💡 Preguntas de Ejemplo")
    example_questions = [
        "¿Qué es LangChain?",
        "¿Quién creó LangChain?",
        "¿Cuáles son los componentes principales?",
        "¿Cómo funciona RAG?"
    ]
    
    for question in example_questions:
        if st.button(f"❓ {question}", key=f"rag_example_{hash(question)}"):
            with st.spinner("Buscando respuesta..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
                    docs = retriever.get_relevant_documents(question)
                    
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
                    
                    answer = rag_chain.invoke(question)
                    
                    st.success("✅ Respuesta:")
                    st.write(answer)
                    
                    # Show relevant sentences for example questions
                    st.subheader("🎯 Oraciones Más Relevantes")
                    for i, doc in enumerate(docs, 1):
                        # Split document into sentences
                        sentences = [s.strip() for s in doc.page_content.split('.') if s.strip()]
                        
                        st.markdown(f"**Del documento {doc.metadata['source']}:**")
                        for j, sentence in enumerate(sentences, 1):
                            if sentence:  # Only show non-empty sentences
                                st.markdown(f"• {sentence}.")
                        st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

def demo_prompt_simplification_tab():
    """Demonstration of how agent approach simplifies complex prompts"""
    st.header("📝 Comparación: Prompt Complejo vs. Agente")
    st.markdown("*Demostración práctica de cómo los agentes simplifican prompts complejos*")
    
    # Show the comparison
    col1, col2 = st.columns(2)
    
    with col1:
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
        st.markdown("• Prompt extremadamente largo (2000+ caracteres)")
        st.markdown("• Difícil de mantener y modificar")
        st.markdown("• Lógica monolítica")
        st.markdown("• Difícil debugging si algo falla")
        st.markdown("• Costoso en tokens")
    
    with col2:
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
        st.markdown("• Prompt principal muy corto (500 caracteres)")
        st.markdown("• Lógica distribuida en herramientas especializadas")
        st.markdown("• Fácil mantener y extender")
        st.markdown("• Debug granular por herramienta")
        st.markdown("• Herramientas reutilizables")
    
    # Interactive demonstration
    st.markdown("---")
    st.subheader("🎮 Demostración Interactiva")
    
    demo_doc = st.text_area(
        "Ingresa contenido de documento IPM para analizar:",
        value="Plan IPM para Fresas 2024. Monitoreo semanal con trampas pegajosas. Umbrales: 5 ácaros por hoja. Prevención: rotación de cultivos y plantas refugio. Resistencia: rotación IRAC.",
        height=100
    )
    
    if st.button("🔍 Analizar con Agente", type="primary"):
        if not check_api_key():
            st.error("⚠️ Configura tu API key para usar la demostración")
            return
            
        llm = init_llm()
        if not llm:
            return
            
        with st.spinner("Analizando documento con agente..."):
            try:
                # Show step-by-step analysis
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
    
    # Code comparison
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

def main():
    """Main application"""
    st.title("🦜 LangChain Demo Application")
    st.markdown("Explora las capacidades de LangChain a través de esta interfaz interactiva")
    
    # Sidebar
    st.sidebar.title("📋 Configuración")
    
    # API Key status
    if check_api_key():
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
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat Básico",
        "🧠 Chat con Memoria", 
        "🤖 Agentes",
        "📚 RAG",
        "🔍 IPM Audit",
        "📝 Demo Prompts"
    ])
    
    with tab1:
        basic_chat_tab()
    
    with tab2:
        memory_chat_tab()
    
    with tab3:
        agents_tab()
    
    with tab4:
        rag_tab()
    
    with tab5:
        ipm_audit_tab()
    
    with tab6:
        demo_prompt_simplification_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Asegúrate de tener configurada tu API key de OpenAI para usar todas las funcionalidades.")

if __name__ == "__main__":
    main() 