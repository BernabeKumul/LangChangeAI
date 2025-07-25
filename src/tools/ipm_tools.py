"""
Herramientas especializadas para auditorías IPM (Integrated Pest Management)
"""

from langchain_core.tools import tool
import json
from typing import Optional


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