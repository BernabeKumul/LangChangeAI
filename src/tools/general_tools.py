"""
Herramientas generales para el agente de LangChain
"""

from langchain_core.tools import tool
from datetime import datetime
import json


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