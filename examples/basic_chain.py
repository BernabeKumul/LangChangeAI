"""
Basic LangChain Example
This example demonstrates the core LangChain functionality including:
- LLM integration
- Prompt templates
- Chains
- Output parsing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from config import Config

def basic_llm_example():
    """Basic LLM usage example"""
    print("=== Basic LLM Example ===")
    
    # Validate configuration
    Config.validate_config()
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Simple prompt
    response = llm.invoke("¿Qué es LangChain y para qué sirve?")
    print(f"Response: {response.content}")
    print("-" * 50)

def prompt_template_example():
    """Prompt template example"""
    print("=== Prompt Template Example ===")
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto en {topic}. Responde de manera clara y concisa."),
        ("human", "{question}")
    ])
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create chain
    chain = prompt | llm | StrOutputParser()
    
    # Execute chain
    response = chain.invoke({
        "topic": "inteligencia artificial",
        "question": "¿Cuáles son las ventajas de usar LangChain?"
    })
    
    print(f"Response: {response}")
    print("-" * 50)

def chain_example():
    """Chain example with multiple steps"""
    print("=== Chain Example ===")
    
    # Create multiple prompt templates
    summary_prompt = ChatPromptTemplate.from_template(
        "Resume el siguiente texto en máximo 50 palabras: {text}"
    )
    
    translation_prompt = ChatPromptTemplate.from_template(
        "Traduce el siguiente texto del español al inglés: {text}"
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=0.3,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create chains
    summary_chain = summary_prompt | llm | StrOutputParser()
    translation_chain = translation_prompt | llm | StrOutputParser()
    
    # Sample text
    text = """
    LangChain es un framework de desarrollo que permite crear aplicaciones 
    potenciadas por modelos de lenguaje grandes (LLM). Proporciona un conjunto 
    de herramientas y abstracciones que facilitan la integración de LLMs con 
    diversas fuentes de datos y servicios. Con LangChain, los desarrolladores 
    pueden crear chatbots, sistemas de preguntas y respuestas, agentes 
    inteligentes, y aplicaciones de análisis de texto de manera más eficiente 
    y escalable.
    """
    
    # Execute chains in sequence
    summary = summary_chain.invoke({"text": text})
    print(f"Summary: {summary}")
    
    translation = translation_chain.invoke({"text": summary})
    print(f"Translation: {translation}")
    print("-" * 50)

def main():
    """Main function to run all examples"""
    try:
        print("🚀 LangChain Basic Examples")
        print("=" * 50)
        
        basic_llm_example()
        prompt_template_example()
        chain_example()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create .env file with your OPENAI_API_KEY")

if __name__ == "__main__":
    main() 