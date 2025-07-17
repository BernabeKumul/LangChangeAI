"""
Agents and Tools Example
This example demonstrates how to create AI agents that can use external tools:
- Custom tools
- Built-in tools
- Agent execution
- Tool calling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import json
from datetime import datetime
from config import Config

@tool
def calculator(operation: str) -> str:
    """
    Performs basic mathematical operations.
    
    Args:
        operation: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the mathematical operation
    """
    try:
        # Security: Only allow basic mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in operation):
            return "Error: Invalid characters in operation"
        
        result = eval(operation)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for a city.
    
    Args:
        city: Name of the city to get weather for
    
    Returns:
        Weather information for the city
    """
    try:
        # Note: This is a mock implementation
        # In a real application, you would use a weather API
        weather_data = {
            "mexico": {"temp": "25°C", "condition": "Sunny", "humidity": "60%"},
            "madrid": {"temp": "18°C", "condition": "Cloudy", "humidity": "70%"},
            "london": {"temp": "12°C", "condition": "Rainy", "humidity": "85%"},
            "new york": {"temp": "22°C", "condition": "Partly cloudy", "humidity": "55%"}
        }
        
        city_lower = city.lower()
        if city_lower in weather_data:
            data = weather_data[city_lower]
            return f"Weather in {city}: {data['temp']}, {data['condition']}, Humidity: {data['humidity']}"
        else:
            return f"Weather data not available for {city}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

@tool
def get_current_time() -> str:
    """
    Get the current date and time.
    
    Returns:
        Current date and time as a string
    """
    try:
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error getting time: {str(e)}"

@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and provide statistics.
    
    Args:
        text: The text to analyze
    
    Returns:
        Text analysis results
    """
    try:
        words = text.split()
        sentences = text.split('.')
        
        analysis = {
            "characters": len(text),
            "words": len(words),
            "sentences": len([s for s in sentences if s.strip()]),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
        
        return json.dumps(analysis, indent=2)
    except Exception as e:
        return f"Error analyzing text: {str(e)}"

def basic_agent_example():
    """Basic agent example with custom tools"""
    print("=== Basic Agent Example ===")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Define tools
    tools = [calculator, get_weather, get_current_time, text_analyzer]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente útil que puede usar herramientas para ayudar al usuario. 
        Tienes acceso a las siguientes herramientas:
        - calculator: para operaciones matemáticas
        - get_weather: para obtener información del clima
        - get_current_time: para obtener la fecha y hora actual
        - text_analyzer: para analizar texto
        
        Usa las herramientas cuando sea necesario y proporciona respuestas útiles en español."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test the agent
    test_queries = [
        "¿Cuál es el resultado de 15 * 23 + 47?",
        "¿Qué hora es?",
        "¿Cómo está el clima en Madrid?",
        "Analiza este texto: 'LangChain es una biblioteca muy útil para crear aplicaciones con IA.'"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        response = agent_executor.invoke({"input": query, "chat_history": []})
        print(f"Response: {response['output']}")
        print("-" * 30)
    
    print("-" * 50)

def conversational_agent_example():
    """Conversational agent with memory"""
    print("=== Conversational Agent Example ===")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Define tools
    tools = [calculator, get_weather, get_current_time]
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente personal inteligente llamado LangBot. 
        Puedes mantener conversaciones y usar herramientas cuando sea necesario.
        
        Herramientas disponibles:
        - calculator: para cálculos matemáticos
        - get_weather: para información del clima
        - get_current_time: para fecha y hora
        
        Mantén un tono amigable y profesional en español."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    
    # Simulate a conversation
    conversation_flow = [
        "Hola, me llamo Carlos. ¿Cómo estás?",
        "¿Puedes calcular cuánto es 100 * 25?",
        "¿Recuerdas mi nombre?",
        "¿Qué hora es?",
        "¿Cómo está el clima en México?",
        "¿Puedes hacer un resumen de nuestra conversación?"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\nConversation {i}: {message}")
        response = agent_executor.invoke({"input": message})
        print(f"LangBot: {response['output']}")
        print("-" * 30)
    
    print("-" * 50)

def search_agent_example():
    """Agent with search capabilities"""
    print("=== Search Agent Example ===")
    
    try:
        # Initialize LLM
        llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.DEFAULT_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Create search tool
        search = DuckDuckGoSearchRun()
        
        # Define tools including search
        tools = [search, calculator, get_current_time]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente de investigación que puede buscar información en internet.
            
            Herramientas disponibles:
            - duckduckgo_search: para buscar información actualizada en internet
            - calculator: para cálculos matemáticos
            - get_current_time: para fecha y hora
            
            Usa la búsqueda cuando necesites información actual o específica."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        
        # Create agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Test search queries
        search_queries = [
            "¿Cuáles son las últimas noticias sobre LangChain?",
            "¿Cuál es el precio actual del Bitcoin?",
            "¿Qué temperatura hace hoy en Madrid?"
        ]
        
        for i, query in enumerate(search_queries, 1):
            print(f"\nSearch Query {i}: {query}")
            try:
                response = agent_executor.invoke({"input": query, "chat_history": []})
                print(f"Response: {response['output']}")
            except Exception as e:
                print(f"Error in search: {e}")
            print("-" * 30)
        
    except Exception as e:
        print(f"Search functionality not available: {e}")
        print("Install duckduckgo-search: pip install duckduckgo-search")
    
    print("-" * 50)

def custom_tool_example():
    """Example of creating custom tools"""
    print("=== Custom Tool Example ===")
    
    @tool
    def password_generator(length: int = 12) -> str:
        """
        Generate a random password.
        
        Args:
            length: Length of the password (default: 12)
        
        Returns:
            Generated password
        """
        import random
        import string
        
        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(random.choice(characters) for _ in range(length))
        return f"Generated password: {password}"
    
    @tool
    def text_reverser(text: str) -> str:
        """
        Reverse the given text.
        
        Args:
            text: Text to reverse
        
        Returns:
            Reversed text
        """
        return f"Reversed text: {text[::-1]}"
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Define custom tools
    tools = [password_generator, text_reverser, calculator]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente con herramientas personalizadas.
        
        Herramientas disponibles:
        - password_generator: genera contraseñas seguras
        - text_reverser: invierte texto
        - calculator: hace cálculos matemáticos
        
        Usa las herramientas según sea necesario."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test custom tools
    custom_queries = [
        "Genera una contraseña de 16 caracteres",
        "Invierte el texto 'LangChain es genial'",
        "Calcula 2^8 y luego genera una contraseña de esa longitud"
    ]
    
    for i, query in enumerate(custom_queries, 1):
        print(f"\nCustom Query {i}: {query}")
        response = agent_executor.invoke({"input": query, "chat_history": []})
        print(f"Response: {response['output']}")
        print("-" * 30)
    
    print("-" * 50)

def main():
    """Main function to run all agent examples"""
    try:
        print("🤖 LangChain Agents and Tools Examples")
        print("=" * 50)
        
        Config.validate_config()
        
        basic_agent_example()
        conversational_agent_example()
        search_agent_example()
        custom_tool_example()
        
        print("✅ All agent examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create .env file with your OPENAI_API_KEY")
        print("3. For search functionality: pip install duckduckgo-search")

if __name__ == "__main__":
    main() 