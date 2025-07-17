"""
Memory and Conversation Example
This example demonstrates how to maintain conversation context using LangChain memory:
- ConversationBufferMemory
- ConversationSummaryMemory
- ConversationBufferWindowMemory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
from config import Config

def buffer_memory_example():
    """Example using ConversationBufferMemory"""
    print("=== Buffer Memory Example ===")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create memory that stores entire conversation
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Simulate a conversation
    print("Conversation 1:")
    response1 = conversation.predict(input="Hola, me llamo José. ¿Cómo estás?")
    print(f"AI: {response1}")
    
    print("\nConversation 2:")
    response2 = conversation.predict(input="¿Cuál es mi nombre?")
    print(f"AI: {response2}")
    
    print("\nConversation 3:")
    response3 = conversation.predict(input="¿Puedes explicarme qué es LangChain?")
    print(f"AI: {response3}")
    
    print("\nMemory content:")
    print(memory.buffer)
    print("-" * 50)

def summary_memory_example():
    """Example using ConversationSummaryMemory"""
    print("=== Summary Memory Example ===")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create memory that summarizes conversation
    memory = ConversationSummaryMemory(
        llm=llm,
        return_messages=True
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Simulate a long conversation
    topics = [
        "Soy un desarrollador de Python interesado en IA",
        "¿Qué proyectos de IA me recomiendas para principiantes?",
        "¿Cómo puedo aprender machine learning?",
        "¿Cuáles son las mejores librerías de Python para IA?",
        "¿Puedes resumir lo que hemos hablado?"
    ]
    
    for i, topic in enumerate(topics, 1):
        print(f"\nConversation {i}:")
        response = conversation.predict(input=topic)
        print(f"Human: {topic}")
        print(f"AI: {response}")
    
    print("\nMemory summary:")
    print(memory.buffer)
    print("-" * 50)

def window_memory_example():
    """Example using ConversationBufferWindowMemory"""
    print("=== Window Memory Example ===")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create memory that keeps only last k interactions
    memory = ConversationBufferWindowMemory(
        k=2,  # Keep only last 2 interactions
        return_messages=True
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    # Simulate multiple conversations
    messages = [
        "Mi nombre es José",
        "Vivo en México",
        "Soy programador",
        "Me gusta la inteligencia artificial",
        "¿Cuál es mi nombre?",  # This should not be remembered due to window size
        "¿Cuál es mi profesión?"  # This should be remembered
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\nConversation {i}:")
        response = conversation.predict(input=message)
        print(f"Human: {message}")
        print(f"AI: {response}")
        print(f"Memory size: {len(memory.chat_memory.messages)}")
    
    print("\nFinal memory content:")
    for msg in memory.chat_memory.messages:
        print(f"{type(msg).__name__}: {msg.content}")
    print("-" * 50)

def custom_memory_example():
    """Example of custom memory implementation"""
    print("=== Custom Memory Example ===")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=Config.DEFAULT_MODEL,
        temperature=Config.DEFAULT_TEMPERATURE,
        openai_api_key=Config.OPENAI_API_KEY
    )
    
    # Create a custom prompt with memory
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Mantén contexto de la conversación."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create chain
    chain = prompt | llm
    
    # Simulate conversation with manual memory management
    chat_history = []
    
    def chat_with_memory(user_input):
        # Add user message to history
        chat_history.append(HumanMessage(content=user_input))
        
        # Generate response
        response = chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # Add AI response to history
        chat_history.append(AIMessage(content=response.content))
        
        return response.content
    
    # Test conversation
    questions = [
        "Hola, soy Ana y estoy aprendiendo LangChain",
        "¿Cuáles son los componentes principales de LangChain?",
        "¿Recuerdas mi nombre?",
        "¿Puedes darme un ejemplo de uso de memoria en LangChain?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nConversation {i}:")
        print(f"Human: {question}")
        response = chat_with_memory(question)
        print(f"AI: {response}")
    
    print(f"\nTotal messages in history: {len(chat_history)}")
    print("-" * 50)

def main():
    """Main function to run all memory examples"""
    try:
        print("🧠 LangChain Memory Examples")
        print("=" * 50)
        
        Config.validate_config()
        
        buffer_memory_example()
        summary_memory_example()
        window_memory_example()
        custom_memory_example()
        
        print("✅ All memory examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create .env file with your OPENAI_API_KEY")

if __name__ == "__main__":
    main() 