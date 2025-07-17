"""
Test Setup Script
This script validates that all dependencies are installed and configurations are correct.
"""

import sys
import importlib
import os
from config import Config

def test_imports():
    """Test that all required packages are installed"""
    print("🔍 Testing imports...")
    
    required_packages = [
        'langchain',
        'langchain_openai',
        'langchain_core',
        'langchain_community',
        'streamlit',
        'chromadb',
        'tiktoken',
        'faiss',
        'requests',
        'beautifulsoup4',
        'PyPDF2',
        'python_dotenv'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'python_dotenv':
                importlib.import_module('dotenv')
            elif package == 'beautifulsoup4':
                importlib.import_module('bs4')
            elif package == 'PyPDF2':
                importlib.import_module('PyPDF2')
            elif package == 'faiss':
                importlib.import_module('faiss')
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_environment():
    """Test environment variables configuration"""
    print("\n🔧 Testing environment configuration...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("  ⚠️  .env file not found")
        print("  Create .env file from env_example.txt")
        return False
    
    # Check API key
    try:
        Config.validate_config()
        print("  ✅ OpenAI API key configured")
    except ValueError as e:
        print(f"  ❌ Configuration error: {e}")
        return False
    
    # Check optional configurations
    if Config.LANGCHAIN_API_KEY:
        print("  ✅ LangSmith API key configured")
    else:
        print("  ⚠️  LangSmith API key not configured (optional)")
    
    print(f"  ✅ Default model: {Config.DEFAULT_MODEL}")
    print(f"  ✅ Temperature: {Config.DEFAULT_TEMPERATURE}")
    
    return True

def test_basic_functionality():
    """Test basic LangChain functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Test basic prompt
        prompt = ChatPromptTemplate.from_template("Say 'Hello from LangChain!' in {language}")
        chain = prompt | llm
        
        response = chain.invoke({"language": "Spanish"})
        print(f"  ✅ Basic chain test: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality"""
    print("\n🔍 Testing vector store...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.schema import Document
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        
        # Create test documents
        docs = [
            Document(page_content="LangChain is a framework for LLM applications", metadata={"source": "test"}),
            Document(page_content="Vector stores enable semantic search", metadata={"source": "test"})
        ]
        
        # Create vector store
        vectorstore = Chroma.from_documents(docs, embeddings)
        
        # Test search
        results = vectorstore.similarity_search("LangChain framework", k=1)
        print(f"  ✅ Vector store test: Found {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector store test failed: {e}")
        return False

def test_agent_functionality():
    """Test agent functionality"""
    print("\n🤖 Testing agent functionality...")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_core.tools import tool
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        # Create simple tool
        @tool
        def test_tool(input_text: str) -> str:
            """A simple test tool"""
            return f"Test tool received: {input_text}"
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=Config.OPENAI_API_KEY)
        
        # Create agent
        tools = [test_tool]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant with access to tools."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools)
        
        # Test agent
        response = agent_executor.invoke({"input": "Use the test tool with 'hello'", "chat_history": []})
        print(f"  ✅ Agent test: {response['output'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 LangChain Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_basic_functionality,
        test_vector_store,
        test_agent_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your LangChain setup is ready.")
        print("\nNext steps:")
        print("1. Run examples: python examples/basic_chain.py")
        print("2. Start web app: streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the configuration.")
        print("See README.md for troubleshooting guide.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 