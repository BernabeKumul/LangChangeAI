# 🦜 LangChain Demo Project

Una demostración completa de **LangChain**, el framework más popular para desarrollar aplicaciones con modelos de lenguaje grandes (LLM). Este proyecto incluye ejemplos prácticos y una interfaz web interactiva.

## 📋 Tabla de Contenidos

- [¿Qué es LangChain?](#qué-es-langchain)
- [Características del Proyecto](#características-del-proyecto)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso](#uso)
- [Ejemplos Incluidos](#ejemplos-incluidos)
- [Interfaz Web](#interfaz-web)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribuir](#contribuir)
- [Licencia](#licencia)

## 🤖 ¿Qué es LangChain?

[LangChain](https://langchain.com/) es un framework de código abierto diseñado para facilitar el desarrollo de aplicaciones potenciadas por modelos de lenguaje grandes. Creado por Harrison Chase, LangChain proporciona las herramientas necesarias para:

- **Integrar LLMs** con diversas fuentes de datos
- **Crear agentes inteligentes** que pueden usar herramientas
- **Implementar memoria conversacional** para mantener contexto
- **Desarrollar sistemas RAG** (Retrieval Augmented Generation)
- **Procesar documentos** y realizar búsquedas semánticas

## ✨ Características del Proyecto

### 🔧 Ejemplos Prácticos
- **Chains básicas**: Integración simple con LLMs
- **Memoria conversacional**: Múltiples tipos de memoria
- **Agentes con herramientas**: Calculadora, tiempo, análisis de texto
- **RAG (Retrieval Augmented Generation)**: Búsqueda en documentos
- **Vector stores**: Almacenamiento y búsqueda de embeddings

### 🌐 Interfaz Web Interactiva
- Aplicación Streamlit con múltiples funcionalidades
- Chat básico con LLM
- Chat con memoria conversacional
- Agentes con herramientas personalizadas
- Sistema RAG para preguntas sobre documentos

### 📚 Documentación Completa
- Ejemplos comentados en español
- Configuración flexible
- Guías de uso paso a paso

## 🚀 Instalación

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd LangChange
```

### 2. Crear Entorno Virtual
```bash
# Python 3.8+ requerido
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
# source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
```bash
# Copiar archivo de ejemplo
cp env_example.txt .env

# Editar .env con tu API key
# OPENAI_API_KEY=tu_api_key_aquí
```

## ⚙️ Configuración

### API Keys Necesarias

1. **OpenAI API Key** (obligatoria)
   - Obtener en: https://platform.openai.com/api-keys
   - Añadir a `.env`: `OPENAI_API_KEY=tu_key_aquí`

2. **LangSmith API Key** (opcional, para tracing)
   - Obtener en: https://smith.langchain.com/
   - Añadir a `.env`: `LANGCHAIN_API_KEY=tu_key_aquí`

### Configuración Personalizada

Edita `config.py` para personalizar:
- Modelo de LLM por defecto
- Temperatura y max tokens
- Configuración de vector stores
- Tamaño de chunks para documentos

## 🎯 Uso

### Ejecutar Ejemplos Individuales

```bash
# Ejemplo básico de chains
python examples/basic_chain.py

# Ejemplo de memoria conversacional
python examples/memory_conversation.py

# Ejemplo de agentes con herramientas
python examples/agents_tools.py

# Ejemplo de RAG
python examples/rag_retrieval.py
```

### Ejecutar Interfaz Web

```bash
streamlit run streamlit_app.py
```

Abre tu navegador en `http://localhost:8501` para usar la interfaz interactiva.

## 📖 Ejemplos Incluidos

### 1. Basic Chain (`examples/basic_chain.py`)
- Uso básico de LLM
- Plantillas de prompts
- Chains secuenciales
- Parsing de outputs

**Ejemplo de uso:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Explica {topic} en {language}")
chain = prompt | llm

response = chain.invoke({"topic": "LangChain", "language": "español"})
```

### 2. Memory Conversation (`examples/memory_conversation.py`)
- ConversationBufferMemory
- ConversationSummaryMemory
- ConversationBufferWindowMemory
- Memoria personalizada

**Tipos de memoria:**
- **Buffer**: Almacena toda la conversación
- **Summary**: Resume conversaciones largas
- **Window**: Mantiene solo las últimas N interacciones

### 3. Agents and Tools (`examples/agents_tools.py`)
- Agentes básicos
- Herramientas personalizadas
- Agentes conversacionales
- Integración con búsqueda web

**Herramientas incluidas:**
- 🧮 Calculadora matemática
- 🕒 Información de tiempo
- 📝 Análisis de texto
- 🔍 Búsqueda web (DuckDuckGo)

### 4. RAG Retrieval (`examples/rag_retrieval.py`)
- Carga y procesamiento de documentos
- Embeddings y vector stores
- Búsqueda por similitud
- Chains de pregunta-respuesta

**Funcionalidades RAG:**
- Splitting inteligente de documentos
- Múltiples estrategias de retrieval
- Filtrado por metadata
- Persistencia de vector stores

## 🌐 Interfaz Web

La aplicación Streamlit incluye cuatro secciones principales:

### 💬 Chat Básico
- Conversación simple con LLM
- Sin memoria entre mensajes
- Ideal para pruebas rápidas

### 🧠 Chat con Memoria
- Conversación con contexto
- Recuerda interacciones previas
- Experiencia más natural

### 🤖 Agentes
- Agentes con herramientas
- Calculadora, tiempo, análisis
- Ejemplos interactivos

### 📚 RAG
- Preguntas sobre documentos
- Base de datos vectorial
- Búsqueda semántica

## 📁 Estructura del Proyecto

```
LangChange/
├── config.py                 # Configuración del proyecto
├── requirements.txt          # Dependencias Python
├── env_example.txt          # Ejemplo de variables de entorno
├── streamlit_app.py         # Aplicación web Streamlit
├── README.md                # Este archivo
└── examples/
    ├── basic_chain.py       # Ejemplos básicos de chains
    ├── memory_conversation.py # Ejemplos de memoria
    ├── agents_tools.py      # Ejemplos de agentes
    └── rag_retrieval.py     # Ejemplos de RAG
```

## 🔧 Personalización

### Añadir Nuevas Herramientas

```python
from langchain_core.tools import tool

@tool
def mi_herramienta(input_text: str) -> str:
    """Descripción de mi herramienta"""
    # Tu lógica aquí
    return resultado
```

### Crear Nuevos Tipos de Memoria

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
```

### Integrar Nuevos LLMs

```python
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Anthropic Claude
llm = ChatAnthropic(model="claude-3-sonnet-20240229")

# Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-pro")
```

## 🚀 Casos de Uso Avanzados

### 1. Sistema de Atención al Cliente
- Agente con acceso a base de conocimientos
- Memoria de conversaciones
- Escalado a agentes humanos

### 2. Asistente de Investigación
- RAG con múltiples fuentes
- Búsqueda web automática
- Generación de reportes

### 3. Automatización de Tareas
- Agentes con herramientas de sistema
- Workflows complejos
- Integración con APIs

## 🐛 Solución de Problemas

### Error: "OpenAI API Key not found"
```bash
# Verificar que .env existe y contiene:
OPENAI_API_KEY=tu_key_aquí
```

### Error: "Module not found"
```bash
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

### Error: "Streamlit not found"
```bash
# Verificar instalación de Streamlit
pip show streamlit
pip install streamlit
```

## 📚 Recursos Adicionales

- [Documentación oficial de LangChain](https://python.langchain.com/)
- [LangSmith para debugging](https://smith.langchain.com/)
- [Ejemplos de la comunidad](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [LangChain YouTube Channel](https://www.youtube.com/@LangChain)

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

Creado con ❤️ para demostrar las capacidades de LangChain.

---

## 🎉 ¡Empezar!

1. **Instala las dependencias**: `pip install -r requirements.txt`
2. **Configura tu API key**: Edita el archivo `.env`
3. **Ejecuta la aplicación**: `streamlit run streamlit_app.py`
4. **Explora los ejemplos**: Prueba cada funcionalidad

¡Disfruta explorando LangChain! 🚀 