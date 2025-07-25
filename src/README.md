# Estructura Modular - Carpeta `src/`

Esta carpeta contiene la nueva arquitectura modular del proyecto LangChain Demo, organizada siguiendo buenas prácticas de desarrollo Python.

## 📁 Organización de Módulos

### 🔧 `tools/`
Herramientas de LangChain organizadas por funcionalidad:
- `imp_tools.py` - 6 herramientas especializadas para auditorías IPM
- `general_tools.py` - Herramientas generales (calculadora, tiempo, análisis)

### ⚙️ `services/`
Capa de servicios con lógica de negocio centralizada:
- `llm_service.py` - Gestión de LLM con caching y validación
- `embedding_service.py` - Gestión de embeddings OpenAI  
- `chain_service.py` - Creación de chains, agentes y RAG

### 📊 `models/`
Modelos de datos con validación Pydantic:
- `imp_models.py` - Modelos para auditorías IPM

### 🛠️ `utils/`
Utilidades reutilizables:
- `document_utils.py` - Procesamiento y análisis de documentos
- `optimization_utils.py` - Optimización de tokens y análisis de costos

### 🎨 `ui/`
Componentes de interfaz de usuario modulares:
- `imp_audit_ui.py` - Interfaz completa para auditorías IPM
- `basic_chat_ui.py` - Chat básico
- `memory_chat_ui.py` - Chat con memoria
- `agents_ui.py` - Interfaz para agentes
- `rag_ui.py` - Interfaz para RAG
- `demo_ui.py` - Demostración de conceptos

## 🔄 Patrones de Uso

### Importar Servicios
```python
from src.services import LLMService, EmbeddingService
llm = LLMService.get_llm()
```

### Usar Herramientas
```python
from src.tools import analyze_pest_monitoring
result = analyze_pest_monitoring.invoke({"document_content": "...", "file_name": "doc.pdf"})
```

### Crear Componentes UI
```python
from src.ui import IMPAuditUI
audit_ui = IMPAuditUI()
audit_ui.render()
```

## 📈 Beneficios

✅ **Mantenibilidad**: Cada módulo tiene una responsabilidad específica
✅ **Testabilidad**: Módulos independientes fáciles de testear  
✅ **Reutilización**: Componentes pueden usarse en múltiples contextos
✅ **Escalabilidad**: Fácil agregar nuevas funcionalidades
✅ **Legibilidad**: Código organizado y fácil de entender

Ver `../STRUCTURE_REFACTORING.md` para documentación completa de la reestructuración. 