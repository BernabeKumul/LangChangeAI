"""
Utilidades para optimización de tokens y costos
"""

import streamlit as st
from typing import Dict, List, Tuple, Any


class TokenOptimizer:
    """Clase para optimización de tokens y análisis de costos"""
    
    # Token pricing (approximate costs per 1K tokens in USD)
    TOKEN_COSTS = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03}
    }
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estima el número de tokens (aproximación: 1 token ≈ 4 caracteres)"""
        return len(text) // 4
    
    @staticmethod
    def calculate_cost(tokens: int, model: str = "gpt-3.5-turbo", operation: str = "input") -> float:
        """Calcula el costo aproximado en USD para el número de tokens"""
        if model not in TokenOptimizer.TOKEN_COSTS:
            model = "gpt-3.5-turbo"  # Default fallback
        
        cost_per_1k = TokenOptimizer.TOKEN_COSTS[model].get(operation, 0.0015)
        return (tokens / 1000) * cost_per_1k
    
    @staticmethod
    def analyze_optimization_impact(
        original_texts: List[str], 
        optimized_texts: List[str],
        model: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        """Analiza el impacto de la optimización en tokens y costos"""
        
        original_tokens = sum(TokenOptimizer.estimate_tokens(text) for text in original_texts)
        optimized_tokens = sum(TokenOptimizer.estimate_tokens(text) for text in optimized_texts)
        
        tokens_saved = original_tokens - optimized_tokens
        reduction_pct = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
        
        original_cost = TokenOptimizer.calculate_cost(original_tokens, model)
        optimized_cost = TokenOptimizer.calculate_cost(optimized_tokens, model)
        cost_saved = original_cost - optimized_cost
        
        return {
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "reduction_percentage": reduction_pct,
            "original_cost": original_cost,
            "optimized_cost": optimized_cost,
            "cost_saved": cost_saved,
            "model": model
        }
    
    @staticmethod
    def display_optimization_summary(analysis: Dict[str, Any]) -> None:
        """Muestra un resumen visual de la optimización en Streamlit"""
        
        st.subheader("📊 Resumen de Optimización")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Tokens Originales", 
                f"{analysis['original_tokens']:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Tokens Optimizados", 
                f"{analysis['optimized_tokens']:,}",
                delta=f"-{analysis['tokens_saved']:,}"
            )
        
        with col3:
            st.metric(
                "Reducción", 
                f"{analysis['reduction_percentage']:.1f}%",
                delta=f"{analysis['tokens_saved']:,} tokens"
            )
        
        with col4:
            st.metric(
                "Ahorro Estimado", 
                f"${analysis['cost_saved']:.4f}",
                delta=f"-{(analysis['cost_saved']/analysis['original_cost']*100):.1f}%" if analysis['original_cost'] > 0 else "0%"
            )
        
        # Detailed breakdown
        with st.expander("📋 Detalles de Costos"):
            st.write(f"**Modelo:** {analysis['model']}")
            st.write(f"**Costo Original:** ${analysis['original_cost']:.4f}")
            st.write(f"**Costo Optimizado:** ${analysis['optimized_cost']:.4f}")
            st.write(f"**Ahorro Total:** ${analysis['cost_saved']:.4f}")
    
    @staticmethod
    def recommend_strategy(document_tokens: List[int], max_tokens: int = 4000) -> str:
        """Recomienda la mejor estrategia de optimización basada en el análisis"""
        
        total_tokens = sum(document_tokens)
        avg_doc_size = total_tokens / len(document_tokens) if document_tokens else 0
        num_docs = len(document_tokens)
        
        if total_tokens <= max_tokens:
            return "No optimization needed - within token limits"
        
        reduction_needed = total_tokens - max_tokens
        reduction_pct = (reduction_needed / total_tokens) * 100
        
        if reduction_pct < 30 and avg_doc_size > 1000:
            return "smart_extraction"  # Light optimization
        elif reduction_pct < 60 and num_docs > 1:
            return "summarization"  # Moderate optimization
        else:
            return "chunking"  # Heavy optimization needed
    
    @staticmethod
    def create_optimization_report(
        documents_info: List[Dict[str, Any]], 
        strategy_used: str,
        analysis: Dict[str, Any]
    ) -> str:
        """Crea un reporte detallado de la optimización realizada"""
        
        report = f"""
## 📊 Reporte de Optimización de Tokens

### Estrategia Utilizada: {strategy_used.replace('_', ' ').title()}

### Documentos Procesados:
"""
        
        for i, doc in enumerate(documents_info, 1):
            original_tokens = doc.get('original_tokens', 0)
            processed_tokens = doc.get('processed_tokens', 0)
            reduction = original_tokens - processed_tokens
            reduction_pct = (reduction / original_tokens * 100) if original_tokens > 0 else 0
            
            report += f"""
**Documento {i}: {doc.get('filename', 'Unknown')}**
- Tokens originales: {original_tokens:,}
- Tokens procesados: {processed_tokens:,}
- Reducción: {reduction:,} tokens ({reduction_pct:.1f}%)
"""
        
        report += f"""
### Resumen Total:
- **Tokens originales:** {analysis['original_tokens']:,}
- **Tokens finales:** {analysis['optimized_tokens']:,}
- **Reducción total:** {analysis['tokens_saved']:,} tokens ({analysis['reduction_percentage']:.1f}%)
- **Ahorro estimado:** ${analysis['cost_saved']:.4f}

### Estrategias Disponibles:
- **Smart Extraction:** Extrae solo párrafos con contenido IPM relevante
- **Summarization:** Resume documentos manteniendo información clave
- **Chunking:** Usa búsqueda semántica para encontrar chunks más relevantes
"""
        
        return report 