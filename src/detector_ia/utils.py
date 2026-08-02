"""
Funciones utilitarias para el detector de IA.
"""

from typing import List, Optional, Dict
import re
import string


def preprocess_text(text: str) -> str:
    """
    Preprocesa texto para análisis: limpia, normaliza y prepara.
    
    Args:
        text: Texto a preprocesar
        
    Returns:
        Texto limpio y normalizado
    """
    if not text:
        return text
    
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Eliminar caracteres de control
    text = ''.join(char for char in text if char in string.printable or char.isspace())
    
    return text


def split_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Divide un texto largo en fragmentos superpuestos.
    
    Args:
        text: Texto a dividir
        max_length: Longitud máxima de cada fragmento (en caracteres)
        overlap: Superposición entre fragmentos (en caracteres)
        
    Returns:
        Lista de fragmentos de texto
    """
    if len(text) <= max_length:
        return [text]
    
    fragments = []
    start = 0
    
    while start < len(text):
        end = min(start + max_length, len(text))
        fragments.append(text[start:end])
        start = end - overlap
        
        if start >= len(text):
            break
    
    return fragments


def text_to_sentences(text: str) -> List[str]:
    """
    Divide un texto en oraciones.
    
    Args:
        text: Texto a dividir
        
    Returns:
        Lista de oraciones
    """
    # Usar expresiones regulares para dividir en oraciones
    # Esta es una aproximación simple
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def generate_sample_texts() -> Dict[str, List[str]]:
    """
    Genera textos de ejemplo para pruebas.
    
    Returns:
        Diccionario con textos humanos y de IA de ejemplo
    """
    human_texts = [
        "El otro día fui al parque con mis amigos. Hacía mucho sol y decidimos tomar un helado. "
        "Fue una tarde muy agradable.",
        
        "La reunión de equipo fue productiva. Discutimos varios puntos importantes del proyecto y "
        "tomamos decisiones clave para el siguiente trimestre.",
        
        "Me encanta leer libros de ciencia ficción. Recientemente terminé una novela de Isaac Asimov "
        "que me dejó pensando durante días.",
        
        "El clima en Barcelona es muy variable. Un día hace sol y al siguiente puede llover sin parar. "
        "Por eso siempre llevo paraguas en la mochila.",
        
        "Ayer cociné pasta para la cena. Usé una receta nueva que encontré en internet y quedó "
        "deliciosa. Mis compañeros de piso quedaron impresionados."
    ]
    
    # Textos que simulan ser generados por IA (patrones típicos)
    ai_texts = [
        "La inteligencia artificial está transformando radicalmente múltiples industrias en todo el "
        "mundo, desde la medicina hasta la manufactura, ofreciendo soluciones innovadoras y "
        "efectivas para problemas complejos.",
        
        "Es fundamental comprender que los modelos de lenguaje grande representan un avance "
        "significativo en la capacidad de las máquinas para entender y generar texto de manera "
        "coherente y contextualmente apropiada.",
        
        "La implementación de sistemas de detección de texto generado por IA se ha vuelto "
        "cada vez más importante en la era digital actual, donde la distinción entre contenido "
        "auténtico y sintético puede ser desafiante.",
        
        "Los algoritmos de procesamiento de lenguaje natural han alcanzado un nivel de sofisticación "
        "tal que pueden generar textos que son virtualmente indistinguibles de aquellos "
        "escritos por humanos en muchas situaciones."
    ]
    
    return {
        "human": human_texts,
        "ai": ai_texts
    }


def format_analysis_results(analyses: List) -> str:
    """
    Formatea resultados de análisis para visualización.
    
    Args:
        analyses: Lista de DetectionAnalysis
        
    Returns:
        String formateado con los resultados
    """
    lines = []
    lines.append("=" * 80)
    lines.append("RESULTADOS DE DETECCIÓN DE IA")
    lines.append("=" * 80)
    
    for i, analysis in enumerate(analyses, 1):
        lines.append(f"\nTexto {i}:")
        lines.append(f"  Texto: {analysis.text[:100]}...")
        lines.append(f"  Perplejidad: {analysis.perplexity:.2f}")
        lines.append(f"  Resultado: {analysis.result.value.upper()}")
        lines.append(f"  Confianza: {analysis.confidence:.2%}")
        lines.append(f"  Explicación: {analysis.explanation}")
        
        if analysis.is_too_short:
            lines.append("  ⚠️  Texto demasiado corto para análisis preciso")
        if analysis.is_empty:
            lines.append("  ⚠️  Texto vacío")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def calculate_accuracy(detector, human_texts: List[str], ai_texts: List[str]) -> Dict[str, float]:
    """
    Calcula la precisión del detector.
    
    Args:
        detector: Instancia de AITextDetector
        human_texts: Lista de textos humanos
        ai_texts: Lista de textos de IA
        
    Returns:
        Diccionario con métricas de precisión
    """
    from .ai_detector import DetectionResult
    
    # Analizar todos los textos
    human_results = detector.batch_analyze(human_texts)
    ai_results = detector.batch_analyze(ai_texts)
    
    # Contar clasificaciones correctas
    correct_human = sum(1 for r in human_results if r.result == DetectionResult.HUMAN)
    correct_ai = sum(1 for r in ai_results if r.result == DetectionResult.AI)
    
    total_human = len(human_results)
    total_ai = len(ai_results)
    
    return {
        "human_accuracy": correct_human / total_human if total_human > 0 else 0,
        "ai_accuracy": correct_ai / total_ai if total_ai > 0 else 0,
        "overall_accuracy": (correct_human + correct_ai) / (total_human + total_ai) if (total_human + total_ai) > 0 else 0,
        "total_samples": total_human + total_ai,
        "correct_predictions": correct_human + correct_ai
    }
