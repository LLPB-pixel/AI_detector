"""
Detector de IA basado en perplejidad de modelos de lenguaje.

Este paquete proporciona herramientas para detectar texto generado por IA
mediante el análisis de perplejidad usando modelos de lenguaje preentrenados.
"""

from .perplexity_scorer import PerplexityScorer
from .ai_detector import AITextDetector

__version__ = "1.0.0"
__all__ = ["PerplexityScorer", "AITextDetector"]
