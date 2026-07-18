"""
Calculador de burstiness para detección de IA.

El burstiness mide la variabilidad de la perplejidad entre oraciones.
- Texto humano: alta variabilidad (burstiness alto)
- Texto generado con IA: baja variabilidad (burstiness bajo)
"""

import re
import statistics
from typing import List, Dict

try:
    from .perplexity_scorer import PerplexityScorer
except (ImportError, ValueError):
    from perplexity_scorer import PerplexityScorer


def split_sentences(text: str) -> List[str]:
    """Divide texto en oraciones usando puntos, signos de interrogación y exclamación."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


class BurstinessScorer:
    """
    Calcula el burstiness de un texto basado en la variabilidad de su perplejidad por oración.
    
    El burstiness se define como: std_ppl / mean_ppl
    - Valores altos: texto humano (variabilidad natural)
    - Valores bajos: texto generado por IA (consistencia artificial)
    """
    
    def __init__(self, perplexity_scorer: PerplexityScorer):
        """
        Args:
            perplexity_scorer: Instancia de PerplexityScorer para calcular perplejidad
        """
        self.ppl_scorer = perplexity_scorer
    
    def score(self, text: str) -> Dict:
        """
        Calcula el burstiness de un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con:
            - burstiness: relación desviación estándar / media de perplejidad por oración
            - mean_ppl: perplejidad media por oración
            - std_ppl: desviación estándar de perplejidad por oración
            - n_sentences: número de oraciones
            - per_sentence_ppl: lista de perplejidades por oración
        """
        sentences = split_sentences(text)
        
        if len(sentences) < 2:
            # No hay suficiente oraciones para calcular variabilidad
            ppl = self.ppl_scorer.score(text)
            return {
                "burstiness": 0.0,
                "mean_ppl": ppl,
                "std_ppl": 0.0,
                "n_sentences": len(sentences),
                "per_sentence_ppl": [ppl]
            }
        
        # Calcular perplejidad para cada oración
        ppls = [self.ppl_scorer.score(s) for s in sentences]
        mean_ppl = statistics.mean(ppls)
        std_ppl = statistics.pstdev(ppls) if len(ppls) > 1 else 0.0
        
        # Burstiness: coeficiente de variación (std/mean)
        burstiness = std_ppl / mean_ppl if mean_ppl > 0 else 0.0
        
        return {
            "burstiness": burstiness,
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "n_sentences": len(sentences),
            "per_sentence_ppl": ppls,
        }
