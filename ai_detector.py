"""
Detector de texto generado por IA basado en análisis de perplejidad.

Este módulo implementa un detector que analiza textos y determina su probabilidad
de haber sido generado por modelos de lenguaje basándose en su perplejidad.

La idea principal:
- Texto humano natural: perplejidad baja (el modelo lo predice bien)
- Texto generado por IA: puede tener perplejidad más alta o más baja dependiendo
del modelo generador y el modelo evaluador
- Ataques adversariales: perplejidad muy alta

El detector usa múltiples estrategias:
1. Perplejidad absoluta
2. Análisis de patrones en la perplejidad
3. Comparación con umbrales ajustables
"""

from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics

from .perplexity_scorer import PerplexityScorer


class DetectionResult(Enum):
    """Resultados posibles de la detección."""
    HUMAN = "human"
    AI = "ai"
    UNCERTAIN = "uncertain"


@dataclass
class DetectionAnalysis:
    """Análisis completo de la detección de IA."""
    
    # Textos analizados
    text: str
    
    # Scores de perplejidad
    perplexity: float
    log_perplexity: float
    
    # Clasificación
    result: DetectionResult
    confidence: float  # 0-1
    
    # Análisis adicional
    is_too_short: bool = False
    is_empty: bool = False
    
    # Métricas para análisis
    avg_perplexity: Optional[float] = None
    std_perplexity: Optional[float] = None
    
    # Explicación
    explanation: str = ""
    
    def to_dict(self) -> dict:
        """Convierte el análisis a diccionario."""
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "perplexity": self.perplexity,
            "log_perplexity": self.log_perplexity,
            "result": self.result.value,
            "confidence": self.confidence,
            "is_too_short": self.is_too_short,
            "is_empty": self.is_empty,
            "explanation": self.explanation
        }


class AITextDetector:
    """
    Detector de texto generado por IA basado en perplejidad.
    
    Este detector usa un modelo de lenguaje para calcular la perplejidad de textos
    y determinar si probablemente fueron generados por IA.
    
    Args:
        model_name: Nombre del modelo para calcular perplejidad (default: "gpt2")
        device: Dispositivo a usar ("cuda", "cpu", None para auto-detectar)
        ai_threshold: Umbral de perplejidad para considerar AI (default: 50.0)
        human_threshold: Umbral inferior para texto humano (default: 10.0)
        min_text_length: Longitud mínima de texto para análisis (default: 20)
        use_log_scale: Usar escala logarítmica para comparaciones (default: True)
    
    Uso:
        detector = AITextDetector()
        result = detector.analyze("Este es un texto para analizar")
        print(f"Resultado: {result.result}, Confianza: {result.confidence}")
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        ai_threshold: float = 50.0,
        human_threshold: float = 10.0,
        min_text_length: int = 20,
        use_log_scale: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.ai_threshold = ai_threshold
        self.human_threshold = human_threshold
        self.min_text_length = min_text_length
        self.use_log_scale = use_log_scale
        
        # Inicializar el calculador de perplejidad
        self.scorer = PerplexityScorer(
            model_name=model_name,
            device=device
        )
        
        # Historial de análisis para estadísticas
        self.analysis_history: List[DetectionAnalysis] = []
    
    def analyze(self, text: str) -> DetectionAnalysis:
        """
        Analiza un texto y determina si probablemente fue generado por IA.
        
        Args:
            text: Texto a analizar
            
        Returns:
            DetectionAnalysis con el resultado completo
        """
        # Guardar el texto original
        original_text = text
        
        # Verificar texto vacío
        if not text or not text.strip():
            analysis = DetectionAnalysis(
                text=original_text,
                perplexity=0.0,
                log_perplexity=0.0,
                result=DetectionResult.UNCERTAIN,
                confidence=0.0,
                is_empty=True,
                explanation="Texto vacío, no se puede analizar"
            )
            self.analysis_history.append(analysis)
            return analysis
        
        # Verificar longitud mínima
        if len(text.strip()) < self.min_text_length:
            analysis = DetectionAnalysis(
                text=original_text,
                perplexity=0.0,
                log_perplexity=0.0,
                result=DetectionResult.UNCERTAIN,
                confidence=0.0,
                is_too_short=True,
                explanation=f"Texto demasiado corto (mínimo {self.min_text_length} caracteres)"
            )
            self.analysis_history.append(analysis)
            return analysis
        
        # Calcular perplejidad
        perplexity = self.scorer.score(text)
        log_perplexity = self._safe_log(perplexity)
        
        # Determinar resultado y confianza
        result, confidence, explanation = self._classify(
            perplexity, 
            log_perplexity,
            len(text)
        )
        
        analysis = DetectionAnalysis(
            text=original_text,
            perplexity=perplexity,
            log_perplexity=log_perplexity,
            result=result,
            confidence=confidence,
            is_too_short=False,
            is_empty=False,
            explanation=explanation
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[DetectionAnalysis]:
        """
        Analiza múltiples textos.
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            Lista de DetectionAnalysis
        """
        return [self.analyze(text) for text in texts]
    
    def _safe_log(self, value: float) -> float:
        """Calcula logaritmo natural de forma segura."""
        import math
        if value <= 0:
            return 0.0
        try:
            return math.log(value)
        except (ValueError, OverflowError):
            return 0.0
    
    def _classify(
        self, 
        perplexity: float, 
        log_perplexity: float,
        text_length: int
    ) -> Tuple[DetectionResult, float, str]:
        """
        Clasifica el texto basado en su perplejidad.
        
        Args:
            perplexity: Score de perplejidad
            log_perplexity: Logaritmo de la perplejidad
            text_length: Longitud del texto en caracteres
            
        Returns:
            Tupla con (resultado, confianza, explicación)
        """
        # Ajustar umbrales basado en longitud del texto
        # Textos más cortos suelen tener mayor variabilidad
        effective_ai_threshold = self.ai_threshold
        effective_human_threshold = self.human_threshold
        
        if text_length < 50:
            effective_ai_threshold *= 1.5  # Más tolerante con textos cortos
            effective_human_threshold *= 0.8
        
        if self.use_log_scale:
            # Usar escala logarítmica para mejor discriminación
            score = log_perplexity
            ai_threshold = self._safe_log(effective_ai_threshold)
            human_threshold = self._safe_log(effective_human_threshold)
        else:
            score = perplexity
            ai_threshold = effective_ai_threshold
            human_threshold = effective_human_threshold
        
        # Lógica de clasificación
        if perplexity == 0.0:
            return DetectionResult.UNCERTAIN, 0.0, "No se pudo calcular perplejidad"
        
        # Texto humano: perplejidad baja
        if score < human_threshold:
            confidence = max(0.0, min(1.0, 1.0 - (score / human_threshold)))
            return DetectionResult.HUMAN, confidence, \
                f"Perplejidad baja ({perplexity:.2f}), típico de texto humano"
        
        # Texto AI: perplejidad muy alta
        elif score > ai_threshold:
            # Confianza basada en qué tan por encima del umbral está
            excess = (score - ai_threshold) / ai_threshold
            confidence = max(0.5, min(1.0, 0.7 + excess))
            return DetectionResult.AI, confidence, \
                f"Perplejidad alta ({perplexity:.2f}), sugiere texto generado por IA"
        
        # Zona intermedia: incierto
        else:
            # Confianza inversa: más cerca del umbral = menos confianza
            distance_to_ai = ai_threshold - score
            distance_to_human = score - human_threshold
            total_distance = distance_to_ai + distance_to_human
            
            if total_distance > 0:
                confidence = distance_to_human / total_distance
            else:
                confidence = 0.5
            
            return DetectionResult.UNCERTAIN, confidence, \
                f"Perplejidad en zona intermedia ({perplexity:.2f}), no claro"
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Obtiene estadísticas de los análisis realizados.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.analysis_history:
            return {"total": 0}
        
        # Filtrar análisis válidos (no vacíos, no demasiado cortos)
        valid_analyses = [
            a for a in self.analysis_history 
            if not a.is_empty and not a.is_too_short
        ]
        
        if not valid_analyses:
            return {"total": len(self.analysis_history)}
        
        perplexities = [a.perplexity for a in valid_analyses]
        
        return {
            "total": len(self.analysis_history),
            "valid": len(valid_analyses),
            "avg_perplexity": statistics.mean(perplexities),
            "median_perplexity": statistics.median(perplexities),
            "min_perplexity": min(perplexities),
            "max_perplexity": max(perplexities),
            "std_perplexity": statistics.stdev(perplexities) if len(perplexities) > 1 else 0,
            "ai_detections": sum(1 for a in valid_analyses if a.result == DetectionResult.AI),
            "human_detections": sum(1 for a in valid_analyses if a.result == DetectionResult.HUMAN),
            "uncertain_detections": sum(1 for a in valid_analyses if a.result == DetectionResult.UNCERTAIN)
        }
    
    def set_thresholds(self, ai_threshold: float, human_threshold: float):
        """
        Ajusta los umbrales de detección.
        
        Args:
            ai_threshold: Nuevo umbral para AI
            human_threshold: Nuevo umbral para texto humano
        """
        self.ai_threshold = ai_threshold
        self.human_threshold = human_threshold
    
    def calibrate(
        self, 
        human_texts: List[str], 
        ai_texts: List[str],
        calibration_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Calibra automáticamente los umbrales basado en ejemplos.
        
        Args:
            human_texts: Lista de textos humanos (etiqueta: humano)
            ai_texts: Lista de textos generados por IA (etiqueta: IA)
            calibration_iterations: Iteraciones de calibración
            
        Returns:
            Diccionario con los nuevos umbrales y métricas de calibración
        """
        if not human_texts or not ai_texts:
            return {"error": "Necesitan proporcionarse ejemplos humanos y de IA"}
        
        # Calcular perplejidad de todos los ejemplos
        human_scores = self.scorer.batch_score(human_texts)
        ai_scores = self.scorer.batch_score(ai_texts)
        
        # Calcular umbral óptimo usando el punto medio
        avg_human = statistics.mean(human_scores)
        avg_ai = statistics.mean(ai_scores)
        
        # Nuevo umbral: punto medio entre los promedios
        new_threshold = (avg_human + avg_ai) / 2
        
        # Ajustar con margen de seguridad
        human_std = statistics.stdev(human_scores) if len(human_scores) > 1 else 1
        ai_std = statistics.stdev(ai_scores) if len(ai_scores) > 1 else 1
        
        new_ai_threshold = new_threshold + (ai_std * 0.5)
        new_human_threshold = new_threshold - (human_std * 0.5)
        
        # Aplicar los nuevos umbrales
        self.ai_threshold = float(new_ai_threshold)
        self.human_threshold = float(new_human_threshold)
        
        return {
            "new_ai_threshold": self.ai_threshold,
            "new_human_threshold": self.human_threshold,
            "avg_human_perplexity": avg_human,
            "avg_ai_perplexity": avg_ai,
            "human_std": human_std,
            "ai_std": ai_std,
            "calibration_samples": len(human_texts) + len(ai_texts)
        }
    
    def get_model_info(self) -> dict:
        """
        Obtiene información sobre el modelo usado.
        
        Returns:
            Diccionario con información del modelo
        """
        return self.scorer.get_model_info()
