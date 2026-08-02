"""
AI text detector based on perplexity and burstiness analysis.

This package provides tools to detect AI-generated text using perplexity and
sentence-level burstiness computed with pretrained language models.
"""

from __future__ import annotations

from .ai_detector import AITextDetector, DetectionAnalysis, DetectionResult
from .burstiness_scorer import BurstinessScorer, split_sentences
from .perplexity_scorer import PerplexityScorer

__version__ = "1.0.0"

__all__ = [
    "AITextDetector",
    "BurstinessScorer",
    "DetectionAnalysis",
    "DetectionResult",
    "PerplexityScorer",
    "split_sentences",
]
