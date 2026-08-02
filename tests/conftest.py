"""Shared fixtures for the detector_ia test suite."""

from __future__ import annotations

import re
from typing import Callable

import pytest

from detector_ia import DetectionAnalysis, DetectionResult


class FakeScorer:
    """Deterministic PerplexityScorer stand-in for tests."""

    def __init__(self, scoring_fn: Callable[[str], float]):
        self._scoring_fn = scoring_fn

    def score(self, text: str) -> float:
        return self._scoring_fn(text)

    def batch_score(self, texts):
        return [self.score(t) for t in texts]

    def get_model_info(self) -> dict:
        return {
            "model_name": "fake",
            "device": "cpu",
            "max_length": 512,
            "num_parameters": 0,
            "dtype": "float32",
        }


@pytest.fixture
def fake_scorer():
    """Returns a FakeScorer factory configured with a scoring function."""

    def make(scoring_fn: Callable[[str], float]) -> FakeScorer:
        return FakeScorer(scoring_fn)

    return make


@pytest.fixture
def classification_scorer(fake_scorer):
    """Returns a scorer that yields human-like or AI-like perplexity by keyword."""
    return fake_scorer(
        lambda text: 200.0 if re.search(r"\bai\b", text.lower()) else 3.0
    )


@pytest.fixture
def human_analysis():
    return DetectionAnalysis(
        text="A perfectly ordinary human sentence written by a person.",
        perplexity=3.0,
        log_perplexity=1.1,
        result=DetectionResult.HUMAN,
        confidence=0.9,
        explanation="Low perplexity (3.00), typical of human text",
    )
