"""Tests for PerplexityScorer (without loading real models)."""

from __future__ import annotations

import importlib.util

import pytest

from detector_ia import PerplexityScorer


class TestScoringContract:
    def test_batch_score_matches_individual_scores(self, fake_scorer):
        scorer = fake_scorer(lambda t: len(t))
        assert scorer.batch_score(["a", "bb", "ccc"]) == [1, 2, 3]

    def test_get_model_info_shape(self, fake_scorer):
        info = fake_scorer(lambda t: 1.0).get_model_info()
        assert set(info) >= {"model_name", "device", "max_length", "num_parameters", "dtype"}


class TestImportErrors:
    def test_raises_helpful_error_without_torch(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ImportError, match="transformers and torch"):
            PerplexityScorer()

    def test_constructor_available_in_public_api(self):
        assert callable(PerplexityScorer)
