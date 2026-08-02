"""Tests for the BurstinessScorer and sentence splitting."""

from __future__ import annotations

from detector_ia.burstiness_scorer import BurstinessScorer, split_sentences


class TestSplitSentences:
    def test_splits_on_sentence_boundaries(self):
        text = "First sentence. Second one! And a question?"
        assert split_sentences(text) == [
            "First sentence.",
            "Second one!",
            "And a question?",
        ]

    def test_handles_newlines(self):
        text = "Line one.\nLine two.\n\nLine three."
        assert split_sentences(text) == ["Line one.", "Line two.", "Line three."]

    def test_single_sentence(self):
        assert split_sentences("Just one sentence.") == ["Just one sentence."]

    def test_empty_text(self):
        assert split_sentences("") == []


class TestScore:
    def test_single_sentence_returns_zero_burstiness(self, fake_scorer):
        scorer = fake_scorer(lambda t: 10.0)
        burst = BurstinessScorer(scorer)
        result = burst.score("Only one sentence here.")
        assert result["burstiness"] == 0.0
        assert result["n_sentences"] == 1

    def test_variable_sentences_produce_burstiness(self, fake_scorer):
        scoring = {
            "First sentence.": 10.0,
            "Second one.": 30.0,
        }
        scorer = fake_scorer(lambda t: scoring.get(t, 10.0))
        burst = BurstinessScorer(scorer)
        result = burst.score("First sentence. Second one.")
        assert result["n_sentences"] == 2
        assert result["mean_ppl"] == 20.0
        assert result["std_ppl"] > 0
        assert result["burstiness"] > 0
        assert len(result["per_sentence_ppl"]) == 2

    def test_uniform_sentences_give_low_burstiness(self, fake_scorer):
        scorer = fake_scorer(lambda t: 15.0)
        burst = BurstinessScorer(scorer)
        result = burst.score("Same value. Same value.")
        assert result["burstiness"] == 0.0
        assert result["mean_ppl"] == 15.0
