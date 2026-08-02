"""Tests for the utility functions."""

from __future__ import annotations

from detector_ia import AITextDetector
from detector_ia.utils import (
    calculate_accuracy,
    format_analysis_results,
    generate_sample_texts,
    preprocess_text,
    split_text,
    text_to_sentences,
)


class TestPreprocessText:
    def test_normalizes_whitespace(self):
        assert preprocess_text("  Hello   world  ") == "Hello world"

    def test_removes_control_characters(self):
        assert "\x00" not in preprocess_text("hello\x00world")
        assert preprocess_text("hello\x00world") == "helloworld"

    def test_empty_input(self):
        assert preprocess_text("") == ""
        assert preprocess_text(None) is None


class TestSplitText:
    def test_short_text_unchanged(self):
        assert split_text("short", max_length=100) == ["short"]

    def test_long_text_splits_with_overlap(self):
        text = "x" * 500
        fragments = split_text(text, max_length=100, overlap=20)
        assert len(fragments) > 1
        assert all(len(f) <= 100 for f in fragments)
        assert all(len(f) >= 80 for f in fragments)


class TestTextToSentences:
    def test_splits_into_sentences(self):
        assert text_to_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]


class TestSampleTexts:
    def test_returns_human_and_ai_groups(self):
        samples = generate_sample_texts()
        assert "human" in samples and "ai" in samples
        assert len(samples["human"]) > 0
        assert len(samples["ai"]) > 0


class TestFormatResults:
    def test_formats_analysis(self, human_analysis):
        output = format_analysis_results([human_analysis])
        assert "DETECTION" in output.upper()
        assert "human" in output.lower()


class TestCalculateAccuracy:
    def test_perfect_accuracy(self, classification_scorer):
        detector = AITextDetector(scorer=classification_scorer)
        metrics = calculate_accuracy(
            detector,
            human_texts=[
                "A normal everyday human written paragraph.",
                "Another human note about daily life.",
            ],
            ai_texts=[
                "This AI generated passage is clearly synthetic.",
                "More AI generated content right here.",
            ],
        )
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["total_samples"] == 4
        assert metrics["correct_predictions"] == 4
