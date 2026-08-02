"""Tests for AITextDetector classification and behavior."""

from __future__ import annotations

from detector_ia import AITextDetector, DetectionResult

LONG_TEXT = "This is a sufficiently long piece of text to be analyzed properly."


def make_detector(fake_scorer) -> AITextDetector:
    return AITextDetector(scorer=fake_scorer)


class TestAnalyze:
    def test_empty_text(self, fake_scorer):
        detector = make_detector(fake_scorer(lambda t: 0.0))
        analysis = detector.analyze("   ")
        assert analysis.result == DetectionResult.UNCERTAIN
        assert analysis.is_empty
        assert analysis.confidence == 0.0

    def test_too_short(self, fake_scorer):
        detector = make_detector(fake_scorer(lambda t: 0.0))
        analysis = detector.analyze("hi")
        assert analysis.result == DetectionResult.UNCERTAIN
        assert analysis.is_too_short
        assert analysis.confidence == 0.0

    def test_human_classification(self, classification_scorer):
        detector = make_detector(classification_scorer)
        analysis = detector.analyze(LONG_TEXT)
        assert analysis.result == DetectionResult.HUMAN
        assert 0.0 <= analysis.confidence <= 1.0

    def test_ai_classification(self, classification_scorer):
        detector = make_detector(classification_scorer)
        analysis = detector.analyze("This text is clearly AI generated for sure.")
        assert analysis.result == DetectionResult.AI
        assert 0.0 <= analysis.confidence <= 1.0

    def test_intermediate_perplexity_is_uncertain(self, fake_scorer):
        detector = make_detector(fake_scorer(lambda t: 25.0))
        analysis = detector.analyze(LONG_TEXT)
        assert analysis.result == DetectionResult.UNCERTAIN

    def test_batch_analyze(self, classification_scorer):
        detector = make_detector(classification_scorer)
        results = detector.batch_analyze([LONG_TEXT, "Text that is AI generated here."])
        assert len(results) == 2
        assert results[0].result == DetectionResult.HUMAN
        assert results[1].result == DetectionResult.AI

    def test_analysis_history(self, classification_scorer):
        detector = make_detector(classification_scorer)
        detector.analyze(LONG_TEXT)
        assert len(detector.analysis_history) == 1

    def test_to_dict(self, classification_scorer, human_analysis):
        data = human_analysis.to_dict()
        assert data["result"] == "human"
        assert "text" in data
        assert "explanation" in data


class TestLazyLoading:
    def test_no_model_loaded_until_first_analyze(self):
        detector = AITextDetector()
        assert detector.scorer is None

    def test_empty_text_does_not_load_model(self):
        detector = AITextDetector()
        analysis = detector.analyze("")
        assert analysis.is_empty
        assert detector.scorer is None


class TestStatistics:
    def test_no_history(self, fake_scorer):
        detector = make_detector(fake_scorer(lambda t: 3.0))
        assert detector.get_statistics() == {"total": 0}

    def test_statistics_counts(self, classification_scorer):
        detector = make_detector(classification_scorer)
        detector.analyze(LONG_TEXT)  # human
        detector.analyze("AI generated text example here.")  # ai
        stats = detector.get_statistics()
        assert stats["total"] == 2
        assert stats["valid"] == 2
        assert stats["human_detections"] == 1
        assert stats["ai_detections"] == 1
        assert stats["uncertain_detections"] == 0


class TestThresholds:
    def test_set_thresholds(self, fake_scorer):
        detector = make_detector(fake_scorer(lambda t: 3.0))
        detector.set_thresholds(ai_threshold=80.0, human_threshold=5.0)
        assert detector.ai_threshold == 80.0
        assert detector.human_threshold == 5.0

    def test_calibrate(self, fake_scorer):
        scoring = {
            "human one": 3.0,
            "human two": 5.0,
            "ai one": 190.0,
            "ai two": 210.0,
        }
        detector = make_detector(fake_scorer(lambda t: scoring.get(t, 100.0)))
        result = detector.calibrate(
            human_texts=["human one", "human two"],
            ai_texts=["ai one", "ai two"],
        )
        assert "new_ai_threshold" in result
        assert "new_human_threshold" in result
        assert result["calibration_samples"] == 4
        assert detector.ai_threshold > detector.human_threshold

    def test_calibrate_requires_both_groups(self, fake_scorer):
        detector = make_detector(fake_scorer(lambda t: 3.0))
        result = detector.calibrate(human_texts=["x"], ai_texts=[])
        assert "error" in result

    def test_model_info_delegates_to_scorer(self, classification_scorer):
        detector = make_detector(classification_scorer)
        info = detector.get_model_info()
        assert info["model_name"] == "fake"
