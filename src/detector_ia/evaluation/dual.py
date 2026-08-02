"""
Evaluation of the AI detector using a dual approach: perplexity + burstiness.

Loads the Ateeqq/AI-and-Human-Generated-Text dataset from Hugging Face.
"""

from __future__ import annotations

from ..burstiness_scorer import BurstinessScorer
from ..perplexity_scorer import PerplexityScorer


def classify_dual(
    text: str,
    ppl_scorer: PerplexityScorer,
    burst_scorer: BurstinessScorer,
    ppl_threshold: float,
    burst_threshold: float,
) -> str:
    """
    Classify a text as AI or human using both criteria.

    Both signals must point to AI for an AI verdict:
    - Low perplexity (< ppl_threshold)
    - Low burstiness (< burst_threshold)

    Args:
        text: Text to classify
        ppl_scorer: Perplexity calculator
        burst_scorer: Burstiness calculator
        ppl_threshold: Perplexity threshold for AI
        burst_threshold: Burstiness threshold for AI

    Returns:
        "ai" if likely AI, "human" otherwise
    """
    global_ppl = ppl_scorer.score(text)
    burst_result = burst_scorer.score(text)
    burstiness = burst_result["burstiness"]

    low_ppl = global_ppl < ppl_threshold
    low_burst = burstiness < burst_threshold

    # Both criteria must hold to classify as AI
    if low_ppl and low_burst:
        return "ai"
    return "human"


def find_optimal_thresholds(
    human_texts: list[str],
    ai_texts: list[str],
    ppl_scorer: PerplexityScorer,
    burst_scorer: BurstinessScorer,
    max_samples: int = 100,
) -> tuple[float, float]:
    """
    Find optimal thresholds using percentiles.

    Perplexity: midpoint between the 25th percentile of humans and 75th of AI.
    Burstiness: midpoint between the 75th percentile of humans and 25th of AI.
    """
    import numpy as np

    human_texts = human_texts[:max_samples]
    ai_texts = ai_texts[:max_samples]

    # Compute perplexities
    human_ppls = [ppl_scorer.score(t) for t in human_texts]
    ai_ppls = [ppl_scorer.score(t) for t in ai_texts]

    # Compute burstiness
    human_bursts = [burst_scorer.score(t)["burstiness"] for t in human_texts]
    ai_bursts = [burst_scorer.score(t)["burstiness"] for t in ai_texts]

    # Perplexity threshold: midpoint between 75th pct of humans and 25th of AI
    ppl_human_75 = np.percentile(human_ppls, 75)
    ppl_ai_25 = np.percentile(ai_ppls, 25)
    ppl_threshold = (ppl_human_75 + ppl_ai_25) / 2

    # Burstiness threshold: midpoint between 25th pct of humans and 75th of AI
    burst_human_25 = np.percentile(human_bursts, 25)
    burst_ai_75 = np.percentile(ai_bursts, 75)
    burst_threshold = (burst_human_25 + burst_ai_75) / 2

    print("\nThresholds computed:")
    print(f"  Perplexity: {ppl_threshold:.2f}")
    print(f"  Burstiness: {burst_threshold:.4f}")

    return ppl_threshold, burst_threshold


def evaluate(model_name: str = "distilgpt2", device: str = "cpu") -> None:
    """Run the dual evaluation on the reference dataset."""
    from datasets import load_dataset

    print("Loading dataset Ateeqq/AI-and-Human-Generated-Text...")
    dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")

    train_data = dataset["train"]
    print(f"Dataset loaded: {len(train_data)} samples\n")

    # Initialize scorers with a lightweight model
    print(f"Initializing scorers (model: {model_name})...")
    ppl_scorer = PerplexityScorer(model_name=model_name, device=device)
    burst_scorer = BurstinessScorer(ppl_scorer)

    # Separate human and AI texts to compute thresholds
    human_texts = [item["abstract"] for item in train_data if item["label"] == 0]
    ai_texts = [item["abstract"] for item in train_data if item["label"] == 1]

    # Compute optimal thresholds
    ppl_threshold, burst_threshold = find_optimal_thresholds(
        human_texts, ai_texts, ppl_scorer, burst_scorer
    )

    # Counters for metrics
    TP = 0  # True positives: AI detected as AI
    FP = 0  # False positives: Human detected as AI
    FN = 0  # False negatives: AI detected as human
    TN = 0  # True negatives: Human detected as human

    total = len(train_data)

    # Iterate over the dataset
    print(f"\nEvaluating {total} samples with the dual approach...")
    print("Progress: ", end="", flush=True)

    for i, item in enumerate(train_data):
        text = item["abstract"]
        true_label = "ai" if item["label"] == 1 else "human"

        # Classify using the dual approach
        predicted = classify_dual(
            text, ppl_scorer, burst_scorer, ppl_threshold, burst_threshold
        )

        # Update counters
        if true_label == "ai":
            if predicted == "ai":
                TP += 1
            else:
                FN += 1
        else:  # human
            if predicted == "ai":
                FP += 1
            else:
                TN += 1

        percent = (i + 1) / total * 100
        print(f"\rProgress: {i + 1}/{total} ({percent:.1f}%)", end="", flush=True)

    print("\n")

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total

    print("\n" + "=" * 50)
    print("RESULTS (Dual Approach: Perplexity + Burstiness)")
    print("=" * 50)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")


def main() -> None:
    """Console entry point for the dual evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the AI detector using perplexity + burstiness."
    )
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help='Device: "cpu" or "cuda"')
    args = parser.parse_args()

    evaluate(model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()
