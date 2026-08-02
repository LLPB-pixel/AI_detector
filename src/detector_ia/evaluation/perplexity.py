"""
Evaluation of the AI detector using perplexity only.

Loads the Ateeqq/AI-and-Human-Generated-Text dataset from Hugging Face.
"""

from __future__ import annotations

from ..ai_detector import AITextDetector


def evaluate(model_name: str = "distilgpt2", device: str = "cpu") -> None:
    """Run the perplexity-only evaluation on the reference dataset."""
    from datasets import load_dataset

    print("Loading dataset Ateeqq/AI-and-Human-Generated-Text...")
    dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")

    train_data = dataset["train"]
    print(f"Dataset loaded: {len(train_data)} samples\n")

    # Create a detector with a lightweight model
    print(f"Initializing detector (model: {model_name})...")
    detector = AITextDetector(model_name=model_name, device=device)

    # Counters for metrics
    TP = 0  # True positives: AI detected as AI
    FP = 0  # False positives: Human detected as AI
    FN = 0  # False negatives: AI detected as human
    TN = 0  # True negatives: Human detected as human

    total = len(train_data)

    # Iterate over the dataset
    print(f"Evaluating {total} samples...")
    print("Progress: ", end="", flush=True)

    for i, item in enumerate(train_data):
        text = item["abstract"]
        true_label = "ai" if item["label"] == 1 else "human"

        # Analyze with the detector
        result = detector.analyze(text)
        predicted = result.result.value  # human, ai, or uncertain

        # Treat uncertain as human
        if predicted == "uncertain":
            predicted = "human"

        # Update counters
        if true_label == "ai":  # label 1 = AI, 0 = human
            if predicted == "ai":
                TP += 1
            else:
                FN += 1
        else:  # human
            if predicted == "ai":
                FP += 1
            else:
                TN += 1

        # Show progress every 10 samples
        if (i + 1) % 10 == 0:
            percent = (i + 1) / total * 100
            print(f"\rProgress: {i + 1}/{total} ({percent:.1f}%)", end="", flush=True)
        if (i + 1) % 100 == 0:
            print("\n" + "=" * 50)
            print("PARTIAL RESULTS")
            print("=" * 50)
            print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
            print(f"Precision:  {TP / (TP + FP):.4f}" if (TP + FP) > 0 else "Precision:  n/a")
            print(f"Recall:     {TP / (TP + FN):.4f}" if (TP + FN) > 0 else "Recall:     n/a")
            print(f"Accuracy:   {(TP + TN) / total:.4f}")

    print("\n")  # New line after progress

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")


def main() -> None:
    """Console entry point for the perplexity-only evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the AI detector using perplexity only."
    )
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help='Device: "cpu" or "cuda"')
    args = parser.parse_args()

    evaluate(model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()
