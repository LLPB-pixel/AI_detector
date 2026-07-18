"""
Evaluación del detector de IA usando enfoque dual: perplejidad + burstiness.
Carga el dataset Ateeqq/AI-and-Human-Generated-Text desde Hugging Face.
"""

from datasets import load_dataset
from perplexity_scorer import PerplexityScorer
from burstiness_scorer import BurstinessScorer


def classify_dual(text: str, ppl_scorer: PerplexityScorer, burst_scorer: BurstinessScorer,
                 ppl_threshold: float, burst_threshold: float) -> str:
    """
    Clasifica texto como AI o humano usando ambos criterios.
    
    Ambas señales deben apuntar a IA:
    - Perplejidad baja (< ppl_threshold)
    - Burstiness bajo (< burst_threshold)
    
    Args:
        text: Texto a clasificar
        ppl_scorer: Calculador de perplejidad
        burst_scorer: Calculador de burstiness
        ppl_threshold: Umbral de perplejidad para IA
        burst_threshold: Umbral de burstiness para IA
        
    Returns:
        "ai" si es probable IA, "human" en caso contrario
    """
    global_ppl = ppl_scorer.score(text)
    burst_result = burst_scorer.score(text)
    burstiness = burst_result["burstiness"]
    
    low_ppl = global_ppl < ppl_threshold
    low_burst = burstiness < burst_threshold
    
    # Ambos criterios deben ser ciertos para clasificar como IA
    if low_ppl and low_burst:
        return "ai"
    else:
        return "human"


def find_optimal_thresholds(human_texts: list, ai_texts: list, 
                            ppl_scorer: PerplexityScorer, burst_scorer: BurstinessScorer):
    """
    Encuentra umbrales óptimos usando percentiles.
    
    Para perplejidad: usar percentil 25 de humanos y 75 de IA
    Para burstiness: usar percentil 75 de humanos y 25 de IA
    """
    import numpy as np
    
    # Calcular perplejidades
    human_ppls = [ppl_scorer.score(t) for t in human_texts[:100]]  # Limitado para velocidad
    ai_ppls = [ppl_scorer.score(t) for t in ai_texts[:100]]
    
    # Calcular burstiness
    human_bursts = [burst_scorer.score(t)["burstiness"] for t in human_texts[:100]]
    ai_bursts = [burst_scorer.score(t)["burstiness"] for t in ai_texts[:100]]
    
    # Umbral de perplejidad: punto medio entre percentil 75 de humanos y 25 de IA
    ppl_human_75 = np.percentile(human_ppls, 75)
    ppl_ai_25 = np.percentile(ai_ppls, 25)
    ppl_threshold = (ppl_human_75 + ppl_ai_25) / 2
    
    # Umbral de burstiness: punto medio entre percentil 25 de humanos y 75 de IA
    burst_human_25 = np.percentile(human_bursts, 25)
    burst_ai_75 = np.percentile(ai_bursts, 75)
    burst_threshold = (burst_human_25 + burst_ai_75) / 2
    
    print(f"\nUmbrales calculados:")
    print(f"  Perplejidad: {ppl_threshold:.2f}")
    print(f"  Burstiness: {burst_threshold:.4f}")
    
    return ppl_threshold, burst_threshold


def evaluate():
    # Cargar dataset de Hugging Face
    print("Cargando dataset Ateeqq/AI-and-Human-Generated-Text...")
    dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")
    
    # Guardar dataset localmente en ./data/
    print("Guardando dataset en ./data/ ...")
    dataset.save_to_disk("./data/ai_human_text")
    
    train_data = dataset["train"]
    print(f"Dataset cargado: {len(train_data)} muestras\n")
    
    # Inicializar calculadores con modelo ligero
    print("Inicializando calculadores (modelo: distilgpt2)...")
    ppl_scorer = PerplexityScorer(model_name="distilgpt2", device="cpu")
    burst_scorer = BurstinessScorer(ppl_scorer)
    
    # Separar textos humanos y AI para calcular umbrales
    human_texts = [item["abstract"] for item in train_data if item["label"] == 0]
    ai_texts = [item["abstract"] for item in train_data if item["label"] == 1]
    
    # Calcular umbrales óptimos
    ppl_threshold, burst_threshold = find_optimal_thresholds(
        human_texts, ai_texts, ppl_scorer, burst_scorer
    )
    
    # Contadores para métricas
    TP = 0  # Verdaderos positivos: AI detectado como AI
    FP = 0  # Falsos positivos: Humano detectado como AI
    FN = 0  # Falsos negativos: AI detectado como humano
    TN = 0  # Verdaderos negativos: Humano detectado como humano
    
    total = len(train_data)
    
    # Iterar sobre el dataset
    print(f"\nEvaluando {total} muestras con enfoque dual...")
    print("Progreso: ", end="", flush=True)
    
    for i, item in enumerate(train_data):
        text = item["abstract"]
        true_label = "ai" if item["label"] == 1 else "human"
        
        # Clasificar usando enfoque dual
        predicted = classify_dual(text, ppl_scorer, burst_scorer, ppl_threshold, burst_threshold)
        
        # Actualizar contadores
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
        print(f"\rProgreso: {i + 1}/{total} ({percent:.1f}%)", end="", flush=True)
    
    print("\n")
    
    # Calcular métricas
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total
    
    print("\n" + "=" * 50)
    print("RESULTADOS (Enfoque Dual: Perplejidad + Burstiness)")
    print("=" * 50)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")


if __name__ == "__main__":
    evaluate()
