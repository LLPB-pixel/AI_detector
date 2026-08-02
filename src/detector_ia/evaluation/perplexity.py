"""
Evaluación simple del detector de IA usando perplejidad.
Carga el dataset Ateeqq/AI-and-Human-Generated-Text desde Hugging Face.
"""

from datasets import load_dataset
from ai_detector import AITextDetector


def evaluate():
    # Cargar dataset de Hugging Face
    print("Cargando dataset Ateeqq/AI-and-Human-Generated-Text...")
    dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")
    
    # Guardar dataset localmente en ./data/
    print("Guardando dataset en ./data/ ...")
    dataset.save_to_disk("./data/ai_human_text")
    
    train_data = dataset["train"]
    print(f"Dataset cargado: {len(train_data)} muestras\n")
    
    # Crear detector con modelo ligero
    print("Inicializando detector (modelo: distilgpt2)...")
    detector = AITextDetector(model_name="distilgpt2", device="cpu")
    
    # Contadores para métricas
    TP = 0  # Verdaderos positivos: AI detectado como AI
    FP = 0  # Falsos positivos: Humano detectado como AI
    FN = 0  # Falsos negativos: AI detectado como humano
    TN = 0  # Verdaderos negativos: Humano detectado como humano
    
    total = len(train_data)
    
    # Iterar sobre el dataset
    print(f"Evaluando {total} muestras...")
    print("Progreso: ", end="", flush=True)
    
    for i, item in enumerate(train_data):
        text = item["abstract"]
        true_label = "ai" if item["label"] == 1 else "human"
        
        # Analizar con el detector
        result = detector.analyze(text)
        predicted = result.result.value  # human, ai, o uncertain
        
        # Tratar uncertain como humano
        if predicted == "uncertain":
            predicted = "human"
        
        # Actualizar contadores
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
        
        # Mostrar progreso cada 10 muestras
        if (i + 1) % 10 == 0:
            percent = (i + 1) / total * 100
            print(f"\rProgreso: {i + 1}/{total} ({percent:.1f}%)", end="", flush=True)
        if (i + 1) % 100 == 0:
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (TP + TN) / total
            print("\n" + "=" * 50)
            print("RESULTADOS PARCIALES")
            print("=" * 50)
            print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
            print(f"Precision:  {precision:.4f}")
            print(f"Recall:     {recall:.4f}")
            print(f"F1-score:   {f1:.4f}")
            print(f"Accuracy:   {accuracy:.4f}")
    
    print("\n")  # Nueva línea después del progreso
    
    # Calcular métricas
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / total
    
    print("\n" + "=" * 50)
    print("RESULTADOS")
    print("=" * 50)
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-score:   {f1:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")


if __name__ == "__main__":
    evaluate()
