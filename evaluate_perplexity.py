"""
Evaluación simple del detector de IA usando perplejidad.
Carga el dataset Ateeqq/AI-and-Human-Generated-Text y calcula precision y recall.
"""

from datasets import load_dataset
from detector_ia import AITextDetector


def evaluate():
    # Cargar dataset
    print("Cargando dataset...")
    dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")
    
    # Obtener los datos de entrenamiento
    train_data = dataset["train"]
    
    # Crear detector
    print("Inicializando detector...")
    detector = AITextDetector(model_name="gpt2", device="cpu")
    
    # Contadores para métricas
    TP = 0  # Verdaderos positivos: AI detectado como AI
    FP = 0  # Falsos positivos: Humano detectado como AI
    FN = 0  # Falsos negativos: AI detectado como humano
    TN = 0  # Verdaderos negativos: Humano detectado como humano
    
    total = len(train_data)
    
    # Iterar sobre el dataset
    print(f"Evaluando {total} muestras...")
    for i, item in enumerate(train_data):
        text = item["text"]
        true_label = item["label"].lower()  # human o ai
        
        # Analizar con el detector
        result = detector.analyze(text)
        predicted = result.result.value  # human, ai, o uncertain
        
        # Tratar uncertain como humano o AI? Para simplificar, lo contamos como error
        # Opción: considerar uncertain como humano
        if predicted == "uncertain":
            predicted = "human"
        
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
        
        # Mostrar progreso cada 100 muestras
        if (i + 1) % 100 == 0:
            print(f"  Proceso: {i + 1}/{total}")
    
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
