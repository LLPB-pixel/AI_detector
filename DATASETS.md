# Datasets para Testing del Detector de IA

Este documento lista datasets públicos adecuados para probar y evaluar el detector de IA basado en perplejidad.

## 📋 Datasets Recomendados

### 1. **Ateeqq/AI-and-Human-Generated-Text** ⭐ *(Recomendado)*

**Descripción:** Dataset con textos humanos y generados por IA (GPT-4, BARD) en diversos géneros como ensayos, historias, poesía y código Python.

**Tamaño:** Pequeño, ideal para experimentación

**Formato:** Textos con etiquetas (human/AI)

**Enlace:** https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text

**Cómo descargar manualmente:**
```bash
# Requiere la librería datasets de Hugging Face
pip install datasets

# Descargar desde Python
from datasets import load_dataset

dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")

# Guardar localmente (opcional)
dataset.save_to_disk("./data/ai_human_text")
```

**Uso con el detector:**
```python
from detector_ia import AITextDetector
from datasets import load_dataset

detector = AITextDetector()
dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")

# Analizar algunos ejemplos
human_texts = dataset["train"]["text"][:10]  # Primeros 10 humanos
ai_texts = dataset["train"]["text"][-10:]   # Últimos 10 AI

for text in human_texts:
    result = detector.analyze(text)
    print(f"Humano: {result.result.value}, Conf: {result.confidence:.2%}")
```

---

### 2. **SuperAnnotate/ai-detector**

**Descripción:** 44,000 pares de texto (22k humanos + 22k AI), balanceado para clasificación binaria.

**Tamaño:** Medio (44k muestras)

**Formato:** Textos con etiquetas binarias

**Enlace:** https://huggingface.co/datasets/SuperAnnotate/ai-detector

**Cómo descargar:**
```bash
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("SuperAnnotate/ai-detector")
# dataset["train"] contiene los datos
```

---

### 3. **Human-vs-AI-Text** (GitHub)

**Descripción:** 4,000 párrafos de entrenamiento (2,016 humanos + 1,984 AI) y 4,000 de test. Diseñado para clasificación binaria.

**Tamaño:** Pequeño-medio (8k textos totales)

**Formato:** JSON

**Enlace:** https://github.com/NicolasHHH/Human-vs-AI-Text

**Cómo descargar manualmente:**
```bash
# Clonar el repositorio
git clone https://github.com/NicolasHHH/Human-vs-AI-Text.git

# Los datos están en el repositorio
# - train.json: Datos de entrenamiento
# - test.json: Datos de test
```

**Estructura del JSON:**
```json
{
  "text": "El texto aquí...",
  "label": 0  // 0 = humano, 1 = AI
}
```

**Uso con el detector:**
```python
import json
from detector_ia import AITextDetector

detector = AITextDetector()

# Cargar datos
with open("Human-vs-AI-Text/train.json", "r") as f:
    data = json.load(f)

# Analizar muestras
for item in data[:20]:
    result = detector.analyze(item["text"])
    actual_label = "human" if item["label"] == 0 else "ai"
    predicted = result.result.value
    print(f"Real: {actual_label}, Predicho: {predicted}, Conf: {result.confidence:.2%}")
```

---

### 4. **An Applied Statistics Dataset** (GitHub)

**Descripción:** 200 respuestas (100 humanos + 100 AI/ChatGPT) a preguntas de estadística aplicada. Dataset específico de dominio.

**Tamaño:** Muy pequeño (200 muestras)

**Formato:** JSON/CSV

**Enlace:** https://github.com/shahidul034/An-Applied-Statistics-Dataset-for-Human-vs-AI-Generated-Answer-Classification

**Ventaja:** Muy pequeño, ideal para pruebas rápidas

**Cómo descargar:**
```bash
git clone https://github.com/shahidul034/An-Applied-Statistics-Dataset-for-Human-vs-AI-Generated-Answer-Classification.git
```

---

### 5. **idajikuu/AI-detection** (Hugging Face)

**Descripción:** Dataset enfocado en detección de texto generado por IA.

**Enlace:** https://huggingface.co/datasets/idajikuu/AI-detection

---

### 6. **AI-GA Dataset** (GitHub)

**Descripción:** Dataset académico para benchmarking de texto humano vs AI (GPT-3, GPT-4, BARD).

**Enlace:** https://github.com/panagiotisanagnostou/AI-GA

---

## 📥 Cómo Descargar Datasets (Manual)

### Para datasets de Hugging Face:

1. **Navega al enlace** del dataset en Hugging Face
2. **Haz clic en "Files"** para ver los archivos
3. **Descarga manualmente** los archivos CSV/JSON/Parquet
4. **Colócalos** en tu proyecto, por ejemplo en `./data/`

### Para datasets de GitHub:

1. **Abre el enlace** del repositorio
2. **Haz clic en "Code" → "Download ZIP"**
3. **Extrae el ZIP** en tu proyecto
4. **Usa los archivos** directamente

---

## 🔧 Script para Evaluar con un Dataset

Aquí tienes un script completo para evaluar el detector con cualquier dataset:

```python
"""
Script para evaluar el detector de IA con un dataset.

Uso:
    python evaluate_detector.py --dataset <nombre_dataset> --path <ruta>
"""

import argparse
import json
import csv
from pathlib import Path
from detector_ia import AITextDetector


def load_json_dataset(path):
    """Carga dataset desde JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_dataset(path):
    """Carga dataset desde CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    # Asumir columnas: 'text' y 'label'
    return df.to_dict('records')


def evaluate_detector(detector, dataset, text_key='text', label_key='label'):
    """Evalúa el detector con un dataset."""
    results = []
    
    for item in dataset:
        text = item[text_key]
        
        # Convertir etiqueta a formato estándar
        label = item[label_key]
        if isinstance(label, (int, float)):
            actual = "human" if label == 0 else "ai"
        elif isinstance(label, str):
            actual = label.lower()
        else:
            actual = str(label).lower()
        
        # Analizar con el detector
        analysis = detector.analyze(text)
        predicted = analysis.result.value
        
        # Verificar si es correcto
        is_correct = actual == predicted
        
        results.append({
            'text': text[:100] + '...',
            'actual': actual,
            'predicted': predicted,
            'correct': is_correct,
            'confidence': analysis.confidence,
            'perplexity': analysis.perplexity
        })
    
    return results


def calculate_metrics(results):
    """Calcula métricas de evaluación."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Métricas por clase
    human_correct = sum(1 for r in results if r['actual'] == 'human' and r['correct'])
    human_total = sum(1 for r in results if r['actual'] == 'human')
    
    ai_correct = sum(1 for r in results if r['actual'] == 'ai' and r['correct'])
    ai_total = sum(1 for r in results if r['actual'] == 'ai')
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'human_accuracy': human_correct / human_total if human_total > 0 else 0,
        'ai_accuracy': ai_correct / ai_total if ai_total > 0 else 0,
        'total': total,
        'correct': correct
    }


def print_results(results, metrics):
    """Imprime los resultados de evaluación."""
    print("=" * 60)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 60)
    print(f"\nPrecisión total: {metrics['accuracy']:.2%}")
    print(f"Precisión en humanos: {metrics['human_accuracy']:.2%}")
    print(f"Precisión en IA: {metrics['ai_accuracy']:.2%}")
    print(f"Total: {metrics['total']} muestras, {metrics['correct']} correctas")
    
    print("\n" + "-" * 60)
    print("EJEMPLOS:")
    print("-" * 60)
    
    # Mostrar algunos ejemplos
    for result in results[:5]:
        status = "✓" if result['correct'] else "✗"
        print(f"{status} Real: {result['actual']:8} | Pred: {result['predicted']:8} | "
              f"Conf: {result['confidence']:.2%} | Pplx: {result['perplexity']:.2f}")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluar detector de IA con dataset')
    parser.add_argument('--path', type=str, required=True, 
                        help='Ruta al archivo del dataset')
    parser.add_argument('--format', type=str, default='json',
                        choices=['json', 'csv', 'hf'],
                        help='Formato del dataset (json, csv, hf)')
    parser.add_argument('--text-key', type=str, default='text',
                        help='Clave para el texto')
    parser.add_argument('--label-key', type=str, default='label',
                        help='Clave para la etiqueta')
    parser.add_argument('--limit', type=int, default=None,
                        help='Límite de muestras a evaluar')
    
    args = parser.parse_args()
    
    # Cargar dataset
    print(f"Cargando dataset desde {args.path}...")
    
    if args.format == 'json':
        dataset = load_json_dataset(args.path)
    elif args.format == 'csv':
        dataset = load_csv_dataset(args.path)
    elif args.format == 'hf':
        from datasets import load_dataset
        dataset = load_dataset(args.path)
        dataset = dataset['train'].to_list()
    else:
        raise ValueError(f"Formato no soportado: {args.format}")
    
    # Limitar muestras
    if args.limit:
        dataset = dataset[:args.limit]
    
    print(f"Dataset cargado: {len(dataset)} muestras")
    
    # Crear detector
    print("Creando detector...")
    detector = AITextDetector(device='cpu')
    
    # Evaluar
    print("Evaluando...")
    results = evaluate_detector(
        detector, 
        dataset, 
        text_key=args.text_key,
        label_key=args.label_key
    )
    
    # Calcular métricas
    metrics = calculate_metrics(results)
    
    # Mostrar resultados
    print_results(results, metrics)


if __name__ == '__main__':
    main()
```

**Guarda este script como `evaluate_detector.py` y úsalo así:**
```bash
# Para dataset JSON
python evaluate_detector.py --path ./data/dataset.json --format json

# Para dataset CSV
python evaluate_detector.py --path ./data/dataset.csv --format csv

# Para dataset de Hugging Face (necesita librería datasets)
python evaluate_detector.py --path Ateeqq/AI-and-Human-Generated-Text --format hf --limit 100
```

---

## 📊 Ejemplo de Testing con el Dataset

Si descargas el dataset **Human-vs-AI-Text**:

```bash
# 1. Descargar el dataset
git clone https://github.com/NicolasHHH/Human-vs-AI-Text.git

# 2. Crear directorio de datos
mkdir -p data

# 3. Copiar los archivos JSON a tu proyecto
cp Human-vs-AI-Text/*.json data/

# 4. Evaluar con 100 muestras
python evaluate_detector.py --path data/train.json --format json --limit 100
```

---

## 🎯 Recomendación

Para **empezar rápido**, te recomiendo:

1. **Dataset más pequeño:** `shahidul034/An-Applied-Statistics-Dataset` (200 muestras)
2. **Dataset equilibrado:** `Ateeqq/AI-and-Human-Generated-Text` (tamaño pequeño, bien etiquetado)
3. **Dataset completo:** `SuperAnnotate/ai-detector` (44k muestras, para evaluación serria)

---

## ⚠️ Notas de Seguridad

- **No descargues datasets de fuentes no confiables**
- **Revisa siempre el contenido** antes de usarlo
- **Verifica las licencias** de los datasets
- **Los datasets mencionados** son de repositorios públicos y académicos de confianza

---

## 📚 Recursos Adicionales

- [Hugging Face Datasets](https://huggingface.co/datasets) - Miles de datasets públicos
- [Kaggle Datasets](https://www.kaggle.com/datasets) - Busca "AI generated text"
- [Papers With Code - Text Generation](https://paperswithcode.com/task/text-generation) - Datasets académicos
