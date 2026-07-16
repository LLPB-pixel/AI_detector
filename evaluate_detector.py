#!/usr/bin/env python3
"""
Script para evaluar el detector de IA con un dataset.

Uso:
    python evaluate_detector.py --path <ruta> [OPCIONES]

Ejemplos:
    # Dataset JSON local
    python evaluate_detector.py --path ./data/train.json --format json --limit 100
    
    # Dataset CSV local
    python evaluate_detector.py --path ./data/dataset.csv --format csv
    
    # Dataset de Hugging Face (requiere librería 'datasets')
    python evaluate_detector.py --path Ateeqq/AI-and-Human-Generated-Text --format hf --limit 50

Opciones:
    --path       Ruta al archivo del dataset (requerido)
    --format     Formato del dataset: json, csv, hf (default: json)
    --text-key   Clave para el campo de texto (default: 'text')
    --label-key  Clave para el campo de etiqueta (default: 'label')
    --limit      Número máximo de muestras a evaluar (default: todas)
    --model      Modelo a usar para el detector (default: 'gpt2')
    --device     Dispositivo: cuda, cpu, auto (default: auto)
"""

import argparse
import json
import sys
import os

# Añadir el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_ia import AITextDetector


def load_json_dataset(path):
    """Carga dataset desde archivo JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_dataset(path):
    """Carga dataset desde archivo CSV."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("La librería 'pandas' es necesaria para CSV. Instala con: pip install pandas")
    
    df = pd.read_csv(path)
    return df.to_dict('records')


def load_huggingface_dataset(path):
    """Carga dataset desde Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "La librería 'datasets' es necesaria para Hugging Face. "
            "Instala con: pip install datasets"
        )
    
    dataset = load_dataset(path)
    # Asumir que el dataset tiene un split 'train'
    if 'train' in dataset:
        return dataset['train'].to_list()
    else:
        # Si no tiene split, usar el primero disponible
        first_split = list(dataset.keys())[0]
        return dataset[first_split].to_list()


def evaluate_detector(detector, dataset, text_key='text', label_key='label'):
    """
    Evalúa el detector con un dataset.
    
    Args:
        detector: Instancia de AITextDetector
        dataset: Lista de diccionarios con claves text_key y label_key
        text_key: Clave para el campo de texto
        label_key: Clave para el campo de etiqueta
        
    Returns:
        Lista de diccionarios con los resultados de cada muestra
    """
    results = []
    
    for i, item in enumerate(dataset):
        # Obtener texto
        if text_key not in item:
            print(f"Advertencia: Clave '{text_key}' no encontrada en el item {i}")
            continue
        
        text = str(item[text_key])
        
        # Convertir etiqueta a formato estándar
        if label_key not in item:
            print(f"Advertencia: Clave '{label_key}' no encontrada en el item {i}")
            continue
        
        label = item[label_key]
        
        # Normalizar etiqueta
        if isinstance(label, (int, float)):
            actual = "human" if label == 0 else "ai"
        elif isinstance(label, str):
            actual = label.strip().lower()
            # Normalizar variantes comunes
            if actual in ['human', 'humano', 'real', 'true', '1']:
                actual = "human"
            elif actual in ['ai', 'ia', 'artificial', 'fake', 'false', '0']:
                actual = "ai"
            else:
                print(f"Advertencia: Etiqueta desconocida '{label}' en item {i}")
                continue
        else:
            actual = str(label).strip().lower()
            if actual not in ['human', 'ai']:
                print(f"Advertencia: Etiqueta desconocida '{label}' en item {i}")
                continue
        
        # Analizar con el detector
        try:
            analysis = detector.analyze(text)
            predicted = analysis.result.value
            
            # Verificar si es correcto
            is_correct = actual == predicted
            
            results.append({
                'index': i,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'actual': actual,
                'predicted': predicted,
                'correct': is_correct,
                'confidence': analysis.confidence,
                'perplexity': analysis.perplexity,
                'log_perplexity': analysis.log_perplexity
            })
        except Exception as e:
            print(f"Error analizando item {i}: {e}")
            continue
    
    return results


def calculate_metrics(results):
    """
    Calcula métricas de evaluación.
    
    Args:
        results: Lista de resultados de evaluación
        
    Returns:
        Diccionario con métricas
    """
    if not results:
        return {
            'accuracy': 0,
            'human_accuracy': 0,
            'ai_accuracy': 0,
            'total': 0,
            'correct': 0,
            'human_total': 0,
            'ai_total': 0
        }
    
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Métricas por clase
    human_results = [r for r in results if r['actual'] == 'human']
    ai_results = [r for r in results if r['actual'] == 'ai']
    
    human_correct = sum(1 for r in human_results if r['correct'])
    ai_correct = sum(1 for r in ai_results if r['correct'])
    
    # Calcular perplejidad promedio por clase
    human_perplexities = [r['perplexity'] for r in human_results]
    ai_perplexities = [r['perplexity'] for r in ai_results]
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'human_accuracy': human_correct / len(human_results) if human_results else 0,
        'ai_accuracy': ai_correct / len(ai_results) if ai_results else 0,
        'total': total,
        'correct': correct,
        'human_total': len(human_results),
        'ai_total': len(ai_results),
        'avg_perplexity_human': sum(human_perplexities) / len(human_perplexities) if human_perplexities else 0,
        'avg_perplexity_ai': sum(ai_perplexities) / len(ai_perplexities) if ai_perplexities else 0
    }


def print_results(results, metrics, verbose=False):
    """Imprime los resultados de evaluación."""
    print("\n" + "=" * 70)
    print(" " * 20 + "RESULTADOS DE EVALUACIÓN")
    print("=" * 70)
    
    print(f"\n📊 MÉTRICAS GLOBALES:")
    print("-" * 40)
    print(f"  Precisión total:    {metrics['accuracy']:.2%}")
    print(f"  Precisión humanos:  {metrics['human_accuracy']:.2%} ({metrics['human_total']} muestras)")
    print(f"  Precisión IA:       {metrics['ai_accuracy']:.2%} ({metrics['ai_total']} muestras)")
    print(f"  Total:              {metrics['total']} muestras analizadas")
    print(f"  Correctas:          {metrics['correct']}")
    print(f"  Error rate:        {1 - metrics['accuracy']:.2%}")
    
    print(f"\n📈 PERPLEJIDAD PROMEDIO:")
    print("-" * 40)
    print(f"  Humano:  {metrics['avg_perplexity_human']:.2f}")
    print(f"  IA:      {metrics['avg_perplexity_ai']:.2f}")
    
    if verbose and results:
        print(f"\n📝 EJEMPLOS DETALLADOS ({min(10, len(results))} primeros):")
        print("-" * 70)
        print(f"{'#':<4} {'Real':<8} {'Pred':<8} {'✓':<4} {'Conf':<8} {'Pplx':<10}")
        print("-" * 70)
        
        for result in results[:10]:
            status = "✓" if result['correct'] else "✗"
            print(f"{result['index']:<4} {result['actual']:<8} {result['predicted']:<8} "
                  f"{status:<4} {result['confidence']:.2%} {result['perplexity']:<10.2f}")
    
    if not verbose and results:
        print(f"\n📝 MUESTRA DE RESULTADOS (5 primeros):")
        print("-" * 70)
        for result in results[:5]:
            status = "✓" if result['correct'] else "✗"
            print(f"  {status} Real: {result['actual']:8} | Pred: {result['predicted']:8} | "
                  f"Conf: {result['confidence']:.2%} | Pplx: {result['perplexity']:.2f}")
    
    print("=" * 70)


def save_results(results, metrics, output_path):
    """Guarda los resultados en un archivo JSON."""
    output = {
        'metrics': metrics,
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Resultados guardados en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar detector de IA con dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--path', type=str, required=True,
                        help='Ruta al archivo del dataset o nombre del dataset en HF')
    parser.add_argument('--format', type=str, default='json',
                        choices=['json', 'csv', 'hf'],
                        help='Formato del dataset (default: json)')
    parser.add_argument('--text-key', type=str, default='text',
                        help='Clave para el campo de texto (default: text)')
    parser.add_argument('--label-key', type=str, default='label',
                        help='Clave para el campo de etiqueta (default: label)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Número máximo de muestras a evaluar (default: todas)')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='Modelo a usar para el detector (default: gpt2)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Dispositivo: cuda, cpu (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Archivo JSON para guardar resultados')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar resultados detallados')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  DETECTOR DE IA - EVALUACIÓN CON DATASET")
    print("=" * 70)
    
    # Cargar dataset
    print(f"\n📂 Cargando dataset desde: {args.path}")
    print(f"   Formato: {args.format}")
    
    try:
        if args.format == 'json':
            dataset = load_json_dataset(args.path)
        elif args.format == 'csv':
            dataset = load_csv_dataset(args.path)
        elif args.format == 'hf':
            dataset = load_huggingface_dataset(args.path)
        else:
            raise ValueError(f"Formato no soportado: {args.format}")
        
        print(f"   ✅ Dataset cargado: {len(dataset)} muestras")
        
    except FileNotFoundError:
        print(f"   ❌ Error: Archivo no encontrado: {args.path}")
        print("   Asegúrate de que la ruta es correcta.")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error cargando dataset: {e}")
        sys.exit(1)
    
    # Limitar muestras
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"   📊 Usando: {len(dataset)} muestras (límite aplicado)")
    
    # Crear detector
    print(f"\n🤖 Creando detector...")
    print(f"   Modelo: {args.model}")
    if args.device:
        print(f"   Dispositivo: {args.device}")
    
    try:
        detector = AITextDetector(
            model_name=args.model,
            device=args.device
        )
        print("   ✅ Detector creado")
        
    except ImportError as e:
        print(f"   ❌ Error: {e}")
        print("\n   Para instalar las dependencias:")
        print("   pip install torch transformers")
        sys.exit(1)
    
    # Evaluar
    print(f"\n🔍 Evaluando {len(dataset)} muestras...")
    print("   Esto puede tardar unos minutos...")
    
    results = evaluate_detector(
        detector,
        dataset,
        text_key=args.text_key,
        label_key=args.label_key
    )
    
    print(f"   ✅ Análisis completado: {len(results)} resultados")
    
    # Calcular métricas
    metrics = calculate_metrics(results)
    
    # Mostrar resultados
    print_results(results, metrics, verbose=args.verbose)
    
    # Guardar resultados si se especifica
    if args.output:
        save_results(results, metrics, args.output)
    
    # Retornar código de salida
    if metrics['accuracy'] >= 0.7:
        print("\n✅ El detector tiene buen rendimiento en este dataset")
    elif metrics['accuracy'] >= 0.5:
        print("\n⚠️  El detector tiene rendimiento moderado")
    else:
        print("\n❌ El detector necesita mejora para este dataset")


if __name__ == '__main__':
    main()
