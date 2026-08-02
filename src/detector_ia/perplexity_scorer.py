"""
Calculador de perplexity usando modelos de lenguaje.

Los ataques adversariales optimizados automáticamente (GCG, sufijos random)
suelen tener perplejidad muy alta respecto a texto natural, incluso cuando
no contienen ninguna keyword sospechosa.

Este módulo proporciona una implementación optimizada para detección de IA.
"""

from typing import Optional, List, Tuple
import warnings


class PerplexityScorer:
    """
    Calculador de perplexity usando modelos de lenguaje para detección de IA.
    
    Args:
        model_name: Nombre del modelo a usar (default: "distilgpt2" - versión ligera de GPT2)
        device: Dispositivo a usar ("cuda", "cpu", None para auto-detectar)
        max_length: Longitud máxima de tokens a procesar (default: 512)
        batch_size: Tamaño de batch para procesamiento (default: 1)
    """
    
    def __init__(
        self, 
        model_name: str = "distilgpt2", 
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 1
    ):
        # Import diferido: esta clase es opcional y pesada (torch + transformers)
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        except ImportError as e:
            raise ImportError(
                "Perplexity calculation requires transformers and torch. "
                "Install with: pip install torch transformers"
            ) from e
        
        # Configurar dispositivo
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA no está disponible, usando CPU en su lugar.")
            self.device = "cpu"
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.torch = torch
        
        # Cargar tokenizador y modelo
        self._load_model()
    
    def _load_model(self):
        """Carga el tokenizador y modelo."""
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        
        print(f"  Cargando tokenizador '{self.model_name}'...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        print(f"  Cargando modelo '{self.model_name}' en {self.device} (puede tardar)...")
        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name, 
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        print("  Modelo cargado correctamente.")
    
    def score(self, text: str) -> float:
        """
        Calcula la perplexity de un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Perplexity del texto (mayor = más "sorpresivo" para el modelo)
        """
        if not text or not text.strip():
            return 0.0
        
        # Tokenizar el texto
        encodings = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        input_ids = encodings.input_ids
        
        # Texto demasiado corto para evaluar
        if input_ids.shape[1] < 2:
            return 0.0
        
        # Calcular loss (cross-entropy)
        with self.torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
        
        # perplexity = exp(loss); loss ya es la cross-entropy media por token
        perplexity = float(self.torch.exp(outputs.loss))
        
        return perplexity
    
    def batch_score(self, texts: List[str]) -> List[float]:
        """
        Calcula la perplexity para múltiples textos.
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            Lista de perplexity scores
        """
        scores = []
        for text in texts:
            scores.append(self.score(text))
        return scores
    
    def get_model_info(self) -> dict:
        """
        Obtiene información sobre el modelo cargado.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(self.model.dtype)
        }
