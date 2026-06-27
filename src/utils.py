import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)

def configurar_gpu_y_cargar_modelo(ruta_checkpoint):
    """
    Evalúa la VRAM disponible y carga el modelo en 8-bit (T4) o bfloat16 (Pro).
    Retorna: (modelo, tokenizer)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó GPU. El pipeline requiere CUDA.")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    modo_8bit = vram_gb < 20

    logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    logger.info(f"Cargando modelo en modo {'8-bit' if modo_8bit else 'bfloat16'}...")

    tokenizer = AutoTokenizer.from_pretrained(ruta_checkpoint, trust_remote_code=True)

    if modo_8bit:
        modelo = AutoModelForCausalLM.from_pretrained(
            ruta_checkpoint,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        modelo = AutoModelForCausalLM.from_pretrained(
            ruta_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map='cuda:0',
            trust_remote_code=True,
        )

    modelo.eval()
    logger.info(f"✓ Modelo cargado exitosamente. Capas ocultas: {modelo.config.num_hidden_layers}")
    
    return modelo, tokenizer