import csv
import re
import yaml
import torch
import numpy as np
import logging
from collections import defaultdict
from pathlib import Path
from safetensors.torch import load_file
from src.setup import cargar_configuracion
from src.utils import configurar_gpu_y_cargar_modelo

logger = logging.getLogger(__name__)

def cargar_config_ar(ruta_checkpoint_ar):
    """Lee el archivo nla_meta.yaml específico del Autoencoder Reconstructor."""
    ruta_yaml = ruta_checkpoint_ar / 'nla_meta.yaml'
    with open(ruta_yaml, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def anular_layernorm_final(modelo):
    """
    El AR opera sobre la salida raw del bloque. Esta función busca y anula
    la capa de normalización final reemplazándola con la función de Identidad.
    """
    inner = modelo.model
    for attr in ('norm', 'final_layernorm', 'ln_f'):
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity())
            logger.info(f"✓ LayerNorm final ({attr}) anulada exitosamente.")
            return
    logger.warning("No se encontró LayerNorm final para anular.")

def cargar_value_head(modelo, ruta_checkpoint_ar):
    """Carga y acopla la capa lineal 'value_head' al modelo."""
    d = modelo.config.hidden_size
    value_head = torch.nn.Linear(d, d, bias=False, dtype=torch.float32)
    ruta_value_head = ruta_checkpoint_ar / 'value_head.safetensors'
    
    if not ruta_value_head.exists():
        raise FileNotFoundError(f"No se encontró {ruta_value_head}")

    value_head.load_state_dict(load_file(str(ruta_value_head)))
    device = next(modelo.parameters()).device
    return value_head.to(device).eval()

@torch.inference_mode()
def reconstruir(descripcion, modelo, tokenizer, value_head, ar_template):
    """Toma un texto explicativo y retorna el vector reconstruido [3584]."""
    device = next(modelo.parameters()).device
    prompt = ar_template.format(explanation=descripcion)
    
    ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'].to(device)
    h = modelo.model(ids, use_cache=False).last_hidden_state
    
    # Extraemos el último token y lo pasamos por la cabeza de valor
    pred = value_head(h[0, -1].float()).cpu()
    return pred

def calcular_metricas(descripcion, v_raw_np, mse_scale, modelo, tokenizer, value_head, ar_template):
    """Calcula y retorna (MSE, Similitud_Coseno)."""
    pred = reconstruir(descripcion, modelo, tokenizer, value_head, ar_template)
    gold = torch.as_tensor(v_raw_np, dtype=torch.float32)

    # Normalización matemática requerida por la arquitectura
    pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
    gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale

    mse = ((pred_n - gold_n) ** 2).mean().item()
    cos = torch.nn.functional.cosine_similarity(pred.unsqueeze(0), gold.unsqueeze(0)).item()
    return mse, cos

def interpretar(cos):
    """Categoriza cualitativamente la similitud coseno."""
    if cos >= 0.9:    return 'EXCELENTE'
    elif cos >= 0.75: return 'BUENO'
    elif cos >= 0.5:  return 'MEDIOCRE'
    else:             return 'POBRE'

def main():
    logger.info("=== INICIANDO FASE DE RECONSTRUCCIÓN Y EVALUACIÓN NLA ===")

    config = cargar_configuracion()
    base_dir = Path(config['entorno']['base_dir'])

    ruta_checkpoint_ar = base_dir / config['directorios']['checkpoints'] / 'nla_ar'
    dir_act = base_dir / config['directorios']['activaciones']
    csv_verb = base_dir / config['directorios']['verbalizaciones'] / 'explicaciones_nla.csv'
    csv_salida = base_dir / config['directorios']['resultados'] / 'resultados_nla.csv'

    # 1. Validar archivos de entrada
    if not csv_verb.exists():
        logger.error(f"No se encontró el archivo CSV de verbalizaciones en {csv_verb}")
        return

    # 2. Cargar configuración NLA
    ar_meta = cargar_config_ar(ruta_checkpoint_ar)
    mse_scale = ar_meta['extraction']['mse_scale']
    ar_template = ar_meta['prompt_templates']['ar']

    logger.info(f"Escala MSE configurada: {mse_scale:.4f}")

    # 3. Preparar Modelo Reconstructor
    ar_backbone, tokenizer = configurar_gpu_y_cargar_modelo(ruta_checkpoint_ar)
    anular_layernorm_final(ar_backbone)
    value_head = cargar_value_head(ar_backbone, ruta_checkpoint_ar)

    # 4. Leer Explicaciones
    with open(csv_verb, encoding='utf-8') as f:
        filas_verb = list(csv.DictReader(f))
    logger.info(f"✓ {len(filas_verb)} entradas cargadas desde el Verbalizer.")

    columnas_csv = [
        'id', 'lang', 'grupo', 'tema', 'texto',
        'senales_colombianas', 'hipotesis_nla',
        'explicacion_nla', 'cos_sim_mean', 'mse_mean', 'fidelidad'
    ]
    resultados = []
    errores = []
    patron_cuartiles = re.compile(r'Cuartil \d+:\s*(.*?)(?=(?:\n\nCuartil \d+:|$))', re.DOTALL)
    total = len(filas_verb)

    # 5. Ejecutar Evaluación
    for i, fila in enumerate(filas_verb):
        pid = fila['id']
        lang = fila['lang']
        logger.info(f"Reconstruyendo [{i+1:3d}/{total}] {pid}_{lang} ...")

        try:
            ruta_npy = dir_act / f"{pid}_{lang}.npy"
            v_raw = np.load(ruta_npy)
            explicaciones = patron_cuartiles.findall(fila['explicacion_nla'])

            if len(explicaciones) != 4:
                raise ValueError(f"Se esperaban 4 cuartiles, se encontraron {len(explicaciones)}")

            cos_sims, mses = [], []

            for j in range(4):
                mse_q, cos_q = calcular_metricas(
                    explicaciones[j], v_raw[j], mse_scale, ar_backbone, tokenizer, value_head, ar_template
                )
                mses.append(mse_q)
                cos_sims.append(cos_q)

            cos_mean = np.mean(cos_sims)
            mse_mean = np.mean(mses)

            resultados.append({
                **fila, # Copiamos la data anterior (id, lang, grupo, etc)
                'cos_sim_mean' : round(float(cos_mean), 4),
                'mse_mean'     : round(float(mse_mean), 4),
                'fidelidad'    : interpretar(cos_mean),
            })

            logger.info(f"  ✓ cos_mean={cos_mean:.3f} | mse_mean={mse_mean:.3f} | {interpretar(cos_mean)}")

        except Exception as e:
            logger.error(f"  ✗ ERROR procesando {pid}_{lang}: {e}")
            errores.append({'id': pid, 'lang': lang, 'error': str(e)})

    # 6. Guardar Resultados en CSV
    with open(csv_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas_csv)
        writer.writeheader()
        writer.writerows(resultados)

    # 7. Resumen Estadístico
    por_grupo_cos = defaultdict(list)
    por_grupo_mse = defaultdict(list)

    for r in resultados:
        por_grupo_cos[r['grupo']].append(r['cos_sim_mean'])
        por_grupo_mse[r['grupo']].append(r['mse_mean'])

    logger.info("=== RESUMEN ESTADÍSTICO POR GRUPO ===")
    for grupo in sorted(por_grupo_cos.keys()):
        vals_cos = por_grupo_cos[grupo]
        vals_mse = por_grupo_mse[grupo]
        logger.info(f"Grupo: {grupo} (n={len(vals_cos)})")
        logger.info(f"  COS -> media={sum(vals_cos)/len(vals_cos):.4f} | max={max(vals_cos):.4f} | min={min(vals_cos):.4f}")
        logger.info(f"  MSE -> media={sum(vals_mse)/len(vals_mse):.4f} | max={max(vals_mse):.4f} | min={min(vals_mse):.4f}")

    logger.info("=== PIPELINE FINALIZADO ===")
    logger.info(f"✓ Resultados guardados: {len(resultados)}")
    logger.info(f"✗ Errores: {len(errores)}")
    logger.info(f"✓ Archivo final exportado a: {csv_salida}")

if __name__ == "__main__":
    main()