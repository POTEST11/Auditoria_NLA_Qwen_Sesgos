import os
import json
import torch
import numpy as np
import logging
from pathlib import Path
from src.setup import cargar_configuracion
from src.utils import configurar_gpu_y_cargar_modelo

logger = logging.getLogger(__name__)

def extraer_activacion_cuartiles(texto, modelo, tokenizer, capa_objetivo):
    """
    Extrae la activación media de la capa objetivo dividida en 4 cuartiles.
    Retorna: (matriz numpy [4, 3584], n_tokens int)
    """
    almacen = {}

    # Hook para interceptar la salida de la capa específica
    def hook_fn(modulo, entrada, salida):
        almacen['act'] = salida.detach().float().cpu()

    handle = modelo.model.layers[capa_objetivo].register_forward_hook(hook_fn)

    chat = [{'role': 'user', 'content': texto}]
    
    salida_tokenizada = tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True
    )

    input_ids = salida_tokenizada['input_ids'].to('cuda')
    n_tokens = input_ids.shape[1]

    with torch.no_grad():
        _ = modelo(input_ids=input_ids)

    handle.remove()

    tensor = almacen['act'][0]            # Shape: [T, 3584]
    inicio = min(10, n_tokens - 1)        # Saltar tokens con normas anómalas
    tensor_valido = tensor[inicio:]       # Shape: [T_valido, 3584]

    # Dividir secuencialmente en 4 cuartiles
    cuartiles = torch.tensor_split(tensor_valido, 4, dim=0)

    medias_cuartiles = []
    for chunk in cuartiles:
        if chunk.shape[0] > 0:
            medias_cuartiles.append(chunk.mean(dim=0))
        else:
            medias_cuartiles.append(torch.zeros(tensor.shape[1]))

    act_matrix = torch.stack(medias_cuartiles)
    
    return act_matrix.numpy(), n_tokens

def main():
    logger.info("=== INICIANDO FASE DE INFERENCIA Y EXTRACCIÓN ===")
    
    config = cargar_configuracion()
    base_dir = Path(config['entorno']['base_dir'])
    
    ruta_checkpoint = base_dir / config['directorios']['checkpoints'] / 'qwen_sujeto'
    ruta_salida = base_dir / config['directorios']['activaciones']
    ruta_prompts = Path("data/prompts_dataset.json") # Asegúrate de colocar tu JSON aquí
    capa = config['parametros']['capa_objetivo']

    if not ruta_prompts.exists():
        logger.error(f"No se encontró el archivo de prompts en {ruta_prompts}")
        return

    # Cargar Dataset
    with open(ruta_prompts, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    logger.info(f"✓ {len(prompts)} prompts cargados para inferencia.")

    # Cargar Modelo
    modelo, tokenizer = configurar_gpu_y_cargar_modelo(ruta_checkpoint)

    metadatos = []
    errores = []
    total_ops = len(prompts) * 2

    # Loop de Inferencia
    for i, entrada in enumerate(prompts):
        pid = entrada['id']

        for lang, campo in [('es', 'prompt_es'), ('en', 'prompt_en')]:
            op_num = i * 2 + (0 if lang == 'es' else 1) + 1
            texto = entrada[campo]
            
            logger.info(f"Procesando [{op_num:3d}/{total_ops}] {pid}_{lang} ...")

            try:
                act_matrix, n_tok = extraer_activacion_cuartiles(texto, modelo, tokenizer, capa)
                nombre_npy = f'{pid}_{lang}.npy'
                
                np.save(ruta_salida / nombre_npy, act_matrix)

                metadatos.append({
                    'id'                 : pid,
                    'lang'               : lang,
                    'grupo'              : entrada['grupo'],
                    'tema'               : entrada['tema'],
                    'texto'              : texto,
                    'senales_colombianas': entrada.get('senales_colombianas', []),
                    'hipotesis_nla'      : entrada.get('hipotesis_nla', ''),
                    'n_tokens'           : n_tok,
                    'capa'               : capa,
                    'forma_tensor'       : list(act_matrix.shape),
                    'archivo_npy'        : nombre_npy,
                })
            except Exception as e:
                logger.error(f"✗ ERROR procesando {pid}_{lang}: {e}")
                errores.append({'id': pid, 'lang': lang, 'error': str(e)})

    # Guardar metadatos JSON
    ruta_meta = ruta_salida / 'metadatos_activaciones.json'
    with open(ruta_meta, 'w', encoding='utf-8') as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)

    logger.info(f"=== RESUMEN DE INFERENCIA ===")
    logger.info(f"✓ Activaciones guardadas: {len(metadatos)}")
    logger.info(f"✗ Errores: {len(errores)}")
    logger.info(f"✓ Metadatos generados en: {ruta_meta}")

if __name__ == "__main__":
    main()