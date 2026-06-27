import os
import json
import csv
import yaml
import re
import torch
import numpy as np
import logging
from pathlib import Path
from safetensors import safe_open
from src.setup import cargar_configuracion
from src.utils import configurar_gpu_y_cargar_modelo

logger = logging.getLogger(__name__)

def cargar_config_nla(ruta_checkpoint_av):
    """Lee el archivo sidecar nla_meta.yaml que contiene los parámetros de inyección."""
    ruta_yaml = ruta_checkpoint_av / 'nla_meta.yaml'
    with open(ruta_yaml, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def cargar_embedding_rapido(checkpoint_dir, dtype):
    """Carga únicamente la capa de embeddings desde los safetensors para ahorrar VRAM."""
    index_path = checkpoint_dir / 'model.safetensors.index.json'
    
    if index_path.exists():
        with open(index_path, 'r') as f:
            weight_map = json.load(f)['weight_map']
        key = [k for k in weight_map if k.endswith('embed_tokens.weight')][0]
        shard = checkpoint_dir / weight_map[key]
    else:
        shard = checkpoint_dir / 'model.safetensors'
        with safe_open(shard, framework='pt') as f_:
            key = [k for k in f_.keys() if k.endswith('embed_tokens.weight')][0]
            
    with safe_open(shard, framework='pt') as f_:
        weight = f_.get_tensor(key).to(dtype)
        
    emb = torch.nn.Embedding(*weight.shape, _weight=weight)
    emb.requires_grad_(False)
    return emb.eval()

def verbalizar(v_raw, av_model, tokenizer, embed_layer, nla_meta, temperatura=1.0, max_nuevos_tokens=200):
    """
    Inyecta el vector de activación en el prompt y genera la explicación en texto natural.
    """
    injection_char = nla_meta['tokens']['injection_char']
    injection_tok_id = nla_meta['tokens']['injection_token_id']
    left_neighbor = nla_meta['tokens']['injection_left_neighbor_id']
    right_neighbor = nla_meta['tokens']['injection_right_neighbor_id']
    injection_scale = nla_meta['extraction']['injection_scale']
    prompt_template = nla_meta['prompt_templates']['av']

    # 1. Preparar el Prompt
    contenido = prompt_template.format(injection_char=injection_char)
    formatted_string = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': contenido}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(formatted_string, add_special_tokens=False)
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    # 2. Obtener Embeddings Base
    with torch.no_grad():
        embeds = (embed_layer(ids_t) * 1.0).float() # embed_scale = 1.0

    # 3. Escalar el Vector de Inyección (Matemática de Norma)
    v = torch.as_tensor(v_raw, dtype=torch.float32)
    norma = v.norm().clamp_min(1e-12)
    v_scaled = v * (injection_scale / norma)

    # 4. Inyectar en la posición exacta
    inyectado = False
    for p in range(1, len(input_ids) - 1):
        if (input_ids[p] == injection_tok_id and
            input_ids[p-1] == left_neighbor and
            input_ids[p+1] == right_neighbor):
            embeds[0, p] = v_scaled
            inyectado = True
            break
            
    if not inyectado:
        raise ValueError("No se encontró la posición de inyección. Revisar prompt template y tokens.")

    # 5. Sincronizar Device y Dtype dinámicamente con el modelo
    referencia_param = next(av_model.parameters())
    embeds_dev = embeds.to(device=referencia_param.device, dtype=referencia_param.dtype)

    # 6. Generar Explicación
    with torch.no_grad():
        tokens_gen = av_model.generate(
            inputs_embeds=embeds_dev,
            max_new_tokens=max_nuevos_tokens,
            temperature=temperatura,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    texto_gen = tokenizer.decode(tokens_gen[0], skip_special_tokens=False)
    
    # 7. Extraer solo el contenido dentro de <explanation>
    m = re.search(r'<explanation>\s*(.*?)\s*</explanation>', texto_gen, re.DOTALL)
    return m.group(1).strip() if m else texto_gen

def main():
    logger.info("=== INICIANDO FASE DE VERBALIZACIÓN NLA ===")
    
    config = cargar_configuracion()
    base_dir = Path(config['entorno']['base_dir'])
    
    ruta_checkpoint_av = base_dir / config['directorios']['checkpoints'] / 'nla_av'
    dir_act = base_dir / config['directorios']['activaciones']
    ruta_meta_json = dir_act / 'metadatos_activaciones.json'
    csv_salida = base_dir / config['directorios']['verbalizaciones'] / 'explicaciones_nla.csv'

    # 1. Cargar metadatos
    if not ruta_meta_json.exists():
        logger.error(f"No se encontró el JSON de metadatos en {ruta_meta_json}")
        return
        
    with open(ruta_meta_json, 'r', encoding='utf-8') as f:
        metadatos = json.load(f)
    logger.info(f"✓ {len(metadatos)} entradas a verbalizar cargadas.")

    # 2. Cargar Configuración NLA y Modelo AV
    nla_meta = cargar_config_nla(ruta_checkpoint_av)
    av_model, tokenizer = configurar_gpu_y_cargar_modelo(ruta_checkpoint_av)

    # Validar inyección
    ids_test = tokenizer.encode(nla_meta['tokens']['injection_char'], add_special_tokens=False)
    if ids_test != [nla_meta['tokens']['injection_token_id']]:
        logger.error(f"El carácter de inyección no coincide con el Token ID esperado.")
        return

    # 3. Cargar Capa de Embeddings Aislada
    referencia_param = next(av_model.parameters())
    dtype_emb = torch.bfloat16 if referencia_param.dtype == torch.bfloat16 else torch.float16
    embed_layer = cargar_embedding_rapido(ruta_checkpoint_av, dtype_emb)
    logger.info("✓ Capa de embedding aislada cargada exitosamente.")

    # 4. Loop Batch
    columnas_csv = ['id', 'lang', 'grupo', 'tema', 'texto', 
                    'senales_colombianas', 'hipotesis_nla', 'explicacion_nla']
    filas = []
    errores = []
    total = len(metadatos)

    for i, entrada in enumerate(metadatos):
        pid = entrada['id']
        lang = entrada['lang']
        logger.info(f"Verbalizando [{i+1:3d}/{total}] {pid}_{lang} ...")

        try:
            ruta_npy = dir_act / entrada['archivo_npy']
            v_raw = np.load(ruta_npy) # Shape esperado: [4, 3584]
            explicaciones_cuartiles = []

            for j, v_cuartil in enumerate(v_raw):
                exp_cuartil = verbalizar(v_cuartil, av_model, tokenizer, embed_layer, nla_meta)
                explicaciones_cuartiles.append(f"Cuartil {j+1}: {exp_cuartil}")

            explicacion_completa = "\n\n".join(explicaciones_cuartiles)

            filas.append({
                'id'                 : pid,
                'lang'               : lang,
                'grupo'              : entrada['grupo'],
                'tema'               : entrada['tema'],
                'texto'              : entrada['texto'],
                'senales_colombianas': '|'.join(entrada.get('senales_colombianas', [])),
                'hipotesis_nla'      : entrada.get('hipotesis_nla', ''),
                'explicacion_nla'    : explicacion_completa,
            })
            
            logger.info(f"  ✓ {explicacion_completa.replace(chr(10), ' ')[:70]}...")

        except Exception as e:
            logger.error(f"  ✗ ERROR en {pid}_{lang}: {e}")
            errores.append({'id': pid, 'lang': lang, 'error': str(e)})

    # 5. Guardar Resultados en CSV
    with open(csv_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columnas_csv)
        writer.writeheader()
        writer.writerows(filas)

    logger.info("=== RESUMEN DE VERBALIZACIÓN ===")
    logger.info(f"✓ Verbalizaciones guardadas: {len(filas)}")
    logger.info(f"✗ Errores: {len(errores)}")
    logger.info(f"✓ CSV exportado a: {csv_salida}")

if __name__ == "__main__":
    main()