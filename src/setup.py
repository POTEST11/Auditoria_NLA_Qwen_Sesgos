import os
import yaml
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. Configuración del sistema de Logging
# Esto enviará los mensajes tanto a la consola como a un archivo pipeline.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../pipeline.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def cargar_configuracion(ruta_yaml='../config.yaml'):
    """Carga el archivo de configuración."""
    try:
        with open(ruta_yaml, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo de configuración en {ruta_yaml}")
        raise

def crear_estructura_directorios(config):
    """Crea la estructura de carpetas definida en el config.yaml."""
    base_dir = Path(config['entorno']['base_dir'])
    
    # Lista de subcarpetas a crear
    subcarpetas = [
        base_dir / config['directorios']['activaciones'],
        base_dir / config['directorios']['verbalizaciones'],
        base_dir / config['directorios']['resultados']
    ]
    
    # Añadir las carpetas específicas para cada checkpoint
    for clave_modelo in config['modelos'].keys():
        ruta_checkpoint = base_dir / config['directorios']['checkpoints'] / clave_modelo
        subcarpetas.append(ruta_checkpoint)

    logger.info("Verificando/Creando estructura de directorios...")
    for carpeta in subcarpetas:
        carpeta.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directorio listo: {carpeta}")

def descargar_modelos(config):
    """Descarga los modelos de HF si no existen localmente."""
    base_dir = Path(config['entorno']['base_dir'])
    checkpoints_dir = base_dir / config['directorios']['checkpoints']
    
    logger.info("Iniciando verificación y descarga de modelos...")
    
    for nombre_carpeta, repo_id in config['modelos'].items():
        ruta_destino = checkpoints_dir / nombre_carpeta
        
        # Validación rápida para evitar llamadas innecesarias a la API
        if (ruta_destino / "config.json").exists():
            logger.info(f"✓ {nombre_carpeta} ya existe en caché. Omitiendo descarga.")
            continue
            
        logger.info(f"Descargando {repo_id} en {ruta_destino}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=ruta_destino,
                ignore_patterns=['*.msgpack', '*.h5', 'flax_model*']
            )
            logger.info(f"✓ Descarga completada para {repo_id}")
        except Exception as e:
            logger.error(f"Error al descargar {repo_id}: {str(e)}")
            raise

def validar_sidecars_nla(config):
    """Verifica que el archivo nla_meta.yaml exista en los autoencoders (Crítico)."""
    base_dir = Path(config['entorno']['base_dir'])
    checkpoints_dir = base_dir / config['directorios']['checkpoints']
    
    modelos_nla = ['nla_av', 'nla_ar']
    logger.info("Validando archivos sidecar críticos (nla_meta.yaml)...")
    
    for modelo in modelos_nla:
        ruta_sidecar = checkpoints_dir / modelo / 'nla_meta.yaml'
        if ruta_sidecar.exists():
            logger.info(f"✓ Sidecar encontrado en {modelo}")
        else:
            logger.error(f"✗ FALTA nla_meta.yaml en {modelo}. El pipeline fallará.")

def main():
    logger.info("=== INICIANDO SETUP DEL PIPELINE ===")
    
    # Precaución para entornos Colab: advertir si Drive no está montado
    if "content/drive" in str(Path.cwd().parent) and not Path("/content/drive/MyDrive").exists():
        logger.warning("Parece que estás en Colab pero Google Drive no está montado.")
        logger.warning("Ejecuta 'from google.colab import drive; drive.mount('/content/drive')' antes de correr este script.")
    
    config = cargar_configuracion()
    crear_estructura_directorios(config)
    descargar_modelos(config)
    validar_sidecars_nla(config)
    
    logger.info("=== SETUP COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()