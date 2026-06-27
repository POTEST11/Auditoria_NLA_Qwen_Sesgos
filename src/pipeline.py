import logging
from src import setup, inference, verbalizer, reconstructor

logger = logging.getLogger(__name__)

def main():
    logger.info("**************************************************")
    logger.info("* INICIANDO PIPELINE DE AUDITORÍA XAI (NLA)      *")
    logger.info("**************************************************")
    
    try:
        # Fase 1: Entorno y Modelos
        setup.main()
        
        # Fase 2: Extracción Mecanística
        inference.main()
        
        # Fase 3: Traducción de Activaciones a Texto
        verbalizer.main()
        
        # Fase 4: Reconstrucción y Evaluación Matemática
        reconstructor.main()
        
        logger.info("**************************************************")
        logger.info("* PIPELINE COMPLETADO EXITOSAMENTE               *")
        logger.info("**************************************************")
        
    except Exception as e:
        logger.error(f"El pipeline falló de manera crítica: {e}")
        raise

if __name__ == "__main__":
    main()