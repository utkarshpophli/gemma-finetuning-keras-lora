from src.data_preprocessing import load_and_preprocess_data
from src.model import load_gemma_model, enable_lora, compile_model
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model(config):
    logger.info("Loading and preprocessing data...")
    data = load_and_preprocess_data(config.DATASET_NAME, config.NUM_SAMPLES)
    
    logger.info("Loading Gemma model...")
    model = load_gemma_model(config.MODEL_PRESET)
    
    logger.info("Enabling LoRA...")
    model = enable_lora(model, config.LORA_RANK)
    
    logger.info("Compiling model...")
    model = compile_model(model, config.LEARNING_RATE, config.WEIGHT_DECAY)
    
    logger.info("Starting training...")
    model.preprocessor.sequence_length = config.MAX_SEQUENCE_LENGTH
    model.fit(data, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    
    logger.info("Training completed.")
    return model

if __name__ == "__main__":
    config = Config()
    trained_model = train_model(config)
    trained_model.save_weights(config.MODEL_SAVE_PATH)
    logger.info(f"Model weights saved to {config.MODEL_SAVE_PATH}")