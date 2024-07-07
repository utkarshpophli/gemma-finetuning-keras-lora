class Config:
    DATASET_NAME = "ignmilton/ign_clean_instruct_dataset_500k"
    NUM_SAMPLES = 1000
    MODEL_PRESET = "gemma_2b_en"
    LORA_RANK = 4
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    MAX_SEQUENCE_LENGTH = 512
    EPOCHS = 1
    BATCH_SIZE = 1
    MODEL_SAVE_PATH = "models/gemma_finetuned_lora.h5"