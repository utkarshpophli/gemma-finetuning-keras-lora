import os
import keras
import keras_nlp
from tensorflow.keras import mixed_precision

def setup_environment():
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
    mixed_precision.set_global_policy('mixed_float16')

def load_gemma_model(preset="gemma_2b_en"):
    setup_environment()
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(preset)
    return gemma_lm

def enable_lora(model, rank=4):
    model.backbone.enable_lora(rank=rank)
    return model

def compile_model(model, learning_rate=5e-5, weight_decay=0.01):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
    
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

if __name__ == "__main__":
    model = load_gemma_model()
    model = enable_lora(model)
    model = compile_model(model)
    model.summary()