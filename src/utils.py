import keras_nlp

def generate_text(model, prompt, max_length=256, k=5, seed=2):
    sampler = keras_nlp.samplers.TopKSampler(k=k, seed=seed)
    model.compile(sampler=sampler)
    return model.generate(prompt, max_length=max_length)

def format_prompt(instruction):
    return f"Instructions\n{instruction}\n\nResponse:\n"