from datasets import load_dataset

def load_and_preprocess_data(dataset_name, num_samples=1000):
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    
    data = []
    for item in dataset:
        instruction = item['input'].strip()
        response = item['output'].strip()
        formatted_string = f"Instruction:\n{instruction}\n\nResponse:\n{response}"
        data.append(formatted_string)
    
    return data

if __name__ == "__main__":
    dataset_name = "ignmilton/ign_clean_instruct_dataset_500k"
    processed_data = load_and_preprocess_data(dataset_name)
    print(f"Processed {len(processed_data)} samples")
    print("Sample:", processed_data[0])