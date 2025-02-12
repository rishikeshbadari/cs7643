import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import os
os.environ["HF_TOKEN"] = "Access Token for Hugging Face"

EXPERIMENT_MODE = False

import numpy as np
import base64

def encode_pixel_values_flat(image):
    """Flatten image into space-separated pixel values"""
    pixel_values = image.flatten()
    return ' '.join(map(str, pixel_values))

def encode_pixel_values_rows(image):
    """Convert to space-separated values, split by rows"""
    rows = []
    for row in image:
        row_string = ' '.join(map(str, row))
        rows.append(row_string)
    return '\n'.join(rows)

def encode_base64_flat(image):
    """Convert entire image to base64"""
    bytes_data = image.astype(np.uint8).tobytes()
    return base64.b64encode(bytes_data).decode('utf-8')

def encode_base64_rows(image):
    """Convert each row to base64 separately"""
    rows = []
    for row in image:
        bytes_data = row.astype(np.uint8).tobytes()
        row_base64 = base64.b64encode(bytes_data).decode('utf-8')
        rows.append(row_base64)
    return '\n'.join(rows)

def encode_binary_flat(image):
    """Convert to binary (0/1) based on threshold, flat"""
    binary = (image > 128).astype(int)
    return ' '.join(map(str, binary.flatten()))

def encode_binary_rows(image):
    """Convert to binary (0/1) based on threshold, split by rows"""
    binary = (image > 128).astype(int)
    rows = []
    for row in binary:
        row_string = ' '.join(map(str, row))
        rows.append(row_string)
    return '\n'.join(rows)

def encode_normalized_flat(image):
    """Normalize pixel values to 0-1 range, flat"""
    normalized = image / 255.0
    return ' '.join(map(lambda x: f"{x:.3f}", normalized.flatten()))

def encode_normalized_rows(image):
    """Normalize pixel values to 0-1 range, split by rows"""
    normalized = image / 255.0
    rows = []
    for row in normalized:
        row_string = ' '.join(map(lambda x: f"{x:.3f}", row))
        rows.append(row_string)
    return '\n'.join(rows)

def encode_mnist_image(image, method='pixel_flat'):
    """Main encoding function with multiple methods"""
    if hasattr(image, 'getdata'):
        image = np.array(image)
    
    if image.ndim == 3:
        image = image.mean(axis=2)
    
    encoding_methods = {
        'pixel_flat': encode_pixel_values_flat,
        'pixel_rows': encode_pixel_values_rows,
        'base64_flat': encode_base64_flat,
        'base64_rows': encode_base64_rows,
        'binary_flat': encode_binary_flat,
        'binary_rows': encode_binary_rows,
        'normalized_flat': encode_normalized_flat,
        'normalized_rows': encode_normalized_rows
    }
    
    if method not in encoding_methods:
        raise ValueError(f"Unknown encoding method: {method}. Available methods: {list(encoding_methods.keys())}")
    
    return encoding_methods[method](image)

def prepare_mnist_dataset(split="train", encoding_method='pixel_flat'):
    dataset = load_dataset("mnist")
    
    full_dataset = dataset[split]
    
    if EXPERIMENT_MODE:
        if split == "train":
            full_dataset = full_dataset.select(range(20000))
        elif split == "test":
            full_dataset = full_dataset.select(range(1000))
    else:
        if split == "test":
            full_dataset = full_dataset.select(range(3000))
    
    labels = full_dataset["label"]
    print(f"Label distribution in {split} dataset: {np.bincount(labels)}")
    
    data = [
        {"input_text": encode_mnist_image(item["image"], method=encoding_method), 
         "label": item["label"]}
        for item in full_dataset
    ]
    return Dataset.from_list(data)

def load_model_and_tokenizer():
    model_name = "meta-llama/Llama-3.2-1B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
        tokenizer.pad_token_id = len(tokenizer) - 1
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=10,
        device_map="auto"
        # torch_dtype=torch.float16
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    tokenizer.save_pretrained("./results")
    
    # for name, module in model.named_modules():
    #     print(name)
    
    return tokenizer, model

def setup_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj", "o_proj"],
        bias="none",
        task_type="SEQ_CLS"
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def predict_digit(model, tokenizer, encoded_image, max_length=900):
    inputs = tokenizer(encoded_image, return_tensors="pt", truncation=True, 
                      max_length=max_length, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_digit = torch.argmax(outputs.logits, dim=1).item()
    # print (predicted_digit)
    
    return predicted_digit

def evaluate_model(model, tokenizer, test_dataset, desc="Evaluating"):
    model.eval()
    true_labels = []
    predicted_labels = []
    
    for item in tqdm(test_dataset, desc=desc):
        prediction = predict_digit(model, tokenizer, item["input_text"])
        true_labels.append(item["label"])
        predicted_labels.append(prediction)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    return accuracy, report, conf_matrix

def train_model(model, tokenizer, train_dataset):
    def tokenize_data(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            padding="max_length",
            max_length=900,
            # return_tensors="pt"
        )

    tokenized_dataset = train_dataset.map(
        tokenize_data,
        batched=True,
        remove_columns=["input_text"]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        fp16=True,
        logging_dir="./logs",
        save_total_limit=1,
        learning_rate=6e-4,
        warmup_steps=50,
        logging_steps=100,
        save_steps=1000,
        save_safetensors=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

def main():
    print("Preparing datasets...")
    train_dataset = prepare_mnist_dataset(split="train")
    test_dataset = prepare_mnist_dataset(split="test")
    
    print("Loading model and tokenizer...")
    tokenizer, model = load_model_and_tokenizer()
    
    print("\nEvaluating base model before training...")
    base_accuracy, base_report, base_conf_matrix = evaluate_model(model, tokenizer, test_dataset, "Testing base model")
    print(f"\nBase Model Accuracy: {base_accuracy:.4f}")
    print("\nBase Model Classification Report:")
    print(base_report)
    print("\nBase Model Confusion Matrix:")
    print(base_conf_matrix)
    
    print("\nSetting up LoRA...")
    model = setup_lora(model)
    # print(model)
    
    print("Starting training...")
    train_model(model, tokenizer, train_dataset)
    
    print("\nEvaluating fine-tuned model...")
    tuned_accuracy, tuned_report, tuned_conf_matrix = evaluate_model(model, tokenizer, test_dataset, "Testing fine-tuned model")
    print(f"\nFine-tuned Model Accuracy: {tuned_accuracy:.4f}")
    print("\nFine-tuned Model Classification Report:")
    print(tuned_report)
    print("\nFine-tuned Model Confusion Matrix:")
    print(tuned_conf_matrix)
    
    print("\nTraining and evaluation completed!")
    
    print("\nAccuracy Comparison:")
    print(f"Base Model: {base_accuracy:.4f}")
    print(f"Fine-tuned Model: {tuned_accuracy:.4f}")
    print(f"Improvement: {(tuned_accuracy - base_accuracy):.4f}")

if __name__ == "__main__":
    main()
