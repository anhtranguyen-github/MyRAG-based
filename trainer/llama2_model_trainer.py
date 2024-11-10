from llama2_model import load_llama_model, prepare_model_for_training
from dotenv import load_dotenv
from datasets import load_dataset
import os
import torch
from accelerate import Accelerator
import transformers
from datetime import datetime
from huggingface_hub import login
load_dotenv()


#torch.cuda.empty_cache()


# Load the API key securely
api_token = os.getenv("HF_API_TOKEN")
if api_token is None:
    raise ValueError("Hugging Face API token not found. Please set the HF_API_TOKEN environment variable.")

print("Logging in to Hugging Face...")
login(api_token)

# Load dataset
print("Loading dataset...")
dataset = load_dataset('csv', data_files='data/train.csv')
print("Splitting dataset into train and evaluation sets...")
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

def formatting_func(example):
    text = f"### Câu hỏi: {example['question']}\n ### Trả lời: {example['context']}"
    return text

def generate_and_tokenize_prompt(prompt, tokenizer):
    return tokenizer(formatting_func(prompt))

# Load Llama model and tokenizer from llama_model.py
base_model_id = "bkai-foundation-models/vietnamese-llama2-7b-120GB"
print(f"Loading Llama model from {base_model_id}...")

model, tokenizer = load_llama_model(base_model_id)

# Tokenize datasets
print("Tokenizing training dataset...")
tokenized_train_dataset = train_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
print("Tokenizing evaluation dataset...")
tokenized_val_dataset = eval_dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer))

# Prepare model for training with LoRA
print("Preparing model for training with LoRA...")
model = prepare_model_for_training(model)

# Initialize Accelerator
print("Initializing Accelerator...")
accelerator = Accelerator()
model = accelerator.prepare_model(model)

if torch.cuda.device_count() > 1:
    print("Using model parallelism for multi-GPU training...")
    model.is_parallelizable = True
    model.model_parallel = True

project = "law-finetune"
base_model_name = "llama2-7b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

print("Setting up the Trainer...")
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        do_eval=True,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Disabling model cache...")
model.config.use_cache = False

print("Starting training...")
trainer.train()
print("Training completed.")
