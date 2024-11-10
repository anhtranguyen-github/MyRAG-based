from datetime import datetime
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer
import time
import wandb

from ultils import get_training_dataset, load_csv_from_path, slice_csv_data, split_datasets



load_dotenv()
# Get the W&B API key from the environment variables
wandb_api_key = os.getenv("WANDB_API_KEY")



basemodel_path = "hiieu/halong_embedding"
model = SentenceTransformer(basemodel_path)

# Load CSV data
print("Loading CSV data...")
csv_data = load_csv_from_path('data/train.csv')
#csv_data = csv_data[:100]
print(f"Loaded {len(csv_data)} entries from the CSV file.")

# Prepare queries, corpus, and relevant documents
print("Preparing queries, corpus, and relevant documents...")
queries, corpus, relevant_docs = slice_csv_data(csv_data)

print(f"Number of Queries: {len(queries)}")
print(f"Number of Corpus Entries: {len(corpus)}")
print(f"Number of Relevant Docs Entries: {len(relevant_docs)}")

# Create evaluation and training sets
print("Splitting relevant documents into evaluation and training sets...")
eval_set, train_set = split_datasets(relevant_docs)

print(f"Created eval_set with {len(eval_set)} entries.")
print(f"Created train_set with {len(train_set)} entries.")

# Setup evaluator
print("Setting up the Information Retrieval Evaluator...")
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=eval_set,
    name=f"dim_128",
    truncate_dim=128,
    score_functions={"cosine": cos_sim},
)

# Evaluate the model
print("Starting evaluation...")
start_time = time.time()
results = evaluator(model)
end_time = time.time()
eval_duration = end_time - start_time

print(f"Evaluation Time: {eval_duration:.2f} seconds")

for k, v in results.items():
    print(f"Result Key: {k}, Value: {v}")

# Prepare training dataset
print("Preparing the training dataset...")
training_dataset = get_training_dataset(queries, corpus, train_set)
eval_dataset = get_training_dataset(queries, corpus, eval_set)

project = "legal-document-finetune"
run_name = "halong_embedding" + "-" + project
output_dir = "./" + run_name

wandb.init(project=project, name=run_name, config={
    "model_name": run_name,
    "output_dir": output_dir
})

# Set up loss functions and training arguments
print("Setting up the loss functions and training arguments...")
inner_loss = MultipleNegativesRankingLoss(model)
loss = MatryoshkaLoss(model, inner_loss, matryoshka_dims=[768, 512, 256, 128])
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    num_train_epochs=5,
    bf16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # No duplicate is good for Multi Negative Ranking Loss
    eval_strategy="steps",
    #evaluation_strategy="no",   
    metric_for_best_model="eval_dim_128_cosine_accuracy@3",  # best score 128 dimension
    load_best_model_at_end=True,
    logging_steps=197*2,
    save_steps=394*2,
    save_total_limit=100,
)

# Initialize the trainer
print("Initializing the SentenceTransformerTrainer...")
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=training_dataset,
     
    #eval_dataset =  eval_dataset,
    loss=loss,
    evaluator=evaluator,
)



print("Logging into WandB...")
# Log in to W&B
wandb.login(key=wandb_api_key)


# Start training
print("Starting training...")
trainer.train()

# Save the model
print("Saving the model...")
trainer.save_model()
print("Model saved local successfully.")

# Save the model as a W&B artifact
print("Logging the model as a W&B artifact...")
artifact = wandb.Artifact(name= run_name, type='model')
artifact.add_dir(output_dir)
wandb.log_artifact(artifact)

wandb.finish()
print("Model artifact logged successfully to W&B.")

final_model = SentenceTransformer(output_dir)

final_results = evaluator(final_model)
for k,v in final_results.items():
    print(k, v)