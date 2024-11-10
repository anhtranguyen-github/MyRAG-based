from datetime import datetime
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer
import wandb

from ultils import get_training_dataset, load_csv_from_path, slice_csv_data, split_datasets

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")

# Load model and checkpoint
basemodel_path = "hiieu/halong_embedding"
model = SentenceTransformer(basemodel_path)

# Load CSV data
csv_data = load_csv_from_path('data/train.csv')
queries, corpus, relevant_docs = slice_csv_data(csv_data)
eval_set, train_set = split_datasets(relevant_docs)

# Setup evaluator
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=eval_set,
    name="dim_128",
    truncate_dim=128,
    score_functions={"cosine": cos_sim},
)

# Prepare training and evaluation datasets
training_dataset = get_training_dataset(queries, corpus, train_set)
eval_dataset = get_training_dataset(queries, corpus, eval_set)

# Set up WandB
project = "legal-document-finetune"
run_name = "halong_embedding" + "-" + project
output_dir = "./" + run_name
wandb.init(project=project, name=run_name, config={
    "model_name": run_name,
    "output_dir": output_dir
})

# Set up loss functions and training arguments
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
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    metric_for_best_model="eval_dim_128_cosine_accuracy@3",
    load_best_model_at_end=True,
    logging_steps=197*2,
    save_steps=394*2,
    save_total_limit=100,
    resume_from_checkpoint="halong_embedding-legal-document-finetune/checkpoint-27580",  # Specify the checkpoint path here
)

# Initialize the trainer with the checkpoint
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=training_dataset,
    loss=loss,
    evaluator=evaluator,
)

# Log in to WandB
wandb.login(key=wandb_api_key)

# Start training from checkpoint
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# Save the model and log as artifact
trainer.save_model()
artifact = wandb.Artifact(name=run_name, type='model')
artifact.add_dir(output_dir)
wandb.log_artifact(artifact)

wandb.finish()
