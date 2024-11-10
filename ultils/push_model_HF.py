from sentence_transformers import SentenceTransformer

model = SentenceTransformer("./halong_embedding-legal-document-finetune")
model.push_to_hub("tranguyen/halong_embedding-legal-document-finetune", exist_ok=True)
