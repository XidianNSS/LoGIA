from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

# Download models
AutoModelForSequenceClassification.from_pretrained("gpt2", cache_dir="./cache")
AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama_v1.1", cache_dir="./cache")

# Download datasets
load_dataset("glue", "cola", cache_dir="./cache")
load_dataset("glue", "sst2", cache_dir="./cache")
load_dataset("rotten_tomatoes", cache_dir="./cache")
