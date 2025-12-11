from transformers import AutoModelForMaskedLM, BertForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased", cache_dir="./cache")
for name, param in model.named_parameters():
    print(name, param.shape)