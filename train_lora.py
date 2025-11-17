import sys
import random
import argparse
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
import os
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, get_scheduler
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

seed = 100
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
transformers.set_seed(seed)
device = 'cuda'

def save_model(model, save_path, train_method):
    print('SAVING', flush=True)
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    try:
        if train_method != 'lora':
            model.save_pretrained(save_path)
        else:
            lora_param = {name:parameters for name, parameters in model.named_parameters() if parameters.requires_grad}
            torch.save(lora_param, save_path)
        print(f"Model saved to {save_path} successfully.", flush=True)
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes', "CR"], default='cola')
    parser.add_argument('--check_every', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=8000)
    parser.add_argument('--pct_mask', type=float, default=None)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--train_method', type=str, default='lora', choices=['full', 'lora'])
    parser.add_argument('--rank', type=int, default=8)  
    parser.add_argument('--models_cache', type=str, default='./cache')
    args = parser.parse_args()
    print(args)
    seq_key = 'text' if args.dataset == 'rotten_tomatoes' or args.dataset == "CR" else 'sentence'
    num_labels = 2
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels, cache_dir='./cache', local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir="./cache", local_files_only=True)
    
    # Configure tokenizer and model
    if tokenizer.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # Load LoRA model if applicable
    if args.train_method == 'lora':
        target_modules_dict = {"gpt2":["c_attn"], "TinyLlama/TinyLlama_v1.1":["q_proj", "v_proj"]}
        target_modules = target_modules_dict[args.model_name]
        lora_config = LoraConfig(
            base_model_name_or_path=args.model_name,
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=target_modules
        )
        model = get_peft_model(model, lora_config)
        

    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola':
        metric = load('./eval/matthews_correlation')
        train_metric = load('./eval/matthews_correlation')
    else:
        metric = load('./eval/accuracy')
        train_metric = load('./eval/accuracy')

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset, cache_dir="cache")
    elif args.dataset == "rotten_tomatoes":
        datasets = load_dataset(args.dataset, cache_dir="cache")
    elif args.dataset == "CR":
        datasets = load_dataset("SetFit/CR", cache_dir="cache")
    
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    elif args.dataset == 'CR':
        tokenized_datasets = tokenized_datasets.remove_columns(['text', 'label_text'])
    else:
        assert False
    
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')
    tokenized_datasets = tokenized_datasets.shuffle(seed=seed)
    train_dataset = tokenized_datasets['train'].select(range(args.save_steps))
    # train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation'] if args.dataset != "CR" else tokenized_datasets['test']
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_training_steps = len(train_loader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    n_steps = 0
    train_loss = 0

    # Run training loop
    for epoch in range(args.num_epochs):
        model.train()
        idx = 0
        for batch in tqdm(train_loader):
            idx += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            train_metric.add_batch(predictions=predictions, references=batch['labels'])
            
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            if args.pct_mask is not None:
                for param in model.parameters():
                    grad_mask = (torch.rand(param.grad.shape).to(device) > args.pct_mask).float()
                    param.grad.data = param.grad.data * grad_mask

            if args.noise is not None:
                for param in model.parameters():
                    param.grad.data = param.grad.data + torch.randn(param.grad.shape).to(device) * args.noise

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()

            n_steps += 1
            if n_steps % args.check_every == 0:
                print('metric train: ', train_metric.compute())
                print('loss train: ', train_loss/n_steps)
                train_loss = 0.0
        model.eval()
        
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            metric.add_batch(predictions=predictions, references=batch['labels'])
            
        print('metric eval: ', metric.compute())
    print('END')
    save_model(model, f'./lora_weight/{args.model_name}_{args.dataset}_r{args.rank}.pt', args.train_method)
    

if __name__ == '__main__':
    main()
