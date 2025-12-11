import os
import sys
import torch
import random
import logging
import numpy as np
import transformers
from evaluate import load
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment

LLAMA_START_TOKEN_ID = 1

class ValueRecorder:
    def __init__(self):
        self.value = None

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

# 设置种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)

# 设置logging信息
def set_loginfo(args):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_method = "full" if args.attk_part == "full" else f"{args.attk_part}_r{args.rank}"
    if args.change_steps is not None and args.init == "svd":
        method = "logia"
    elif args.position_swap and args.init == "candidate_random":
        method = "lamp"
    else:
        method = args.loss
    pathdir = f"./logs/{method}/{args.model_name}/{args.dataset}"
    filename = f"./logs/{method}/{args.model_name}/{args.dataset}/{train_method}_{args.init}_inputs{args.inputs}_b{args.batch_size}_iter{args.max_iters}_{now}.log" if not args.test else "./logs/test/test.log"
    
    if not os.path.exists(pathdir):
        os.makedirs(pathdir)
    # 设置file_hander
    file_hander = logging.FileHandler(filename=filename, encoding="utf-8")
    file_formatter = logging.Formatter('[%(levelname)s][%(asctime)s.%(msecs)03d] [%(filename)s:%(lineno)d]: %(message)s')
    file_hander.setFormatter(file_formatter)
    # 设置stream_hander
    stream_hander = logging.StreamHandler()
    stream_formatter = logging.Formatter("%(message)s")
    stream_hander.setFormatter(stream_formatter)
    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        datefmt='(%Y-%m-%d) %H:%M:%S',
        handlers=[file_hander, stream_hander]
    )

# 对于Llama模型的攻击，添加<s>; 对于填充的token，将其变为pad_token
def fix_special_token(args, x_embeds, embedding_weight, pad_token_id, pads):
    if pads is not None: 
        for sen_id in range(x_embeds.shape[0]):
            x_embeds.data[sen_id, pads[sen_id]:] = embedding_weight[pad_token_id]

# 获取填充token
def get_pads(args, batch, tokenizer):
    pads = None
    if args.know_padding and args.batch_size > 1:
        pads = [batch['input_ids'].shape[1]]*batch['input_ids'].shape[0]
        for sen_id in range(batch['input_ids'].shape[0]):
            for i in range(batch['input_ids'].shape[1]-1, 0, -1):
                if batch['input_ids'][sen_id][i] == tokenizer.pad_token_id:
                    pads[sen_id] = i
                else:
                    break
    return pads

# 获取攻击的数据
def prepare_data(args):
    # 需要选取的总数据条数
    total_num = args.batch_size * args.inputs
    # 对于文本生成任务
    if args.task_type == "text-generation":
        assert args.dataset in ["wikitext-2"]
        # 加载数据集
        if args.dataset == "wikitext-2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="./cache")[args.split]
            # dataset = dataset.filter(lambda data: len(data["text"]) > 10 and len(data["text"]) <= 50 and ("=" not in data["text"]) and ("@" not in data["text"]))   # 筛去空行
            dataset = dataset.map(lambda data: {"text": f"{data['text'].split('.', 1)[0]}."})
            dataset = dataset.filter(lambda data: len(data["text"]) > 10 and len(data["text"]) < 50 and ("=" not in data["text"]) and ("@" not in data["text"]) and ("." in data["text"]))
            
            # 随机挑选total_num个数据
            dataset = dataset.shuffle(seed=args.seed)
            select_range = range(total_num)
            dataset = dataset.select(select_range)  
    # 对于分类任务
    elif args.task_type == "text-classification":
        assert args.dataset in ["cola", "sst2", "rotten_tomatoes", "CR"]
        if args.dataset in ["cola", "sst2"]:
            dataset = load_dataset("glue", args.dataset, cache_dir="./cache", download_mode="reuse_dataset_if_exists")[args.split]
        elif args.dataset == "rotten_tomatoes":
            dataset = load_dataset(args.dataset, cache_dir="./cache", download_mode="reuse_dataset_if_exists")[args.split]
        # 随机挑选total_num个数据
        dataset = dataset.shuffle(args.seed)
        select_range = range(total_num)
        dataset = dataset.select(select_range)
    else:
        raise ValueError
    
    # 获取tokenizer并设置pad_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir="./cache", local_files_only=True)
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token_id = 0

    # 用于记录共inputs个batch的数据
    ground_truth_data_list = [] 
    
    for i in range(0, total_num, args.batch_size):
        batch = dataset[i : i + args.batch_size]

        # 为数据添加label
        if args.task_type == "text-generation":
            # 获取要恢复的数据
            ground_truth_text = batch["text"]
            # ground_truth_text = [" The Tower Building of the Little Rock Arsenal, also known as U.S."]
            ground_truth_data = {"text":ground_truth_text, **tokenizer(ground_truth_text, padding=True, return_tensors="pt")}
            # 文本生成任务的labels和inputs_ids一致
            ground_truth_data["labels"] = ground_truth_data["input_ids"]
        elif args.task_type == "text-classification":
            ground_truth_text = batch["sentence"] if args.dataset in ["cola", "sst2"] else batch["text"]
            ground_truth_data = {"text":ground_truth_text, **tokenizer(ground_truth_text, padding=True, return_tensors="pt")}
            ground_truth_data["labels"] = torch.tensor(batch["label"])
        ground_truth_data_list.append(ground_truth_data)

    # len_seq = []
    # for ground_truth_data in ground_truth_data_list:
    #     for input_ids in ground_truth_data["input_ids"]:
    #         len_seq.append(len(input_ids))
    # print(f"avg_token: {sum(len_seq) / len(len_seq)}")

    return ground_truth_data_list, tokenizer


# 用于生成虚拟数据和标签
def generate_dummy_data(args, data_size):
    # TODO 不同的初始化方式
    dummy_data = (torch.randn(size=data_size, dtype=torch.float32, device=args.device) * 0.1).clamp(-0.1, 0.1)
    dummy_data.requires_grad =True
    dummy_data.grad = torch.zeros_like(dummy_data)

    return dummy_data

# 获取初始化标签
def get_init(args, model, data_size, target_gradient, true_labels, embedding_weight:torch.Tensor, pads, tokenizer):
    # from train import get_dummy_gradient, compute_loss
    
    embedding_norm = embedding_weight.data.norm(p=2, dim=1).mean()
    # print(f"embedding_norm: {embedding_norm}")

    if args.init == "clip_random":
        # CLIP RANDOM LOSS
        dummy_data = (torch.randn(size=data_size, dtype=torch.float32, device=args.device) * 0.1).clamp(-0.1, 0.1)
        
        dummy_data.requires_grad =True
        dummy_data.grad = torch.zeros_like(dummy_data)

    elif args.init == "random":
        # RANDOM LOSS
        dummy_data = torch.randn(size=data_size, dtype=torch.float32, device=args.device)
        dummy_data.requires_grad =True
        dummy_data.grad = torch.zeros_like(dummy_data)

    elif args.init == "svd":
        # 取出第一个transformer块中A矩阵的梯度
        grad_A = target_gradient[0]
        rank_A = np.linalg.matrix_rank(grad_A.cpu())
        grad_A.to(args.device)
        logging.info(f"A rank:{rank_A}")
        U, S, Vh = torch.linalg.svd(grad_A)
        V = Vh.T
        base_vector = V[:, :rank_A].T
        length = data_size[0] * data_size[1]
        indices = torch.arange(length) % rank_A
        dummy_data = base_vector[indices]
        
        # NORMAL SVD LOSS
        dummy_data = (dummy_data / dummy_data.norm(p=2, dim=1, keepdim=True)) * embedding_norm
        dummy_data = dummy_data.view(data_size[0], data_size[1], -1)
        dummy_data.requires_grad =True
        dummy_data.grad = torch.zeros_like(dummy_data)

    elif args.init == "candidate_random":
        from train import get_dummy_gradient, compute_loss
        num_inits = data_size[0]

        new_shape = [500*num_inits] + list(data_size[1:])
        embeds = torch.randn(new_shape).to(args.device)

        # Pick candidates based on rec loss
        best_x_embeds, best_rec_loss = None, None
        for i in range(500):
            tmp_embeds = embeds[i*num_inits:(i+1)*num_inits]
            fix_special_token(args, tmp_embeds, embedding_weight, tokenizer.pad_token_id, pads)
            dummy_grad, task_loss = get_dummy_gradient(args, model, tmp_embeds, true_labels, embedding_weight=embedding_weight)
            rec_loss = compute_loss(args, target_gradient, dummy_grad)
            if (best_rec_loss is None) or (rec_loss < best_rec_loss):
                best_rec_loss = rec_loss
                best_x_embeds = tmp_embeds
                logging.info(f'[Init] best rec loss: {best_rec_loss.item()}')
        
        # Pick best permutation of candidates
        for i in range(500):
            idx = torch.cat((torch.tensor([0], dtype=torch.int32), torch.randperm(data_size[1]-2)+1, torch.tensor([data_size[1]-1], dtype=torch.int32) ))
            tmp_embeds = best_x_embeds[:, idx].detach()
            dummy_grad, task_loss = get_dummy_gradient(args, model, tmp_embeds, true_labels, embedding_weight=embedding_weight)
            rec_loss = compute_loss(args, target_gradient, dummy_grad)
            if (rec_loss < best_rec_loss):
                best_rec_loss = rec_loss
                best_x_embeds = tmp_embeds
                logging.info(f'[Init] best perm rec loss: {best_rec_loss.item()}')
        
        
        best_x_embeds /= best_x_embeds.norm(dim=2,keepdim=True)
        best_x_embeds *= embedding_norm

        dummy_data = best_x_embeds.detach().clone()
        dummy_data = dummy_data.requires_grad_(True)
    
    return dummy_data

# 将文本数据映射到embedding向量
def map_to_real_embedding(tokenizer, batch_token, embedding_weight):

    embeddings = embedding_weight[batch_token]
    embeddings.requires_grad = True
    
    return embeddings

# 将embedding转换为token,并解码为字符串
def get_closest_tokens(args, model, dummy_data):

    if args.model_name == "gpt2":
        embedding_matrix = model.transformer.wte.weight.data
    elif args.model_name == "TinyLlama/TinyLlama_v1.1":
        embedding_matrix = model.model.model.embed_tokens.weight.data
        
    with torch.no_grad():
        # 使用欧几里得距离度量
        if args.distance_metric == "euclidean":
            token_list = []
            for seq in dummy_data:
                seq_token = []
                for embd in seq:
                    embd = embd.unsqueeze(0)
                    dist = torch.norm(embedding_matrix - embd, dim=1)
                    seq_token.append(torch.argmin(dist).item())
                token_list.append(seq_token)
            token_list = torch.tensor(token_list)
        # 使用余弦相似度作为距离度量
        elif args.distance_metric == "cosine":

            # 对embedding_matrix每行进行归一化
            embedding_matrix = embedding_matrix - embedding_matrix.mean(dim=-1, keepdim=True)
            embedding_matrix = embedding_matrix / torch.sqrt(torch.sum(embedding_matrix ** 2, dim=-1, keepdim=True))

            # 对dummy_data进行归一化
            dummy_embedding = dummy_data
            dummy_embedding = dummy_embedding - dummy_embedding.mean(dim=-1, keepdim=True)
            dummy_embedding = dummy_embedding / torch.sqrt(torch.sum(dummy_embedding ** 2, dim=-1, keepdim=True))

            similarity = torch.matmul(embedding_matrix, torch.transpose(dummy_embedding, 1, 2))
            token_list = torch.argmax(similarity, dim=1)
    
    return token_list

# 检查向量相似度的作用
def check_utility(token_topk_list, recovered_token, ground_truth_token, tokenizer):
    recovered_minus_ground_token = set(recovered_token.tolist()) - set(ground_truth_token.tolist())
    ground_minus_recovered_token = set(ground_truth_token.tolist()) - set(recovered_token.tolist())
    if len(recovered_minus_ground_token) == 0:
        return float("inf")
    valid_num = 0
    # print(f"difference token num: {len(recovered_minus_ground_token)}")
    for token_id in recovered_minus_ground_token:
        if len(set(token_topk_list[token_id]) & ground_minus_recovered_token) > 0:
            print(f"{tokenizer.decode(token_id)} -> {tokenizer.decode(list(set(token_topk_list[token_id]) & ground_minus_recovered_token))}")
            valid_num += 1

    return valid_num / len(recovered_minus_ground_token)

# 去除pad token
def remove_padding(args, tokenizer, ids):
    for i in range(ids.shape[0] - 1, -1, -1):
        if ids[i] != tokenizer.pad_token_id:
            ids = ids[:i+1]
            break
    if args.model_name == "TinyLlama/TinyLlama_v1.1" and ids[0] == 1:
        ids = ids[1:]
    return ids

# 用于评估攻击效果
def eval_attack(args, recovered_batch_data, ground_truth_batch_data):
    scorer = load("./eval/rouge")

    cost = np.zeros((args.batch_size, args.batch_size))
    for i in range(args.batch_size):
        for j in range(args.batch_size):
            fm = scorer.compute(predictions=[recovered_batch_data[i]], references=[ground_truth_batch_data[j]])['rouge1']
            cost[i, j] = 1.0 - fm
    row_ind, col_ind = linear_sum_assignment(cost)

    ids = list(range(args.batch_size))
    ids.sort(key=lambda i: col_ind[i])
    sorted_recovered_data = []
    for i in range(args.batch_size):
        sorted_recovered_data.append(recovered_batch_data[ids[i]])
    recovered_batch_data = sorted_recovered_data
    
    metrics = scorer.compute(predictions=recovered_batch_data, references=ground_truth_batch_data)
    return metrics, recovered_batch_data