import sys
import json
import time
import random
import torch
import logging
import datasets
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
from transformers import get_scheduler
from peft import LoraConfig, get_peft_model, TaskType, LoraModel
from torch.nn.attention import SDPBackend, sdpa_kernel
from utils import get_closest_tokens, eval_attack, fix_special_token, remove_padding, ValueRecorder, check_utility, map_to_real_embedding

LLAMA_START_TOKEN = 1

# 获取攻击所需的梯度
def get_params_list_for_grad(args, model):
    parameters_list = []
    for name, param in model.named_parameters():
        if args.attk_part == "full":    # 使用所有参数的梯度进行攻击
            param.requires_grad = True
            parameters_list.append(param)
        # elif args.attk_part == "AB" and param.requires_grad == True:    # 使用A和B矩阵的梯度进行攻击
        elif args.attk_part == "AB" and "lora" in name:
            parameters_list.append(param)
        elif args.attk_part == "A" and "lora_A" in name:    # 使用A矩阵的梯度进行攻击
            parameters_list.append(param)
        elif args.attk_part == "B" and "lora_B" in name:    # 使用B矩阵的梯度进行攻击
            parameters_list.append(param)
    
    return parameters_list

# 获取模型
def get_pertrained_model(args, tokenizer):
    cache_dir = "./cache"
    # 获取模型
    if args.task_type == "text-generation":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device, cache_dir=cache_dir, local_files_only=True)

    elif args.task_type == "text-classification":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, device_map=args.device, cache_dir=cache_dir, local_files_only=True)
    
    model.config.pad_token_id = tokenizer.pad_token_id

    # 根据攻击设置选择是否添加LoRA
    if args.attk_part == "full":
        return model
    else:
        # 获取LoRA配置
        target_modules_dict = {"gpt2":["c_attn"], "TinyLlama/TinyLlama_v1.1":["q_proj", "v_proj"]}
        target_modules = target_modules_dict[args.model_name]
        lora_config = LoraConfig(
            base_model_name_or_path=args.model_name,
            task_type=TaskType.CAUSAL_LM if args.task_type == "text-generation" else TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.rank * args.alpha_multi,
            target_modules=target_modules
        )
        
        # 添加LoRA矩阵
        model = get_peft_model(model, peft_config=lora_config)
        logging.info(f"---------{args.model_name} add Lora matrix successfully!-----------")
        if args.finetuned:
            # 使用微调过的LoRA矩阵进行攻击
            finetuned_path = f"./lora_weight/{args.model_name}_{args.dataset}_r{args.rank}.pt"
            lora_param = torch.load(finetuned_path, map_location=torch.device('cpu'))
            model.load_state_dict(lora_param, strict=False)
            logging.info(f"---------{args.model_name} load parameters from {finetuned_path}-----------")
    return model

# 获取梯度和模型信息
def get_model(args, tokenizer):
    model = get_pertrained_model(args, tokenizer).eval()
    if args.model_name == "gpt2":
        embedding_layer = model.transformer.wte
    elif args.model_name == "TinyLlama/TinyLlama_v1.1":
        embedding_layer = model.model.model.embed_tokens
    
    if args.position_swap:
        from transformers import AutoModelForCausalLM
        lm_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device, cache_dir="./cache", local_files_only=True).eval()
        logging.info(f"---------Load {args.model_name} LMHeadModel successfully!-----------")
    else:
        lm_model = None
    
    return model, lm_model, embedding_layer.weight.data.clone()

# 获取真实的梯度信息
def get_target_gradient(args, model, data):
    output = model(
        input_ids=data["input_ids"].to(args.device),
        attention_mask=data["attention_mask"].to(args.device),
        labels=data["labels"].to(args.device)
    )
    loss = output.loss

    gradient = torch.autograd.grad(
        outputs=loss,
        # inputs=get_params_list_for_grad(args, model),
        inputs=[param for param in model.parameters() if param.requires_grad],
        allow_unused=True,
        materialize_grads=True
    )
    # 对梯度添加防御
    apply_gradient_defense(args, gradient)
    return gradient[:-1]

# 防御实验
def apply_gradient_defense(args, gradient):
    # 添加DP防御
    if args.defense == "DP":
        C = args.thresholds
        sigma = args.sigma
        # 计算梯度范数
        grads_sq = torch.tensor(0.0, device=args.device)
        for grad in gradient:
            grad_sq = grad.pow(2).sum()
            grads_sq += grad_sq
        grads_norm = grads_sq.sqrt()
        # 缩放系数
        scale = min(1.0, C / (grads_norm + 1e-8))
        
        # 根据缩放系数进行裁剪
        for grad in gradient:
            grad.mul_(scale)
            # 加噪
            noise = torch.normal(
                mean=0.0,
                std=sigma * C,
                size=grad.shape,
                device=args.device,
                dtype=grad.dtype
            )
            grad.add_(noise)
    # 进行稀疏化
    elif args.defense == "Sparsification":
        spar_rate = args.spar_rate
        # 收集所有梯度
        all_grads = torch.cat([grad.view(-1) for grad in gradient])
        
        # 保留k个梯度
        k = int((1 - spar_rate) * all_grads.numel())
        if k > 0:
            # 阈值
            threshold = torch.topk(all_grads.abs(), k).values.min()

            # 按照阈值稀疏化
            for grad in gradient:
                mask = (grad.abs() >= threshold)
                grad.mul_(mask)
    # 进行梯度量化
    elif args.defense == "Quantization":
        quanti_method = args.quanti_method
        for grad in gradient:
            grad.data = grad.data.to(torch.float16).to(torch.float32) if quanti_method == "float16" else grad.data.to(torch.bfloat16).to(torch.float32)


# 获取optimizer和scheduler
def get_optimizer_and_scheduler(args, params):
    optimizer = torch.optim.AdamW(params=params, lr=args.lr)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=args.max_iters)
    return optimizer, scheduler

# 获取使用虚拟数据和标签训练获得的梯度
def get_dummy_gradient(args, model, dummy_data, true_labels, embedding_weight, dummy_label=None):
    model.zero_grad()
    
    if args.task_type == "text-generation":
        output = model(inputs_embeds=dummy_data)
        # 获取logits
        logits = output.logits
        labels = F.softmax(dummy_label, dim=-1)
        
        # 使logit和label对应
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:, :].contiguous()
        
        # 计算损失
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1, shift_labels.shape[-1])
        )
    # 分类任务
    else:
        if args.model_name == "TinyLlama/TinyLlama_v1.1":
            prefix_embed = embedding_weight[LLAMA_START_TOKEN].unsqueeze(0).unsqueeze(0)
            prefix_embed = prefix_embed.repeat(dummy_data.size(0), 1, 1)
            dummy_data = torch.cat([prefix_embed, dummy_data], dim=1)
        output = model(inputs_embeds=dummy_data, labels=true_labels.to(args.device))
        loss = output.loss

    gradient = torch.autograd.grad(
        outputs=loss,
        inputs=get_params_list_for_grad(args, model),
        create_graph=True,
        allow_unused=True,
        materialize_grads=True
    )

    return gradient, loss


# 用于计算梯度之间的损失
def compute_loss(args, target_gradient, dummy_gradient, iter_step=None):
    # DLG,使用欧几里得距离度量
    loss = 0
    if args.loss == "dlg":

        for dum_grad, tar_grad in zip(dummy_gradient, target_gradient):
            loss += (dum_grad - tar_grad).pow(2).sum()
    elif args.loss == "L2L1":

        for dum_grad, tar_grad in zip(dummy_gradient, target_gradient):
            loss += (dum_grad - tar_grad).pow(2).sum() + args.tag_alpha * (dum_grad - tar_grad).abs().sum()
    elif args.loss == "cos":

        for dum_grad, tar_grad in zip(dummy_gradient, target_gradient):
            loss += 1.0 - (dum_grad * tar_grad).sum() / ((dum_grad.view(-1).norm(p=2) * tar_grad.view(-1).norm(2)) + 1e-8)
        loss /= len(dummy_gradient)
    elif args.loss == "KL":
        for dum_grad, tar_grad in zip(dummy_gradient, target_gradient):
            KL_loss_col = torch.nn.functional.kl_div(input=torch.log(torch.nn.functional.softmax(dum_grad.T, dim=0)), target=torch.nn.functional.softmax(tar_grad.T, dim=0), reduction="batchmean")
            KL_loss_row = torch.nn.functional.kl_div(input=torch.log(torch.nn.functional.softmax(dum_grad, dim=0)), target=torch.nn.functional.softmax(tar_grad, dim=0), reduction="batchmean")
            cos_loss = 1.0 - (dum_grad * tar_grad).sum() / ((dum_grad.view(-1).norm(p=2) * tar_grad.view(-1).norm(2)) + 1e-8)
            loss += KL_loss_col + KL_loss_row + cos_loss
        loss /= len(dummy_gradient)

    return loss


# 对位置进行重排
def position_swap(args, model, lm_model, dummy_data, true_labels, target_gradient, max_len, embedding_weight):
    closest_batch_token = get_closest_tokens(args=args, model=model, dummy_data=dummy_data)
    best_embedding, best_loss = None, None
    changed = None
    best_ids = closest_batch_token
    for sen_id in range(dummy_data.data.shape[0]):
        logging.info(f"---------------swap {sen_id+1}/{args.batch_size} sequence")
        for sample_idx in range(200):
            perm_ids = np.arange(dummy_data.shape[1])
            try:
                if sample_idx != 0:
                    if sample_idx % 4 == 0 and max_len[sen_id] > 2: # swap two tokens
                        i, j = 1 + np.random.randint(max_len[sen_id] - 2), 1 + np.random.randint(max_len[sen_id] - 2)
                        perm_ids[i], perm_ids[j] = perm_ids[j], perm_ids[i]
                        # swap_info = f"swap {tokenizer.decode(best_ids[sen_id][i])} with {tokenizer.decode(best_ids[sen_id][j])}"
                    elif sample_idx % 4 == 1 and max_len[sen_id] > 2: # move a token to another place
                        i = 1 + np.random.randint(max_len[sen_id] - 2)
                        j = 1 + np.random.randint(max_len[sen_id] - 1)
                        if i < j:
                            perm_ids = np.concatenate([perm_ids[:i], perm_ids[i+1:j], perm_ids[i:i+1], perm_ids[j:]])
                            # swap_info = f"move {tokenizer.decode(best_ids[sen_id][i])} before {tokenizer.decode(best_ids[sen_id][j])}"
                        else:
                            perm_ids = np.concatenate([perm_ids[:j], perm_ids[i:i+1], perm_ids[j:i], perm_ids[i+1:]])
                            # swap_info =f"move {tokenizer.decode(best_ids[sen_id][j])} before {tokenizer.decode(best_ids[sen_id][i])}"
                    elif sample_idx % 4 == 2: # move a sequence to another place
                        b = 1 + np.random.randint(max_len[sen_id] - 1)
                        e = 1 + np.random.randint(max_len[sen_id] - 1)
                        if b > e:
                            b, e = e, b
                        p = 1 + np.random.randint(max_len[sen_id] - 1 - (e-b))
                        if p >= b:
                            p += e-b
                        if p < b:
                            perm_ids = np.concatenate([perm_ids[:p], perm_ids[b:e], perm_ids[p:b], perm_ids[e:]])
                        elif p >= e:
                            perm_ids = np.concatenate([perm_ids[:b], perm_ids[e:p], perm_ids[b:e], perm_ids[p:]])
                        else:
                            assert False
                    elif sample_idx % 4 == 3 and max_len[sen_id] > 2: # take some prefix and put it at the end
                        i = 1 + np.random.randint(max_len[sen_id] - 2)
                        perm_ids = np.concatenate([perm_ids[:1], perm_ids[i:max_len[sen_id] - 1], perm_ids[1:i], perm_ids[max_len[sen_id] - 1:]])
                        # swap_info = f"put 1~{tokenizer.decode(best_ids[sen_id][i])} to the end"
                new_ids = best_ids.clone()
                new_ids[sen_id] = best_ids[sen_id, perm_ids]
                
                new_embedding = dummy_data.clone()
                new_embedding[sen_id] = dummy_data[sen_id, perm_ids, :].clone()
            
                with sdpa_kernel(SDPBackend.MATH):
                    dummy_gradient, _ = get_dummy_gradient(args=args, model=model, dummy_data=new_embedding, embedding_weight=embedding_weight, true_labels=true_labels)
                rec_loss = compute_loss(args, target_gradient, dummy_gradient)
                ppl = lm_model(input_ids=new_ids, labels=new_ids).loss
                new_loss = rec_loss + ppl * args.coeff_ppl

                if (best_loss is None) or (new_loss < best_loss):
                    logging.info(f"new_loss:{new_loss}(rec_loss:{rec_loss}, ppl:{ppl})")
                    best_embedding = new_embedding
                    best_loss = new_loss
                    if args.enable_multi_swap:
                        best_ids = new_ids
                        if sample_idx != 0:
                            changed = sample_idx % 4
                    # if not( changed is None ):
                            change = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][changed]
                            logging.info(change)
                            # logging.info(swap_info)
                        dummy_data.data = best_embedding
                    else:
                        if sample_idx != 0:
                            changed = sample_idx % 4
            except Exception as e:
                logging.info(e)
                logging.info(max_len, i, j, len(perm_ids))
        if not( changed is None ) and not args.enable_multi_swap:
            change = ['Swapped tokens', 'Moved token', 'Moved sequence', 'Put prefix at the end'][changed]
            logging.info( change )
            dummy_data.data = best_embedding


# 进行梯度反演攻击
def gradient_inversion_attack(args, model, lm_model, target_gradient, dummy_data, dummy_label, tokenizer, ground_truth_data, embedding_weight, pads):
    # 获取optimizer和scheduler
    params = [dummy_data, dummy_label] if args.task_type == "text-generation" else [dummy_data]
    data_size = dummy_data.shape
    optimizer, scheduler = get_optimizer_and_scheduler(args, params=params)
    task_loss_recorder = ValueRecorder()
    time_start = time.time()
    embedding_norm = embedding_weight.data.norm(p=2, dim=1).mean()
    max_norm = torch.max(embedding_weight.data.norm(p=2, dim=1))
    min_norm = torch.min(embedding_weight.data.norm(p=2, dim=1))
    

    for iter_step in range(args.max_iters):
        # 使用获取的梯度优化数据
        def closure():
            optimizer.zero_grad()
            with sdpa_kernel(SDPBackend.MATH):
                dummy_gradient, task_loss = get_dummy_gradient(args=args, model=model, dummy_data=dummy_data, true_labels=ground_truth_data["labels"], embedding_weight=embedding_weight, dummy_label=dummy_label)
            attack_loss = compute_loss(args=args, target_gradient=target_gradient, dummy_gradient=dummy_gradient, iter_step=iter_step)
            if args.embedding_norm:
                norm_loss = ( dummy_data.norm(p=2,dim=2).mean() - embedding_norm ).square() * args.coeff_reg
                attack_loss += norm_loss
            task_loss_recorder.setValue(task_loss)

            attack_loss.backward(
                inputs=params,
                create_graph=False
            )

            # 对梯度进行裁剪
            if args.clip is not None:
                norm_list = []
                with torch.no_grad():
                    for data in params:
                        grad_norm = data.grad.norm()
                        # print(grad_norm)
                        norm_list.append(grad_norm)
                        if grad_norm > args.clip:
                            data.grad.mul_(args.clip / (grad_norm + 1e-8))
            # print(f"| iter:{iter_step} | attack_loss:{attack_loss} | task_loss:{task_loss} | grad_norm:{norm_list}")
            return attack_loss

        attack_loss = optimizer.step(closure=closure)
        scheduler.step()

        #对于填充的token，将其变为pad_token
        fix_special_token(args, dummy_data, embedding_weight, tokenizer.pad_token_id, pads)
        

        # 输出损失
        if (iter_step + 1) % args.print_iters == 0:
        
            loss_info = f"Atk.loss: {attack_loss.item():2.5f}"
            norm = dummy_data.norm(p=2,dim=2).mean()
            time_used = time.time() - time_start
            logging.info(f"| It:{iter_step + 1} | {loss_info} | Task.loss: {task_loss_recorder.getValue():2.5f} | Norm: {norm:2.5f} | Time {time_used:2.5f}s |")

        # 优化过程中对序列进行处理
        if (iter_step + 1) % args.swap_every == 0:
            
            if pads is None:
                max_len = [dummy_data.shape[1]]*dummy_data.shape[0]
            else:
                max_len = pads
            print("mex_len:", max_len)
            # 进行随机重排，恢复序列的位置信息
            if args.position_swap:
                recovered_batch_token = get_closest_tokens(args=args, model=model, dummy_data=dummy_data)

                recovered_batch_data, ground_truth_batch_data = [], []
                for i in range(args.batch_size):
                    # recovered_batch_data.append(remove_padding(args, tokenizer, recovered_batch_token[i]))
                    # ground_truth_batch_data.append(remove_padding(args, tokenizer, ground_truth_data["input_ids"][i]))
                    recovered_batch_data.append(tokenizer.decode(recovered_batch_token[i]))
                    ground_truth_batch_data.append(tokenizer.decode(ground_truth_data["input_ids"][i]))
                # 使用余弦距离恢复出的数据
                logging.info(f"============{iter_step + 1} recovered data before swap================")
                for idx, data in enumerate(recovered_batch_data):
                    logging.info(f"recovered_text_{idx}: {data}")
                logging.info("===========================================================")

                logging.info("---------------Attempt to swap position---------------")
                position_swap(args, model, lm_model, dummy_data, ground_truth_data["labels"], target_gradient, max_len, embedding_weight)

        # 计算一次还原出的数据
        if (iter_step + 1) % args.snapshot_iters == 0:
            recovered_batch_token = get_closest_tokens(args=args, model=model, dummy_data=dummy_data)

            recovered_batch_data, ground_truth_batch_data = [], []
            for i in range(args.batch_size):
                recovered_batch_data.append(tokenizer.decode(remove_padding(args, tokenizer, recovered_batch_token[i])))
                ground_truth_batch_data.append(tokenizer.decode(remove_padding(args, tokenizer, ground_truth_data["input_ids"][i])))
                # recovered_batch_data.append(tokenizer.decode(recovered_batch_token[i]))
                # ground_truth_batch_data.append(tokenizer.decode(ground_truth_data["input_ids"][i]))
            
            # 使用余弦距离恢复出的数据
            logging.info(f"============{iter_step + 1} recovered data================")
            for idx, data in enumerate(recovered_batch_data):
                logging.info(f"recovered_text_{idx}: {data}")
            logging.info("===========================================================")

            # 计算rouge指标
            metrics, recovered_batch_data = eval_attack(args=args, ground_truth_batch_data=ground_truth_batch_data, recovered_batch_data=recovered_batch_data)
            logging.info(f"attack result:{metrics}")

    return recovered_batch_data, ground_truth_batch_data, recovered_batch_token
