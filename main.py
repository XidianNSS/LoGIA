import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import time
import json
import logging
import argparse
from evaluate import load
from utils import *
from train import *
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def get_args():
    parser = argparse.ArgumentParser()
    
    # model/task
    parser.add_argument("--model_name", type=str, default="gpt2", choices=["gpt2", "TinyLlama/TinyLlama_v1.1"])
    parser.add_argument("--task_type", type=str, default="text-classification")
    parser.add_argument("--dataset", type=str, default="cola", choices=["cola", "sst2", "rotten_tomatoes", "wikitext-2"])
    parser.add_argument("--split", type=str, default="train")

    # LoRA hyparameters
    parser.add_argument("--attk_part", type=str, default="AB", choices=["full", "AB", "A", "B"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha_multi", type=int, default=2)    # alpha是rank的几倍，一般设置为2~4
    parser.add_argument("--finetuned", action="store_true", default=True)

    # attack hyparameters
    parser.add_argument("--loss", type=str, default="cos", choices=["dlg", "L2L1", "cos"])
    parser.add_argument("--inputs", type=int, default=100)         # 一次攻击处理多少个批次的数据
    parser.add_argument("--batch_size", type=int, default=1)     # 批次大小
    parser.add_argument("--distance_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--tag_alpha", type=float, default=0.01)
    parser.add_argument("--know_padding", type=bool, default=True)
    parser.add_argument("--init", type=str, default="random", choices=["random", "svd", "clip_random", "candidate_random"])
    # embedding范数正则化
    parser.add_argument("--embedding_norm", action="store_true", default=False)
    parser.add_argument("--coeff_reg", type=float, default=0.1)     # embedding长度正则化
    parser.add_argument("--norm_clamp", action="store_true", default=False)
    # 位置重排
    parser.add_argument("--position_swap", action="store_true", default=False)   # 是否进行token的交换
    parser.add_argument("--swap_every", type=int, default=100)      # 每多少步迭代进行一次优化
    parser.add_argument("--coeff_ppl", type=float, default=0.1)
    parser.add_argument("--enable_multi_swap", action="store_true", default=False)
    
    # parser.add_argument("--map_init", action="store_true", default=False)
    
    # multi stages attack
    parser.add_argument("--change_steps", type=int, default=None)
    parser.add_argument("--map_to_real", action="store_true", default=False)

    # defense experiment
    parser.add_argument("--defense", type=str, default=None, choices=[None, "DP", "Sparsification", "Quantization"])
    parser.add_argument("--thresholds", type=float, default=10)
    parser.add_argument("--sigma", type=float, default=1e-5)
    parser.add_argument("--spar_rate", type=float, default=0.3)
    parser.add_argument("--quanti_method", type=str, default="float16", choices=["float16", "bfloat16"])

    # training hyparameters
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--max_iters", type=int, default=4000)
    parser.add_argument("--snapshot_iters", type=int, default=100)
    parser.add_argument("--print_iters", type=int, default=25)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = get_args()
    set_loginfo(args)
    logging.info(f"{args}")
    set_seed(args.seed)
    scorer = load("./eval/rouge")

    # 1、获取用于恢复的数据，共inputs * batch_size个
    ground_truth_data_list, tokenizer = prepare_data(args)
    # 用于记录所有的攻击结果和真实数据
    total_recovered_data = []
    total_ground_truth_data = []

    # 若执行映射操作，则每优化change_steps次执行一次映射操作 
    if args.change_steps is not None:
        num_stages = args.max_iters // args.change_steps
        args.max_iters = args.change_steps
        multi_stages = True
    else:
        multi_stages = False

    model, lm_model, embedding_weight = get_model(args, tokenizer)

    for idx, ground_truth_data in enumerate(ground_truth_data_list):
        logging.info(f"\n------------Start recovering id_{idx+1}/{args.inputs} data------------\n")
        logging.info(f"========Ground Truth Data==========")
        for i, seq in enumerate(ground_truth_data["text"]):
            logging.info(f"text_{i}: {seq}")
        logging.info("===================================")
        logging.info(f"seq_length: {[len(input_ids) for input_ids in ground_truth_data['input_ids']]}")

        # 2、获取真实梯度信息
        target_gradient = get_target_gradient(args, model, ground_truth_data)

        # 3、生成虚拟数据和标签
        seq_length = len(ground_truth_data["input_ids"][0]) if args.model_name == "gpt2" else len(ground_truth_data["input_ids"][0]) - 1
        pads = get_pads(args, ground_truth_data, tokenizer) # 填充信息
        data_size = (args.batch_size, seq_length, embedding_weight.data.shape[-1])  # 虚拟数据大小
        label_size =(args.batch_size, seq_length, tokenizer.vocab_size) if args.task_type == "text-generation" else (args.batch_size, 2)
        # 初始化
        dummy_data = get_init(args, model, data_size, target_gradient, ground_truth_data["labels"], embedding_weight, pads, tokenizer)
        dummy_label = generate_dummy_data(args, label_size)
        fix_special_token(args, dummy_data, embedding_weight, tokenizer.pad_token_id, pads)

        # 查看初始化的文本数据
        initial_token_list = get_closest_tokens(args=args, model=model, dummy_data=dummy_data)
        initial_text_list = []
        logging.info(f"=============Initial Text:=============")
        for idx, initial_token in enumerate(initial_token_list):
            init_text = tokenizer.decode(remove_padding(args, tokenizer, initial_token))
            initial_text_list.append(init_text)
            logging.info(f"initial_text_{idx}:{init_text}")
        logging.info("===================================")

        # 初始化时是否将dummy embedding 转换为真实的embedding
        # if args.map_init:
        #     dummy_data = map_to_real_embedding(tokenizer, initial_token_list, embedding_weight)
        #     logging.info("------------map the dummy embedding to real embedding------------")

        # 4、优化虚拟标签和数据
        if multi_stages:
            
            args.loss = "cos"
            for stage in range(num_stages):
                logging.info(f"\n------------Stage {stage + 1}/{num_stages}: Start attacking part {args.attk_part} with loss {args.loss}------------")
                target_gradient = get_target_gradient(args, model, ground_truth_data)
                recovered_batch_data, ground_truth_batch_data, recovered_batch_token = gradient_inversion_attack(args, model, lm_model, target_gradient, dummy_data, dummy_label, tokenizer, ground_truth_data, embedding_weight, pads)

                # 将dummy embedding 转换为真实的embedding
                if args.map_to_real:
                    logging.info("------------map the dummy embedding to real embedding------------")
                    dummy_data = map_to_real_embedding(tokenizer, recovered_batch_token, embedding_weight)
                    print(dummy_data.shape)
        else:
            
            recovered_batch_data, ground_truth_batch_data, recovered_batch_token = gradient_inversion_attack(args, model, lm_model, target_gradient, dummy_data, dummy_label, tokenizer, ground_truth_data, embedding_weight, pads)

        # 计算当轮和至今为止的攻击指标
        total_recovered_data += recovered_batch_data
        total_ground_truth_data += ground_truth_batch_data

        avg_metrics = scorer.compute(predictions=total_recovered_data, references=total_ground_truth_data)
        logging.info(f"Avg Metrics:{avg_metrics}")


if __name__ == "__main__":
    main()
    # init_test()