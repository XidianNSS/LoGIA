# SEMG

## 1. Installation & Running Steps

### **Step 1: Create Conda Environment**

```bash
conda create -n semg python==3.12
```

### **Step 2: Install Required Dependencies**

```bash
conda activate semg
pip install -r requirements.txt
```

### **Step 3: LoRA Matrix Fine-tuning**

This step prevents the LoRA **A matrix** from having extremely small gradients.

```bash
python train_lora.py \
    --model_name gpt2 \
    --dataset cola \
    --rank 8
```

The fine-tuned LoRA matrices will be saved in the **`lora_weight/`** directory.

### **Step 4: Run the SEMG Attack Script**

Inside the *scripts* directory:

```bash
bash scripts/semg_example.sh
```

---

## 2. Key Parameters

| Parameter           | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| **model_name**      | Model name                                                  |
| **dataset**         | Dataset used for attack experiments                         |
| **loss**            | Distance metric for dummy gradient vs. true gradient        |
| **inputs**          | Number of samples used in the attack                        |
| **batch_size**      | Batch size of data                                   |
| **distance_metric** | Distance metric for mapping dummy embeddings to real tokens |
| **init**            | Initialization strategy for dummy inputs                    |
| **embedding_norm**  | Whether to add embedding norm regularization                |
| **position_swap**   | Whether to randomly permute token positions                 |
| **change_steps**    | Interval for performing Embedding Discrete Mapping          |
| **map_to_real**     | Whether to perform Embedding Discrete Mapping               |
| **defense**         | Defense strategy used against SEMG                          |
| **lr**              | Learning rate                                               |
| **max_iters**       | Maximum optimization iterations                             |

---

## 3. Directory and File Overview

```
SEMG/
│
├── cache/                # Cached model files and datasets
├── eval/                 # Evaluation utilities for attack metrics
├── logs/                 # Training and attack logs
├── lora_weight/          # Stored LoRA matrices
├── scripts/              # Executable experiment scripts
├── utils/                # Dataset processing, dummy input initialization, token utilities
│
├── main.py               # Main entry point of SEMG
├── train.py              # Dummy gradient computation, loss calculation, optimization
├── train_lora.py         # LoRA fine-tuning script
│
└── README.md
