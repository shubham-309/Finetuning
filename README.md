# Complete Guide to Building and Deploying a Large Language Model Pipeline

This README provides a comprehensive guide to developing, fine-tuning, and deploying a GPT-2-based large language model (LLM) using the Alpaca dataset. It covers the entire pipeline—data preprocessing, model customization, training preparation, and deployment—offering detailed explanations of each step, the techniques used, and their rationale. Whether you're an AI/ML engineer or a developer, this document serves as a standalone resource to understand and implement the workflow.

## Table of Contents
1. [Overview](#overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Architecture Customization](#model-architecture-customization)
4. [Training Preparation](#training-preparation)
5. [Deployment Strategy](#deployment-strategy)
6. [Execution Instructions](#execution-instructions)
7. [Conclusion](#conclusion)

---

## Overview

This pipeline is designed to fine-tune and deploy a GPT-2 model from Hugging Face’s `transformers` library, using the [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca) for instruction-following tasks. The process includes:

- **Data Preprocessing:** Preparing raw text data for causal language modeling.
- **Model Customization:** Enhancing GPT-2 with efficient fine-tuning techniques.
- **Training Preparation:** Setting up the model and data for fine-tuning.
- **Deployment:** Serving the model to handle high-concurrency inference requests.

Rather than relying on external libraries like PEFT, this pipeline implements fine-tuning techniques from scratch, showcasing a deep understanding of transformer modifications and offering full control over the process.

---

## Data Preprocessing

### Purpose
Data preprocessing transforms the Alpaca dataset—a set of instruction-following examples—into a format suitable for training GPT-2 in a causal language modeling framework.

### Implementation
- **File:** `data_pipeline.py`
- **Steps:**
  - **Loading:** The `AlpacaDataset` class reads the dataset (`alpaca_data.json`), treating each JSON entry as a sample with "instruction," "input" (optional), and "output" fields.
  - **Formatting:** Each sample is combined into a single string:
    - With "input": `Instruction: {instruction}\nInput: {input}\nResponse: {output}`
    - Without "input": `Instruction: {instruction}\nResponse: {output}`
  - **Tokenization:** The GPT-2 tokenizer (`GPT2Tokenizer`) converts strings into token IDs, using the `eos_token` as the padding token.
  - **Batching:** A custom `collate_fn` in the `Dataloader` tokenizes and dynamically pads sequences to the longest in each batch. Labels match `input_ids`, with padding tokens set to `-100` to exclude them from the loss.

### Techniques and Rationale
- **Dynamic Padding:** Adjusts padding to each batch’s maximum length, reducing memory usage compared to fixed padding.
- **Causal Language Modeling Format:** Merges instruction and response into one sequence, aligning with GPT-2’s autoregressive design.
- **On-the-Fly Tokenization:** Performed during batching to save disk space and support flexible batch sizes.

### Why GPT-2 Tokenizer?
It ensures compatibility with the GPT-2 model and efficiently processes diverse text using byte-pair encoding (BPE).

---

## Model Architecture Customization

### Purpose
Customizing GPT-2 enhances its fine-tuning efficiency and adaptability for specific tasks, avoiding full retraining of the model’s parameters.

### Implementation
- **File:** `modify_llm.py`
- **Base Model:** `GPT2LMHeadModel` from `transformers`.
- **Techniques:**
  1. **LoRA (Low-Rank Adaptation):**
     - Adds low-rank matrices (`A` and `B`) to attention layers, keeping original weights frozen.
     - Implemented via a `LoRAAttention` class, adjustable with `--lora_rank` (e.g., 8).
  2. **Adapters:**
     - Inserts small MLPs after attention and feed-forward layers, with residual connections.
     - Uses a `GPT2BlockWithAdapters` class, configurable with `--adapter_dim` (e.g., 64).
  3. **Extra Transformer Block:**
     - Appends an additional `GPT2Block` to increase model depth.
     - Applied via the `add_extra_block` function.
- **Command-Line Interface:** `argparse` enables dynamic modification selection (e.g., `--modification_type lora`).

### Techniques and Rationale
- **LoRA:** Reduces trainable parameters (e.g., from millions to thousands), making fine-tuning efficient.
- **Adapters:** Adds task-specific capacity with minimal overhead (~1-2% of parameters).
- **Extra Block:** Boosts model capacity for complex tasks, though it increases compute needs.

### Why Custom Implementation?
Building these techniques from scratch provides flexibility and demonstrates a thorough understanding of transformer architecture.

---

## Training Preparation

### Purpose
This stage integrates the preprocessed data and customized model, preparing them for fine-tuning without running the full training loop.

### Implementation
- **File:** `train_llm.py`
- **Steps:**
  - Imports `AlpacaDataset` and `custom_collate_fn` from `data_pipeline.py`.
  - Loads GPT-2 and applies modifications from `modify_llm.py`.
  - Sets up the `Dataloader` with the dataset.
  - Configures the `AdamW` optimizer with trainable parameters.
- **Command-Line Options:** Supports `--modification_type`, `--lora_rank`, `--adapter_dim`, `--batch_size`, `--lr`, and `--data_path`.

### Techniques and Rationale
- **Modular Design:** Separates components into reusable files for maintainability.
- **Dynamic Configuration:** Allows experimentation via command-line arguments.
- **AdamW Optimizer:** Stabilizes training with adaptive learning rates and weight decay.

### Why This Approach?
It validates the pipeline’s components before resource-heavy training, ensuring compatibility and correctness.

---

## Deployment Strategy

### Purpose
The strategy ensures the fine-tuned model can handle ~500 concurrent inference requests with low latency and high throughput.

### Implementation
- **Hardware:** Multi-GPU servers (e.g., 4 NVIDIA A100s per server, 5-10 servers total).
- **Framework:** vLLM for optimized LLM inference.
- **Techniques:**
  - **In-Flight Batching:** Processes requests dynamically as they arrive.
  - **KV Caching:** Stores attention key-value pairs to speed up generation.
  - **Prompt Caching:** Uses Redis to cache frequent prompt responses.
- **Setup:**
  - **Load Balancer:** Distributes requests (e.g., Nginx).
  - **Orchestration:** Kubernetes with horizontal pod autoscaling (HPA).
  - **Optimizations:** FP16 precision and paged attention.

### Techniques and Rationale
- **In-Flight Batching:** Minimizes latency for real-time workloads.
- **KV Caching:** Accelerates autoregressive generation.
- **Prompt Caching:** Reduces redundant computation for common prompts.
- **vLLM:** Offers built-in optimizations for LLMs.

### Why This Strategy?
It balances performance, scalability, and efficiency, meeting high-concurrency demands.

---

## Execution Instructions

### Prerequisites
- Python 3.8+
- Libraries: `torch`, `transformers`
- Dataset: `alpaca_data.json`

### Commands
1. **Data Preprocessing:**
   ```bash
   python data_pipeline.py
   ```
   ### Output:
   *Sample batch:*
    *input_ids shape: torch.Size([16, 108])*
    *attention_mask shape: torch.Size([16, 108])*
    *labels shape: torch.Size([16, 108])*
    *Sample input_ids: tensor([ 6310,  2762,    25,  7343,   642, 15162,  4568,   661,   714,   466])*
2. **Model Modification:**
    ```bash
    python modify_llm.py --modification_type lora --lora_rank 8
    python modify_llm.py --modification_type adapter --adapter_dim 64
    python modify_llm.py --modification_type extra_block
    ```
    ### Output:
    # Applied LoRA with rank 8
    ```bash
    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): LoRAAttention(
              (c_attn): Conv1D(nf=2304, nx=768)
              (c_proj): Conv1D(nf=768, nx=768)
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D(nf=3072, nx=768)
              (c_proj): Conv1D(nf=768, nx=3072)
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
    ```

    # Applied adapters with dimension 64
    ``` bash
    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2BlockWithAdapters(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): Conv1D(nf=2304, nx=768)
              (c_proj): Conv1D(nf=768, nx=768)
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D(nf=3072, nx=768)
              (c_proj): Conv1D(nf=768, nx=3072)
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (attn_adapter): Sequential(
              (0): Linear(in_features=768, out_features=64, bias=True)
              (1): ReLU()
              (2): Linear(in_features=64, out_features=768, bias=True)
            )
            (mlp_adapter): Sequential(
              (0): Linear(in_features=768, out_features=64, bias=True)
              (1): ReLU()
              (2): Linear(in_features=64, out_features=768, bias=True)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
    ```
    # Added an extra transformer block
    ```bash
    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-12): 13 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): Conv1D(nf=2304, nx=768)
              (c_proj): Conv1D(nf=768, nx=768)
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D(nf=3072, nx=768)
              (c_proj): Conv1D(nf=768, nx=3072)
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )
    ```
3. **Training Preparation:**
    ```bash
    python train_llm.py --modification_type lora --lora_rank 8 --data_path alpaca_data.json --batch_size 16 --lr 2e-5
    ```
        
        
