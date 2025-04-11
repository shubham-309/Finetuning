import argparse
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block

# Custom LoRA Attention Layer
class LoRAAttention(GPT2Attention):
    def __init__(self, config, rank=8):
        super().__init__(config)
        self.rank = rank
        d_model = config.n_embd
        # LoRA parameters for c_attn (d_model, 3*d_model)
        self.lora_A = nn.Parameter(torch.randn(d_model, rank) * 0.01)  # Gaussian init
        self.lora_B = nn.Parameter(torch.zeros(rank, 3 * d_model))    # Zero init
    
    def forward(self, hidden_states, *args, **kwargs):
        # Original qkv computation
        qkv = self.c_attn(hidden_states)
        # Add LoRA adaptation: (batch, seq, d_model) @ (d_model, r) @ (r, 3*d_model)
        lora_term = (hidden_states @ self.lora_A) @ self.lora_B
        qkv = qkv + lora_term
        return super().forward(hidden_states, *args, **kwargs)

# Custom Block with Adapters
class GPT2BlockWithAdapters(GPT2Block):
    def __init__(self, config, adapter_dim=64):
        super().__init__(config)
        d_model = config.n_embd
        # Adapter after attention
        self.attn_adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, d_model)
        )
        # Adapter after MLP
        self.mlp_adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, d_model)
        )
        # Initialize adapters with small weights
        nn.init.zeros_(self.attn_adapter[2].weight)
        nn.init.zeros_(self.mlp_adapter[2].weight)
    
    def forward(self, hidden_states, *args, **kwargs):
        # Attention with adapter
        attn_output, *rest = self.attn(hidden_states, *args, **kwargs)
        attn_output = attn_output + self.attn_adapter(attn_output)  # Residual
        # MLP with adapter
        mlp_output = self.mlp(attn_output)
        mlp_output = mlp_output + self.mlp_adapter(mlp_output)      # Residual
        return (mlp_output, *rest)

# Modification Functions
def apply_lora(model, rank):
    # Replace attention layers with LoRA attention.
    for block in model.transformer.h:
        block.attn = LoRAAttention(model.config, rank)
    return model

def apply_adapters(model, adapter_dim):
    # Replace blocks with adapter-augmented blocks.
    for i, block in enumerate(model.transformer.h):
        model.transformer.h[i] = GPT2BlockWithAdapters(model.config, adapter_dim)
    return model

def add_extra_block(model):
    # Append an extra transformer block.
    new_block = GPT2Block(model.config)
    model.transformer.h.append(new_block)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Modify GPT-2 architecture.")
    parser.add_argument('--modification_type', type=str, required=True,
                        choices=['lora', 'adapter', 'extra_block'],
                        help="Type of architecture modification")
    parser.add_argument('--lora_rank', type=int, default=8,
                        help="Rank for LoRA modification")
    parser.add_argument('--adapter_dim', type=int, default=64,
                        help="Hidden dimension for adapters")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load base model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Apply modification
    if args.modification_type == 'lora':
        model = apply_lora(model, args.lora_rank)
        print(f"Applied LoRA with rank {args.lora_rank}")
    elif args.modification_type == 'adapter':
        model = apply_adapters(model, args.adapter_dim)
        print(f"Applied adapters with dimension {args.adapter_dim}")
    elif args.modification_type == 'extra_block':
        model = add_extra_block(model)
        print("Added an extra transformer block")
    
    # Print modified architecture summary
    print(model)
    # Model is ready for training

if __name__ == "__main__":
    main()

# # Apply LoRA with rank 8
# python3 modify_llm.py --modification_type lora --lora_rank 8

# # Apply adapters with dimension 64
# python3 modify_llm.py --modification_type adapter --adapter_dim 64

# # Add an extra transformer block
# python3 modify_llm.py --modification_type extra_block