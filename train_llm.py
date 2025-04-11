import argparse
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
from modify_llm import apply_lora, apply_adapters, add_extra_block
from data_pipeline import AlpacaDataset, custom_collate_fn
from tqdm import tqdm
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLM for training.")
    parser.add_argument('--modification_type', type=str, required=True,
                        choices=['lora', 'adapter', 'extra_block'],
                        help="Type of architecture modification")
    parser.add_argument('--lora_rank', type=int, default=8,
                        help="Rank for LoRA modification")
    parser.add_argument('--adapter_dim', type=int, default=64,
                        help="Hidden dimension for adapters")
    parser.add_argument('--data_path', type=str, default='alpaca_data.json',
                        help="Path to Alpaca dataset")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for dataloader")
    parser.add_argument('--lr', type=float, default=2e-5,
                        help="Learning rate for optimizer")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    dataset = AlpacaDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer)
    )
    
    # Load and modify model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    if args.modification_type == 'lora':
        model = apply_lora(model, args.lora_rank)
    elif args.modification_type == 'adapter':
        model = apply_adapters(model, args.adapter_dim)
    elif args.modification_type == 'extra_block':
        model = add_extra_block(model)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1):
        for batch in tqdm(dataloader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1} done. Last batch loss: {loss.item()}")
    
    # Verify data and model integration
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        print(f"Batch prepared with input_ids shape: {inputs['input_ids'].shape}")
        break

    os.makedirs("results", exist_ok=True)
    model.save_pretrained("results/final_model")
    tokenizer.save_pretrained("results/final_model")

    print("Model and data are initialized and ready for training.")

if __name__ == "__main__":
    main()