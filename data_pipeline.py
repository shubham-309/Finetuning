import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

class AlpacaDataset(Dataset):
    def __init__(self, data):
        """Initialize dataset with loaded Hugging Face dataset."""
        self.data = data
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        print(f"Item {idx}: {item}")
        return item

def custom_collate_fn(batch, tokenizer, max_length=512):
    """Tokenize and pad a batch of Alpaca-style samples."""
    prompts = []
    for sample in batch:
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        output = sample["output"]

        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            prompt = f"Instruction: {instruction}\nResponse: {output}"
        
        prompts.append(prompt)

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    labels = input_ids.clone()
    labels[input_ids == tokenizer.pad_token_id] = -100  # Mask padding in loss

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    # Load the Alpaca dataset
    raw_dataset = load_dataset("tatsu-lab/alpaca")['train']
    alpaca_dataset = AlpacaDataset(raw_dataset)

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloader
    dataloader = DataLoader(
        alpaca_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, tokenizer)
    )

    # Test the pipeline
    for batch in dataloader:
        print("Sample batch:")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"Sample input_ids: {batch['input_ids'][0][:10]}")
        break

if __name__ == "__main__":
    main()