import transformers
import torch
from transformers import BertTokenizerFast

def tokenize_unlabeled_text(text, max_length=256):
    tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
    encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,  # Explicitly activate truncation
            padding='max_length',  # Pad to max_length
            return_attention_mask=True,
            return_tensors="pt"
    )
    
    return encoded["input_ids"], encoded["attention_mask"]

    