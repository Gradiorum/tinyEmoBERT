from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path, task="sentiment", max_length=256):
    df = pd.read_csv(file_path)
    
    # Convert column names to lowercase for case-insensitive operations
    df.columns = df.columns.str.lower()
    
    # Drop rows with NA values
    df.dropna(subset=['text', task.lower()], inplace=True)
    
    # Convert labels to integers
    le = LabelEncoder()
    df[task.lower()] = le.fit_transform(df[task.lower()])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_mapping}")
    
    texts = df['text'].tolist()
    labels = df[task.lower()].tolist()
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader

def prepare_and_save_dataset(file_path, save_path, task="sentiment", max_length=256):
    df = pd.read_csv(file_path)
    
    # Convert column names to lowercase for case-insensitive operations
    df.columns = df.columns.str.lower()
    
    # Drop rows with NA values
    df.dropna(subset=['text', task.lower()], inplace=True)
    
    # Convert labels to integers
    le = LabelEncoder()
    df[task.lower()] = le.fit_transform(df[task.lower()])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_mapping}")
    
    texts = df['text'].tolist()
    labels = df[task.lower()].tolist()
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    # Append '_32' to indicate batch size of 32
    save_path = f"{save_path.rstrip('.pt')}_no_batch.pt"
    
    # Save TensorDataset to disk
    torch.save(dataset, save_path)



