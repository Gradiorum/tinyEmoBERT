import os
import transformers
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.nn as nn

class tinyBERT_finetune(nn.Module):
    def __init__(self, num_sentiment_labels, num_emotion_labels):
        super(tinyBERT_finetune, self).__init__()
        self.tinyBERT = BertForSequenceClassification.from_pretrained(
            "prajjwall/bert-tiny",
        )
        self.sentiment_head = nn.Linear(768, num_sentiment_labels)
        self.emotion_head = nn.Linear(768, num_emotion_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.tinyBERT(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        sentiment_logits = self.sentiment_head(pooled_output)
        emotion_logits = self.emotion_head(pooled_output)
        return sentiment_logits, emotion_logits
    
