from transformers import BertModel, BertTokenizerFast
import torch.nn as nn

class tinyBERT_finetune(nn.Module):
    def __init__(self, num_sentiment_labels, num_emotion_labels):
        super(tinyBERT_finetune, self).__init__()
        self.tinyBERT = BertModel.from_pretrained("prajjwal1/bert-tiny")
        self.sentiment_head = nn.Linear(128, num_sentiment_labels)
        self.emotion_head = nn.Linear(128, num_emotion_labels)

    def forward(self, input_ids_sentiment, attention_mask_sentiment, input_ids_emotion, attention_mask_emotion):
        outputs_sentiment = self.tinyBERT(input_ids=input_ids_sentiment, attention_mask=attention_mask_sentiment)
        outputs_emotion = self.tinyBERT(input_ids=input_ids_emotion, attention_mask=attention_mask_emotion)
        
        # Using pooler_output
        pooled_output_sentiment = outputs_sentiment.pooler_output
        pooled_output_emotion = outputs_emotion.pooler_output
        
        # Debugging: Check shapes
        #print(f"Shape of pooled_output_sentiment: {pooled_output_sentiment.shape}")
        #print(f"Shape of pooled_output_emotion: {pooled_output_emotion.shape}")
        
        # Debugging print statements
        #print("Debugging outputs_sentiment:")
        #print(type(outputs_sentiment))
        #print(outputs_sentiment)
        
        #print("Debugging outputs_emotion:")
        #print(type(outputs_emotion))
        #print(outputs_emotion)
        
        sentiment_logits = self.sentiment_head(pooled_output_sentiment)
        emotion_logits = self.emotion_head(pooled_output_emotion)
        
        return sentiment_logits, emotion_logits