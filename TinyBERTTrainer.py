# Standard library imports (if any)

# Third-party library imports
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.optim as optim
from torch.nn import CrossEntropyLoss
# Local application/library specific imports
from models.tinyEmoBERT import tinyBERT_finetune
from utils.Metrics import blank
import DataProcessing.load_data as DataLoad


class EmotionSentimentClassifier:
    def __init__(self, model, device, num_sentiment_labels, num_emotion_labels):
        self.model = model.to(device)
        self.device = device
        self.sentiment_criterion = CrossEntropyLoss()
        self.emotion_criterion = CrossEntropyLoss()
    
    #Custom Loss Function .80 Weight for Emotion and .20 Weight For Sentiment    
    def compute_loss(self, sentiment_logits,emotion_logits,sentiment_labels,emotion_labels):
        sentiment_loss = self.sentiment_criterion(sentiment_logits,sentiment_labels)
        emotion_loss = self.emotion_criterion(emotion_logits,emotion_labels)
        
    
        return 0.2 * sentiment_loss + 0.8 * emotion_loss #Return Custom Loss
    
    def train_step(self, dataloader, optimizer):
        hello
        

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning GPU Training is not avaliable CPU Training Enabled")
        
    NUM_SENTIMENT_LABELS = 3
    NUM_EMOTION_LABELS = 7
    
    model = tinyBERT_finetune(num_sentiment_labels=NUM_SENTIMENT_LABELS,num_emotion_labels=NUM_EMOTION_LABELS)
    
    classifier = EmotionSentimentClassifier(num_sentiment_labels=NUM_SENTIMENT_LABELS,num_emotion_labels=NUM_EMOTION_LABELS)
    
    


    

