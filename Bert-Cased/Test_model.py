# Standard library imports (if any)
import sys
sys.path.append(r'D:\Users\WillR\VsCodeProjects\Natural Language Processing')
from tinyEmoBERT.models.tinyEmoBERT import tinyBERT_finetune
import os
# Third-party library imports
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
# Local application/library specific imports
from model import Bert_Base_uncased
from tinyEmoBERT.utils.Metrics import AdvancedMetrics
import tinyEmoBERT.DataProcessing.load_data as DataLoad
from tinyEmoBERT.utils.Metrics import TinyEmoBoard
import torchmetrics
from tqdm import tqdm
from tinyEmoBERT.utils.callbacks import EarlyStopping

class EmotionSentimentClassifier:
    def __init__(self, model, device, num_sentiment_labels, num_emotion_labels, log_dir):
        self.model = model.to(device)
        self.device = device
        self.sentiment_criterion = CrossEntropyLoss()
        self.emotion_criterion = CrossEntropyLoss()
        self.writer = TinyEmoBoard(log_dir=log_dir)
        
        self.sentiment_accuracy = torchmetrics.Accuracy(num_classes=num_sentiment_labels, task='multiclass').to(device)
        self.emotion_accuracy = torchmetrics.Accuracy(num_classes=num_emotion_labels, task='multiclass').to(device)
        
        self.sentiment_precision = torchmetrics.Precision(num_classes=num_sentiment_labels, task='multiclass').to(device)
        self.emotion_precision = torchmetrics.Precision(num_classes=num_emotion_labels, task='multiclass').to(device)
        
        self.sentiment_recall = torchmetrics.Recall(num_classes=num_sentiment_labels, task='multiclass').to(device)
        self.emotion_recall = torchmetrics.Recall(num_classes=num_emotion_labels, task='multiclass').to(device)
        
        self.f1_sentiment = torchmetrics.F1Score(num_classes=num_sentiment_labels,task = 'multiclass').to(device)
        self.f1_emotion = torchmetrics.F1Score(num_classes=num_emotion_labels, task = 'multiclass').to(device)
        
        self.mcc_sentiment = torchmetrics.MatthewsCorrCoef(num_classes=num_sentiment_labels,task = 'multiclass').to(device)
        self.mcc_emotion = torchmetrics.MatthewsCorrCoef(num_classes=num_emotion_labels,task = 'multiclass').to(device)
    
    def compute_loss(self, sentiment_logits, emotion_logits, sentiment_labels, emotion_labels):
        sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
        emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
        return 0.2 * sentiment_loss + 0.8 * emotion_loss # Custom Loss
    
    def test_step(self, sentiment_dataloader, emotion_dataloader):
        self.model.eval()
        aggregated_metrics = {}
        num_batches =  min(len(sentiment_dataloader),len (emotion_dataloader))
        
        #intialize metric counters
        aggregated_metrics = {
    'total_sentiment_accuracy': 0.0,
    'total_emotion_accuracy': 0.0,
    'total_sentiment_precision': 0.0,
    'total_emotion_precision': 0.0,
    'total_sentiment_recall': 0.0,
    'total_emotion_recall': 0.0,
    'total_sentiment_f1': 0.0,
    'total_emotion_f1': 0.0,
    'total_sentiment_mcc': 0.0,
    'total_emotion_mcc': 0.0
}
        
        with torch.no_grad():  # disable gradient calculation
            pbar = tqdm(total=min(len(sentiment_dataloader), len(emotion_dataloader)), desc=f"Testing")
            iter_sentiment = iter(sentiment_dataloader)
            iter_emotion = iter(emotion_dataloader)

            for batch_idx in range(min(len(sentiment_dataloader), len(emotion_dataloader))):
                input_ids_sentiment, attention_mask_sentiment, labels_sentiment = next(iter_sentiment)
                input_ids_emotion, attention_mask_emotion, labels_emotion = next(iter_emotion)

                # Move to device
                input_ids_sentiment, attention_mask_sentiment, labels_sentiment = input_ids_sentiment.to(self.device), attention_mask_sentiment.to(self.device), labels_sentiment.to(self.device)
                input_ids_emotion, attention_mask_emotion, labels_emotion = input_ids_emotion.to(self.device), attention_mask_emotion.to(self.device), labels_emotion.to(self.device)

                # Forward pass
                sentiment_logits, emotion_logits = self.model(input_ids_sentiment, attention_mask_sentiment, input_ids_emotion, attention_mask_emotion)

                # Compute Metrics and Log to TensorBoard
                acc_sentiment = self.sentiment_accuracy(sentiment_logits.argmax(dim=1), labels_sentiment).item()
                acc_emotion = self.emotion_accuracy(emotion_logits.argmax(dim=1), labels_emotion).item()

                # Compute weighted average accuracy
                avg_acc = (acc_sentiment + acc_emotion) /2

                # Compute other metrics for sentiment
                prec_sentiment = self.sentiment_precision(sentiment_logits.argmax(dim=1), labels_sentiment).item()
                recall_sentiment = self.sentiment_recall(sentiment_logits.argmax(dim=1), labels_sentiment).item()
                f1_sentiment = self.f1_sentiment(sentiment_logits, labels_sentiment).item()
                mcc_sentiment = self.mcc_sentiment(sentiment_logits.argmax(dim=1),labels_sentiment).item()
                

                # Compute other metrics for emotion
                prec_emotion = self.emotion_precision(emotion_logits.argmax(dim=1), labels_emotion).item()
                recall_emotion = self.emotion_recall(emotion_logits.argmax(dim=1), labels_emotion).item()
                f1_emotion = self.f1_emotion(emotion_logits, labels_emotion).item()
                mcc_emotion = self.mcc_emotion(emotion_logits.argmax(dim=1),labels_emotion).item()

                # Compute weighted average for precision, recall, and F1-score
                avg_prec = (prec_sentiment+prec_emotion)/2
                avg_recall = (recall_sentiment+recall_emotion)/2
                avg_f1 = (f1_sentiment+f1_emotion)/2
                avg_mcc = (mcc_emotion+mcc_sentiment)/2

                pbar.set_postfix({
                "Sentiment Accuracy": acc_sentiment, 
                "Emotion Accuracy": acc_emotion,
                "Avg Accuracy": avg_acc  
                })
                
                
                # Update aggregated metrics (inside your loop)
                aggregated_metrics['total_sentiment_accuracy'] += acc_sentiment
                aggregated_metrics['total_emotion_accuracy'] += acc_emotion
                aggregated_metrics['total_sentiment_precision'] += prec_sentiment
                aggregated_metrics['total_emotion_precision'] += prec_emotion
                aggregated_metrics['total_sentiment_recall'] += recall_sentiment
                aggregated_metrics['total_emotion_recall'] += recall_emotion
                aggregated_metrics['total_sentiment_f1'] += f1_sentiment
                aggregated_metrics['total_emotion_f1'] += f1_emotion
                aggregated_metrics['total_sentiment_mcc'] += mcc_sentiment
                aggregated_metrics['total_emotion_mcc'] += mcc_emotion

                pbar.update(1)
                
                
                # Log Accuracy Metrics
                self.writer.log_scalar('Test/Sentiment Accuracy', acc_sentiment, batch_idx)
                self.writer.log_scalar('Test/Emotion Accuracy', acc_emotion, batch_idx)
                self.writer.log_scalar('Test/Average Accuracy', avg_acc, batch_idx)

                # Log Precision Metrics
                self.writer.log_scalar('Test/Sentiment Precision', prec_sentiment, batch_idx)
                self.writer.log_scalar('Test/Emotion Precision', prec_emotion, batch_idx)
                self.writer.log_scalar('Test/Average Precision', avg_prec, batch_idx)

                # Log Recall Metrics
                self.writer.log_scalar('Test/Sentiment Recall', recall_sentiment, batch_idx)
                self.writer.log_scalar('Test/Emotion Recall', recall_emotion, batch_idx)
                self.writer.log_scalar('Test/Average Recall', avg_recall, batch_idx)

                # Log F1-score Metrics
                self.writer.log_scalar('Test/Sentiment F1', f1_sentiment, batch_idx)
                self.writer.log_scalar('Test/Emotion F1', f1_emotion, batch_idx)
                self.writer.log_scalar('Test/Average F1', avg_f1, batch_idx)

                # Log MCC Metrics
                self.writer.log_scalar('Test/Sentiment MCC', mcc_sentiment, batch_idx)
                self.writer.log_scalar('Test/Emotion MCC', mcc_emotion, batch_idx)
                self.writer.log_scalar('Test/Average MCC', avg_mcc, batch_idx)
                
            for key in aggregated_metrics.keys():
                    aggregated_metrics[key] /= num_batches
        pbar.close()
        return aggregated_metrics

            


def main(mode = "full"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data from saved tensors
    
    sentiment_data_test = torch.load(r"E:\Bert-Base-Uncased\saved\test_sentimen_bert-cased.pt")
    emotion_data_test = torch.load(r"E:\Bert-Base-Uncased\saved\test_emotion_bert-cased.pt")
    
    

   
    
    
    
        # Create DataLoader objects with batch_size=None, assuming data is already batched
    
    sentiment_dataloader_test = DataLoader(sentiment_data_test, batch_size=256)
    
    
    emotion_dataloader_test = DataLoader(emotion_data_test, batch_size=256)
    NUM_SENTIMENT_LABELS = 3
    NUM_EMOTION_LABELS = 8
    LOG_DIR = r"D:\Users\WillR\VsCodeProjects\Natural Language Processing\Bert-Base-EmoSen\logging-Bert-Base-Pretrained"
    

    model = Bert_Base_uncased(num_sentiment_labels=NUM_SENTIMENT_LABELS, num_emotion_labels=NUM_EMOTION_LABELS)
    optimizer = torch.optim.AdamW(model.parameters(),lr =1e-5, weight_decay=1e-6)
    classifier = EmotionSentimentClassifier(model, device, NUM_SENTIMENT_LABELS, NUM_EMOTION_LABELS, LOG_DIR)

   

    
    # Assuming you have test_step implemented in classifier
    test_results = classifier.test_step(sentiment_dataloader_test, emotion_dataloader_test)
    print("Test Results:", test_results)


if __name__ == "__main__":
    main(mode="test")  # or "train" or "test"    