# Standard library imports (if any)
import sys
sys.path.append(r'D:\Users\WillR\VsCodeProjects\Natural Language Processing\tinyEmoBERT')
from models.tinyEmoBERT import tinyBERT_finetune
import os
# Third-party library imports
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
# Local application/library specific imports
from models.tinyEmoBERT import tinyBERT_finetune
from utils.Metrics import AdvancedMetrics
import DataProcessing.load_data as DataLoad
from utils.Metrics import TinyEmoBoard
import torchmetrics
from tqdm import tqdm
from utils.callbacks import EarlyStopping

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
    
    def train_step(self, sentiment_dataloader, emotion_dataloader, optimizer, epoch):
        pbar = tqdm(total=min(len(sentiment_dataloader), len(emotion_dataloader)), desc=f"Training Epoch {epoch}")
        iter_sentiment = iter(sentiment_dataloader)
        iter_emotion = iter(emotion_dataloader)
        
        total_train_loss = 0.0
        num_batches = 0
        for batch_idx in range(min(len(sentiment_dataloader), len(emotion_dataloader))):
            input_ids_sentiment, attention_mask_sentiment, labels_sentiment = next(iter_sentiment)
            input_ids_emotion, attention_mask_emotion, labels_emotion = next(iter_emotion)
                
                
            # Move to device
            input_ids_sentiment, attention_mask_sentiment, labels_sentiment = input_ids_sentiment.to(self.device), attention_mask_sentiment.to(self.device), labels_sentiment.to(self.device)
            input_ids_emotion, attention_mask_emotion, labels_emotion = input_ids_emotion.to(self.device), attention_mask_emotion.to(self.device), labels_emotion.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            sentiment_logits, emotion_logits = self.model(input_ids_sentiment, attention_mask_sentiment, input_ids_emotion, attention_mask_emotion)
            
            # Compute Loss
            loss = self.compute_loss(sentiment_logits, emotion_logits, labels_sentiment, labels_emotion)
            
            # Accumulate the loss
            total_train_loss += loss.item()
            num_batches +=1
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
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

        # Update tqdm description
            pbar.set_postfix({
                "Loss": loss.item(), 
                "Sentiment Accuracy": acc_sentiment, 
                "Emotion Accuracy": acc_emotion,
                "Avg Accuracy": avg_acc  
            })
            pbar.update(1)

            #compute average training loss for the epoch
            
            avg_train_loss = total_train_loss / num_batches
            # Log metrics to TensorBoard
            self.writer.log_scalar('Validation/Loss', avg_train_loss, epoch)
            self.writer.log_scalar('Training/Sentiment Accuracy', acc_sentiment, epoch * len(sentiment_dataloader) + batch_idx)
            self.writer.log_scalar('Training/Emotion Accuracy', acc_emotion, epoch * len(emotion_dataloader) + batch_idx)
            self.writer.log_scalar('Training/Average Accuracy', avg_acc, epoch * len(emotion_dataloader) + batch_idx)

            self.writer.log_scalar('Training/Sentiment Precision', prec_sentiment, epoch)
            self.writer.log_scalar('Training/Sentiment Recall', recall_sentiment, epoch)
            self.writer.log_scalar('Training/Sentiment F1', f1_sentiment, epoch)

            self.writer.log_scalar('Training/Emotion Precision', prec_emotion, epoch)
            self.writer.log_scalar('Training/Emotion Recall', recall_emotion, epoch)
            self.writer.log_scalar('Training/Emotion F1', f1_emotion, epoch)

            self.writer.log_scalar('Training/Average Precision', avg_prec, epoch)
            self.writer.log_scalar('Training/Average Recall', avg_recall, epoch)
            self.writer.log_scalar('Training/Average F1', avg_f1, epoch)
            self.writer.log_scalar('Training/Sentiment MCC', mcc_sentiment, epoch * len(sentiment_dataloader) + batch_idx)
            self.writer.log_scalar('Training/Emotion MCC', mcc_emotion, epoch * len(emotion_dataloader) + batch_idx)
            self.writer.log_scalar('Training/Average MCC', avg_mcc, epoch * len(emotion_dataloader) + batch_idx)
        pbar.close()
        avg_train_loss = total_train_loss / num_batches
        self.writer.log_scalar('Training/Loss', avg_train_loss, epoch)


    def val_step(self, sentiment_dataloader, emotion_dataloader, epoch):
        pbar = tqdm(total=min(len(sentiment_dataloader), len(emotion_dataloader)), desc=f"Validation {epoch}")
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        iter_sentiment = iter(sentiment_dataloader)
        iter_emotion = iter(emotion_dataloader)

        with torch.no_grad():  # disable gradient calculation
            for batch_idx in range(min(len(sentiment_dataloader), len(emotion_dataloader))):
                input_ids_sentiment, attention_mask_sentiment, labels_sentiment = next(iter_sentiment)
                input_ids_emotion, attention_mask_emotion, labels_emotion = next(iter_emotion)
                
                # Move to device
                input_ids_sentiment = input_ids_sentiment.to(self.device)
                attention_mask_sentiment = attention_mask_sentiment.to(self.device)
                labels_sentiment = labels_sentiment.to(self.device)
                input_ids_emotion = input_ids_emotion.to(self.device)
                attention_mask_emotion = attention_mask_emotion.to(self.device)
                labels_emotion = labels_emotion.to(self.device)
                
                # Forward pass
                sentiment_logits, emotion_logits = self.model(input_ids_sentiment, attention_mask_sentiment, input_ids_emotion, attention_mask_emotion)
                
                # Compute Loss
                loss = self.compute_loss(sentiment_logits, emotion_logits, labels_sentiment, labels_emotion)
                total_loss += loss.item()
                num_batches += 1
           
           
            
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
                    "Loss": loss.item(), 
                    "Sentiment Accuracy": acc_sentiment, 
                    "Emotion Accuracy": acc_emotion,
                    "Avg Accuracy": avg_acc  
                })
                pbar.update(1)
                
                
                self.writer.log_scalar('Validation/Sentiment Accuracy', acc_sentiment, epoch * len(sentiment_dataloader) + batch_idx)
                self.writer.log_scalar('Validation/Emotion Accuracy', acc_emotion, epoch * len(emotion_dataloader) + batch_idx)
                self.writer.log_scalar('Validation/Average Accuracy', avg_acc, epoch * len(emotion_dataloader) + batch_idx)

                self.writer.log_scalar('Validation/Sentiment Precision', prec_sentiment, epoch)
                self.writer.log_scalar('Validation/Sentiment Recall', recall_sentiment, epoch)
                self.writer.log_scalar('Validation/Sentiment F1', f1_sentiment, epoch)

                self.writer.log_scalar('Validation/Emotion Precision', prec_emotion, epoch)
                self.writer.log_scalar('Validation/Emotion Recall', recall_emotion, epoch)
                self.writer.log_scalar('Validation/Emotion F1', f1_emotion, epoch)

                self.writer.log_scalar('Validation/Average Precision', avg_prec, epoch)
                self.writer.log_scalar('Validation/Average Recall', avg_recall, epoch)
                self.writer.log_scalar('Validation/Average F1', avg_f1, epoch)

                self.writer.log_scalar('Validation/Sentiment MCC', mcc_sentiment, epoch * len(sentiment_dataloader) + batch_idx)
                self.writer.log_scalar('Validation/Emotion MCC', mcc_emotion, epoch * len(emotion_dataloader) + batch_idx)
                self.writer.log_scalar('Validation/Average MCC', avg_mcc, epoch * len(emotion_dataloader) + batch_idx)
       
        
        avg_val_loss = total_loss / num_batches
        # Log metrics to TensorBoard
        self.writer.log_scalar('Validation/Loss', avg_val_loss, epoch)
        
        pbar.close()
        return avg_val_loss
   
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
    sentiment_data_train = torch.load(r"E:\text_datasets\saved\train_sentimen_no_batch.pt")
    sentiment_data_val = torch.load(r"E:\text_datasets\saved\val_sentimen_no_batch.pt")
    sentiment_data_test = torch.load(r"E:\text_datasets\saved\test_sentimen_no_batch.pt")
    emotion_data_train = torch.load(r"E:\text_datasets\saved\train_emotion_no_batch.pt")
    emotion_data_val = torch.load(r"E:\text_datasets\saved\val_emotion_no_batch.pt")
    emotion_data_test = torch.load(r"E:\text_datasets\saved\test_emotion_no_batch.pt")
    
    

   
    
    
    
        # Create DataLoader objects with batch_size=None, assuming data is already batched
    sentiment_dataloader_train = DataLoader(sentiment_data_train, batch_size=256, shuffle=True)
    sentiment_dataloader_val = DataLoader(sentiment_data_val, batch_size=256)
    sentiment_dataloader_test = DataLoader(sentiment_data_test, batch_size=256)
    for batch in sentiment_dataloader_train:
        print(f"Shape of first batch from sentiment_dataloader_train: {batch[0].shape if torch.is_tensor(batch[0]) else 'NA'}")
        break  # We only want the first batch
    emotion_dataloader_train = DataLoader(emotion_data_train, batch_size=256, shuffle=True)
    emotion_dataloader_val = DataLoader(emotion_data_val, batch_size=256)
    emotion_dataloader_test = DataLoader(emotion_data_test, batch_size=256)
    NUM_SENTIMENT_LABELS = 3
    NUM_EMOTION_LABELS = 8
    LOG_DIR = r"D:\Users\WillR\VsCodeProjects\Natural Language Processing\tinyEmoBERT\logging"
    

    model = tinyBERT_finetune(num_sentiment_labels=NUM_SENTIMENT_LABELS, num_emotion_labels=NUM_EMOTION_LABELS)
    optimizer = torch.optim.AdamW(model.parameters(),lr =1e-5, weight_decay=1e-6)
    classifier = EmotionSentimentClassifier(model, device, NUM_SENTIMENT_LABELS, NUM_EMOTION_LABELS, LOG_DIR)

    if mode in ["train", "full"]:
        # Your training logic here
        early_stopping = EarlyStopping(patience=25, min_delta=0.0001)  # Initialize Early Stopping
        num_epochs = 100
        for epoch in range(num_epochs):
            classifier.train_step(sentiment_dataloader_train, emotion_dataloader_train, optimizer, epoch)
            val_loss = classifier.val_step(sentiment_dataloader_val, emotion_dataloader_val, epoch)

            if early_stopping.step(val_loss, classifier.model):
                print("Early stopping triggered. Restoring best model weights.")
                classifier.model.load_state_dict(early_stopping.best_state)
                break

        if early_stopping.best_state is not None:
            torch.save(early_stopping.best_state, 'best_model.pth')

    if mode in ["test", "full"]:
        if os.path.exists('best_model.pth'):
            classifier.model.load_state_dict(torch.load('best_model.pth'))
    # Assuming you have test_step implemented in classifier
    test_results = classifier.test_step(sentiment_dataloader_test, emotion_dataloader_test)
    print("Test Results:", test_results)


if __name__ == "__main__":
    main(mode="test")  # or "train" or "test"    
    
    
    

