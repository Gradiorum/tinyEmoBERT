import sys
import threading
from tkinter import Tk, Text, Label, Entry
import time

# Your existing imports
sys.path.append(r'D:\Users\WillR\VsCodeProjects\Natural Language Processing\tinyEmoBERT')
from models.tinyEmoBERT import tinyBERT_finetune
from utils.Tokenizer import tokenize_unlabeled_text
from utils.Metrics import AdvancedMetrics
import torch

# Initialize Model and Metrics
NUM_SENTIMENT_LABELS = 3
NUM_EMOTION_LABELS = 9
model = tinyBERT_finetune(NUM_SENTIMENT_LABELS, NUM_EMOTION_LABELS)
model.load_state_dict(torch.load(r'D:\Users\WillR\VsCodeProjects\Natural Language Processing\tinyEmoBERT\best_model.pth'))
model.eval()

# Label mappings
sentiment_label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
emotion_label_mapping = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'neutral', 5: 'sad', 6: 'sadness', 7: 'surprise', 8: 'worry'}

root = Tk()
text_widget = Text(root)
result_label = Label(root, text="Sentiment & Emotion will appear here")
inference_time_label = Label(root, text="Inference Time:")
flops_label = Label(root, text="FLOPs:")

# Entry widgets for displaying metrics
token_time_entry = Entry(root)
inf_time_entry = Entry(root)

# Initialize metrics object
metrics = AdvancedMetrics(model)

def update_metrics(text):
    start_time = time.time()
    input_ids, attention_mask = tokenize_unlabeled_text(text)
    token_time = time.time() - start_time
    
    inf_time = metrics.measure_inference_time(input_ids, attention_mask)
    
    # Update Entry widgets
    token_time_entry.delete(0, 'end')
    token_time_entry.insert(0, str(token_time))
    
    inf_time_entry.delete(0, 'end')
    inf_time_entry.insert(0, str(inf_time))
    
# Function for making inference
def inference(model, text):
    input_ids, attention_mask = tokenize_unlabeled_text(text)
    sentiment_logits, emotion_logits = model(input_ids, attention_mask, input_ids, attention_mask)
    sentiment = torch.argmax(sentiment_logits, dim=1).item()
    emotion = torch.argmax(emotion_logits, dim=1).item()
    
    return sentiment_label_mapping[sentiment], emotion_label_mapping[emotion]

def make_inference(event):
    current_text = text_widget.get("1.0", 'end-1c')
    
    # Offload metrics calculation to a separate thread
    threading.Thread(target=update_metrics, args=(current_text,)).start()
    
    # Your existing inference code
    sentiment, emotion = inference(model, current_text)
    result_label.config(text=f"Sentiment: {sentiment}, Emotion: {emotion}")

# Layout using grid
text_widget.grid(row=0, columnspan=4)
result_label.grid(row=1, columnspan=4)
inference_time_label.grid(row=2, column=0)
flops_label.grid(row=2, column=2)
token_time_entry.grid(row=2, column=1)
inf_time_entry.grid(row=2, column=3)

# Binding the spacebar event
text_widget.bind('<space>', make_inference)

root.mainloop()

