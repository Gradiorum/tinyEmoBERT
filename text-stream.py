import sys
import threading
import time
import psutil
from tkinter import Tk, Text, Label, Entry

# Your existing imports
sys.path.append(r"D:\Users\WillR\VsCodeProjects\Natural Language Processing\LowLatency-TextClass_Emotion_Sentiment")
from tinyEmoBERT.models.tinyEmoBERT import tinyBERT_finetune
from tinyEmoBERT.utils.Tokenizer import tokenize_unlabeled_text
from tinyEmoBERT.utils.Metrics import AdvancedMetrics
import torch

# Initialize Model and Metrics
NUM_SENTIMENT_LABELS = 3
NUM_EMOTION_LABELS = 8
model = tinyBERT_finetune(NUM_SENTIMENT_LABELS, NUM_EMOTION_LABELS)
model.load_state_dict(torch.load(r"D:\Users\WillR\VsCodeProjects\Natural Language Processing\best_model.pth"))
model.eval()

# Label mappings
sentiment_label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
emotion_label_mapping = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'love', 4: 'neutral', 5: 'sad', 6: 'surprise', 7: 'worry'}

root = Tk()
text_widget = Text(root)
result_label = Label(root, text="Sentiment & Emotion will appear here")
inference_time_label = Label(root, text="Inference Time:")
flops_label = Label(root, text="FLOPs:")
num_inferences_label = Label(root, text="Number of Inferences: 0")
memory_usage_label = Label(root, text="Memory Usage: 0 MB")

# Entry widgets for displaying metrics
token_time_entry = Entry(root)
inf_time_entry = Entry(root)

# Initialize metrics object
metrics = AdvancedMetrics(model)

# Global variables for tracking metrics
total_inference_time = 0
num_inferences = 0
start_time = time.time()

def update_metrics(text):
    global total_inference_time, num_inferences
    start_time = time.time()
    input_ids, attention_mask = tokenize_unlabeled_text(text)
    token_time = time.time() - start_time
    
    inf_time = metrics.measure_inference_time(input_ids, attention_mask)
    total_inference_time += inf_time
    num_inferences += 1
    
    # Update Entry widgets
    token_time_entry.delete(0, 'end')
    token_time_entry.insert(0, str(token_time))
    
    inf_time_entry.delete(0, 'end')
    inf_time_entry.insert(0, str(inf_time))
    
    # Update inference count and average time
    num_inferences_label.config(text=f"Number of Inferences: {num_inferences}")

def make_inference(event):
    current_text = text_widget.get("1.0", 'end-1c')
    
    # Offload metrics calculation to a separate thread
    threading.Thread(target=update_metrics, args=(current_text,)).start()
    
    # Your existing inference code
    sentiment, emotion = inference(model, current_text)
    result_label.config(text=f"Sentiment: {sentiment}, Emotion: {emotion}")

def inference(model, text):
    input_ids, attention_mask = tokenize_unlabeled_text(text)
    sentiment_logits, emotion_logits = model(input_ids, attention_mask, input_ids, attention_mask)
    sentiment = torch.argmax(sentiment_logits, dim=1).item()
    emotion = torch.argmax(emotion_logits, dim=1).item()
    
    return sentiment_label_mapping[sentiment], emotion_label_mapping[emotion]

# Layout adjustments
text_widget.grid(row=0, columnspan=4)
result_label.grid(row=1, columnspan=4)
inference_time_label.grid(row=2, column=0)
flops_label.grid(row=2, column=2)
token_time_entry.grid(row=2, column=1)
inf_time_entry.grid(row=2, column=3)
num_inferences_label.grid(row=3, columnspan=4)
memory_usage_label.grid(row=4, columnspan=4)

# Binding the spacebar event
text_widget.bind('<space>', make_inference)

def calculate_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert bytes to MB

def update_memory_usage():
    while True:
        memory_usage = calculate_memory_usage()
        memory_usage_label.config(text=f"Memory Usage: {memory_usage:.2f} MB")
        time.sleep(1)

# Start memory usage tracking in a separate thread
threading.Thread(target=update_memory_usage, daemon=True).start()

# Program closure handler
def on_close():
    print(f"Average Inference Time: {total_inference_time / num_inferences:.2f} s")
    print(f"Total Runtime: {time.time() - start_time:.2f} s")
    # Additional print statements for other metrics
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()

