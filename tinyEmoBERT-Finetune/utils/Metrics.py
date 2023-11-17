import time
import torch
from ptflops import get_model_complexity_info
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter

class AdvancedMetrics:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def speed_per_token(self, total_time, max_length):
        return total_time / max_length
    
    def compute_flops(self):
        # Assuming you have a method to compute FLOPs
        macs, _ = get_model_complexity_info(self.model, (1, 256), as_strings=True)
        return macs
    
    def measure_inference_time(self, input_ids, attention_mask):
        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_ids.to(self.device), attention_mask.to(self.device), 
                                input_ids.to(self.device), attention_mask.to(self.device))
        end_time = time.time()
        total_time = end_time - start_time
        return total_time

    def measure_tokenization_time(self, text, tokenizer):
        start_time = time.time()
        tokens = tokenizer.tokenize(text)
        end_time = time.time()
        total_time = end_time - start_time
        return total_time
    def measure_batch_inference_time(self, input_ids, attention_mask):
        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_ids.to(self.device), attention_mask.to(self.device))
        end_time = time.time()
        total_time = end_time - start_time
        batch_size = input_ids.size(0)
        average_time_per_sample = total_time / batch_size
        return total_time, average_time_per_sample
    def compute_balanced_accuracy(self, y_true, y_pred):
        # Assuming y_true and y_pred are PyTorch tensors.
        # Detach and bring to CPU for metric computation
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
        return balanced_accuracy_score(y_true, y_pred)
class TinyEmoBoard:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)
        
    def log_histogram(self, tag, values, global_step=None):
        self.writer.add_histogram(tag, values, global_step)
        
    def log_image(self, tag, img_tensor, global_step=None):
        self.writer.add_image(tag, img_tensor, global_step)
        
    def log_text(self, tag, text, global_step=None):
        self.writer.add_text(tag, text, global_step)
        
    def log_custom_metrics(self, tag, metrics_dict, global_step=None):
        for metric_name, metric_value in metrics_dict.items():
            self.writer.add_scalar(f"{tag}/{metric_name}", metric_value, global_step)
            
    def close(self):
        self.writer.close()
        