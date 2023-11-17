import os
import load_data as ld

sentiment_file_path = r"E:\text_datasets\test_sentiment.csv"
emotion_file_path = r"E:\text_datasets\test_emotion.csv"

ld.prepare_and_save_dataset(sentiment_file_path,r"E:\Bert-Base-Uncased\saved\test_sentiment",task = "sentiment", max_length=256)
ld.prepare_and_save_dataset(emotion_file_path,r"E:\Bert-Base-Uncased\saved\test_emotion", task = "emotion", max_length=256)