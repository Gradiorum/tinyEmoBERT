import create_dataset as cd
import load_data as ld

"""
Instructions:

1. Declare file_paths for root_folder

2. call cd.convert_files_to_csv if there are non csv files in the folder (supports txt and json.gz)

3. call cd.normalize_data to drop IDs from the data and rename labels
"""
emotion_folder_path = "E:\text_datasets\EmotionClassification"
sentiment_folder_path = "E:\text_datasets\SentimentAnalysis"

cd.convert_files_to_csv(emotion_folder_path)
cd.convert_files_to_csv(sentiment_folder_path)

cd.normalize_dataset(emotion_folder_path)

cd.process_dataset(emotion_folder_path,mode='unbalanced')
cd.process_dataset(sentiment_folder_path,mode='unbalanced')

