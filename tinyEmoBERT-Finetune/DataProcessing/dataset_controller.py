import create_dataset as cd
import load_data as ld
import os

"""
Instructions:

1. Declare file_paths for root_folder

2. call cd.convert_files_to_csv if there are non csv files in the folder (supports txt and json.gz)

3. call cd.normalize_data to drop IDs from the data and rename labels
"""


def save_dfs_to_csv(folder_path, **dataframes):
    """
    Save given pandas DataFrames to CSV files.
    
    Parameters:
        folder_path (str): The folder where to save the CSV files.
        **dataframes : Arbitrarily many keyword arguments representing the dataframes to save.
        
    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for name, df in dataframes.items():
        file_path = os.path.join(folder_path, f"{name}.csv")
        df.to_csv(file_path, index=False)

# Sample usage:
# folder_path = "your/folder/path"
# save_dfs_to_csv(folder_path, train_emotion=train_emotion, val_emotion=val_emotion, test_emotion=test_emotion, train_sentiment=train_sentiment, val_sentiment=val_sentiment, test_sentiment=test_sentiment)

emotion_folder_path = r"E:\text_datasets\EmotionClassification"
sentiment_folder_path = r"E:\text_datasets\SentimentAnalysis"




emotion_dataset_aggregated = cd.process_dataset_from_preformatted(emotion_folder_path,mode='unbalanced')

emotion_dataset_aggregate = emotion_dataset_aggregated.iloc[:, 1:]

emotion_dataset_aggregated.to_csv(r"E:\text_datasets\EmotionClassificationAggregated.csv",index=False)

sentiment_dataset_aggregated = cd.process_dataset_from_preformatted(sentiment_folder_path,mode='unbalanced')

sentiment_dataset_aggregate = sentiment_dataset_aggregated.iloc[:, 1:]

sentiment_dataset_aggregated.to_csv(r"E:\text_datasets\SentimentClassificationAggregated.csv",index=False)

train_emotion, val_emotion, test_emotion = cd.split_aggregated_dataset(r"E:\text_datasets\EmotionClassificationAggregated.csv",include_val=True,show_class_split=True,show_info="all")

train_sentiment, val_sentiment, test_sentiment = cd.split_aggregated_dataset(r"E:\text_datasets\SentimentClassificationAggregated.csv",include_val=True,show_info='all',show_class_split=True)

save_dfs_to_csv(
    r"E:\text_datasets",
    train_emotion=train_emotion,
    val_emotion=val_emotion,
    test_emotion=test_emotion,
    train_sentiment=train_sentiment,
    val_sentiment=val_sentiment,
    test_sentiment=test_sentiment
)

# Usage example
file_paths = [
    r"E:\text_datasets\test_emotion.csv",
    r"E:\text_datasets\test_sentiment.csv",
    r"E:\text_datasets\train_emotion.csv",
    r"E:\text_datasets\train_sentiment.csv",
    r"E:\text_datasets\val_emotion.csv",
    r"E:\text_datasets\val_sentiment.csv"
]

save_paths = [
    r"E:\text_datasets\saved\test_emotion",
    r"E:\text_datasets\saved\test_sentiment",
    r"E:\text_datasets\saved\train_emotion",
    r"E:\text_datasets\saved\train_sentiment",
    r"E:\text_datasets\saved\val_emotion",
    r"E:\text_datasets\saved\val_sentiment"
]

tasks = ['emotion', 'sentiment', 'emotion', 'sentiment', 'emotion', 'sentiment']

for file_path, save_path, task in zip(file_paths, save_paths, tasks):
    ld.prepare_and_save_dataset(file_path, save_path, task=task)