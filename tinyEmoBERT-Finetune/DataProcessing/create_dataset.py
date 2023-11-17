import os
import pandas as pd
import csv
from collections import Counter
import json
import gzip
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

"""
For the dataset collection method we personally will vet the dataset, even if there are some
categories in datasets that are not present in other datasets, we still would be able to add it to the dataset
there would just be less examples of that emotion present in the dataset, which is something that
we would hope to avoid, however this is the reality of Artificial Intelligence and we must deal with it

"""

"""
Labels contained in datasets:
Joy
Sadness
Anger
Fear
Surprise
Love
happy (mapped to joy)
"""


#This function works
def delimiter_detection(file_path):

                
                
    delimiter_counts = Counter()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                delimiter_counts.update(line.strip())
        delimiter_types = ['\t', ';', ',']
        detected = max([(delimiter, delimiter_counts.get(delimiter, 0)) for delimiter in delimiter_types], key=lambda x: x[1])
        return detected[0]
    except FileNotFoundError:
        print(f"File {file_path} not found.")

#This works and needs no further functionalities
def convert_files_to_csv(root_folder):
    try:
        for folder_path , _, filenames in os.walk(root_folder):
            for filename in filenames:
                file_path = os.path.join(folder_path, filename)
                
                # Check if the file is a .json.gz
                if filename.endswith('.json.gz'):
                    with gzip.GzipFile(file_path, 'r') as f:
                        json_data = json.loads(f.read().decode('utf-8'))
                    df = pd.DataFrame(json_data)
                    csv_file_path = os.path.join(folder_path, filename.replace('.json.gz', '.csv'))
                    df.to_csv(csv_file_path, index=False)
                
                # Check if the file is a .txt
                elif filename.endswith('.txt'):
                    delimiter = delimiter_detection(file_path)
                    if delimiter is not None:
                        df = pd.read_csv(file_path, delimiter=delimiter)
                        csv_file_path = os.path.join(folder_path, filename.replace('.txt', '.csv'))
                        df.to_csv(csv_file_path, index=False)
                
                else:
                    print(f"The given file {filename} is not supported for conversion.")
                    continue

    except FileNotFoundError:
        print(f"Directory {root_folder} not found.")
        


def convert_file_jsonl_gz_to_csv(filepath):
    # Initialize an empty list to hold the JSON objects
    json_list = []
    
    # Open the .jsonl.gz file and read lines
    with gzip.open(filepath, 'rb') as f:
        for line in f:
            # Each line is one JSON object
            json_object = json.loads(line)
            json_list.append(json_object)
            
    # Convert the list of JSON objects to a DataFrame
    df = pd.DataFrame(json_list)
    
    # Save to CSV
    csv_filepath = os.path.splitext(os.path.splitext(filepath)[0])[0] + '.csv'
    df.to_csv(csv_filepath, index=False)
    
    return csv_filepath  # Return the new CSV filepath

# Example usage:
# new_csv_path = convert_file_jsonl_gz_to_csv("E:\\text_datasets\\EmotionClassification\\EmotionClassificationDataset6\\data.jsonl.gz")

import json
import pandas as pd

def convert_jsonl_to_csv(jsonl_file_path, csv_file_path):
    """
    Converts a JSONL file to a CSV file.
    
    Parameters:
    jsonl_file_path (str): The file path for the input JSONL file.
    csv_file_path (str): The file path for the output CSV file.
    
    Returns:
    None: Writes the CSV to disk at the location specified by csv_file_path.
    """
    
    # Initialize an empty list to hold the JSON objects
    data_list = []
    
    try:
        # Open the JSONL file and read line by line
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse each line as a JSON object and append to the list
                data_list.append(json.loads(line.strip()))
        
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data_list)
        
        # Write the DataFrame to a CSV file
        df.to_csv(csv_file_path, index=False)
        
        print(f"Successfully converted {jsonl_file_path} to {csv_file_path}.")
        
    except Exception as e:
        print(f"An error occurred: {e}")


def normalize_dataset(root_folder, mode='sentiment'):
    mode = mode.lower()
    for folder_path, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                
                # Drop ID columns (case-insensitive)
                df.drop(columns=[col for col in df.columns if 'id' in col.lower()], inplace=True)
                
                # Column name regularization
                # Renaming to 'Text' and 'Emotion' based on some rules
                rename_dict = {}
                text_related_keywords = ['text', 'sentence', 'content']
                label_related_keywords = ['emotion', 'label', 'sentiment']
                
                for col in df.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in text_related_keywords):
                        rename_dict[col] = 'Text'
                    elif any(keyword in col_lower for keyword in label_related_keywords):
                        rename_dict[col] = 'Emotion'
                
                # If no renaming occurred, fall back to heuristics
                if not rename_dict:
                    for col in df.columns:
                        sample_value = str(df[col].iloc[0])
                        if len(sample_value.split()) > 1:  # More than one word
                            rename_dict[col] = 'Text'
                        else:
                            rename_dict[col] = 'Emotion'
                
                df.rename(columns=rename_dict, inplace=True)
                
                # Reordering columns
                if 'Text' in df.columns and 'Emotion' in df.columns:
                    df = df[['Text', 'Emotion']]
                
                # Class label regularization for 'emotion'
                if mode == 'emotion':
                    emotion_class_mapping = {
    # Mapping to 'joy'
    'happy': 'joy',
    'elated': 'joy',
    'glad': 'joy',
    'pleased': 'joy',
    'overjoyed': 'joy',
    'thrilled': 'joy',
    'happiness': 'joy',
    'fun': 'joy',
    'enthusiasm': 'joy',

    # Mapping to 'anger'
    'irate': 'anger',
    'livid': 'anger',
    'furious': 'anger',
    'incensed': 'anger',
    'enraged': 'anger',
    'hate': 'anger',
    'disgust': 'anger',

    # Mapping to 'sad'
    'sadness': 'sad',
    'unhappy': 'sad',
    'downcast': 'sad',
    'mournful': 'sad',
    'dismal': 'sad',
    'melancholic': 'sad',
    'sorrowful': 'sad',
    'shame': 'sad',

    # Mapping to 'fear'
    'scared': 'fear',
    'frightened': 'fear',
    'afraid': 'fear',
    'terrified': 'fear',
    'anxious': 'fear',

    # Mapping to 'surprise'
    'astonished': 'surprise',
    'amazed': 'surprise',
    'stunned': 'surprise',

    # Mapping to 'love'
    'affectionate': 'love',
    'loving': 'love',
    'devoted': 'love',
    'fond': 'love',

    # Mapping to 'neutral'
    'relief': 'neutral',
    'empty': 'neutral',
    'boredom': 'neutral',
}

                    df['Emotion'] = df['Emotion'].str.lower().replace(emotion_class_mapping)
                elif mode == 'sentiment':
                    sentiment_class_mapping = {
                        'positive': 'Positive',
                        'neutral': 'Neutral',
                        'negative': 'Negative'
                    }
                    
                    df['Emotion'] = df['Emotion'].str.capitalize().replace(sentiment_class_mapping)
                
                # Save the normalized DataFrame back to disk
                df.to_csv(file_path, index=False)
                   
            else: 
                print(f"File: {filename} is not a csv; it has been ignored.")




#This function does not currently work           
def process_dataset(root_folder, mode='unbalanced', convert=True, normalize=True, analysis_type='sentiment'):
    # Step 1: Optionally convert all files to CSV
    mode = mode.lower()
    analysis_type = analysis_type.lower()
    
    if convert:
        convert_files_to_csv(root_folder)
    
    # Step 2: Optionally normalize the datasets
    if normalize:
        normalize_dataset(root_folder, mode=analysis_type)  # Pass analysis_type to normalize_dataset
    
    # Initialize empty list to hold dataframes
    all_data = []
    
    # Step 3: Aggregation and Balancing
    for folder_path, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                all_data.append(df)
    
    # Concatenate all dataframes
    aggregated_data = pd.concat(all_data, ignore_index=True)
    
    if mode == 'balanced':
        # Resample or otherwise balance the classes
        pass
    elif mode == 'unbalanced':
        # Leave as is
        pass
    else:
        raise ValueError("Invalid mode. Choose between 'balanced' and 'unbalanced'")
    
    return aggregated_data

def process_dataset_from_preformatted(root_folder, mode='unbalanced'):
    # Initialize empty list to hold dataframes
    all_data = []
    
    # Step 1: Aggregation and column consistency checking
    for folder_path, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                all_data.append(df)
    
    # Check if all DataFrames have the same columns
    unique_column_sets = {frozenset(df.columns) for df in all_data}
    if len(unique_column_sets) != 1:
        raise ValueError("All DataFrames must have the same columns to proceed.")
    
    # Step 2: Concatenate all dataframes
    aggregated_data = pd.concat(all_data, ignore_index=True)
    
    # Step 3: Optionally balance the dataset
    if mode == 'balanced':
        # Resample or otherwise balance the classes
        pass
    elif mode == 'unbalanced':
        # Leave as is
        pass
    else:
        raise ValueError("Invalid mode. Choose between 'balanced' and 'unbalanced'")
    
    return aggregated_data

def split_aggregated_dataset(file_path,include_val = True,split = "80/10/10",show_info = "all",show_class_split = True):
    
    ratios = list(map(int, split.replace(" ", "").split('/')))
    if include_val == True:
        if len(ratios) != 3 or sum(ratios) != 100:
            raise ValueError("Invalid Split Ratio, Must sum to 100 and contain 3 splits")
    elif include_val == False:
        if len(ratios) != 2 or sum(ratios) != 100:
            raise ValueError("Invalid Split Ratio, Must sum to 100 and contain 2 splits")
    
    df = pd.read_csv(file_path)
    
    
    if show_info.lower in ["all","basic"]:
        print(f"Dataset size: {len(df)}")
        
    if show_class_split:
        print("Initial class distribution:")
        print(df.iloc[:, 1].value_counts())
               
    train_size = ratios[0] / 100.0
    val_size = ratios[1] / 100.0
    test_size = ratios[2] / 100.0
    
    #Perform Splitting
    
    train_df, temp_df = train_test_split(df, test_size=1-train_size, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=42)
    
    if show_info in ["All", "Split"]:
        print(f"Training set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
    
    if show_class_split:
        print("Class distribution after splitting:")
        print("Train set:")
        print(train_df.iloc[:, 1].value_counts())
        print("Validation set:")
        print(val_df.iloc[:, 1].value_counts())
        print("Test set:")
        print(test_df.iloc[:, 1].value_counts())  
    
    if include_val:
        return train_df, val_df, test_df
    else:
        return train_df, test_df
    
    