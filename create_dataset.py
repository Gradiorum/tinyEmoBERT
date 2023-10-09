import os
import pandas as pd
import csv
from collections import Counter
import json
import gzip



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
        
        
def normalize_dataset(root_folder):
    for folder_path, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path,filename)
                df = pd.read_csv(file_path)


def process_dataset(root_folder, mode='unbalanced'):
    
    all_data = []
    
    if mode == 'balanced':
        holder = holder
    elif mode == 'unbalanced':
        holder = holder
    else:
        raise ValueError("Invalid mode. Choose between 'balanced and 'unbalanced")
    
    