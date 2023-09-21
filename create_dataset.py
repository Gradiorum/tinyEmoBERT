import os
import pandas as pd
import csv
from collections import Counter
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

def text_to_csv(root_folder):
    try:
        for folder_path , _, filenames in os.walk(root_folder):
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    
                    delimiter = delimiter_detection(file_path)
                    
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    
                    csv_file_path = os.path.join(folder_path, filename.replace('.txt', '.csv'))
                    df.to_csv(csv_file_path, index=False)
                else:
                    print(f"The given file is not a txt file, cancelling conversion")
                    continue
        
    except FileNotFoundError:
        print(f"Directory {file_path} not found. ")
        

        
