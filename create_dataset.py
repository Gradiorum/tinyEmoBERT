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

def txt_to_csv(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    #Check if file is a TXT file
    
    if file_extension == 'txt':
        