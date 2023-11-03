import pandas as pd
from transformers import pipeline

# Set the pipeline
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")


# Classify the values in a list
def zero_shot_classification(data, labels):
    classification_list = []
    
    for value in data:
        output = pipe(value, labels, multi_label = False)
        max_score_index = output['scores'].index(max(output['scores']))
        label_with_highest_score = output['labels'][max_score_index]
        classification_list.append(label_with_highest_score)
    
    return classification_list


# Create a new column
def store_new_column(data, column_name, data_list):
    data[column_name] = data_list
