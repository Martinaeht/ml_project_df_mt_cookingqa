#%%
# Install required libraries (if not already installed)
# %pip install transformers evaluate optuna sklearn

import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import ipykernel
import optuna

from pandas import json_normalize
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
#%%
# Define dataset folder path
dataset_folder = "/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset"

# Verify that the dataset folder exists
if not os.path.exists(dataset_folder):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

# List files in dataset folder for confirmation
print("Dataset folder contains:", os.listdir(dataset_folder))

#%%
# Define dataset file paths
file_path_train = f"{dataset_folder}/train_recipeqa.json"
file_path_val = f"{dataset_folder}/val_recipeqa.json"
file_path_test = f"{dataset_folder}/test_recipeqa.json"

# Verify all files exist before loading
for file_path in [file_path_train, file_path_val, file_path_test]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

# Load datasets
with open(file_path_train, "r") as file:
    data_train = json.load(file)

with open(file_path_val, "r") as file:
    data_val = json.load(file)

with open(file_path_test, "r") as file:
    data_test = json.load(file)

# Convert datasets into pandas DataFrames
df_train = pd.DataFrame(data_train)
df_val = pd.DataFrame(data_val)
df_test = pd.DataFrame(data_test)

# Display dataset samples
print("Training dataset preview:")
print(df_train.head(10))

print("Validation dataset preview:")
print(df_val.head(10))

print("Test dataset preview:")
print(df_test.head(10))

#%%
# Merge train and validation sets (test set has a missing column)
dataset_qa = pd.concat([df_train, df_val], ignore_index=True)

# Display structure of merged dataset
print("Columns in merged dataset:", dataset_qa.columns)
print("Merged dataset shape:", dataset_qa.shape)

# Save merged dataset as JSON
merged_dataset_path = f"{dataset_folder}/merged_dataset.json"
dataset_qa.to_json(merged_dataset_path, orient="records", lines=False)
print(f"Merged dataset saved to: {merged_dataset_path}")

#%%
# Extract the 'data' column containing nested information
if "data" in dataset_qa.columns:
    nested_data = dataset_qa["data"]
    flattened_dataset_qa = pd.json_normalize(nested_data)
else:
    raise KeyError("Column 'data' not found in dataset_qa")

# Display flattened dataset preview
print("Flattened dataset preview:\n", flattened_dataset_qa.head(5))
print("Columns in flattened dataset:", flattened_dataset_qa.columns)

# Filter dataset for the 'textual_cloze' task
dataset_text_train = flattened_dataset_qa[flattened_dataset_qa["task"] == "textual_cloze"]

print("Filtered dataset preview:\n", dataset_text_train.head(5))

# Define required columns and remove rows with missing values
required_columns = [
    "recipe_id", "context_modality", "context", "choice_list",
    "answer", "qid", "question_modality", "question_text", "question", "task"
]

dataset_text_train = dataset_text_train.dropna(subset=required_columns)

# Clean 'context' column by removing 'images' and 'videos'
for index, row in dataset_text_train.iterrows():
    if isinstance(row["context"], list):  # Ensure context is a list before modifying
        for context_item in row["context"]:
            context_item.pop("images", None)
            context_item.pop("videos", None)

print("Context column cleaned successfully.")

# Display dataset structure after cleanup
num_rows, num_columns = dataset_text_train.shape
print(f"The DataFrame now has {num_rows} rows and {num_columns} columns.")

# Save filtered dataset
filtered_dataset_path = f"{dataset_folder}/dataset_text_train.json"
dataset_text_train.to_json(filtered_dataset_path, orient="records", lines=True)
print(f"Filtered dataset saved to: {filtered_dataset_path}")

#%%
# Ensure no missing values in stratification column
if dataset_text_train["answer"].isnull().any():
    raise ValueError("The 'answer' column contains missing values, which will break stratification.")

# Check stratification feasibility (ensuring multiple unique labels)
if dataset_text_train["answer"].nunique() > 1:
    # Step 1: First split into train and temp (val + test)
    train_df, temp_df = train_test_split(
        dataset_text_train,
        test_size=0.3,  # 30% goes to val + test
        random_state=42,
        stratify=dataset_text_train["answer"]  # Preserve label balance
    )

    # Step 2: Split temp into val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 50% of temp → 15% test, 15% val overall
        random_state=42,
        stratify=temp_df["answer"]
    )
else:
    # If stratification isn't feasible, split without stratify
    train_df, temp_df = train_test_split(dataset_text_train, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Define dataset file paths
train_dataset_path = f"{dataset_folder}/train_qafilt_true.json"
val_dataset_path = f"{dataset_folder}/val_qafilt_true.json"
test_dataset_path = f"{dataset_folder}/test_qafilt_true.json"

# Save datasets to JSON files
train_df.to_json(train_dataset_path, orient="records", lines=True)
val_df.to_json(val_dataset_path, orient="records", lines=True)
test_df.to_json(test_dataset_path, orient="records", lines=True)

print(f"Datasets saved successfully:\nTrain: {train_dataset_path}\nValidation: {val_dataset_path}\nTest: {test_dataset_path}")
#%%