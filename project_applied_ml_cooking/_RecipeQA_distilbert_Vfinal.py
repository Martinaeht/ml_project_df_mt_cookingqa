#%%

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #adapt

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

import sys
import accelerate
print("Python:", sys.executable)
print("Accelerate:", accelerate.__version__)
print("Torch:", torch.__version__)

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipykernel
import optuna
import rich.progress
import evaluate  # for computing accuracy
from pandas import json_normalize
from datasets import load_dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from rich.progress import Progress
from accelerate import Accelerator

# Import Transformers modules
from transformers import (
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    AutoTokenizer,
    RobertaForQuestionAnswering,
    set_seed,
    DistilBertForMultipleChoice,
    DataCollatorForMultipleChoice,
)

# %%

set_seed(42)  

train_dataset_path = '/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset/train_qafilt_true.json'
val_dataset_path = '/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset/val_qafilt_true.json'
test_dataset_path = '/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset/test_qafilt_true.json'

train_dataset = load_dataset('json', data_files=train_dataset_path, split='train')
val_dataset = load_dataset('json', data_files=val_dataset_path, split='train')
test_dataset = load_dataset('json', data_files=test_dataset_path, split='train')

print("Train dataset loaded successfully.")
print("Validation dataset loaded successfully.")
print("Test dataset loaded successfully.")

# Check first few rows of train dataset
print("Train dataset preview:")
print(train_dataset[:2])
print("Validation dataset preview:")
print(val_dataset[:2])
print("Test dataset preview:")
print(test_dataset[:2])

# Check number of samples in each dataset
print(f"Number of samples in train dataset: {len(train_dataset)}")
print(f"Number of samples in validation dataset: {len(val_dataset)}")
print(f"Number of samples in test dataset: {len(test_dataset)}")


#%%

recipeqa_dataset = DatasetDict({
    "train": train_dataset.shuffle(seed=42),
    "val": val_dataset.shuffle(seed=42),
    "test": test_dataset.shuffle(seed=42),
})

print(recipeqa_dataset)


'''
# Select a small subset for faster experimentation
small_recipeqa_dataset = DatasetDict({
    "train": train_dataset.shuffle(seed=333).select(range(64)),
    "val": val_dataset.shuffle(seed=333).select(range(16)),
    "test": test_dataset.shuffle(seed=333).select(range(16)),
})

print(small_recipeqa_dataset)
'''

#%%

# Initialize tokenizer 

max_length = 512 #128 and 256 before
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#%%

# Tokenize function 


def tokenize_function(examples):
    all_input_ids = []
    all_attention_masks = []
    labels = []
    for i in range(len(examples["context"])):
        context_bodies = " ".join([step["body"] for step in examples["context"][i]])
        question_variants = [
            f"{examples['question_text'][i]} " + 
            " ".join([q if q != "@placeholder" else choice for q in examples["question"][i]])
            for choice in examples["choice_list"][i]
        ]

        # 3. Tokenize with context as the first sentence, variant as second
        tokenized = tokenizer(
            [context_bodies] * len(question_variants),   
            question_variants,
            truncation="only_first",
            padding="max_length",
            max_length=max_length,
            return_tensors=None,        
        )
        # 4. Create attention masks
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        labels.append(examples["answer"][i])
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": labels,
    }



# Tokenize the dataset

print("Tokenization starts...")
tokenized_dataset = recipeqa_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,
    remove_columns= recipeqa_dataset["train"].column_names
)
print("Tokenization finished.")
print(tokenized_dataset["train"].select(range(10)))

model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-uncased")


# %%

# datacollator and metrics

from transformers import default_data_collator

#data_collator = default_data_collator 
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

accuracy_metric = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    precision_score = precision.compute(predictions=predictions, references=labels, average="weighted")
    recall_score = recall.compute(predictions=predictions, references=labels, average="weighted") 
    f1_score_result = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {
        'accuracy': acc['accuracy'],
        'precision': precision_score['precision'],
        'recall': recall_score['recall'],
        'f1': f1_score_result['f1']  # Use the f1 score from the evaluate library
    }


#%%

# objective function for Optuna

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 6)
    weight_decay = trial.suggest_uniform('weight_decay', 1e-5, 0.1)

    model = DistilBertForMultipleChoice.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir="/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/distilbert_recipeqa_optuna_large_new",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=8,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],  
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    eval_results = trainer.evaluate()
    accuracy_result = eval_results['eval_accuracy']
    print(f"Trial completed with validation accuracy: {accuracy_result:.4f}")
    print(f"Trial completed with validation loss: {eval_results['eval_loss']:.4f}")
    print(f"Trial completed with f1 score: {eval_results['eval_f1']:.4f}")
    return accuracy_result


#%%

# Optuna with progress bar
n_trials = 10 

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=n_trials) 

print(f"Best trial: {study.best_trial.params}")

# Save best parameters to a file
save_directory = "/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/distilbert_recipeqa_optuna_large_new"
os.makedirs(save_directory, exist_ok=True)

best_params_path = os.path.join(save_directory, "best_params.json")
with open(best_params_path, 'w') as f:
    json.dump(study.best_trial.params, f)
print(f"Best trial parameters saved to {best_params_path}")

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Tokenizer and best model saved successfully.")

# %%

# final training with best parameters

# Load best parameters
with open(best_params_path, 'r') as f:
    best_params = json.load(f)
print("Best trial parameters loaded: ", best_params)

final_model = DistilBertForMultipleChoice.from_pretrained(save_directory)
tokenizer = DistilBertTokenizer.from_pretrained(save_directory)

final_training_args = TrainingArguments(
    output_dir=os.path.join(save_directory, "final_model_trainer"),
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=best_params["num_train_epochs"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    logging_steps=8,
    load_best_model_at_end=True,
    report_to="none",
    seed=42,
)

final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

final_trainer.train()

save_directory_final = "/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/distilbert_recipeqa_final_model"
os.makedirs(save_directory, exist_ok=True)

# Save the final model and tokenizer
final_model.save_pretrained(save_directory_final)
tokenizer.save_pretrained(save_directory_final)
print(f"Final model and tokenizer saved to {save_directory_final}")

# Evaluate the final model on the validation set
final_results = final_trainer.evaluate(tokenized_dataset["val"])
print(f"Final Model Evaluation on Validation Set: {final_results}")

#make predictions on the val set
val_predictions = final_trainer.predict(tokenized_dataset["val"])
val_predicted_labels = np.argmax(val_predictions.predictions, axis=1)
val_true_labels = val_predictions.label_ids


print("\nSample Predictions:")
for idx in range(5):  # Show first 5 examples
    example = recipeqa_dataset["val"][idx]
    full_question = f"{example['question_text']} " + " ".join([
        q if q != "@placeholder" else example["choice_list"][0] 
        for q in example["question"]
    ])
    print(f"Q{idx + 1}: {full_question}")
    print("Choices:")
    for i, choice in enumerate(example["choice_list"]):
        selected = "✓" if i == val_predicted_labels[idx] else ""
        print(f"  {i}: {choice} {selected}")
    print(f"Predicted Answer: {example['choice_list'][val_predicted_labels[idx]]}")
    print(f"Actual Answer: {example['answer']}")


# accuracy scores
accuracy_val = accuracy_score(val_true_labels, val_predicted_labels)
f1_val = f1_score(val_true_labels, val_predicted_labels, average='weighted')  

# Print the results 
print(f"\n Accuracy: {accuracy_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")

#%%

final_model = DistilBertForMultipleChoice.from_pretrained(save_directory_final)
tokenizer = DistilBertTokenizer.from_pretrained(save_directory_final)
best_params_path = "/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/distilbert_recipeqa_optuna_large_new/best_params.json"

# best parameters from Optuna
with open(best_params_path, 'r') as f:
    best_params = json.load(f)
print("Best trial parameters loaded: ", best_params)


# Dummy TrainingArguments 
final_training_args = TrainingArguments(
    output_dir="./results",  
    per_device_eval_batch_size=16,  
    report_to="none",
    seed=42,
    eval_strategy="no",
)

# Setup trainer only for evaluation
trainer_test_set = Trainer(
    model=final_model,
    args=final_training_args,
    eval_dataset=tokenized_dataset["test"], 
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

test_results = trainer_test_set.evaluate(tokenized_dataset["test"])
print(f"Final evaluation results on test set: {test_results}")


test_predictions = trainer_test_set.predict(tokenized_dataset["test"])
test_predicted_labels = np.argmax(test_predictions.predictions, axis=1)
test_true_labels = test_predictions.label_ids


print("\nSample Predictions:")
for idx in range(5): 
    example = recipeqa_dataset["test"][idx]
    full_question = f"{example['question_text']} " + " ".join([
        q if q != "@placeholder" else example["choice_list"][0] 
        for q in example["question"]
    ])
    print(f"Q{idx + 1}: {full_question}")
    print("Choices:")
    for i, choice in enumerate(example["choice_list"]):
        selected = "✓" if i == test_predicted_labels[idx] else ""
        print(f"  {i}: {choice} {selected}")
    print(f"Predicted Answer: {example['choice_list'][test_predicted_labels[idx]]}")
    print(f"Actual Answer: {example['answer']}")


