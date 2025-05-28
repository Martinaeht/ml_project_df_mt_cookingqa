#Dimi Test 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #adapt

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipykernel
import optuna
import evaluate  # for computing accuracy
from pandas import json_normalize
from datasets import load_dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import f1_score

# Import Transformers modules
from transformers import (
    DataCollatorWithPadding,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    AutoTokenizer,
    set_seed,
    DistilBertForMultipleChoice,
    DataCollatorForMultipleChoice,
)

# %%

print(f"Using device: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

set_seed(42)

# %%
train_dataset_path = '/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset/train_qafilt_true.json'
val_dataset_path = '/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset/val_qafilt_true.json'
test_dataset_path = '/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/Dataset_RecipeQA/RecipeQA_dataset/test_qafilt_true.json'

train_dataset = load_dataset('json', data_files=train_dataset_path, split='train')
val_dataset = load_dataset('json', data_files=val_dataset_path, split='train')
test_dataset = load_dataset('json', data_files=test_dataset_path, split='train')

recipeqa_dataset = DatasetDict({
    "train": train_dataset.shuffle(seed=42),
    "val": val_dataset.shuffle(seed=42),
    "test": test_dataset.shuffle(seed=42),
})

print(recipeqa_dataset)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
max_length = 512

def preprocess_function(examples):
    all_input_ids = []
    all_attention_masks = []
    labels = []
        # 1. Combine context bodies into one string; 2. Construct the full question text (question_text + actual question with placeholder replaced)
    for i in range(len(examples["context"])):
        context_bodies = " ".join([step["body"] for step in examples["context"][i]])
        question_variants = [
            f"{examples['question_text'][i]} " + 
            " ".join([q if q != "@placeholder" else choice for q in examples["question"][i]])
            for choice in examples["choice_list"][i]
        ]

        # 3. Tokenize with context as the first sentence, variant as second
        tokenized = tokenizer(
            [context_bodies] * len(question_variants),    # Repeat context for each choice
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

print("Tokenization starts...")
tokenized_dataset = recipeqa_dataset.map(preprocess_function, batched=True, remove_columns=recipeqa_dataset["train"].column_names)
print("Tokenization finished.")
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy_metric = accuracy.compute(predictions=predictions, references=labels)
    precision_score = precision.compute(predictions=predictions, references=labels, average="weighted")
    recall_score = recall.compute(predictions=predictions, references=labels, average="weighted")
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")

    return {
        **accuracy_metric,
        'precision': precision_score['precision'],
        'recall': recall_score['recall'],
        'f1': f1_score['f1']
    }

#Training with Optuna
def objective(trial):
    # Define the hyperparameters to be optimized
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)  # Weight decay to regularize the model
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)  # Dropout rate for regularization
    adam_epsilon = trial.suggest_loguniform('adam_epsilon', 1e-8, 1e-6)  # Epsilon for Adam optimizer
    
    roberta_model = AutoModelForMultipleChoice.from_pretrained("roberta-base")

    # Define the model configuration
    roberta_model.config.attention_probs_dropout_prob = dropout_rate
    roberta_model.config.hidden_dropout_prob = dropout_rate

    # Set up training arguments
    arguments = TrainingArguments(
      output_dir="/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/roberta_model_optuna",
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      logging_steps=8,
      num_train_epochs=num_train_epochs,
      eval_strategy="epoch", 
      save_strategy="epoch",
      learning_rate = learning_rate,
      weight_decay=weight_decay,
      adam_epsilon=adam_epsilon,
      load_best_model_at_end=True,
      report_to='none',
      seed=42
      )

    trainer = Trainer(
      model=roberta_model,
      args=arguments,
      train_dataset=tokenized_dataset['train'],
      eval_dataset=tokenized_dataset['val'], # change to test when you do your final evaluation!
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
      )

    trainer.train()
    results = trainer.evaluate()
    evaluation_accuracy = results['eval_accuracy']
    return evaluation_accuracy

study = optuna.create_study(direction="maximize")  # We want to maximize validation accuracy
study.optimize(objective, n_trials=10)

best_params = study.best_trial.params
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation accuracy: {study.best_value}")

# Save best parameters to a file
save_directory = "/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/roberta_model_optuna_best_params"
os.makedirs(save_directory, exist_ok=True)

best_params_path = os.path.join(save_directory, "best_params.json")
with open(best_params_path, 'w') as f:
    json.dump(study.best_trial.params, f)
print(f"Best trial parameters saved to {best_params_path}")

tokenizer.save_pretrained(save_directory)

roberta_model = AutoModelForMultipleChoice.from_pretrained("roberta-base")
# Save the best model
roberta_model.save_pretrained("/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/roberta_model_optuna_best_model")
print("Best model saved successfully.")

# %%

# final training with best parameters
with open(best_params_path, 'r') as f:
    best_params = json.load(f)
print("Best trial parameters loaded: ", best_params)

#das finale Model musste ich auf Collab trainieren, da der Server nicht genug RAM hatte

final_roberta_model = AutoModelForMultipleChoice.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained(save_directory)

final_training_args = TrainingArguments(
    output_dir="/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/roberta_model_optuna_final_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=best_params["num_train_epochs"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
    adam_epsilon=best_params["adam_epsilon"],
    logging_steps=8,
    load_best_model_at_end=True,
    report_to="none",
    seed=42,
)

final_trainer = Trainer(
    model=final_roberta_model,
    args=final_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

final_trainer.train()

save_directory_final = "/home/mlt_ml3/project_cookingqa/Project_Applied_ML_Cooking/roberta_recipeqa_final_best_model"
os.makedirs(save_directory_final, exist_ok=True)
# Save the final model and tokenizer
final_roberta_model.save_pretrained(save_directory_final)
tokenizer.save_pretrained(save_directory_final)
print(f"Final model and tokenizer saved to {save_directory_final}")

# Evaluate the final model on the validation set
final_results = final_trainer.evaluate(tokenized_dataset["val"])
print(f"Final Model Evaluation on Validation Set: {final_results}")

#Make predictions for deeper analysis
val_predictions = final_trainer.predict(tokenized_dataset["val"])
val_predicted_labels = np.argmax(val_predictions.predictions, axis=1)
val_true_labels = val_predictions.label_ids

# 3. Show some example predictions
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
accuracy_val= accuracy_score(val_true_labels, val_predicted_labels)
f1_val = f1_score(val_true_labels, val_predicted_labels, average='weighted')  # Weighted F1 for multi-class classification

# Print the results 
print(f"\n Accuracy: {accuracy_val:.4f}")
print(f"F1 Score: {f1_val:.4f}")

############

test_results = final_trainer.evaluate(tokenized_dataset['test'])
print(f"Final evaluation results on test set: {test_results}")

test_predictions = final_trainer.predict(tokenized_dataset["test"])
test_predicted_labels = np.argmax(test_predictions.predictions, axis=1)
test_true_labels = test_predictions.label_ids


for idx in range(10):  
    example = recipeqa_dataset["test"][idx]
    print(f"\nQ{idx + 1}: {example['question_text']}")
    print("Choices:")
    for i, choice in enumerate(example["choice_list"]):
        selected = "✓" if i == test_predicted_labels[idx] else ""
        print(f"  {i}: {choice} {selected}")
    print(f"Predicted Answer: {example['choice_list'][test_predicted_labels[idx]]}")
    print(f"Actual Answer: {example['answer']}")

'''Hyperparameter Visualization wollte ich machen, aber es war am server nicht möglich nach dem Trainieren'''

#trial_numbers = list(range(1, len(study.trials) + 1))
#accuracy_values = [trial.value for trial in study.trials]

#plt.plot(trial_numbers, accuracy_values)
#plt.xlabel('Trial Number')
#plt.ylabel('Validation Accuracy')
#plt.title('Optuna Hyperparameter Optimization')
#plt.show()

