
import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from datasets import Dataset


num_choices = 4
max_features = 10000
model_path = "/home/mlt_ml3/project_df_mt/ml_project_df_mt_cookingqa/nb_model"

train_data = [json.loads(line) for line in open('/home/mlt_ml3/project_df_mt/ml_project_df_mt_cookingqa/_RecipeQA_dataset/train_qafilt_true.json')]
val_data = [json.loads(line) for line in open('/home/mlt_ml3/project_df_mt/ml_project_df_mt_cookingqa/_RecipeQA_dataset/val_qafilt_true.json')]
test_data = [json.loads(line) for line in open('/home/mlt_ml3/project_df_mt/ml_project_df_mt_cookingqa/_RecipeQA_dataset/test_qafilt_true.json')]

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)

recipeqa_dataset = DatasetDict({
    "train": train_dataset.shuffle(seed=42),
    "val": val_dataset.shuffle(seed=42),
    "test": test_dataset.shuffle(seed=42),
})

# Preprocess function
def prepare_examples(dataset):
    texts, labels = [], []
    for example in dataset:
        context = " ".join([step["body"] for step in example["context"]])
        for idx, choice in enumerate(example["choice_list"]):
            question_text = f"{example['question_text']} " + " ".join([
                q if q != "@placeholder" else choice for q in example['question']
            ])
            combined = context + " " + question_text
            texts.append(combined)
            labels.append(1 if idx == example['answer'] else 0)
    return texts, labels

print("Preparing training data...")
X_train_texts, y_train = prepare_examples(train_dataset)
print("Preparing validation data...")
X_val_texts, y_val = prepare_examples(val_dataset)
print("Preparing test data...")
X_test_texts, y_test = prepare_examples(test_dataset)

# Vectorization
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=max_features)
X_train = vectorizer.fit_transform(X_train_texts)
X_val = vectorizer.transform(X_val_texts)
X_test = vectorizer.transform(X_test_texts)

# Train Naive Bayes classifier
print("Training Naive Bayes classifier...")
clf = MultinomialNB()
clf.fit(X_train, y_train)

def evaluate_grouped(model, X_texts, y_true, dataset_name="Dataset", group_size=num_choices):
    X = vectorizer.transform(X_texts)
    probs = model.predict_proba(X)[:, 1]  # Use probability for class 1
    preds, y_grouped = [], []

    for i in range(0, len(probs), group_size):
        group_probs = probs[i:i+group_size]
        pred_idx = np.argmax(group_probs)
        true_idx = np.argmax(y_true[i:i+group_size])
        preds.append(pred_idx)
        y_grouped.append(true_idx)

    acc = accuracy_score(y_grouped, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_grouped, preds, average='macro', zero_division=0
    )

    print(f"\nEvaluation on {dataset_name} (Grouped):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

# === Evaluate ===
evaluate_grouped(clf, X_val_texts, y_val, dataset_name="Validation Set")
evaluate_grouped(clf, X_test_texts, y_test, dataset_name="Test Set")

def show_predictions(model, X_texts, y_true, dataset, dataset_name="Dataset", group_size=num_choices, num_examples=5):
    X = vectorizer.transform(X_texts)
    probs = model.predict_proba(X)[:, 1]

    print(f"\nSample Predictions from {dataset_name}:")
    for idx in range(num_examples):
        start = idx * group_size
        end = start + group_size
        example = dataset[idx]
        group_probs = probs[start:end]
        pred_idx = np.argmax(group_probs)
        true_idx = np.argmax(y_true[start:end])

        # Reconstruct full question using the first choice (like your example)
        full_question = f"{example['question_text']} " + " ".join([
            q if q != "@placeholder" else example["choice_list"][0]
            for q in example["question"]
        ])

        print(f"\nQ{idx + 1}: {full_question}")
        print("Choices:")
        for i, choice in enumerate(example["choice_list"]):
            selected = "✓" if i == pred_idx else ""
            print(f"  {i}: {choice} {selected}")
        print(f"Predicted Answer: {example['choice_list'][pred_idx]}")
        print(f"Actual Answer: {example['choice_list'][true_idx]}")


"""
def show_predictions(model, X_texts, y_true, dataset, dataset_name="Dataset", group_size=num_choices, num_examples=5):
    X = vectorizer.transform(X_texts)
    probs = model.predict_proba(X)[:, 1]
    print(f"\nSample Predictions from {dataset_name}:")

    count = 0
    for i in range(0, len(probs), group_size):
        group_probs = probs[i:i+group_size]
        pred_idx = np.argmax(group_probs)
        true_idx = np.argmax(y_true[i:i+group_size])
        example = dataset[i // group_size]

        print(f"\nQuestion: {count + 1}: {example['question_text']} "+ " ".join([
            q if q != "@placeholder" else example['choice_list'][0] 
        for q in example['question']
        ])) #changed
        #print(f"Context: {' '.join([step['body'] for step in example['context']])}")
        for j, choice in enumerate(example["choice_list"]):
            label = ""
            if j == true_idx:
                label += "(Correct) "
            if j == pred_idx:
                label += "(Predicted) "
            print(f"  Choice {j}: {choice} {label}")

        count += 1
        if count >= num_examples:
            break
            
"""


# Show example predictions
show_predictions(clf, X_val_texts, y_val, recipeqa_dataset["val"], dataset_name="Validation Set")
show_predictions(clf, X_test_texts, y_test, recipeqa_dataset["test"], dataset_name="Test Set")

os.makedirs(model_path, exist_ok=True)
joblib.dump(clf, os.path.join(model_path, "nb_model.joblib"))
joblib.dump(vectorizer, os.path.join(model_path, "vectorizer.joblib"))
print(f"\nModel and vectorizer saved to: {model_path}")