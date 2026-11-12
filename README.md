# ml_project_df_mt_cookingqa

Comparison of Naïve Bayes, DistilBERT and RoBERTa-base on the textual cloze task of RecipeQA (Yagcioglu et al., 2018).


Results

Naïve Bayes (TF-IDF): Accuracy: 31.29 %,   F1: 31.46 %

DistilBERT:	          Accuracy: 78.41 %,   F1: 78.52 %

RoBERTa-base:         Accuracy: 84.1 %,    F1: 84.2 %


Approach
- Text-only textual cloze questions (4 choices each); new split of 6,158 / 1,320 / 1,320 samples
- Each choice is inserted into the question and paired with the full recipe context (truncation="only_first")
- Transformers fine-tuned with Hugging Face; hyperparameters tuned with Optuna (10 trials)
