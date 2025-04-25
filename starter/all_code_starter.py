# Download and load spacy english nlp pipeline

#! python -m spacy download en_core_web_sm

import spacy

# Load the spaCy language model
nlp = spacy.load('en_core_web_sm')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

# Load data
df = pd.read_csv(
    'data/reviews.csv',
)

df.info()
df.head()

# Create X and y features

data = df

# separate features from labels
X = data.drop('Recommended IND', axis=1)
y = data['Recommended IND'].copy()

print('Labels:', y.unique())
print('Features:')
display(X.head())



# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    shuffle=True,
    random_state=27,
)

# Filtering relevant features

# Numerical features
num_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Clothing ID','Recommended IND'])

print('Numerical features:', num_features)

# Categorical features
cat_features = df.select_dtypes(include=['object']).columns.drop(["Review Text", 'Title']).tolist()
cat_features.append('Clothing ID')
print('Categorical features:', cat_features)

# Text features
text_features = df[['Review Text']].columns
#text_features = df[['Review Text','Title']].columns


print('Text features:', text_features)

# Numerical pipeline

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler()) ])

num_pipeline

# Categorical pipeline

cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
cat_pipeline





from sklearn.base import BaseEstimator, TransformerMixin

# Class that outputs the character count in reviews

class CountCharacter(BaseEstimator, TransformerMixin):

#''' Outputs the number times that character appears in reviews '''

    def __init__(self, character: str):
        self.character = character

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[text.count(self.character)] for text in X]
    




from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Pipeline for reshaping the review tex data into a 1-dimensional array
initial_text_preprocess = Pipeline([
    (
        'dimension_reshaper',
        FunctionTransformer(
            np.reshape,
            kw_args={'newshape':-1},
        ),
    ),
])

# Pipeline for counting the number of spaces, `!`, and `?` using class CountCharacter()
feature_engineering = FeatureUnion([
    ('count_spaces', CountCharacter(character=' ')),
    ('count_exclamations', CountCharacter(character='!')),
    ('count_question_marks', CountCharacter(character='?')),
])

# Combining the two pipelines to count characters in review data
character_counts_pipeline = Pipeline([
    (
        'initial_text_preprocess',
        initial_text_preprocess,
    ),
    (
        'feature_engineering',
        feature_engineering,
    ),
])

character_counts_pipeline







# SpacyLemmatizer that removes stopwords and lemmatizes reviews
class SpacyLemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lemmatized = [
            ' '.join(
                token.lemma_ for token in doc
                if not token.is_stop
            )
            for doc in self.nlp.pipe(X)
        ]
        return lemmatized   
    




# Create a TF_IDF vectorizer that creates matrix of tfidf scores

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_pipeline = Pipeline([
    (
        'dimension_reshaper',
        FunctionTransformer(
            np.reshape,
            kw_args={'newshape':-1},
        ),
    ),
    (
        'lemmatizer',
        SpacyLemmatizer(nlp=nlp),
    ),
    (
        'tfidf_vectorizer',
        TfidfVectorizer(
            stop_words='english',
        ),
    ),
])
tfidf_pipeline 









# Combine multiple preprocessing & nlp pipelines into fetaure engineering
from sklearn.compose import ColumnTransformer

feature_engineering = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
        ('character_counts', character_counts_pipeline, text_features),
        ('tfidf_text', tfidf_pipeline, text_features),
])

feature_engineering






from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

model_pipeline = make_pipeline(
    feature_engineering,
    RandomForestClassifier(random_state=27),
)

model_pipeline.fit(X_train, y_train)

# Fine-tune the classification model predictions

model_pipeline.get_params().keys()

# Evaluate the preditctions of of our classification model 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred_forest_pipeline = model_pipeline.predict(X_test)
accuracy_forest_pipeline = accuracy_score(y_test, y_pred_forest_pipeline)
precision = precision_score(y_test, y_pred_forest_pipeline)
recall = recall_score(y_test, y_pred_forest_pipeline)
f1 = f1_score(y_test, y_pred_forest_pipeline)

print('Accuracy:', accuracy_forest_pipeline)
print(f"Precision:", precision)
print(f"Recall:", recall)
print(f"F1 Score:", f1)




# use grid search to find the best hyperparameters for the random forest classifier
from sklearn.model_selection import GridSearchCV

# try 4x4x2=32 combinations of hyperparameters
param_grid = [{'randomforestclassifier__n_estimators': [10, 50], # number of trees
               'randomforestclassifier__max_depth': [3], # max depth
               'randomforestclassifier__max_features': [2, 4]}]  # number of features to consider at each split 


# train the model across 5 folds, that's a total of 32x5=160 rounds of training
grid_search = GridSearchCV(model_pipeline, param_grid, cv=2, verbose=3)
grid_search.fit(X_train, y_train)


# retrieve the best parameters
grid_search.best_params_


# retrieve the best model
model_forest_best = grid_search.best_estimator_
model_forest_best



# calculate the final metrics of the best model
y_pred_forest_best = model_forest_best.predict(X_test)
accuracy_forest_best = accuracy_score(y_test, y_pred_forest_best)

precision_best = precision_score(y_test, y_pred_forest_best)
recall_best = recall_score(y_test, y_pred_forest_best)
f1_best = f1_score(y_test, y_pred_forest_best)

print('Final accuracy:', accuracy_forest_best)
print(f"Final precision:", precision_best)
print(f"Final recall:", recall_best)
print(f"Final f1 score:", f1_best)

print(classification_report(y_test, y_pred_forest_best))