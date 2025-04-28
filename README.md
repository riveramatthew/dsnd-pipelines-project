# StyleSense AI: Predicting Product Recommendations from Fashion Reviews

## Using Natural Language Processing & Classification

# Description 

StyleSense is a fast-growing online women's clothing retailer that is facing a challenge. They have a growing number of product reviews which have missing recommendation labels. While customers still provide feedback through review text and other product details, the recommendation field is often left blank.

As a data scientist at StyleSense, my task was to build a predictive model that analyzes these incomplete reviews—considering factors like customer age, product category, and review text to determine whether the customer would recommend the product.

By automating this process, StyleSense will gain valuable insights into customer satisfaction and be better equipped to spot trends and improve their customer's shopping experience.

I. To gain insight into customer satisfaction.

II. To automatically label missing recommendation fields.

III. To better forecast trending products and customer sentiment.

# File Description


1. starter.ipynb:	Main Jupyter notebook used for exploratory data analysis, feature engineering, and building the initial ML pipeline.
2. data/reviews.csv: The main dataset containing customer reviews, used for training and testing the model.
3. README.md: Project documentation including setup instructions, project overview, testing guide, and more
4. test_pipeline.py: Contains Pytest-based unit tests to validate the functionality of custom class transformers and ensure data is being loaded correctly.
5. test.py:	Holds the functions and custom transformer classes such as CountCharacter and SpacyLemmatizer used within the main notebook starter.ipynb.


## Getting Started

Instructions for how to get a copy of the project running on your local machine:

### Dependencies

```
This project can be run using Python version 3. Ensure the following packages are installed before running the code:

1. pandas
2. numpy
3. Spacy: ! python -m spacy download en_core_web_sm
4. scikit-learn
5. Pytest
```

### Installation

This ia a step by step explanation of how to get a dev environment running:

```
1. Clone the repository

git clone https://github.com/udacity/dsnd-pipelines-project.git
cd dsnd-pipelines-project

2. Install requirements

python -m pip install -r requirements.txt

3. Download spaCy’s English NLP model

python -m spacy download en_core_web_sm

4. Install dependencies

pip install pandas numpy scikit-learn pytest spacy

5. Load the spacy English NLP model

nlp = spacy.load('en_core_web_sm')
```

## Testing

Automated tests ensure that the components in this project work as expected. This project uses `pytest` for testing purposes, which allows us to run the tests efficiently and identify any issues quickly. There are 3 tests

All tests can be found here: [Tests](dsnd-pipelines-project/starter/test_pipeline.py)

### Break Down Tests

```
1. test_data_load(): verifies that the data source is accessible, structured as expected and contains the review data.

2. test_count_character_spaces(): verifies that the custom transformer class is performing as expected; for example it counts blank spaces in review texts.

3. test_spacy_lemmatizer(): verifies that the custom spacy transformer class is performing as a lemmatizer should. A lemmatizer is a tool used in natural language processing (NLP) that reduces words to their base or dictionary form, called a lemma.

```

## Project Instructions

```
1. Handle missing data for numerical, categorical, and text fields.

2. Build a preprocessing pipeline using Pipeline and ColumnTransformer.

3. Apply TF-IDF vectorization to review text to highlight unique and informative terms.

4. Train and evaluate a classification model (e.g., RandomForestClassifier).

5. Assess model performance with precision, recall, and F1-score.

6. Discuss challenges and how it affects predictions.
```

## Results

### 1. Metrics Explanation

Class 1 = Customer would recommend the product (positive review)
Class 0 = Customer would not recommend the product (negative review)

Accuracy:
Percentage of total reviews (both positive and negative) that the model correctly classified.
(High accuracy means the model usually predicts customer sentiment correctly.)

Precision:
Of all reviews the model predicted as "would recommend" (positive), how many were actually correct.
(Important when we want to avoid falsely assuming a customer is happy.)

Recall:
Of all actual positive reviews, how many the model successfully identified.
(High recall means we are capturing almost all true happy customers.)

F1 Score:
F1 Score is particularly useful when the classes (like class 0 and class 1 in this case) are imbalanced, as it gives a balance between precision and recall, favoring neither class. This is why it is crucial in our case where identifying both recommending and non-recommending customers accurately is important.


In our case, F1 score is the most important metric because it ensures that the model is not only good at detecting positive reviews (Class 1) but also improves at detecting the less common negative reviews (Class 0).


### 2. Models Evaluated:


Model	Description
Results 1	Random Forest (default)
Results 2	Random Forest (after basic GridSearchCV tuning)
Results 3	XGBoost Classifier (with scale_pos_weight=2)
Results 4	Gradient Boosting Classifier (default settings)
Results 5	XGBoost Classifier (with scale_pos_weight=5)
Results 6	XGBoost Classifier (with scale_pos_weight=1)

### 3. Performance Overview

Result	               Accuracy	Precision  Recall	F1_Score  Class_0_F1_Score	Class_1_F1_Score
1 (RandomForest 
        default)	    0.848	 0.850	   0.990	  0.915	      0.31	              0.91
2 (RandomForest tuned)	0.823	 0.823	   1.000	  0.903	      0.00	              0.90
3 (XGBoost 
scale_pos_weight=2)	    0.867	 0.877	   0.975	  0.924	      0.49	              0.92
4 (GradientBoosting)	0.854	 0.864	   0.975	  0.916	      0.41	              0.92
5 (XGBoost 
scale_pos_weight=5)	    0.854	 0.864	   0.975	  0.916	      0.41	              0.92
6 (XGBoost 
scale_pos_weight=1)	    0.870	 0.893	   0.958	  0.924	      0.56	              0.92

### 4. Summary

The XGBoost model with scale_pos_weight=1, Result 6, should be deployed as the production model for predicting product recommendations, based on customer reviews at StyleSense. We evaluated several machine learning models to predict whether customers would recommend StyleSense products based on their text-based reviews and other details.

The best model achieved 87% accuracy and the highest F1 score of 92.4%, meaning it balances identifying both happy and unhappy customer reviews better than any other model tested. While the model still favors customer that would recommend StyleSense products, due to the class imbalance caused by higher number of positive reviews, it is also much better than previous models at spotting negative reviews, helping StyleSense quickly address product issues. This model is now ready to help StyleSense track customer satisfaction at scale and drive smarter business decisions.

## Future Scope

1. Further Data Balancing:

To further enhance the detection of the minority class (Class 0, negative reviews), we can experiment with data balancing techniques. Applying oversampling methods like SMOTE (Synthetic Minority Over-sampling Technique) can generate synthetic examples of Class 0, helping the model learn more from the underrepresented class. Alternatively, undersampling Class 1 (positive reviews) can balance the dataset by reducing the number of overrepresented positive reviews.

2. Threshold Optimization:

The classification probability threshold plays a key role in balancing precision and recall. By adjusting this threshold, we can fine-tune the model’s ability to detect Class 0 without sacrificing the performance on Class 1. This could involve lowering the threshold slightly to increase recall for Class 0, while still maintaining a good precision-recall trade-off.

3. Advanced Hyperparameter Tuning:

Further optimizing the hyperparameters of the XGBoost model can significantly enhance its performance. We can perform a randomized search or Bayesian optimization to find the best combination of hyperparameters such as max_depth, learning_rate, min_child_weight, and others. This process could help us find a configuration that improves model generalization and performance, particularly for Class 0.

4. Model Explainability:

To make the model more interpretable for the StyleSense business team, we can use model explainability tools such as SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations). These techniques will allow us to understand which features (e.g., review text, product category, customer age) are most influential in the model's decision-making process. This transparency will help StyleSense stakeholders trust the model and make informed decisions based on the predictions.

## Built With

* [pandas](https://pandas.pydata.org/) - Data manipulation
* [numpy](https://numpy.org/) - Data manipulation
* [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library
* [spaCy](https://spacy.io/) - Natural Language Processing
* [Jupyter Notebook](https://jupyter.org/) - Interactive coding environment
* [Pytest](https://docs.pytest.org/en/stable/) - Testing Code



## License

[License](LICENSE.txt)
