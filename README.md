# StyleSense AI: Predicting Product Recommendations from Fashion Reviews

## Using Natural Language Processing & Classification

# Description 

StyleSense is a fast-growing online women's clothing retailer that is facing a challenge. They have a growing number of product reviews which have missing recommendation labels. While customers still provide feedback through review text and other product details, the recommendation field is often left blank.

As a data scientist at StyleSense, my task was to build a predictive model that analyzes these incomplete reviews—considering factors like customer age, product category, and review text to determine whether the customer would recommend the product.

By automating this process, StyleSense will gain valuable insights into customer satisfaction and be better equipped to spot trends and improve their customer's shopping experience.

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

All tests can be found here:[Tests](test_pipeline.py)

### Break Down Tests

```
1. test_data_load(): verifies that the data source is accessible, structured as expected and contains the review data.

2. test_count_character_spaces(): verifies that the custom transformer class is performing as expected; for example it counts blank spaces in review texts.

3. test_spacy_lemmatizer(): verifies that the custom spacy transformer class is performing as a lemmatizer should. A lemmatizer is a tool used in natural language processing (NLP) that reduces words to their base or dictionary form, called a lemma.

```

## Project Instructions

```
Handle missing data for numerical, categorical, and text fields.

Build a preprocessing pipeline using Pipeline and ColumnTransformer.

Apply TF-IDF vectorization to review text to highlight unique and informative terms.

Train and evaluate a classification model (e.g., RandomForestClassifier).

Assess model performance with precision, recall, and F1-score.

Discuss challenges such as data imbalance and how it affects predictions.
```

## Built With

* [pandas](https://pandas.pydata.org/) - Data manipulation
* [numpy](https://numpy.org/) - Data manipulation
* [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library
* [spaCy](https://spacy.io/) - Natural Language Processing
* [Jupyter Notebook](https://jupyter.org/) - Interactive coding environment
* [Pytest](https://docs.pytest.org/en/stable/) - Testing Code



## License

[License](LICENSE.txt)
