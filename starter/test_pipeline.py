import pandas as pd
import pytest
import spacy
from test import CountCharacter, SpacyLemmatizer  # Importing from starter.py

nlp = spacy.load("en_core_web_sm")

@pytest.fixture
def sample_data():
    return ["This is a test!", "Another sentence."]

def test_data_load():
    '''Test if review data is being extracted from the data source'''
    try:
        df = pd.read_csv("data/reviews.csv")
        assert not df.empty
        assert "Review Text" in df.columns
    except Exception as e:
        pytest.fail(f"An error occurred while reading the file: {e}")

def test_count_character_spaces(sample_data):
    '''Tests if character counter class works as expected'''
    count_spaces = CountCharacter(" ")
    result = count_spaces.transform(sample_data)
    expected = [[3], [1]]
    assert result == expected

def test_spacy_lemmatizer(sample_data):
    '''Tests if my spacy lemmatizer class works as a lemmatizer should'''
    lemmatizer = SpacyLemmatizer(nlp=nlp)
    result = lemmatizer.transform(sample_data)
    
    # Log the result for debugging purposes
    print(f"Lemmatizer result: {result}")
    
    # Check if lemmatization is happening correctly
    assert isinstance(result[0], str)
    assert "test" in result[0]  # lemma of "test"
    assert "sentence" in result[1]  # lemma of "sentence"
    assert "cat" not in result[0]  # 'cat' should not appear in the result if 'Cats' is not in the input.
