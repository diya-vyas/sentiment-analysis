"""
preprocess.py
Text cleaning and preprocessing utilities for sentiment analysis.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Remove URLs, mentions, special characters, and lowercase text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'@\w+', '', text)                  # Remove mentions
    text = re.sub(r'#\w+', '', text)                  # Remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)              # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()          # Normalize whitespace
    return text


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords."""
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w not in STOP_WORDS])


def lemmatize_text(text: str) -> str:
    """Reduce words to their base/lemma form."""
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens])


def full_pipeline(text: str) -> str:
    """Apply full preprocessing: clean → remove stopwords → lemmatize."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """Apply full_pipeline to a DataFrame column and return cleaned df."""
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(full_pipeline)
    return df


if __name__ == '__main__':
    sample = "This product is AMAZING!! 😍 Check it out: https://example.com #NLP @user123"
    print("Original:", sample)
    print("Cleaned: ", full_pipeline(sample))
