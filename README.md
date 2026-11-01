# NLP_Preprocessing_on_twitter_Tarining_data
# Text Preprocessing & Tokenization Pipeline

This repository contains a structured workflow for cleaning raw text and preparing it for deep learning models using `pandas`, `nltk`, and `TensorFlow`/`Keras`. It features a text-cleaning pipeline designed to remove noise, filter out stop words, and tokenize unstructured text columns into index-mapped integer sequences.

## 🚀 Features
* **Text Normalization:** Standardizes string formats by lowering case strings and compressing uneven spacing.
* **Stopword & Punctuation Stripping:** Cleans input phrases using Python's native `string` library and `nltk.corpus.stopwords`.
* **DataFrame Refactoring:** Automates structural operations via `pandas` to prune null rows, drop duplicates, and clear identity metrics.
* **Vector Vectorization:** Uses Keras's `Tokenizer` engine to transform structured words into standardized numeric sequence streams (`texts_to_sequences`).

---

## 🛠️ Getting Started

### Prerequisites
Install the required dependencies directly via pip:
```bash
pip install pandas tensorflow nltk
```

### Resource Acquisition
Run the following script configuration within your setup environment to download the text corpus dependencies:
```python
import nltk
nltk.download("stopwords")
```

---

## 📖 Usage Guide

The dataset preprocessing pipeline is divided into three distinct execution phases:

### 1. Data Cleaning & Normalization
Define the textual sanitizer to strip out punctuation characters, filter stop words, and handle string formatting:
```python
import re
import string
from nltk.corpus import stopwords

def text_preprocessor(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)  # Collapse extra spacing
    text = re.sub(r"\d+", " ", text)  # Purge numerical digits
    
    # Remove punctuation
    no_punc_text = "".join([char for char in text if char not in string.punctuation])
    
    # Filter language stopwords
    clean_text = " ".join([word for word in no_punc_text.split() if word not in stopwords.words("english")])
    return clean_text
```

### 2. DataFrame Optimization
Load the matrix using `pandas`, drop structural anomalies, prune duplicate entries, and clean target columns:
```python
import pandas as pd

# Load dataset columns
data = pd.read_csv("dataset.csv")

# Clean duplicate and missing indices
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# ALWAYS run text cleaning before mapping arrays
data["reviees"] = data["reviees"].apply(text_preprocessor)
```

### 3. Keras Vocabulary Tokenization
Fit the vocabulary space and transform cleaned strings into sequential numerical streams ready for training layers:
```python
from tensorflow.keras.preprocessing.text import Tokenizer

# Build word indexing map
tok = Tokenizer()
tok.fit_on_texts(data["reviees"])

# Sequence mapping transformation
data["reviees"] = tok.texts_to_sequences(data["reviees"])
```

---

## ⚠️ Pipeline Execution Warning
You must call `.apply(text_preprocessor)` **before** running `texts_to_sequences`. Reversing this layout transforms column values into integer arrays, causing your string methods to fail with an explicit `AttributeError: 'list' object has no attribute 'lower'`.
