import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
def remove_special_chars_and_floats(text):
    text = re.sub(r'\b\d+\.\d+\b', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text
def remove_duplicate_sentences(text):
    sentences = sent_tokenize(text)
    unique_sentences = list(dict.fromkeys(sentences))
    return ' '.join(unique_sentences)
df = pd.read_csv(r'C:\Users\Admin\cuda_crawler\cuda_document.csv')
df = df.drop('section', axis=1)
for column in df.columns:
    if column != 'url':
        df[column] = df[column].apply(remove_special_chars_and_floats)
        df[column] = df[column].str.lower()
        df[column] = df[column].apply(remove_duplicate_sentences)
df.to_csv('processed_file.csv', index=False)
print("CSV file has been processed and saved as 'processed_file.csv'")