import re
import spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "lemmatizer"])
stopwords = nlp.Defaults.stop_words  # Default stopwords (326) for spacy.
stopwords.add("umph")  # You can add more stopwords if you like.

def tokenizer(text): 
    tokens = [token.text.lower() for token in nlp(cleaner(text))]
    # tokens = [word for word in tokens if not word in stopwords]  # In case you want to remove stop words.
    return tokens

def cleaner(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    return text.strip()