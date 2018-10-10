import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')

df = pd.read_csv(r'C:\Users\joshu\Downloads\Data\reddit\reddit_pf1.csv', engine='python', index_col=[0])

def create_tokens(text):
    doc = nlp(text)
    tokens = [token for token in doc]
    return tokens

df['tokenized_title'] = df['clean_title'].apply(lambda x: create_tokens(x))

def lemmatize(text):
    lemmas = [token.lemma_ for token in text if token.lemma_ not in '-PRON-']
    return lemmas

df['lemmatized_title'] = df['tokenized_title'].apply(lambda x: lemmatize(x))

def create_NER(text, label = False):
    doc = nlp(text)
    if label is False:
        NER_list = [(ent.text) for ent in doc.ents]
    else:
        NER_list = [(ent.label_) for ent in doc.ents]
    return NER_list    

df['named_entities'] = df['title'].apply(lambda x: create_NER(x))
df['entity_labels'] = df['title'].apply(lambda x: create_NER(x, label = True))
