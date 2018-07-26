"""
Simple script to extract nouns and noun chunks in English originally posted by
Joel
"""

import spacy
import pandas as pd

filename = "text_to_analyze_by_lines.csv"

df = pd.read_csv(filename, sep='\t')

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')

for index, line in df.iterrows():
    print(line)
    doc = nlp(line)

    for token in doc:
       if token.pos_ == 'VERB':
           print(token.text)
    for noun_ph in doc.noun_chunks:
        print(noun_ph)
    print('****')
    for verb_ph in doc.verb_chunks:
        print(verb_ph)

print(list(df))