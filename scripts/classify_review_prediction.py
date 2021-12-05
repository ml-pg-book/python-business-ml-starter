import sys
import pandas as pd
from joblib import load

import spacy

nlp = spacy.load('ja_ginza')


def predict(x: str):
    estimator = load('../models/classify_review.joblib')
    doc = nlp(x)
    return estimator.predict([doc.vector])[0]

def main(x: str):
	y_pred = predict(x)
	print(f"{y_pred}です。")

if __name__ == "__main__":
	main(sys.argv[1])