# Load libraries
import pandas as pd
import numpy as np
import spacy
from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

# Load the Spacy NLP model and the sentiment classifier
nlp_model = spacy.load('model-best')
sentiment_classifier = SentimentClassifier("multilingual")

# Function to use the Spacy NER model to locate the dishes in a given review
def tag_dish_aspects(sentence):
    tagged_review = sentence

    # Add tags to the review to indicate the start and end of dish aspects
    for entity in reversed(list(nlp_model(sentence).ents)):
        if entity.label_ == 'DISH':
            tagged = f'[B-ASP]{entity.text}[E-ASP]'
            start = entity.start_char
            end = entity.end_char
            tagged_review = tagged_review[:start] + tagged + tagged_review[end:]

    return tagged_review

# Function to do sentiment analysis using the sentiment classifier
def predict_sentiment(tagged_review):
    result = sentiment_classifier.predict(tagged_review,
                save_result=False,
                print_result=False,
                ignore_error=True,
                pred_sentiment=True)
    
    if result['aspect'] != ['Global Sentiment']:
        return [(aspect, sentiment, confidence) for aspect, sentiment, confidence 
                in zip(result['aspect'], result['sentiment'], result['confidence'])]

    return []

# Main function to combine dish and sentiment analysis
def absa(reviews):
    data = []

    for row_num, review in enumerate(reviews):
        sentences = sent_tokenize(review)

        for sentence in sentences:
            tagged_review = tag_dish_aspects(sentence)
            data.extend([(row_num, aspect, sentiment, confidence) for aspect, sentiment, confidence 
                         in predict_sentiment(tagged_review)])

    analyzed_data = pd.DataFrame(data, columns=['review_id', 'dish', 'sentiment', 'confidence'])

    return analyzed_data

