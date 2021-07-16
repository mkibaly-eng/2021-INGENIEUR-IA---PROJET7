import streamlit as st
import numpy as np
import pandas as pd
import pickle  
import time
from io import StringIO
import joblib
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import re
import string



def load_model():
    global RNN
    global tokenizer
    global nlp
    
    RNN = tensorflow.keras.models.load_model('saved_model/saved_model', compile=True)
    data_df = pd.read_csv('tweet_100000_clean.csv',
                          encoding='latin',
                          delimiter=",")
    data_df['clean_tweet'] = data_df['clean_tweet'].astype(str)
    tokenizer = Tokenizer(nb_words=2500, lower=True, split=' ')
    tokenizer.fit_on_texts(data_df['clean_tweet'].values)
    nlp = spacy.load('en_core_web_sm')

    

def preprocess_reviews(reviews):
    REPLACE_NO_SPACE = re.compile("[.;:!\?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(')|[0-9]")
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews


def get_lemmatized_text(corpus):
    lem = [' '.join([token.lemma_ for token in nlp(review)]) for review in corpus]
    return lem
    


def remove_stop_words(corpus, stop_words):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in stop_words])
            )
    return removed_stop_words



def predict(tweet):
    
    MAX_WORDS = 34
    
    out = [tweet]
    out = preprocess_reviews(out)
    
    sp = string.punctuation
    out = list(map(lambda t: ''.join(["" if c.isdigit() else c for c in t]), out))
    out = list(map(lambda t: ''.join(["" if c in sp else c for c in t]), out))
    out = list(map(str.lower, out))
    
    out = get_lemmatized_text(out)
    stop_words_ot = ['a', 'about', 'above', 'the', 'after', 'my', 'me', 'our',
                     'your', 'i', 'we', 'you', 'he', 'she', 'they',
                     'their', 'in', 'until', 'before', 'it', 'and', 'at', 'of']

    out = remove_stop_words(out, stop_words_ot)       
    
    X = tokenizer.texts_to_sequences(out)

    vect = pad_sequences(X, maxlen=MAX_WORDS, padding='post')
   
    sentiment = RNN.predict(vect)
    
    my_prediction = np.argmax(sentiment, axis=1) 
    
    result = ['positif' if lbl == 1 else 'négatif' for lbl in my_prediction]
    
    return(result)

def run():
    
    st.sidebar.info('Vous pouvez soit saisir un texte en ligne ou telecharger un fichier txt')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    add_selectbox = st.sidebar.selectbox("Mode de prediction?", ("En ligne", "Fichier txt"))
    st.title("Analyse du sentiment d'un tweet")
    st.header('Cette application permet de prédire le sentiment d un texte en anglais')
    if add_selectbox == "En ligne":
        text1 = st.text_area('Entrer le texte')
        output = ""
        if st.button("Predire"):
            output = predict(text1)
            output = str(output[0])
            st.success(f"Le sentiment du texte est {output}")
            st.balloons()
    elif add_selectbox == "Fichier txt":
        output = ""
        file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])
        if st.button("Predire"):
             text_news = file_buffer.read()

        st_version = st.__version__ 
        versions = st_version.split('.')
        if int(versions[1]) > 67:
             text_news = text_news.decode('utf-8')
        print(text_news)
            
        output = predict(text_news)
        output = str(output[0])
        st.success(f"Le sentiment du texte est {output}")
        st.balloons()
            
if __name__ == "__main__":
    load_model()
    run()
