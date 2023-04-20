# Import necessary libraries
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora, models

# Function to load data from folder path
def load_data(folder_path):
    articles = []
    
	# Iterate through all files in the folder path
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), "r", encoding="latin") as f:
            content = f.read()
            articles.append(content)
    return articles

# Function to preprocess the loaded data
def preprocess_data(articles):
    cleaned_articles = []
     # Iterate through all articles in the input list   

    for article in articles:
	# Remove excessive whitespace characters and convert to lowercase
        text = re.sub(r"\s+", " ", article)
        text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
	# Tokenize text into words
        tokens = word_tokenize(text)
	# Remove stop words (common words with little meaning such as "the" or "a")
        stop_words = set(stopwords.words("english"))
        custom_stopwords = {"mr", "said", "u"}  # Add any additional stopwords here
        stop_words = stop_words.union(custom_stopwords)
	# Remove any remaining short words and lemmatize 
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [token for token in tokens if len(token) > 1 or token in {"i", "a"}]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token for token in tokens if token != "u"]
        cleaned_articles.append(tokens)
    return cleaned_articles

# Function to create a corpus and dictionary from the preprocessed data
def create_corpus_dictionary(cleaned_articles):
    dictionary = corpora.Dictionary(cleaned_articles)
    corpus = [dictionary.doc2bow(article) for article in cleaned_articles]
    return corpus, dictionary

# Function to train an LDA model
def train_lda(corpus, dictionary, num_topics, passes=15):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
    return lda_model

# Function to print and plot the topics from an LDA model
def plot_topics(lda_model):
    num_topics = lda_model.num_topics
    topic_words = lda_model.print_topics(num_words=10)
    
    for topic in range(num_topics):
        print(f"Topic {topic}:")
        print(topic_words[topic])

# Function to run the LDA analysis and save the results to a file
def run_and_save_lda(folder_path, num_topics, num_words, passes, output_file):
    articles = load_data(folder_path)
    cleaned_articles = preprocess_data(articles)
    corpus, dictionary = create_corpus_dictionary(cleaned_articles)
    lda_model = train_lda(corpus, dictionary, num_topics, passes)
    topic_words = lda_model.print_topics(num_words=num_words)
    
    with open(output_file, "w") as f:
        for topic in range(num_topics):
            f.write(f"Topic {topic}:\n")
            f.write(str(topic_words[topic]) + "\n\n")


folder_path = r"C:/Users/jasin/OneDrive/Documents/Text Mining/Mini_Project_3/Articles/Articles"

# Define different parameters to build the LDA model
params = [{"num_topics": 5, "num_words": 10, "passes": 15},{"num_topics": 6, "num_words": 12, "passes": 20},{"num_topics": 7, "num_words": 8, "passes": 10},{"num_topics": 10, "num_words": 10, "passes": 15}]

# Iterating through the parameters and building different LDA models
for i, p in enumerate(params):
    output_file = f"output_{i + 1}.txt"
    run_and_save_lda(folder_path,  p["num_topics"], p["num_words"], p["passes"], output_file)
    

