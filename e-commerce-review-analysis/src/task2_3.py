#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Bharath Narayanan Venkatesh
# #### Student ID: s4033348
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * gensim.downloader
# * re
# * numpy
# * sklearn
# 
# ## Introduction
# For task 2, we are going to take that pre-processed csv and vocab.txt and generate different feature representations for clothing item reviews. This will help in enhancing text analysis and at the same time help in performing Machine Learning analysis.This is a very important part in NLP process.We will convert raw data to numerical data which can be used in ML tasks. We are performing 3 steps, Count vector which is bag of words, unweighted embeddings and Tfidf weighted embeddings.
# 
# For task3, we are going to create a logistic regression based classification model to classify dress reviews based on their recommendation.We will combine the models we found in Task2 with logistic regression model and see which one performs best. Also logistic regression model uses different input combinations like title only, description only and combination of title and description to find whether adding more information improves the performance of the model.

# ## Importing libraries 

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ## Task 2. Generating Feature Representations for Clothing Items Reviews
# 1. I am first loading vocab.txt into a dictionary and processed.csv files to get Processed_Review column.
# 2. Preparing Count Vector: We are creating a empty list to store the bag of words. The code then iterates through each review and a dictionary is used to count the occurrences of each word.This is then formatted as word_index:frequency.
# 3. Pre-trained fasttext word embedding model is loaded.
# 4. I am using this fasttext to calculate unweighted embeddings in Processed_Review column. The function get_unweighted_embedding calculates the mean word vectors from Processed_Review column.
# 5. I am initializing TfidfVectorizer using vocab.txt to calculate Term Frequency and Inverse Document Frequency for Processed_Review column. I am defining a get_weighted_embedding function to calculate weighted_embedding for all review.

# In[2]:


# Loading vocabulary
vocab = {}
with open('vocab.txt', 'r') as f:
    for line in f:
        word, index = line.strip().split(':')
        vocab[word] = int(index)

# Loading processed reviews from processed.csv file
df = pd.read_csv('processed.csv')  

# Handling missing values
df['Processed_Review'] = df['Processed_Review'].astype(str).fillna('')

# Preparing count vector(bag of words)
sparse_count_vectors = []  
for idx, review in enumerate(df['Processed_Review']):
    if isinstance(review, str) and review.strip():
        words = review.split()
        word_freq = defaultdict(int)
        for word in words:
            if word in vocab:
                word_freq[vocab[word]] += 1

# Formatting the count vector in word_index:frequency
        sparse_rep = ','.join([f'{word_idx}:{freq}' for word_idx, freq in sorted(word_freq.items())])
        if sparse_rep:  
            sparse_count_vectors.append(f"#{idx},{sparse_rep}\n")


# In[3]:


embedding_model = api.load("fasttext-wiki-news-subwords-300")


# In[4]:


# Defining a function to calculate unweighted FastText embeddings
def get_unweighted_embedding(review, model, embedding_dim=300):
    words = review.split()  
    word_vectors = [model[word] for word in words if word in model]  
    
    # Calculate the mean embedding if words are found
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(embedding_dim)

# Generating unweighted embeddings for each review using Processed_Review column
unweighted_embeddings = np.array([get_unweighted_embedding(review, embedding_model) for review in df['Processed_Review']])


# In[5]:


# Loading vocabulary from vocab.txt
vocab_dict = vocab  

# Extracting reviews from DataFrame
reviews = df['Processed_Review']

# Initializing TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab_dict)
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

# Getting feature names and their TF-IDF scores
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()

# Function to calculate TF-IDF weighted embeddings
def get_weighted_embedding(review, model, tfidf_scores, feature_names, embedding_dim=300):
    words = review.split()  
    word_vectors = []
    weights = []
    
    for word in words:
        if word in model and word in feature_names:
            word_idx = np.where(feature_names == word)[0][0]
            word_vector = model[word]
            tfidf_weight = tfidf_scores[word_idx]
            word_vectors.append(word_vector)
            weights.append(tfidf_weight)
    
    if len(word_vectors) > 0:
        weighted_embedding = np.average(word_vectors, axis=0, weights=weights)
        return weighted_embedding
    else:
        return np.zeros(embedding_dim)

# Generating TF-IDF weighted embeddings for each review
weighted_embeddings = np.array([get_weighted_embedding(review, embedding_model, tfidf_scores[i], feature_names, embedding_dim=300) for i, review in enumerate(reviews)])


# ### Saving outputs
# I am saving these nparray based unweighted_embeddings and weighted_embeddings to a text named unweighted_embeddings.txt and tfidf_weighted_embeddings.txt and finally creating a file named count_vectors.txt to write all the sparse_count_vectors list to that file.

# In[6]:


# Saving unweighted embeddings
np.savetxt('unweighted_embeddings.txt', unweighted_embeddings, delimiter=',')

# Saving weighted embeddings
np.savetxt('tfidf_weighted_embeddings.txt', weighted_embeddings, delimiter=',')

# Saving sparse_count_vectors to count_vectors.txt
with open('count_vectors.txt', 'w') as outfile:
    outfile.writelines(sparse_count_vectors)


# ## Task 3. Clothing Review Classification

# 1. We are extracting Title and Processed_Review columns from df and target variable is Recommended IND and concatinating title and description into a single feature named X_combined.
# 2. I am creating a Logistic Regression Model with maximum iteration of 1000. 5 fold Cross validation is performed to evaluate the performance of the model using count vector and target variable.We are calculating the accuracy of the model. The model correctly predicted 88.10% of the time across different folds. 
# 3. I am loading the weighted_embeddings np array. The model is uses TF-IDF weighted embeddings and I am performing the same 5 fold Cross validation to compute the accuracy of the model based on this TF-IDF weighted embeddings data. The model correctly predicted 82.91% from the 5 fold cv.
# 4. I am loading the unweighted_embeddings np array. Based on this data, the model predicted a accuracy percentage of 82.80%.
# 5. I am using tfidf_vectorizer to generate TF-IDF features for title of clothing review. We are using the same Logistic Regression Model and again performing 5 fold Cross validation on the data to get the accuracy of the model. This yields a percentage of 88.29%. 
# 6. I am generating TF-IDF features for processed review description. The Logistic Regression Model gives 88.25% of accuracy.
# 7. I am creating a new column in df named combined_text to concatinate Title and Processed_Review Description. The Logistic Regression Model for this data gives a accuracy of 89.84%.

# In[7]:


X_title = df['Title'].fillna('')  
X_description = df['Processed_Review'].fillna('')  
y = df['Recommended IND']  

# Combining title and description
X_combined = X_title + " " + X_description 


# In[8]:


with open('vocab.txt', 'r') as f:
    vocab = {line.split(':')[0]: int(line.split(':')[1]) for line in f}

# Initializing CountVectorizer
count_vectorizer = CountVectorizer(vocabulary=vocab)
X_count = count_vectorizer.fit_transform(X_description)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)

# 5-fold cross-validation
kf = KFold(n_splits=5)
cv_scores = cross_val_score(log_reg, X_count, y, cv=kf, scoring='accuracy')

print(f"Count Vector (Processed_Review) - 5-Fold CV Accuracy: {cv_scores.mean():.4f}")


# In[9]:


X_tfidf_weighted = weighted_embeddings

# 5-fold cross-validation
cv_scores = cross_val_score(log_reg, X_tfidf_weighted, y, cv=kf, scoring='accuracy')

# Print results
print(f"TF-IDF Weighted Embedding (Processed_Review) - 5-Fold CV Accuracy: {cv_scores.mean():.4f}")


# In[10]:


X_unweighted = unweighted_embeddings

cv_scores = cross_val_score(log_reg, X_unweighted, y, cv=kf, scoring='accuracy')

print(f"Unweighted Embedding (Processed_Review) - 5-Fold CV Accuracy: {cv_scores.mean():.4f}")


# In[11]:


# Generating TF-IDF features for title
tfidf_vectorizer = TfidfVectorizer()
X_title_tfidf = tfidf_vectorizer.fit_transform(X_title)

# 5-fold cross-validation
cv_scores = cross_val_score(log_reg, X_title_tfidf, y, cv=kf, scoring='accuracy')
print(f"Title Only - 5-Fold CV Accuracy: {cv_scores.mean():.4f}")


# In[12]:


# Generating TF-IDF features for Processed_Review (description)
tfidf_vectorizer = TfidfVectorizer()
X_description_tfidf = tfidf_vectorizer.fit_transform(X_description)

cv_scores = cross_val_score(log_reg, X_description_tfidf, y, cv=kf, scoring='accuracy')
print(f"Processed_Review (Description Only) - 5-Fold CV Accuracy: {cv_scores.mean():.4f}")


# In[13]:


# Concatenating Title and Processed_Review
df['combined_text'] = df['Title'].fillna('') + ' ' + df['Processed_Review'].fillna('')

# Generating TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_combined_tfidf = tfidf_vectorizer.fit_transform(df['combined_text'])

cv_scores_combined = cross_val_score(log_reg, X_combined_tfidf, y, cv=kf, scoring='accuracy')
print(f"Combined Title and Processed_Review - 5-Fold CV Accuracy: {cv_scores_combined.mean():.4f}")


# ## Summary
# To answer to Question 1 in the assignment specification, Count Vector outshines the remaining weighted and unweighted word embeddings. The model's accuracy percentage is 88.10 which is high when compared to other 2 models percentage. This shows that Count Vector model correctly identifies the main features of clothing review than other 2 models.
# 
# For Question 2 in the assignment specification, the concatination of Title and Description has higher accuracy percentage of 89.84% when compared to their separate percentage accuracy. So, the additional inputs from both the features improves models performance compared to using each feature lonely.
# 
# For task 2, we have generated Feature Representations for Clothing Items Reviews. We have used count vector model and fasttext word embedding based weighted and unweighted TF-IDF model. They were evaluated using Logistic Regression Model with Bag of words model coming out with highest accuracy.
# 
# For task3, we have deployed Logistic Regression Model on Title, Description and combination of Title and Description and the accuracy of combination of Title and description came out to be higher when compared to their individual model accuracy percentage.

# ## Reference 
# 1. https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
# 2. https://stackoverflow.com/questions/42002859/creating-a-tf-idf-matrix-python-3-6
# 3. Scikit-learn. (n.d.). sklearn.feature_extraction.text.TfidfVectorizer. Retrieved September 27, 2024, from https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 
