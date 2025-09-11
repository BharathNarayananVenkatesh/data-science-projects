#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Bharath Narayanan Venkatesh
# #### Student ID: s4033348
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * numpy
# * os
# * warnings
# 
# ## Introduction
# Online e-commerce platforms are highly developing and in order to improve their recommendation process, many companies are hiring data analysts to work on their data which comprises of customer reviews and recommendations.We can build a classification model to analyse the customer reviews. For that we have to first perform the essential text pre-processing on Review Text column present in the dataset. Text Pre Processing is one of the basic steps in NLP. They are used to remove noise and irrelevant information from the target column. This Review Text column contains customer reviews of different clothes. This column provides good information about the customers mindset and their feelings about the product. Since the content present in that column is not structured, I am transforiming this raw text into a more structured format for further analysis using various tasks like Tokenization, lowercasing, removing short words and stop words, calculate term frequency and document frequency, removing words that appear only once, and finally saving them. By doing this, we are giving a new dimension to the column but at the same time we are keeping the content as it is. This makes it easier to perform Machine Learning analysis too.

# ## Importing libraries 

# In[1]:


import re
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import os


# ### 1.1 Examining and loading data
# I am defining the filepath in the first step. In the second step, I am trying to get the information about the file size before loading the file.Last step is the one which provides information about the structure of the file by printing out few lines. Then I am loading the file and then to know the structure, I am using .info and head to print out few lines. Clothing ID, Age, Rating, Recommended IND, Positive Feedback Count are all int type. Title, Review Text, Division Name, Department Name and Class Name are all string.

# In[2]:


# Path to the CSV file
file_path = 'assignment3.csv'

# Getting information about file size
file_size = os.path.getsize(file_path)
print(f"File size: {file_size / (1024 * 1024):.2f} MB")  # Convert bytes to MB

# Reading the first few lines to know the structure of file
with open(file_path, 'r') as file:
    for i in range(5):
        print(file.readline())


# In[3]:


# Loading the data
df = pd.read_csv(file_path)

# Structure details
print(df.info())  
print(df.head())  


# ### 1.2 Pre-processing data
# Since we know the data type and structure of the file, we move on to the next step which is Pre-processing. The steps include
# 1. Loading stopwords_en.txt, this contains large number of stopwords, which we usually remove during pre-processing and then printing the first 10 lines of that file.
# 2. Tokenization process is the next step. I am using the same regular expression as mentioned in the Assignment specification.I am defining a function preprocessing_text to perform multiple tasks like Tokenization and lowercasing the review text column, removing short words that are less than 2 characters, removing stop words and finally storing them in a new column named Processed_Review.
# 3. We are removing the words in Processed_Review based on term frequency and document frequency. I am removing the words that appear only 1 time and also top 20 most frequent words. To view the words I am converting filtered_vocab to a list named filtered_vocab_list.

# In[4]:


# Path to stopwords file
stopwords = 'stopwords_en.txt'

# Read and print first 10 lines of stopwords
with open(stopwords, 'r') as file:
    for i in range(10):
        print(file.readline().strip())


# In[5]:


# Tokenization regular expression
token_pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"

def preprocessing_text(review):
    #Tokenization and lowercasing
    tokens = re.findall(token_pattern, review.lower())
    
    # Removing the short words that are less than 2 characters
    tokens = [token for token in tokens if len(token) > 2]
    
    # Removing stopwords
    tokens = [token for token in tokens if token not in stopwords]
    
    return tokens

# Storing processed tokens in new column
df['Processed_Review'] = df['Review Text'].fillna('').apply(preprocessing_text)


# In[6]:


# Initializing counters for term frequency and document frequency
term_freq = Counter()
doc_freq = Counter()

# Looping through  Processed review column(newly created column) to calculate term and document frequencies
for tokens in df['Processed_Review']:
    term_freq.update(tokens)
    doc_freq.update(set(tokens))
    
#Removing words that appear only once and also I am removing top 20 most frequent words
top_20_words = {word for word, _ in doc_freq.most_common(20)}
filtered_vocab = {word for word, freq in term_freq.items() if freq > 1 and word not in top_20_words}

# Converting filtered_vocab to list to get first 100 words
filtered_vocab_list = list(filtered_vocab)[:100]
print(filtered_vocab_list)


# ## Saving required outputs
# I am rejoining the tokens that are present in the filtered_vocab and then saving them to a csv named processed.csv. I am sorting the filtered_vocab first and then I am creating a file named vocab.txt to save these words.

# In[7]:


# Rejoining tokens that are in the filtered vocabulary for each review
df['Processed_Review'] = df['Processed_Review'].apply(lambda tokens: ' '.join([token for token in tokens if token in filtered_vocab]))

# Save to a CSV file
df.to_csv('processed.csv', index=False)

# Building the vocabulary by sorting first and then save it to vocab.txt
vocab = sorted(filtered_vocab)
with open('vocab.txt', 'w') as f:
    for idx, word in enumerate(vocab):
        f.write(f"{word}:{idx}\n")


# ## Summary
# We have performed the basic Pre-Processing of the Review Text column which includes Tokenization, Lowercasing, removal of short and stop words and removed words that appear only once as well as the top 20 most frequent words. We have saved the work to a new csv named processed.csv.
