from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# Download NLTK tokenizer data
nltk.download('punkt')

# Initializing the stemmer
stemmer = PorterStemmer()

# Defining a custom tokenizer function that stems the words
def stemmed_words(text):
    tokens = word_tokenize(text.lower())  
    return [stemmer.stem(word) for word in tokens]  

# Creating a stemmed version of the stop words list
stemmed_stop_words = [stemmer.stem(word) for word in ENGLISH_STOP_WORDS]

# Loading dataset
df = pd.read_csv('assignment3_II.csv')
df.columns = df.columns.str.strip()  # Strip column names

X = df['Clothes Description']
y = df['Recommended IND']

# Spliting them into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF with stemming and stemmed stop words
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=stemmed_words,         
    stop_words=stemmed_stop_words,    
    max_features=5000                 
)

# Transforming the text data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Saving the trained model and vectorizer
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

app = Flask(__name__)

# Loading the trained logistic regression model and TF-IDF vectorizer
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

def get_similar_items(item_id, top_n=5):
    # Loading the dataset again to fetch all items
    df = pd.read_csv('assignment3_II.csv')
    df.columns = df.columns.str.strip()

    # Filtering the items with positive recommendations
    positive_items = df[df['Recommended IND'] == 1]

    # Getting the description of current item
    item_description = df[df['Clothing ID'] == item_id]['Clothes Description'].values[0]

    # Vectorizing the description of positive recommended items 
    positive_descriptions_tfidf = tfidf_vectorizer.transform(positive_items['Clothes Description'].tolist())

    # Vectorizing the description of current item
    current_item_tfidf = tfidf_vectorizer.transform([item_description])

    # Calculating cosine similarity between current item and all positive recommended items
    similarities = np.dot(positive_descriptions_tfidf, current_item_tfidf.T).toarray()

    # Getting the indices of the most similar items, excluding the current item
    sorted_indices = np.argsort(similarities, axis=0).flatten()[::-1]

    # Fetching top N similar items
    similar_items = []
    for idx in sorted_indices[:top_n]:
        similar_item_id = positive_items.iloc[idx]['Clothing ID']
        similar_item_title = positive_items.iloc[idx]['Clothes Title']
        similar_item_description = positive_items.iloc[idx]['Clothes Description']
        
        # Avoiding recommending the same item
        if similar_item_id != item_id:
            similar_items.append((similar_item_id, similar_item_title, similar_item_description))

    return similar_items


# Defining home route
@app.route('/')
def home():
    return render_template('index.html')

# Defining route to search for items
@app.route('/search', methods=['POST'])
def search():
    keyword = request.form['keyword']
    if not isinstance(keyword, str) or not keyword:
        return render_template('search_results.html', items=[], keyword=keyword, matches=0)

    # Loading item descriptions, titles, and ID from dataset
    df = pd.read_csv('assignment3_II.csv')
    df.columns = df.columns.str.strip()
    descriptions = df['Clothes Description'].tolist()
    titles = df['Clothes Title'].tolist()
    item_ids = df['Clothing ID'].tolist()

    # Processing the search using TF-IDF vectorizer
    descriptions_tfidf = tfidf_vectorizer.transform(descriptions)

    # Transforming keyword to TF-IDF
    keyword_tfidf = tfidf_vectorizer.transform([keyword])
    similarities = np.dot(descriptions_tfidf, keyword_tfidf.T).toarray()

    # Sorting results based on similarity scores
    sorted_indices = np.argsort(similarities, axis=0).flatten()[::-1]

    # Creating a list of tuples for matched item in order to ensure uniqueness
    matched_items = []
    seen_titles = set()

    for i in sorted_indices:
        title = titles[i]
        if title not in seen_titles:
            seen_titles.add(title)
            matched_items.append((item_ids[i], titles[i], descriptions[i]))

        if len(matched_items) >= 10:  # I am limiting results to top 10 unique items
            break

    # Return the number of matched items and previews
    return render_template('search_results.html', items=matched_items, keyword=keyword, matches=len(matched_items))

@app.route('/item/<int:item_id>')
def view_item(item_id):
    # Loading the dataset again to fetch item details
    df = pd.read_csv('assignment3_II.csv')
    df.columns = df.columns.str.strip()

    # Fetching item details for given item_id
    if 'Clothing ID' not in df.columns:
        return f"Error: 'Clothing ID' column not found in assignment3_II.csv.", 500

    try:
        item = df[df['Clothing ID'] == item_id].iloc[0]
    except IndexError:
        return f"Error: No item found with Clothing ID {item_id}", 404

    # Original reviews from dataset
    original_reviews = item['Review Text'].split('\n')

    # Load submitted reviews from the file
    try:
        submitted_reviews_df = pd.read_csv('submitted_reviews.csv')
        submitted_reviews_df.columns = submitted_reviews_df.columns.str.strip()
        submitted_reviews = submitted_reviews_df[submitted_reviews_df['Clothing ID'] == item_id]['Review Text'].tolist()
    except FileNotFoundError:
        submitted_reviews = []
    except KeyError as e:
        return f"Error: {e}. Please ensure 'Clothing ID' exists in submitted_reviews.csv.", 500

    # Combining original and submitted reviews
    all_reviews = original_reviews + submitted_reviews

    item_details = {
        'name': item['Clothes Title'],
        'description': item['Clothes Description'],
        'rating': item['Rating'],
        'recommended': item['Recommended IND'],
        'reviews': all_reviews
    }

    # Getting similar items that were positively recommended by other customers
    similar_items = get_similar_items(item_id, top_n=5)

    # Rendering item details page with similar items list
    return render_template('item_details.html', item=item_details, item_id=item_id, similar_items=similar_items)


@app.route('/submit_review_page/<string:item_title>', methods=['GET'])
def submit_review_page(item_title):
    return render_template('submit_review.html', item_title=item_title)

@app.route('/submit_review', methods=['POST'])
def submit_review():
    review_text = request.form.get('review_text', '').strip()
    item_title = request.form.get('item_title', '').strip()

    if not review_text:
        return "Error: Review text cannot be empty.", 400
    if not item_title:
        return "Error: Invalid item title.", 400

    df = pd.read_csv('assignment3_II.csv')
    df.columns = df.columns.str.strip()

    # Fetching Clothing ID based on Clothes Title
    matching_item = df[df['Clothes Title'] == item_title]
    if matching_item.empty:
        return f"Error: No item found with title '{item_title}'", 404
    
    item_id = matching_item['Clothing ID'].values[0]  

    # Transforming the review text
    review_tfidf = tfidf_vectorizer.transform([review_text])

    # Predicting recommendation (0 or 1)
    recommendation = lr_model.predict(review_tfidf)[0]

    # Rendering confirmation page with LR model's recommendation
    return render_template('confirm_review.html', item_title=item_title, review_text=review_text, recommendation=recommendation)

@app.route('/finalize_review', methods=['POST'])
def finalize_review():
    final_recommendation = int(request.form['final_recommendation'])
    review_text = request.form['review_text']
    item_title = request.form['item_title']

    df = pd.read_csv('assignment3_II.csv')
    df.columns = df.columns.str.strip()

    matching_item = df[df['Clothes Title'] == item_title]
    if matching_item.empty:
        return f"Error: No item found with title '{item_title}'", 404
    
    item_id = matching_item['Clothing ID'].values[0]  

    new_review = pd.DataFrame({
        'Clothing ID': [item_id],
        'Review Text': [review_text],
        'Final Recommendation': [final_recommendation]
    })

    # Appending new review to existing file or creating one if it doesn't exist
    try:
        reviews_df = pd.read_csv('submitted_reviews.csv')
        reviews_df.columns = reviews_df.columns.str.strip()

        if 'Clothing ID' not in reviews_df.columns:
            return "Error: 'Clothing ID' column missing in submitted_reviews.csv.", 500

        reviews_df = pd.concat([reviews_df, new_review], ignore_index=True)
    except FileNotFoundError:
        reviews_df = new_review
    except Exception as e:
        return f"An error occurred while saving the review: {e}", 500

    reviews_df.to_csv('submitted_reviews.csv', index=False)

    return view_item(item_id)


if __name__ == '__main__':
    app.run(debug=True)
