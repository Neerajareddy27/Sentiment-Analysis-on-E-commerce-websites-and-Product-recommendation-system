from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)

# Sample dataset containing brand mentions and sentiments
data = pd.read_csv('brand_sentiment_data.csv')  # Replace with your dataset

# Preprocess the data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['sentiment']

# Train a sentiment analysis model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sentiment_model = RandomForestClassifier()
sentiment_model.fit(X_train, y_train)

# Function to predict sentiment
def predict_sentiment(text):
    text_vectorized = tfidf_vectorizer.transform([text])
    sentiment = sentiment_model.predict(text_vectorized)
    return sentiment[0]

# Function to recommend brands based on sentiment
def recommend_brands(sentiment):
    if sentiment == 'positive':
        # Recommend positive brands
        return ['Brand A', 'Brand B']
    elif sentiment == 'negative':
        # Recommend negative brands
        return ['Brand X', 'Brand Y']
    else:
        # Neutral sentiment, no specific recommendation
        return ['No specific brand recommendation']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    brands = recommend_brands(sentiment)
    return render_template('index.html', sentiment=sentiment, recommended_brands=brands)

if __name__ == '__main__':
    app.run()
