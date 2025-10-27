import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing function
stemmer = PorterStemmer()
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([stemmer.stem(word) for word in text if word not in stopwords.words('english')])

# Streamlit UI
st.title("📊 Tweet Sentiment Analyzer")
tweet = st.text_area("Enter a tweet")

if st.button("Analyze Sentiment"):
    if tweet:
        processed = preprocess(tweet)
        vector = vectorizer.transform([processed]).toarray()
        prediction = model.predict(vector)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a tweet!")