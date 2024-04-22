#Import necessary libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file to get a glimpse of its structure and content
file_path = 'Dataset/mood_detection_data.csv'
data = pd.read_csv(file_path, encoding="ISO-8859-1")
# Display the first few rows of the dataset
data.head()

import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from nltk.tokenize import word_tokenize
from tensorflow.keras.callbacks import Callback

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Display the first few rows and summary of the dataset
mood_data_head = data.head()
mood_data_info = data.info()
mood_data_head, mood_data_info

# Ensure that NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(data):
    # Convert text to lowercase
    data = data.lower()
    # Remove punctuation
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    # Tokenize text
    tokens = word_tokenize(data)
    # Remove stopwords and lemmatize
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(lemmatized)

# Apply preprocessing to the text column
data['processed_text'] = data['text'].apply(preprocess_text)

data[['text', 'processed_text']].head()

# Define a simplified preprocessing function that doesn't rely on NLTK downloads
def simple_preprocess_text(data):
    # Convert text to lowercase
    data = data.lower()
    # Remove non-alphanumeric characters
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
    # Tokenize text
    tokens = data.split()
    # Remove short words
    processed = [word for word in tokens if len(word) > 2]
    return ' '.join(processed)

# Apply simplified preprocessing to the text column
data['simple_processed_text'] = data['text'].apply(simple_preprocess_text)

data[['text', 'simple_processed_text']].head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Prepare the TF-IDF vectorization of the processed text
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for memory management
X = tfidf_vectorizer.fit_transform(data['simple_processed_text'])
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming 'data' is your DataFrame and it's already loaded with 'processed_text' and 'label'
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['processed_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy}")

def chatbot():
    print("Hello! I'm your Mental Health Assistant. How are you feeling today?")

    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye! Thanks for chatting.")
        return

    # Preprocess the input using the simplified preprocess function
    processed_input = simple_preprocess_text(user_input)

    # Vectorize the input
    vectorized_input = tfidf_vectorizer.transform([processed_input])

    # Predict the mood
    mood_prediction = model.predict(vectorized_input)

    # Define responses based on mood
    mood_responses = {
        0: "It's great to hear that you're feeling good! Keeping up with positive vibes is wonderful.",
        1: "It sounds like you might be having a tough time. It’s perfectly okay to feel this way."
    }

    # Suggest coping strategies based on mood
    coping_strategies = {
        0: "Continuing what you're doing is key, and maybe share your positivity with friends or family!",
        1: "Taking a moment to relax and breathe can really help. Here's a helpful link: [Mindfulness Resources](https://healthlibrary.stanford.edu/books-resources/mindfulness-meditation.html)"
    }

    # Generate a response based on the mood
    base_response = mood_responses[mood_prediction[0]]
    additional_help = coping_strategies[mood_prediction[0]]

    response = f"{base_response} {additional_help}"
    print("Chatbot:", response)

# Example usage
chatbot()

