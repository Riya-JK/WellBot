import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import re
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from string import punctuation
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def start_chatting():
    nRowsRead = 1000 # specify 'None' if want to read whole file
    path_to_csv = 'MentalHealth_Dataset/mentalhealth.csv'
    data = pd.read_csv(path_to_csv, delimiter=',', nrows = nRowsRead)
    data.dataframeName = 'Mental_Health_FAQ.csv'

    # Preprocess the data
    data['Questions'] = data['Questions'].str.lower()

    # Assuming data is your DataFrame
    data['Questions'] = data['Questions'].str.replace('[^a-zA-Z]', '')

    # Handle missing values if any
    data.dropna(inplace=True)

    data['Intent'] = data['Questions']

    # Intent Prediction Model
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['Questions'], data['Intent'], test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the model
    model = LinearSVC()
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred)

    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['Questions']))
    # plt.figure(figsize=(10, 5))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis('off')
    # plt.show()

    print("Chatbot: Do you have anymore questions?")
    answer = input("User: ")
    if("yes" not in answer.lower()):
        print("Chatbot: Thank you for using our services!")
        return
    
    print("\nChatbot: Ask a question or enter 'quit' to exit.")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye! Thanks for chatting!")
            break
        
        # Vectorize user input
        user_input_vec = vectorizer.transform([user_input.lower()])
        
        # Predict the intent
        predicted_intent = model.predict(user_input_vec)[0]
        
        # Implement response generation mechanism based on predicted intent
        response = data[data['Questions'] == predicted_intent]['Answers'].values[0] if predicted_intent in data['Questions'].values else "I'm sorry, I don't have a response for that question."
        sentences = re.split(r'(?<=[.!?]) +', response)
        if len(sentences) > 5:
            limited_response = ' '.join(sentences[:5])
        else:
            limited_response = ' '.join(sentences)
        print("Chatbot: "+limited_response)