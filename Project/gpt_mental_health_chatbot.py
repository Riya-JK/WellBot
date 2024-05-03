import pandas as pd
import numpy as np
import warnings
import torch
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
warnings.filterwarnings("ignore")

def get_gpt_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def start_chatting():
    nRowsRead = 1000 # specify 'None' if want to read whole file
    path_to_csv = 'MentalHealth_Dataset/mentalhealth.csv'
    data = pd.read_csv(path_to_csv, delimiter=',', nrows=nRowsRead)
    data.dataframeName = 'Mental_Health_FAQ.csv'

    # Preprocess the data
    data['Questions'] = data['Questions'].str.lower()
    data['Questions'] = data['Questions'].str.replace('[^a-zA-Z]', '')
    data.dropna(inplace=True)

    # Load GPT-2 model and tokenizer
    model, tokenizer = get_gpt_model()

    print("Chatbot: Do you have any more questions?")
    answer = input("User: ")
    if "yes" not in answer.lower():
        print("Chatbot: Thank you for using our services!")
        return

    print("\nChatbot: Ask a question or enter 'quit' to exit.")

    while True:
        user_input = input("User: ")

        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye! Thanks for chatting!")
            break

        # Generate response using GPT-2
        input_ids = tokenizer.encode(user_input, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)  # Set attention mask to 1 for all tokens
        output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove user input from response
        response_without_input = response[len(user_input):].strip()
        # Remove special characters and white spaces
        clean_response = re.sub(r'[^a-zA-Z0-9\s]', '', response_without_input)
        print("Chatbot:", clean_response)

start_chatting()
