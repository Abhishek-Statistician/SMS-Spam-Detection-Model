import streamlit as st
import pickle

# Load the model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title("Machine Learning Prediction App")
st.write("*By Abhishek Chakraborty*")

# Input field for user text
user_input = st.text_input("Enter text for prediction:")

if user_input:
    # Preprocess input and make prediction
    user_vectorized = vectorizer.transform([user_input])  # Vectorize input
    prediction = model.predict(user_vectorized)          # Make prediction

    
    # Display result
    if(prediction == 'spam'):
        st.write("Prediction: Spam")
    else:
        st.write("Prediction: Not Spam")
