import pickle
import streamlit as st

# Load the trained pipeline/model
model = pickle.load(open("pipe.pkl", "rb"))

# App title
st.title("Twitter Sentiment Assessment ML")

# Text input from user
user_input = st.text_input("Enter a tweet for sentiment analysis:")

# When user enters text, make prediction
if user_input:
    prediction = model.predict([user_input])[0]
    st.write("### Sentiment Prediction:")
    if prediction == 1 or prediction == "Positive":
        st.success("Positive 😀")
    else:
        st.error("Negative 😟")