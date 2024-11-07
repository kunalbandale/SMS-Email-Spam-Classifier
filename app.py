import streamlit as st
import time
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer (make sure these files exist in the same directory)
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Model or vectorizer file not found: {e}")
    st.stop()

# Set up Streamlit page configuration at the start
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main content
st.header("Email/SMS Spam Classifier")
st.write("For better results, please paste the complete SMS/EMAIL.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Show loading spinner
        with st.spinner('Predicting...'):
            time.sleep(3)  # Simulate prediction time
            # Preprocess input text
            transformed_sms = transform_text(input_sms)
            # Vectorize input text
            vector_input = tfidf.transform([transformed_sms])
            # Predict using the model
            result = model.predict(vector_input)[0]
            # Display result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")

# Footer content
st.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <p>Build by Kunal Bandale ⚡</p>
        <p>
            <a href="https://github.com/kunalbandale" style="text-decoration: none;">GitHub</a> |
            <a href="https://www.linkedin.com/in/kunalbandale" style="text-decoration: none;">LinkedIn</a> |
            <a href="https://www.kunalbandale.in" style="text-decoration: none;">Website</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Mobile responsiveness CSS
st.markdown(
    """
    <style>
        /* Adjust text area height on mobile */
        @media (max-width: 600px) {
            .stTextArea textarea {
                min-height: 100px !important;
            }
            .css-vfskoc {
                flex-direction: column !important;
                align-items: center !important;
                justify-content: center !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)
