import streamlit as st
import time
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

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


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page Layout
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main Content
st.header("Email/SMS Spam Classifier")

# Pop-up window for instructions
if st.session_state.first_load:
    st.session_state.first_load = False
    st.sidebar.write("For better results, please paste the complete SMS/EMAIL.")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Check if input_sms is empty
    if input_sms.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Show loading spinner
        with st.spinner('Predicting...'):
            time.sleep(3)  # Simulate prediction time
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")

# Footer
st.markdown(
    """
    <hr>
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <div id="footer" style="text-align: center; font-size: 0.9rem; color: white;">
            <p>Build by Kunal Bandale ⚡</p>
            <p>Follow: 
                <a href="https://github.com/kunalbandale" style="color: white;"><i class="fab fa-github"></i></a>
                <a href="https://www.linkedin.com/in/kunalbandale" style="color: white;"><i class="fab fa-linkedin"></i></a>
                <a href="https://www.kunalbandale.in" style="color: white;"><i class="fas fa-globe"></i></a>  
            </p>
        </div>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    </div>
    """,
    unsafe_allow_html=True
)

# CSS for mobile responsiveness
st.markdown(
    """
    <style>
        /* Mobile responsiveness */
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

if 'first_load' not in st.session_state:
    st.session_state.first_load = True
