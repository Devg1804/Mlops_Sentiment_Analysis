import streamlit as st
import numpy as np
import joblib
import os

# Load the vectorizer and model
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("saved_models/NBayes.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

# Predict sentiment
def predict_sentiment(review_text, model, vectorizer):
    # Convert text to vector
    vectorized_text = vectorizer.transform([review_text])
    
    # Predict sentiment
    prediction = model.predict(vectorized_text)
    return prediction[0]

# Streamlit UI
def main():
    st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
    st.title("📝 Customer Review Sentiment Analyzer")

    st.markdown("""
        Enter a customer review below, and we'll predict if it's **Positive** or **Negative**!
    """)

    review_text = st.text_area("Enter your review here:", height=150, placeholder="Type something like 'The product quality is amazing!'")

    if st.button("Predict"):
        if review_text.strip() == "":
            st.warning("⚠️ Please enter a review before clicking Predict.")
        else:
            try:
                model, vectorizer = load_model_and_vectorizer()
                vectorized_text = vectorizer.transform([review_text])
                sentiment = predict_sentiment(review_text, model, vectorizer)
                probas = model.predict_proba(vectorized_text)[0]
                confidence = round(np.max(probas) * 100, 2)
                
                result = "😊 Positive" if sentiment == 1 else "😞 Negative"
                st.success(f"**Sentiment:** {result}")
                st.info(f"**Confidence:** {confidence}%")
            except Exception as e:
                st.error(f"Something went wrong during prediction: {str(e)}")

if __name__ == "__main__":
    main()

