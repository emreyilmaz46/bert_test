import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the saved model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("./my_bert_model")
    tokenizer = BertTokenizer.from_pretrained("./my_bert_model")
    return model, tokenizer

model, tokenizer = load_model()

# Define the category labels
categories = ["Content-related", "Technical", "Administrative"]

# Streamlit app
st.title("AI Course Question Classifier")

# Text input for the question
question = st.text_input("Enter your question about the AI course:")

# Button to trigger classification
if st.button("Classify Question"):
    if question:
        # Tokenize the input
        inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits).item()
        
        # Display the result
        st.write(f"Question category: **{categories[predicted_class]}**")
        
        # Display confidence scores
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for i, category in enumerate(categories):
            st.write(f"{category}: {probabilities[0][i].item():.2%}")
    else:
        st.write("Please enter a question.")

# Add some information about the categories
st.sidebar.header("Category Information")
st.sidebar.write("0 - Content-related: Questions about course topics and AI concepts")
st.sidebar.write("1 - Technical: Questions about programming, AI techniques, or tools")
st.sidebar.write("2 - Administrative: Questions about logistics, prerequisites, or assessments")