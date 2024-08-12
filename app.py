import streamlit as st
from transformers import pipeline

# Load the model from another account
model_repo_id = "dbmdz/bert-base-turkish-cased"  # Replace with the actual model ID
model_pipeline = pipeline("text-classification", model=model_repo_id)  # Replace "task-name" with the appropriate task

st.title("Model Demo with Hugging Face Spaces")
user_input = st.text_input("Enter your input here:")

if user_input:
    result = model_pipeline(user_input)
    st.write("Output:", result)