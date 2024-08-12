import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-turkish-text-classification")
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-turkish-text-classification")

def classify(text):
    cls= pipeline("text-classification",model=model, tokenizer=tokenizer)
    return cls(text)[0]['label']

st.title("Savas Y Test")

user_input = st.text_input("Enter your input here:")
submit_btn = st.button("Start")

if submit_btn:
    result = classify(text=user_input)
    st.write(result)





