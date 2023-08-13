import streamlit as st
import numpy as np
import pandas as pd

import requests

API_KEY = st.secrets["API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3"
headers = {"Authorization": f"Bearer {API_KEY}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


with st.form(key='my_form'):
  labels = st.multiselect('Choose your labels', ["cold", "hot"])
  input_sentence = st.text_area('Text to analyze semantics')
  
  payload = {
      "inputs": input_sentence,
      "parameters": {"candidate_labels": labels},     
    }
  
  submit_button = st.form_submit_button('Classify')

  if submit_button:
    output = query(payload)
    st.write(output)







  








