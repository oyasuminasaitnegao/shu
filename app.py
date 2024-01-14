import streamlit as st

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch import nn
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
saved_model_path = "RTa_5"
model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)

def predict(text):
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    outputs = model(**inputs)
    ps = nn.Softmax(1)(outputs.logits)

    return ps

pred = predict('おいしそうだね～').detach().numpy()

def calculate_fig_prob(pred):
    fig = np.argsort(pred[0])[::-1]
    prob = np.sort(pred[0])[::-1]
    prob = prob/np.sum(prob)

    return fig, prob
    
fig, prob= calculate_fig_prob(pred)
ans = str(np.random.choice(a=fig,p=prob))
st.title(ans)