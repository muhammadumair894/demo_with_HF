import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd

st.title('Question Answering')
st.write('Enter a question and a context.')
question = st.text_input('Question')
context = st.text_input('Context')
if st.button('Submit'):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    st.write(res['answer'])
