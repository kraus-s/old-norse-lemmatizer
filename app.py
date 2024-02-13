import streamlit as st
from keras.models import Sequential
import keras.models as mod
from src.config import *
import src.importer as importer

st.write("Hello world")
# model = mod.load_model(LEMMATIZER_MODEL)
available_docs = importer.load_data()
all_docs_dict = {f"{doc.name}-{doc.ms}": doc for doc in available_docs}
doc_select = st.selectbox("Choose a document", list(all_docs_dict.keys()))
current_doc = all_docs_dict[doc_select]

