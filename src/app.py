import streamlit as st
from keras.models import Sequential
import keras.models as mod
from config import *
import src.importer as importer
import pickle

st.write("Hello world")
# model = mod.load_model(LEMMATIZER_MODEL)
available_docs = importer.load_data()
available_docs = [doc for doc in available_docs if not doc.lemmatized]
all_docs_dict = {f"{doc.name}-{doc.ms}": doc for doc in available_docs}
doc_select = st.selectbox("Choose a document", list(all_docs_dict.keys()))
current_doc = all_docs_dict[doc_select]

unique_forms_lookup: dict[str, str] = pickle.load(open(UNIQUE_DICT_PICKLE, "rb"))

fresh_tokens = []
for token in current_doc.tokens:
    lemma = unique_forms_lookup.get(token.normalized, "N/A")
    token.lemma = str(lemma).replace("{", "").replace("}", "").replace("'", "")
    fresh_tokens.append(token)

from_index = 0
to_index = 20

st.write(" ".join([tok.normalized for tok in fresh_tokens[from_index:to_index]]))
try:
    st.write(" ".join([tok.lemma for tok in fresh_tokens[from_index:to_index]]))
except:
    for tok in fresh_tokens[from_index:to_index]:
        st.write(tok.lemma)

if st.button("Next"):
    from_index += 20
    to_index += 20