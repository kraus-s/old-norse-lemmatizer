import src.importer as importer
import numpy as np
from gensim.models import Word2Vec
from src.menota_parser import NorseDoc, token
from collections import defaultdict
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import argmax
from sklearn.metrics import classification_report
from src.config import *
import pandas as pd
import pickle
import random



def create_word_index(words) -> dict[str, int]:
    word_count = defaultdict(lambda: 0)
    for word in words:
        word_count[word] += 1

    # Sort words by frequency (most to least)
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)

    # Create the word index (word to integer mapping)
    word_index = {word: i + 1 for i, word in enumerate(sorted_words)}  # start indexing from 1
    word_index["<PAD>"] = 0  # add a <PAD> token
    return word_index


def vectorize_doc_list(docs_list: list[list[str]], min_occurence: int = 1):
    # Split each sublist into smaller sublists of length 500
    print(f"Training word2vec on {sum([len(x) for x in docs_list])} tokens")
    # docs_list_windowed = [sublist[i:i+500] for sublist in docs_list for i in range(0, len(sublist), 500)]
    vectorized_model = Word2Vec(docs_list, min_count=min_occurence, window=5, workers=8, vector_size=VECTOR_SPACE, epochs=30)
    return vectorized_model


def _basic_preprocessor(parsed_docs_list: list[NorseDoc]) -> tuple[list[NorseDoc], list[NorseDoc]]:
    best_list = [x for x in parsed_docs_list if x.lemmatized and x.normalized]
    print(f"Found {len(best_list)} documents with lemmatized and normalized tokens")
    all_norms_list = [x for x in parsed_docs_list if x.normalized]
    print(f"Found {len(all_norms_list)} documents with normalized tokens")
    no_of_tokens_best = sum([len(x.tokens) for x in best_list])
    no_of_tokens_all = sum([len(x.tokens) for x in all_norms_list])
    print(f"Found {no_of_tokens_best} tokens in the best list and {no_of_tokens_all} tokens in the all norms list")
    return best_list, all_norms_list


def _lemma_based_stop_filter(input_list: list[NorseDoc]) -> list[list[token]]:
    # This function filters out tokens if their lemma is in the stop word list, except for "er"
    training_list: list[list[str]] = []
    for doc in input_list:
        doc_list: list[str] = []
        for tok in doc.tokens:
            if len(tok.lemma) > 1:
                if tok.lemma not in STOPWORDS:
                    doc_list.append(tok)
                else:
                    if tok.normalized == "er":
                        doc_list.append(tok)
        training_list.append(doc_list)
    return training_list


def _filter_normalized_w2v_vocab(wv_model: Word2Vec, input_list: list[NorseDoc]) -> list[list[token]]:
    """This function will make sure all tokens used for learning are represented in the w2v vocab; 
    this is the function for an embedding model trained on the normalized tokens""" 
    prefiltered_list_0 = [[tok for tok in doc.tokens if tok.lemma != "-"] for doc in input_list]
    prefiltered_list_1 = [[tok for tok in doc if tok.normalized in wv_model.wv] for doc in prefiltered_list_0]
    print(f"Got at total of {sum([len(x) for x in prefiltered_list_1])} tokens for training from the initial {sum([len(doc.tokens) for doc in input_list])} tokens")
    return prefiltered_list_1


def _filter_lemmatized_w2v_vocab(wv_model: Word2Vec, input_list: list[list[token]]) -> list[list[token]]:
    """This function will make sure all tokens used for learning are represented in the w2v vocab; 
    this is the function for an embedding model trained on the lemmatized tokens""" 
    prefiltered_list_0 = [[tok for tok in doc if tok.lemma != "-"] for doc in input_list]
    prefiltered_list_1 = [[tok for tok in doc if tok.lemma in wv_model.wv] for doc in prefiltered_list_0]
    print(f"Got at total of {sum([len(x) for x in prefiltered_list_1])} tokens for training from the initial {sum([len(doc) for doc in input_list])} tokens")
    return prefiltered_list_1


def _lemma_vector_to_normalized_embedding_matrix(tokens_list: list[list[token]], w2v_model: Word2Vec, word_index: dict[str, int]) -> np.ndarray:
    embedding_matrix = np.zeros((len(word_index), w2v_model.vector_size))

    # Populate the embedding matrix
    for token_list in tokens_list:
        for token in token_list:
            normalized_index = word_index.get(token.normalized)
            if normalized_index is not None:  # If the normalized word is in the word_index
                lemma_vector = w2v_model.wv[token.lemma] if token.lemma in w2v_model.wv else np.zeros(w2v_model.vector_size)
                embedding_matrix[normalized_index] = lemma_vector
    return embedding_matrix


def _normalized_embedding_matrix(pretrained_w2v_vectors: Word2Vec, word_index: dict[str, int]):
    embedding_matrix = np.zeros((len(word_index), pretrained_w2v_vectors.vector_size))
    for word, i in word_index.items():
        if word in pretrained_w2v_vectors.wv:
            embedding_matrix[i] = pretrained_w2v_vectors.wv[word]
    return embedding_matrix


def _preprocess_data_for_prediction(docs: list[NorseDoc], word_index: dict[str, int], ambiguous_forms: dict[str, set[str]], sampling: bool = False, stand_off: bool = True) -> tuple[np.ndarray, list[tuple[list[str], str]]]:
    """This function will take a list of NorseDoc objects and return a list of sequences ready for prediction using the model
    It returns a list of tuples, the first element of the tuple is the context and the second element is the target word.
    If stand_off is True, the function will filter the sequences based on ambiguous forms, only returnin samples with ambiguous forms and expecting uniqe forms to be lemmatized another way."""
    tokenized_list = [tok for doc in docs for tok in doc.tokens if tok.normalized in word_index.keys()]
    if stand_off:
        tokenized_list = [tok for tok in tokenized_list if tok.normalized in ambiguous_forms.keys()]
    sequences = importer._create_context_sequences_for_prediction(tokenized_list, window_size=CONTEXT_LENGTH)
    if sampling:
        sequences = random.sample(sequences, 20)
    sequences_seq = importer.text_to_sequence([seq[0] for seq in sequences], word_index)
    sequences_pad = pad_sequences(sequences_seq, maxlen=CONTEXT_LENGTH * 2 + 1)  
    return sequences_pad, sequences


def create_keras_model() -> Sequential:
    # Step 2: Create Keras Model
    model = Sequential()
    model.add(Embedding(len(word_index), lemma_based_vectors.vector_size, 
                        weights=[embedding_matrix], 
                        input_length=maxlen, 
                        trainable=False))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dense(256, activation='relu'))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(num_lemmas, activation='softmax'))  # num_lemmas is the number of unique lemmata

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()
    return model


def _build_unambigous_vocab(input_list: list[NorseDoc]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    form_to_lemmas: dict[str, set[str]] = {}
    
    for doc in input_list:
        for token in doc.tokens:
            normalized = token.normalized
            lemma = token.lemma
            if lemma != "-" and lemma != "" and normalized != "-":
                if normalized not in form_to_lemmas:
                    form_to_lemmas[normalized] = set()
                form_to_lemmas[normalized].add(lemma)
    
    # Filter out non-ambiguous forms
    numbers_resolver_dict: dict[str, str] = {}
    for form, lemmas in form_to_lemmas.items():
        for lemma in lemmas:
            if lemma.isnumeric():
                numbers_resolver_dict[form] = lemmas
    numbers_resolver_dict_1 = {}
    for form, lemmas in numbers_resolver_dict.items():
        for lemma in lemmas:
            if not lemma.isnumeric():
                numbers_resolver_dict_1[form] = lemma
    
    ambiguous_forms = {k: v for k, v in form_to_lemmas.items() if len(v) > 1}
    unique_forms = {k: v for k, v in form_to_lemmas.items() if len(v) == 1}
    print(f"Found {len(ambiguous_forms)} ambiguous forms and {len(unique_forms)} unique forms")
    return ambiguous_forms, unique_forms
    

def _test_model_apply(model: Sequential, all_norms_list: list[NorseDoc], word_index: dict[str, int], lemma_index: dict[str, int], ambiguous_forms: dict[str, set[str]]):
    print("Trying out the model")
    unseen_sequences, hrf_sequences_and_word = _preprocess_data_for_prediction(all_norms_list, word_index, sampling=True, stand_off=True, ambiguous_forms=ambiguous_forms)
    predictions = model.predict(unseen_sequences)
    inverse_lemma_index = {v: k for k, v in lemma_index.items()}
    predicted_indices = np.argmax(predictions, axis=-1)
    predicted_lemmata = [inverse_lemma_index[i] for i in predicted_indices]
    for i in range(len(hrf_sequences_and_word)):
        print(f"HRF: {hrf_sequences_and_word[i]} - Predicted: {predicted_lemmata[i]}")


def standoff_lemmatization(model: Sequential, doc_to_process: NorseDoc):
    pass


if __name__ == "__main__":
    corpus = importer.load_data()
    best_list, all_norms_list = _basic_preprocessor(corpus)
    ambiguous_forms, unique_forms = _build_unambigous_vocab(best_list)
    pickle.dump(ambiguous_forms, open(AMIBGUOUS_DICT_PICKLE, "wb"))
    pickle.dump(unique_forms, open(UNIQUE_DICT_PICKLE, "wb"))


    lemma_based_vectors = vectorize_doc_list([[tok.lemma for tok in doc.tokens] for doc in best_list], min_occurence=MIN_WORD_COUNT)
    lemma_based_vectors.save(LEMMA_W2V_MODEL)
    # pretrained_w2v_vectors = vectorize_doc_list(training_list, min_occurence=MIN_WORD_COUNT)


    prefiltered_list_1 = _filter_lemmatized_w2v_vocab(lemma_based_vectors, [[tok for tok in doc.tokens] for doc in best_list])

    word_index = create_word_index([tok.normalized for doc in prefiltered_list_1 for tok in doc])
    lemma_index = create_word_index([tok.lemma for doc in prefiltered_list_1 for tok in doc])

    all_tokens = [tok for doc in prefiltered_list_1 for tok in doc]


    # Split the dataset into training and testing sets
    maxlen = CONTEXT_LENGTH * 2 + 1

    sequences = importer.create_context_sequences_with_ambiguous_filter(all_tokens, ambiguous_forms, window_size=CONTEXT_LENGTH)
    x_raw = [x[0] for x in sequences]
    y_raw = [x[1] for x in sequences]

    x_train, x_test, y_train, y_test = importer.manual_split(x_raw, y_raw, test_size=0.2)

    x_train_seq = importer.text_to_sequence(x_train, word_index)
    x_test_seq = importer.text_to_sequence(x_test, word_index)

    y_train_seq = [lemma_index[lemma] for lemma in y_train if lemma in lemma_index]
    y_test_seq = [lemma_index[lemma] for lemma in y_test if lemma in lemma_index]
  
    x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen)
    x_test_pad = pad_sequences(x_test_seq, maxlen=maxlen)

    y_train_cat = to_categorical(y_train_seq, num_classes=len(lemma_index))
    y_test_cat = to_categorical(y_test_seq, num_classes=len(lemma_index))

    num_lemmas = y_train_cat.shape[1]


    embedding_matrix = _lemma_vector_to_normalized_embedding_matrix(prefiltered_list_1, lemma_based_vectors, word_index)

    model = create_keras_model()
    

    model.fit(x_train_pad, y_train_cat, epochs=15, validation_split=0.2, use_multiprocessing=True, workers=8, batch_size=64)

    loss, accuracy = model.evaluate(x_test_pad, y_test_cat)
    print(f'Test Accuracy: {accuracy}, Test Loss: {loss}')

    # Predicting the labels for the test set
    y_pred = model.predict(x_test_pad)
    y_pred_classes = argmax(y_pred, axis=1)

    # Print classification report
    classrep = classification_report(y_test_seq, y_pred_classes, target_names=lemma_index.keys(), labels=list(lemma_index.values()), zero_division=0, output_dict=True)

    report_df = pd.DataFrame(classrep).transpose()

    # Save the DataFrame to a CSV file
    report_df.to_csv('classification_report.csv', index=True)
    model.save(LEMMATIZER_MODEL)
    # Try it out
    print("Trying out the model")
    _test_model_apply(model, all_norms_list, word_index, lemma_index, ambiguous_forms)
    