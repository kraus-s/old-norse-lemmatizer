import src.menota_parser as menota_parser
from src.menota_parser import NorseDoc, token
from src.config import *
import glob
import random
from sklearn.model_selection import train_test_split
import itertools
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from gensim.models import Word2Vec


def import_menota_data(path: str = OLD_NORSE_CORPUS_FILES, test: bool = True, test_amount: int = 10) -> list[NorseDoc]:
    """
    Imports the Norse corpus from the Menota XML files
    """
    path_list = glob.glob(f"{path}*.xml")
    if test:
        docs_to_parse = random.sample(path_list, test_amount)
    else:
        docs_to_parse = path_list
    parsed_docs_list = [menota_parser.get_regular_text(path) for path in docs_to_parse]

    return parsed_docs_list


def create_context_sequences(tokens: list[token], window_size=5) -> list[tuple[list[str], str]]:
    sequences = []
    for i in range(len(tokens)):
        start_index = max(i - window_size, 0)
        end_index = min(i + window_size + 1, len(tokens))
        
        context: list[token] = tokens[start_index:end_index]
        target = tokens[i].lemma  # Target is the lemma
        if target == "-":
            import pdb; pdb.set_trace()
        if target == None:
            import pdb; pdb.set_trace()

        context_new: list[str] = [tok.normalized for tok in context]

        sequences.append((context_new, target))
    
    return sequences


def create_context_sequences_with_ambiguous_filter(tokens: list[token], ambiguous_forms: dict[str, set[str]], window_size=5) -> list[tuple[list[str], str]]:
    sequences = []
    for i in range(len(tokens)):
        if tokens[i].normalized in ambiguous_forms.keys():
            start_index = max(i - window_size, 0)
            end_index = min(i + window_size + 1, len(tokens))
            
            context: list[token] = tokens[start_index:end_index]
            target = tokens[i].lemma  # Target is the lemma
            if target == "-":
                import pdb; pdb.set_trace()
            if target == None:
                import pdb; pdb.set_trace()

            context_new: list[str] = [tok.normalized for tok in context]

            sequences.append((context_new, target))
    
    print(f"After filtering for ambiguous forms, the dataset has {len(sequences)} sequences")
    return sequences


def _create_context_sequences_for_prediction(tokens: list[token], window_size: int=5) -> list[tuple[list[str], str]]:
    # This will create a list of context sequences for the given tokens to predict lemmata on unlemmatized data
    sequences = []
    for i in range(len(tokens)):
        start_index = max(i - window_size, 0)
        end_index = min(i + window_size + 1, len(tokens))
        context: list[token] = tokens[start_index:end_index]
        context_new: list[str] = [tok.normalized for tok in context]
        sequences.append((context_new, tokens[i].normalized))
    
    return sequences


def _create_context_sequences_for_prediction_ambiguous_filter(tokens: list[token], ambiguous_forms: dict[str, set[str]], window_size: int=5, ) -> list[tuple[list[str], str]]:
    # This will create a list of context sequences for the given tokens to predict lemmata on unlemmatized data
    sequences = []
    for i in range(len(tokens)):
        if tokens[i].normalized in ambiguous_forms.keys():
            start_index = max(i - window_size, 0)
            end_index = min(i + window_size + 1, len(tokens))
            context: list[token] = tokens[start_index:end_index]
            context_new: list[str] = [tok.normalized for tok in context]
            sequences.append((context_new, tokens[i].normalized))
    
    return sequences


# Split the dataset into training and testing sets
def manual_split(contexts, lemmata, test_size=0.2):
    # Combine the contexts and lemmata into a single list of tuples
    combined = list(zip(contexts, lemmata))
    
    # Sort by lemmata to group the same labels together
    combined.sort(key=lambda x: x[1])

    # Split each group and collect train and test samples
    train_samples, test_samples = [], []
    for _, group in itertools.groupby(combined, key=lambda x: x[1]):
        group = list(group)
        if len(group) > 4:
            train, test = train_test_split(group, test_size=test_size)
            train_samples.extend(train)
            test_samples.extend(test)
        else:
            pass

    # Shuffle the train and test samples to randomize the order
    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # Separate the contexts and lemmata again
    X_train, y_train = zip(*train_samples)
    X_test, y_test = zip(*test_samples)

    return list(X_train), list(X_test), list(y_train), list(y_test)


def evaluate_model(model, X_test_vec, y_test_enc, encoder):
    # Predicting the labels for the test set
    y_pred = model.predict(X_test_vec)
    y_pred_classes = argmax(y_pred, axis=1)

    # Generate the classification report
    print(classification_report(y_test_enc, y_pred_classes, target_names=encoder.classes_))


    # Compute the confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred_classes)

    # Plotting the confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.show()


# Function to convert tokens to their integer indices
def text_to_sequence(texts: list[list[str]], word_index: dict[str, int]):
    sequences = []
    for text in texts:
        sequence = [word_index[word] for word in text if word in word_index]
        sequences.append(sequence)
    return sequences


def create_word_index(words):
    word_count = defaultdict(lambda: 0)
    for word in words:
        word_count[word] += 1

    # Sort words by frequency (most to least)
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)

    # Create the word index (word to integer mapping)
    word_index = {word: i + 1 for i, word in enumerate(sorted_words)}  # start indexing from 1
    word_index["<PAD>"] = 0  # add a <PAD> token
    return word_index


def vectorize_doc_list(list: list[list[str]]):
    # Split each sublist into smaller sublists of length 500
    docs_list_windowed= [sublist[i:i+500] for sublist in list for i in range(0, len(sublist), 500)]
    vectorized_model = Word2Vec(docs_list_windowed, min_count=1, window=5, workers=8, vector_size=VECTOR_SPACE)
    return vectorized_model


if __name__ == "__main__":
    pass