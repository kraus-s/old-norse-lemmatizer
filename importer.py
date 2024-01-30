import menota_parser
from menota_parser import NorseDoc, token
from config import *
import glob
import random
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import tensorflow as tf
from sklearn.metrics import classification_report
from numpy import argmax
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


path_list = glob.glob(f"{OLD_NORSE_CORPUS_FILES}*.xml")
# test_list = random.sample(path_list, 20)
parsed_docs_list = [menota_parser.get_regular_text(path) for path in path_list]

best_list = [x for x in parsed_docs_list if x.lemmatized and x.normalized]
print(f"Found {len(best_list)} documents with lemmatized and normalized tokens")
no_of_tokens = sum([len(x.tokens) for x in best_list])
print(f"Found {no_of_tokens} tokens in total")

def create_context_sequences(tokens: list[token], window_size=5) -> list[tuple[list[tuple[str, str]], str]]:
    sequences = []
    for i in range(len(tokens)):
        start_index = max(i - window_size, 0)
        end_index = min(i + window_size + 1, len(tokens))
        
        context = tokens[start_index:end_index]
        target = tokens[i].lemma  # Target is the lemma
        if target == "-":
            import pdb; pdb.set_trace()
        if target == None:
            import pdb; pdb.set_trace()

        # Include only normalized word forms and morpho-syntactic annotations in the context
        context: list[tuple[str, str]] = [(token.normalized, token.msa) for token in context]

        sequences.append((context, target))
    
    return sequences

all_tokens = [token for doc in best_list for token in doc.tokens if token.lemma != "-"]
sequences = create_context_sequences(all_tokens)




X_raw = [x[0] for x in sequences]
y = [x[1] for x in sequences]
X = []
for context in X_raw:
    context_list = []
    for word, msa in context:
        new_token = f"{word}-{msa.replace(' ', '-')}"
        context_list.append(new_token)
    X.append(context_list)


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

# Usage of the function
X_train_raw, X_test_raw, y_train, y_test = manual_split(X, y, test_size=0.2)

X_train = []
for context_list in X_train_raw:
    context = " ".join([str(x) for x in context_list])
    X_train.append(context)

X_test = []
for context_list in X_test_raw:
    context = " ".join([str(x) for x in context_list])
    X_test.append(context)


max_length = max(len(seq) for seq in X_train)  # You can choose a different strategy for setting max_length


# Encode the labels
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

input_dim = X_train_vec.shape[1]  # Number of features
output_dim = len(encoder.classes_)

# Define model
model = Sequential()
model.add(Dense(64, input_dim=input_dim, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training model...")
model.fit(X_train_vec, y_train_enc, epochs=50, verbose=False, validation_data=(X_test_vec, y_test_enc), batch_size=10)
print("Done!")


loss, accuracy = model.evaluate(X_test_vec, y_test_enc)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
model.save("test-run-1.mdl")


# Predicting the labels for the test set
y_pred = model.predict(X_test)
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
import pdb; pdb.set_trace()