import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Download nltk resources if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the data - Changed to spam.csv
df = pd.read_csv('spam.csv', encoding='latin1')

# Data Cleaning
# Keep the column dropping lines (as spam.csv has these columns)
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.drop_duplicates(inplace=True)

# Text Preprocessing Function (rest of the code remains the same)
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

# Model Building
X = df['transformed_text']
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# Tokenization
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
maxlen = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Save tokenizer
tokenizer_save_path = "tokenizer_en.pkl"
with open(tokenizer_save_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Model BiLSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=100, input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, recurrent_dropout=0.2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train_pad,
                    y_train, epochs=10,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=1)

# Generate predictions for the test set
y_pred_proba = model.predict(X_test_pad)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Accuracy on test set:", accuracy)

# Classification report
class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
print("Classification Report:")
print(class_report)

# Save model
model.save("spam_detection_model_en.h5")

# Save training history
history_save_path = "RNN-Results/training_history.pkl"
with open(history_save_path, 'wb') as f:
    pickle.dump(history.history, f)