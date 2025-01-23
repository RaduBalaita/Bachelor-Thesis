import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data (needed for X_test_pad and y_test, if not loading from saved files)
df = pd.read_csv('spam.csv', encoding='latin1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.drop_duplicates(inplace=True)

# Text Preprocessing Function (needed to recreate X_test_pad)
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

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

X = df['transformed_text']
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# Tokenization
max_words = 10000
tokenizer_save_path = "tokenizer_en.pkl"
with open(tokenizer_save_path, 'rb') as f:
    tokenizer = pickle.load(f)

X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
maxlen = 50
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Load the model
model = tf.keras.models.load_model('spam_detection_model_en.h5')

# Generate predictions for the test set
y_pred_proba = model.predict(X_test_pad)
y_pred = (y_pred_proba > 0.5).astype(int)

# 1. Learning Curve
# Load training history
history_save_path = "RNN-Results/training_history.pkl"  # Load from RNN-Results directory
with open(history_save_path, 'rb') as f:
    history = pickle.load(f)

# Plot learning curve
if history:
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()

    # Save the learning curve plot
    learning_curve_path = 'RNN-Results/learning_curve.png' # Save in RNN-Results directory
    plt.savefig(learning_curve_path)
    plt.close()

# 2. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Ham', 'Spam'], rotation=45)
plt.yticks(tick_marks, ['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text to the Confusion Matrix plot
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, f'{conf_matrix[i, j]}', horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

# Save the Confusion Matrix plot
confusion_matrix_path = 'RNN-Results/confusion_matrix.png' # Save in RNN-Results directory
plt.savefig(confusion_matrix_path)
plt.close()

# 3. ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print("AUC Score for ROC Curve:", roc_auc)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the ROC curve plot
roc_curve_path = 'RNN-Results/roc_curve.png' # Save in RNN-Results directory
plt.savefig(roc_curve_path)
plt.close()

# 4. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)
print("Average Precision Score:", average_precision)

# Plot the Precision-Recall curve
plt.figure()
plt.plot(recall, precision, color='b', lw=2, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

# Save the Precision-Recall curve plot
precision_recall_curve_path = 'RNN-Results/precision_recall_curve.png' # Save in RNN-Results directory
plt.savefig(precision_recall_curve_path)
plt.close()