from flask import Flask, render_template, request
import imaplib
import email as em
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the trained text spam detection model
text_model = load_model("spam_detection_model_en.h5")
text_model.compile(optimizer='rmsprop',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Load the tokenizer for text spam detection
with open("tokenizer_en.pkl", 'rb') as f:
    text_tokenizer = pickle.load(f)

max_sequence_length = 50  # Maximum sequence length for padding

ps = PorterStemmer()

# Function for preprocessing email content
def preprocess_email(email_body):
    email_body = email_body.lower()
    email_body = re.sub(r'\d+', '', email_body)  # Remove digits
    email_body = re.sub(r'[^\w\s]', '', email_body)  # Remove special characters
    tokens = word_tokenize(email_body)
    stop_words = set(stopwords.words('english'))  # English stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatizer to reduce words to base form
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Function for preprocessing text for spam detection (using the loaded tokenizer)
def preprocess_text(text):
    sequences = text_tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# Function to classify text as spam or not spam
def predict_spam(text):
    preprocessed_text = preprocess_text(text)
    prediction = text_model.predict(preprocessed_text)
    spam_probability_percentage = prediction[0][0] * 100
    return spam_probability_percentage

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/detect_text_spam", methods=["GET", "POST"])
def detect_text_spam_route():
    spam_probability = None
    result = None
    if request.method == "POST":
        text = request.form["text"]
        spam_probability = predict_spam(text)
        threshold = 0.5  # Adjust threshold as needed
        result = "Spam" if spam_probability >= threshold else "Not Spam"
    return render_template("text_spam_detection.html", spam_probability=spam_probability, result=result)

@app.route("/detect_email_spam", methods=["GET", "POST"])
def detect_email_spam_route():
    emails = []
    if request.method == "POST":
        user_email = request.form["email"]
        user_password = request.form["password"]

        try:
            # Connect to email server
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(user_email, user_password)
            mail.select("inbox")

            # Search for emails
            result, data = mail.search(None, "ALL")

            for num in data[0].split()[:30]:  # Load the first 30 emails
                result, data = mail.fetch(num, "(RFC822)")
                raw_email = data[0][1]
                msg = em.message_from_bytes(raw_email)

                email_body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            email_body += part.get_payload(decode=True).decode('utf-8')
                else:
                    email_body = msg.get_payload(decode=True).decode('utf-8')

                preprocessed_email = preprocess_email(email_body)
                sequences = text_tokenizer.texts_to_sequences([preprocessed_email])
                padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

                predicted_probabilities = text_model.predict(padded_sequences)
                prediction = "spam" if (predicted_probabilities > 0.5).astype(int) else "non-spam"

                emails.append({
                    "subject": msg["subject"],
                    "classification": prediction
                })

            mail.close()
            mail.logout()

        except Exception as e:
            emails.append({"subject": "Error fetching emails", "classification": str(e)}) # Handle exceptions

    return render_template("email_spam_detection.html", emails=emails)

if __name__ == "__main__":
    app.run(debug=True)