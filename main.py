import re
import string
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential

import seaborn as sns

# Load pre-existing dataset
# Assume X and y are loaded from the dataset

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


def load_and_pre_clean_csv_data(csv_file_path):
    """ load the csv file data and preprocess it by removing row with any emptry column 
        or rows with any invalid data for the particular column"""
    # Load CSV data into DataFrame
    df = pd.read_csv(csv_file_path)

    # Remove rows where the "text" column is empty
    df = df.dropna(subset=['text'])

    # Remove rows where the "target" column is non-numeric
    df = df[pd.to_numeric(df['target'], errors='coerce').notna()]

    # Convert "target" column to numeric type
    df['target'] = pd.to_numeric(df['target'])

    # Optional: Reset index after filtering
    df.reset_index(drop=True, inplace=True)

    return df

def preprocess_tweet(tweet):
    """
    Preprocess a single tweet.
    """
    # Remove links
    tweet = re.sub(r'http\S+', '', tweet)

    # Remove emojis
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')

    # Remove punctuation
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet

def tokenize_and_clean(tweet):
    """
    Tokenize, remove stopwords, and lemmatize a single tweet.
    """
    # Tokenize
    tokens = word_tokenize(tweet)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def pad_vectors(dt):
    """
        padd the rows and columns of the matrix of the vectors
    """
    width = 100
    height = 20
    for index,i  in enumerate(dt):
        diff = width - len(i)
        if diff > 0:
            dt[index] = np.append(dt[index], [0] * diff)
    
    if height > len(dt):
        dt.append([0] * width)
    
    return dt


# Load data from CSV file
FILE_NAME = './tweets.csv'
data = load_and_pre_clean_csv_data(FILE_NAME)

# Preprocess the 'text' column
data['preprocessed_text'] = data['text'].apply(preprocess_tweet)

# Tokenize, remove stopwords, and lemmatize each tweet
data['tokenized_text'] = data['preprocessed_text'].apply(tokenize_and_clean)

# remove rows where column "tokenized_text" is empty
data = data[data['tokenized_text'].apply(len) > 0]

# Display the first few rows of the preprocessed and tokenized data
print(data.head())


# vectorize tweets
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['preprocessed_text'])
y = data['target']


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\n\n")

# Supervised Learning Models
print("Supervised Learning Models".upper())
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")


print("\n\n\n")


# Deep Learning Models
print("Deep Learning Models".upper())
X_train = X_train.toarray()
X_test = X_test.toarray()

VOCAB_SIZE =10000
EMBEDDING_DIM = 16

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
print("\n\n\n")




# Ensemble Method (Voting Classifier)
print("Ensemble Method (Voting Classifier)".upper())
estimators = list(models.items())
voting_classifier = VotingClassifier(estimators)
voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Method Accuracy: {accuracy}")
