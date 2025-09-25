import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import pickle
import os

# Load the dataset
data_path = 'enron_spam_data.csv'
print("Loading dataset...")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")

# Display initial dataset information
print("\nDataset Info:")
print(df.info())

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

print("\nPreprocessing data...")
# Combine Subject and Message for better feature extraction
df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Convert labels to binary
print("Converting labels...")
y = (df['Spam/Ham'].str.lower() == 'spam').astype(int)

# Vectorization with reduced features
print("\nVectorizing text data...")
tfidf = TfidfVectorizer(
    max_features=3000,  # Reduced from 5000
    min_df=2,          # Ignore terms that appear in less than 2 documents
    max_df=0.95        # Ignore terms that appear in more than 95% of documents
)
X = tfidf.fit_transform(df['processed_text'])

# Split the data with stratification
print("\nSplitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensure balanced split
)

# Train XGBoost model with adjusted parameters
print("\nTraining XGBoost model...")
model = XGBClassifier(
    eval_metric='logloss',
    max_depth=2,           # Further reduced from 3
    learning_rate=0.01,
    n_estimators=300,
    min_child_weight=5,    # Increased from 3
    subsample=0.6,         # Reduced from 0.7
    colsample_bytree=0.6,  # Reduced from 0.7
    gamma=2,               # Increased from 1
    reg_alpha=0.5,         # Increased L1 regularization
    reg_lambda=2.0,        # Increased L2 regularization
    scale_pos_weight=0.8   # Reduced from 1 to be less aggressive on spam
)

# Train with validation data
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Predictions with custom threshold
print("\nMaking predictions...")
y_pred_proba = model.predict_proba(X_test)
custom_threshold = 0.6  # Increased threshold for spam classification
y_pred = (y_pred_proba[:, 1] >= custom_threshold).astype(int)

# Calculate and print key metrics
print("\nModel Performance Metrics:")
print("=" * 50)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Save the model and vectorizer
print("\nSaving model and vectorizer...")
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

# Test with example emails
print("\nTesting with example emails...")
example_emails = [
    "URGENT: Your account needs verification. Please verify your account details immediately to prevent suspension.",
    "Hi everyone, just a reminder about our weekly team meeting tomorrow at 10 AM."
]

# Preprocess and predict example emails with custom threshold
for i, email in enumerate(example_emails, 1):
    processed_email = preprocess_text(email)
    email_vector = tfidf.transform([processed_email])
    probability = model.predict_proba(email_vector)
    prediction = (probability[0][1] >= custom_threshold).astype(int)
    
    print(f"\nExample {i}:")
    print(f"Message: {email}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
    print(f"Confidence: {probability[0][1]:.2%}")

print("\nDone! Model and vectorizer have been saved.")
