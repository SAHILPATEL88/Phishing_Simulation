from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Sample training data (spam and non-spam messages)
messages = ["Free entry in a contest", "Hello, how are you?", "Win money now!", "Let's meet tomorrow", "Free prizes available"]
labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit the vectorizer on your training data
X = tfidf.fit_transform(messages)

# Save the vectorizer to a file
with open('E:/ML/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Vectorizer saved successfully!")

