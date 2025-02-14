from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample training data (spam and non-spam messages)
messages = ["Free entry in a contest", "Hello, how are you?", "Win money now!", "Let's meet tomorrow", "Free prizes available"]
labels = [1, 0, 1, 0, 1]  # 1 = Spam, 0 = Not Spam

# Initialize TF-IDF vectorizer and fit it on the messages
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(messages)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)

# Save the trained model and vectorizer
with open('E:/CP_Project_09_43/responsive-dashboard-main/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
    
with open('E:/CP_Project_09_43/responsive-dashboard-main/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved successfully!")
