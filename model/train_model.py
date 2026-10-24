import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = pd.read_csv("../dataset/fake_jobs.csv")

# 2. Combine important text columns
data["text"] = (
    data["title"].fillna("") + " " +
    data["description"].fillna("") + " " +
    data["requirements"].fillna("")
)

# 3. Input and output
X = data["text"]
y = data["fraudulent"]

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 5. Convert text into numerical features
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. Train model
model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

# 7. Prediction
y_pred = model.predict(X_test_tfidf)

# 8. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Save model
joblib.dump(model, "fake_job_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")