import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("dataset/cleaned_jobs.csv")

print("Dataset Shape:", data.shape)


# ==========================================
# 2. Input and Target
# ==========================================

X = data["combined_text"].fillna("")
y = data["fraudulent"]


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 4. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF Training Shape:", X_train_tfidf.shape)


# ==========================================
# 5. Random Forest Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train_tfidf, y_train)

print("Training completed!")


# ==========================================
# 6. Prediction
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 7. Performance
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n========== RANDOM FOREST PERFORMANCE ==========")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")


# ==========================================
# 8. Classification Report
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fake"],
        zero_division=0
    )
)


# ==========================================
# 9. Confusion Matrix
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# ==========================================
# 10. Save Model
# ==========================================

joblib.dump(
    model,
    "random_forest_model.pkl"
)

joblib.dump(
    vectorizer,
    "rf_tfidf_vectorizer.pkl"
)

print("\nRandom Forest model saved successfully!")