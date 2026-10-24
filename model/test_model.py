import joblib

model = joblib.load("model/fake_job_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

job = """
Work from home.
Earn Rs. 60,000 per month.
No interview required.
Pay Rs. 2,000 registration fee to get the job.
Guaranteed placement.
"""

job_vector = vectorizer.transform([job])

prediction = model.predict(job_vector)[0]
probability = model.predict_proba(job_vector)[0]

if prediction == 1:
    print("Result: FAKE / SUSPICIOUS")
else:
    print("Result: GENUINE")

print("Fake Probability:", round(probability[1] * 100, 2), "%")