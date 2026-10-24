import joblib

from explainable_ai import (
    find_risk_factors,
    get_risk_level,
    generate_recommendation
)


# Load model
model = joblib.load("logistic_model.pkl")

# Load TF-IDF
vectorizer = joblib.load("rf_tfidf_vectorizer.pkl")


def predict_job(job_text):

    # Convert text into TF-IDF
    job_vector = vectorizer.transform([job_text])

    # Prediction
    prediction = model.predict(job_vector)[0]

    # Probability
    probability = model.predict_proba(job_vector)[0]

    fake_probability = probability[1] * 100

    # Prediction result
    if prediction == 1:
        result = "FAKE / SUSPICIOUS"
    else:
        result = "GENUINE"

    # Risk level
    risk_level = get_risk_level(fake_probability)

    # Find suspicious factors
    reasons = find_risk_factors(job_text)

    # Recommendation
    recommendations = generate_recommendation(risk_level)

    return (
        result,
        fake_probability,
        risk_level,
        reasons,
        recommendations
    )


# Test job
job = """
Work From Home Data Entry Job.
Earn Rs. 50,000 per month.
No interview required.
Pay Rs. 2,000 registration fee.
Guaranteed job opportunity.
"""


result, score, risk, reasons, recommendations = predict_job(job)


print("\n========== AI JOB ANALYSIS ==========")

print("Prediction :", result)
print("Risk Score :", round(score, 2), "%")
print("Risk Level :", risk)


print("\n========== WHY IS THIS JOB SUSPICIOUS? ==========")

if reasons:

    for reason in reasons:
        print("✓", reason)

else:
    print("No major suspicious factors detected.")


print("\n========== RECOMMENDATION ==========")

for recommendation in recommendations:
    print("•", recommendation)