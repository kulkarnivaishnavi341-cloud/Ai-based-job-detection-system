import re


# Suspicious patterns
RISK_PATTERNS = {
    "registration fee": "Registration fee requested",
    "security deposit": "Security deposit requested",
    "processing fee": "Processing fee requested",
    "pay money": "Payment requested before joining",
    "pay rs": "Payment requested",
    "guaranteed job": "Guaranteed job claim",
    "guaranteed placement": "Guaranteed placement claim",
    "no interview": "No interview required",
    "earn huge money": "Unrealistic earning claim",
    "earn unlimited": "Unlimited earning claim",
    "work from home": "Work-from-home claim"
}


def find_risk_factors(job_text):

    text = job_text.lower()

    reasons = []

    for pattern, reason in RISK_PATTERNS.items():

        if pattern in text:
            reasons.append(reason)

    return reasons


def get_risk_level(score):

    if score >= 70:
        return "HIGH RISK"

    elif score >= 40:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


def generate_recommendation(risk_level):

    if risk_level == "HIGH RISK":

        return [
            "Do not make any payment.",
            "Verify the company before applying.",
            "Do not share sensitive personal information."
        ]

    elif risk_level == "MEDIUM RISK":

        return [
            "Verify the company and job posting.",
            "Check the official company website.",
            "Avoid making advance payments."
        ]

    else:

        return [
            "Job appears relatively safe based on the available information.",
            "Still verify the company before applying."
        ]