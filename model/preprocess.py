import pandas as pd

# Load dataset
data = pd.read_csv("dataset/fake_job_postings.csv")

print("Original Shape:", data.shape)

# Remove duplicate rows
data = data.drop_duplicates()

# Text columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

# Fill missing text values
for column in text_columns:
    if column in data.columns:
        data[column] = data[column].fillna("")

# Convert text to lowercase
for column in text_columns:
    if column in data.columns:
        data[column] = data[column].astype(str).str.lower()

# Remove rows where target is missing
data = data.dropna(subset=["fraudulent"])

# Create combined text
data["combined_text"] = (
    data["title"] + " " +
    data["company_profile"] + " " +
    data["description"] + " " +
    data["requirements"] + " " +
    data["benefits"]
)

# Save cleaned dataset
data.to_csv("dataset/cleaned_jobs.csv", index=False)

print("Cleaned Shape:", data.shape)
print("Cleaned dataset saved successfully!")

# Show class distribution
print("\nFraudulent Distribution:")
print(data["fraudulent"].value_counts())