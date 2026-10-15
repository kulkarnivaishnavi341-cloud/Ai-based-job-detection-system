import pandas as pd

data = pd.read_csv("dataset/fake_job_postings.csv")

print("Dataset Shape:")
print(data.shape)

print("\nColumns:")
print(data.columns)

print("\nFirst 5 Rows:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nFraudulent Distribution:")
print(data["fraudulent"].value_counts())