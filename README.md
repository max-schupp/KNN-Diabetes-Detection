---
title: "NB-fakenews"
author:
- Maximilian
Reviewer:
- Le Minh Quan
output:
  html_document:
    toc: true
    toc_depth: 2
date: "2023-04-07"
---

# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Business Understanding
# Define the problem and determine what the stakeholders are looking for in the output.

# Step 2: Data Understanding
# Load and explore the dataset.

# Load the dataset
df = pd.read_csv('NB-fakenews.csv')

# View the first five rows of the dataset
print(df.head())

# Get the shape of the dataset
print("Number of rows and columns:", df.shape)

# Get the distribution of labels
print(df['label'].value_counts())

# Step 3: Data Preparation
# Clean and preprocess the data.

# Remove any missing values
df.dropna(inplace=True)

# Step 4: Modeling
# Split the data into training and testing sets, fit the model, and make predictions.

# Split the data into training and testing sets
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label'] 

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Make predictions on the test set
# Note: In practice, you would need to split the data into separate training and testing sets.
# This is just an example of how to make predictions using the trained model.
test = ['This is a real news article', 'This is a fake news article']
test_transformed = vectorizer.transform(test)
predictions = model.predict(test_transformed)

# Step 5: Evaluation
# Evaluate the performance of the model.

# Print the accuracy of the model
# Note: In practice, you would need to evaluate the model using more appropriate metrics (e.g., precision, recall, F1 score).
print("Accuracy:", model.score(X, y))

# Step 6: Deployment
# Deploy the model into production.

# Save the model to a file
# Note: In practice, you would likely need to save the vectorizer and the model separately and load them in
# a separate script or application.
import joblib
joblib.dump(model, 'naive_bayes_model.joblib')
