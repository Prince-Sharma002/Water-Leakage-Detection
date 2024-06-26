
import pickle

import spacy
import joblib

nlp = spacy.load("en_core_web_lg")

import pandas as pd

df = pd.read_csv("leakage_data.csv")
df.drop('DATE', axis=1, inplace=True)
df.drop('TIME', axis=1, inplace=True)

selected_col= ['NO' , 'FLOW05' , 'FLOW06' , 'FLOW07' , 'FLOW08' , "FLOW09" , "FLOW10",'LEAK']
df = df[selected_col]


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming df is your DataFrame containing the dataset
# Splitting the DataFrame into features (X) and target variable (y)
X = df.drop(['LEAK', 'NO'], axis=1)  # Assuming 'LEAK' is the target variable
y = df['LEAK']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Save the trained Random Forest classifier to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)

# pickle.dump(sv, open('iri.pkl', 'wb'))
