import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("../data/Cleaned_Startup_Dataset.csv")

# Example: create target column - let's say startups with > $1M funding are considered successful
df['successful'] = (df['funding_total_usd'] > 1_000_000).astype(int)

# Drop or fill NA if needed
df = df.dropna(subset=['country_code', 'region', 'city', 'category'])  # Adjust as needed

# Select features and target
X = df[['country_code', 'region', 'city', 'funding_rounds', 'category']]
y = df['successful']

# Define preprocessing
categorical_features = ['country_code', 'region', 'city', 'category']
numerical_features = ['funding_rounds']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Create pipeline with classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, '../scripts/foundrscore_model.pkl')
