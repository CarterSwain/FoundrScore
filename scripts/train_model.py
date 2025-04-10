import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load & Preprocess Dataset
# -----------------------------
df = pd.read_csv("data/Cleaned_Startup_Dataset.csv")

df['successful'] = (df['funding_total_usd'] > 1_000_000).astype(int)
df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
df['days_to_funding'] = (df['first_funding_at'] - df['founded_at']).dt.days

df = df.dropna(subset=['country_code', 'region', 'city', 'category', 'days_to_funding'])

X = df[['country_code', 'region', 'city', 'funding_rounds', 'category', 'days_to_funding']]
y = df['successful']

categorical_features = ['country_code', 'region', 'city', 'category']
numerical_features = ['funding_rounds', 'days_to_funding']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# -----------------------------
# 2. Define Pipeline & Param Grid
# -----------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 5],
    'classifier__class_weight': ['balanced']
}

# -----------------------------
# 3. Train/Test Split & Grid Search
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# -----------------------------
# 4. Evaluation
# -----------------------------
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)
cv_score = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

print("\n🧪 Evaluation on Test Set:\n")
print(report)
print(f"\n📊 Cross-validation accuracy: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})")
print("\n🔥 Best Parameters:")
print(grid_search.best_params_)

# -----------------------------
# 5. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/foundrscore_model.pkl")

# -----------------------------
# 6. Save Report
# -----------------------------
with open("model_results.txt", "w") as f:
    f.write(report)
    f.write(f"\nCross-validation accuracy: {cv_score.mean():.4f} (+/- {cv_score.std():.4f})\n")
    f.write(f"Best Params: {grid_search.best_params_}")

# -----------------------------
# 7. Feature Importance Plot
# -----------------------------
clf = best_model.named_steps['classifier']
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances = clf.feature_importances_

# Get top 10 important features
top_idx = importances.argsort()[-10:]
plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[top_idx], align='center')
plt.yticks(range(10), feature_names[top_idx])
plt.xlabel("Feature Importance")
plt.title("Top 10 Features Predicting Startup Success")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.close()
