from flask import Blueprint, render_template, request
import joblib
import numpy as np
import pandas as pd

main = Blueprint("main", __name__)

# Load model once
model = joblib.load("models/foundrscore_model.pkl")

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from form
        country = request.form.get("country_code")
        region = request.form.get("region")
        city = request.form.get("city")
        category = request.form.get("category")
        funding_rounds = int(request.form.get("funding_rounds"))
        days_to_funding = int(request.form.get("days_to_funding"))

        # Create input DataFrame
        input_df = pd.DataFrame([{
            "country_code": country,
            "region": region,
            "city": city,
            "category": category,
            "funding_rounds": funding_rounds,
            "days_to_funding": days_to_funding
        }])

        # Predict
        prediction = model.predict(input_df)[0]
        score = model.predict_proba(input_df)[0][1]  # probability of success

        return render_template("result.html", prediction=prediction, score=round(score * 100, 2))

    return render_template("index.html")
