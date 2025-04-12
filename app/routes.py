from flask import Blueprint, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
import requests
from dotenv import load_dotenv
load_dotenv()
# OpenAI Setup
import openai
from openai import OpenAI
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


main = Blueprint("main", __name__)


# Hugging Face model download
MODEL_PATH = "models/foundrscore_model.pkl"
HF_MODEL_URL = "https://huggingface.co/CarterSwain/foundrscore-model/resolve/main/foundrscore_model.pkl"

# Make sure model exists before loading
os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading FoundrScore model from Hugging Face...")
    response = requests.get(HF_MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded and saved.")
    
    
# Load model 
model = joblib.load("models/foundrscore_model.pkl")


# Get suggestions for improvement of Score from OpenAI
def get_openai_suggestions(user_input, probability):
    prompt = f"""
    A startup founder has just received a FoundrScore of {round(probability * 100, 2)}%.
    Based on the following inputs:

    - Country: {user_input['country_code']}
    - Region: {user_input['region']}
    - City: {user_input['city']}
    - Category: {user_input['category']}
    - Funding Rounds: {user_input['funding_rounds']}
    - Days to Funding: {user_input['days_to_funding']}

    Please give this founder 2–3 specific suggestions to improve their chance of success.
    Be constructive, clear, and speak like you're advising a real founder.
    """

    response = openai.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": prompt}],
       temperature=0.7,
       max_tokens=1000
    )


    return response.choices[0].message.content.strip()



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

        # Create user input dictionary for GPT
        user_input = {
            "country_code": country,
            "region": region,
            "city": city,
            "category": category,
            "funding_rounds": funding_rounds,
            "days_to_funding": days_to_funding
        }

        # Get OpenAI suggestions
        suggestions = get_openai_suggestions(user_input, score)

        return render_template(
            "result.html",
            prediction=prediction,
            score=round(score * 100, 2),
            suggestions=suggestions
        )

    return render_template("index.html")

