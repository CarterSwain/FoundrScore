# 🧠 FoundrScore

**FoundrScore** is an AI-powered web app that predicts the likelihood of a startup's success based on historical data.  
It uses a trained machine learning model to analyze key startup features and outputs a probability score — your "FoundrScore."

---

## Features

- Predict startup success based on location, funding history, and category
- Trained with a Random Forest classifier + GridSearchCV tuning
- Flask web app with interactive input form
- Built-in ML model for real-time predictions

---

## ⚠️ Model Download Required

The trained model file is too large to host on GitHub.

👉 [Download FoundrScore model here](https://huggingface.co/CarterSwain/foundrscore-model/resolve/main/foundrscore_model.pkl)

Then place it in the `models/` directory of this project:

/models/foundrscore_model.pkl


---

## Installation

1. Clone this repo:

```bash
git clone https://github.com/CarterSwain/FoundrScore.git
cd FoundrScore


2. Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows


3. Install dependencies:

pip install -r requirements.txt


---


🖥️ Run the App:

python3 run.py

Then go to: http://localhost:5000


---


## Coming Soon:

- OpenAI-powered suggestions based on your FoundrScore

- User-friendly UI improvements

- Visualizations of feature importance


---

## Author:

Carter Swain
carterswaindev.com  • @CarterSwain

---

## Acknowledgments

This project was made possible thanks to the dataset:

Big Startup Success/Fail Dataset from Crunchbase
Published by @yanmaksi on Kaggle

The dataset compiles startup company data from Crunchbase, including funding history, location, categories, and outcomes.
It provided the foundation for training and evaluating the FoundrScore prediction model.


