# UK Football Match Outcome Prediction (H/D/A)

## Project Overview
This project predicts the outcomes of UK football matches (Home/Draw/Away) **before kickoff**.  
We use **historical match data** from [Football-Data.co.uk](https://www.football-data.co.uk) and **upcoming fixtures** from the [football-data.org API](https://www.football-data.org).  

The system builds **pre-match features** such as rolling team form, rest days, and implied win probabilities from betting odds.  
Models include:
- Logistic Regression (baseline)
- TensorFlow MLP (tabular features)
- TensorFlow Team-Embedding Model (advanced)

The best model (based on Macro-F1 and Log-loss) is saved and used to predict **upcoming fixtures**, stored in `outputs/fixtures_predictions.csv`.

---

## Research Question
> **Can a TensorFlow model with team embeddings and pre-match tabular features outperform a logistic regression baseline for football outcome prediction under a time-aware split?**

**Sub-questions:**
1. Does a TensorFlow MLP outperform logistic regression on Macro-F1 and Log-loss?  
2. Do team embeddings improve prediction quality further?  
3. Are the predicted probabilities calibrated enough for real-world use?  

**Metrics:** Macro-F1 (primary), Log-loss & Brier Score (secondary).

---

## Project Structure

uk-footy-pred/
├─ data/
│ ├─ raw/ # raw Football-Data CSVs (not in repo)
│ └─ processed/ # processed data (optional)
├─ models/ # trained models (.joblib / .keras)
├─ notebooks/
│ └─ 01_eda.ipynb # Exploratory Data Analysis
├─ outputs/
│ ├─ figures/ # saved plots
│ └─ fixtures_predictions.csv
├─ src/
│ ├─ data/ # data loaders & API fetch
│ ├─ features/ # feature engineering
│ ├─ models/ # ML + DL models
│ ├─ eval/ # evaluation metrics
│ └─ run_pipeline.py # main pipeline entry point
├─ .env # template for API key
├─ requirements.txt # pinned dependencies
├─ README.md # this file
└─ LICENSE # MIT License


## ⚙️ Installation & Setup

### 1. Clone the repository

git clone https://github.com/Shankar226/uk-footy-pred.git 
cd uk-footy-pred

### 2. Create a virtual environment

python -m venv .venv
.venv\Scripts\activate   # Windows

### 3. Install dependencies

pip install -r requirements.txt

### 4. Prepare data

Download CSVs (E0.csv, E1.csv, etc.) from [Football-Data.co.uk ](https://www.football-data.co.uk/englandm.php)

Place them into data/raw/.

### 5. Configure API

.env # template for API key(file)

Add your football-data.org API key:

FOOTBALL_DATA_API_KEY=your_api_key_here
COMPETITION=PL

### Ethics

Not betting advice: This project is for research and educational purposes only.

Risks: Predictions could be misused for gambling; probabilities are not guarantees.

Transparency: All features are pre-match; splits are time-aware to avoid leakage.