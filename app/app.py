# coding: utf-8
"""
app.py — Flask app for Telco churn prediction (local use, no secret keys)

Place in app/ and run from that folder or project root:
    python app.py
Assumes:
- data/reference/first_telc.csv exists
- model/model.sav exists
- optional: model/model_columns.pkl (list of feature names saved at training)
"""

import os
import pickle
import logging
from typing import List, Optional

import pandas as pd
from flask import Flask, request, render_template

# ---------- App setup ----------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Paths ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REFERENCE_CSV = os.path.join(BASE_DIR, "data", "reference", "first_telc.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.sav")
MODEL_COLS_PATH = os.path.join(BASE_DIR, "model", "model_columns.pkl")  # optional

# ---------- Expected columns ----------
FORM_COLS = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure'
]

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_group'
]

NUMERIC_COLS = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']

# ---------- Load reference data and model ----------
if not os.path.exists(REFERENCE_CSV):
    raise FileNotFoundError(f"Reference CSV not found: {REFERENCE_CSV}")
df_ref = pd.read_csv(REFERENCE_CSV)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
logger.info(f"Loaded model from {MODEL_PATH}")

def _load_model_columns() -> Optional[List[str]]:
    # 1) try model_columns.pkl
    if os.path.exists(MODEL_COLS_PATH):
        try:
            with open(MODEL_COLS_PATH, "rb") as f:
                cols = pickle.load(f)
            logger.info(f"Loaded model columns from {MODEL_COLS_PATH} ({len(cols)} cols)")
            return list(cols)
        except Exception as e:
            logger.warning(f"Could not load model_columns.pkl: {e}")

    # 2) try estimator attribute feature_names_in_
    cols_attr = getattr(model, "feature_names_in_", None)
    if cols_attr is not None:
        cols = list(cols_attr)
        logger.info(f"Using model.feature_names_in_ ({len(cols)} cols)")
        return cols

    # 3) derive from df_ref by applying same preprocessing (best-effort)
    try:
        df_tmp = df_ref.copy()
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        df_tmp['tenure'] = pd.to_numeric(df_tmp.get('tenure', 0), errors='coerce').fillna(0).astype(int)
        df_tmp['tenure_group'] = pd.cut(df_tmp.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
        if 'tenure' in df_tmp.columns:
            df_tmp = df_tmp.drop(columns=['tenure'])
        numeric_part = df_tmp[['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']].copy()
        dummies = pd.get_dummies(df_tmp[CATEGORICAL_COLS], dummy_na=False)
        final = pd.concat([numeric_part, dummies], axis=1)
        cols = list(final.columns)
        logger.info(f"Derived {len(cols)} model columns from reference CSV")
        return cols
    except Exception as e:
        logger.error(f"Failed to derive model columns from reference CSV: {e}")
        return None

MODEL_COLUMNS = _load_model_columns()
if MODEL_COLUMNS is None:
    logger.warning("MODEL_COLUMNS is None. Predictions may fail due to feature mismatch.")

# ---------- Preprocessing helper ----------
def preprocess_single_input(new_row: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    new_row : DataFrame with one row and columns = FORM_COLS
    reference_df : original reference DataFrame (unchanged)
    returns processed DataFrame (not yet reindexed to MODEL_COLUMNS)
    """
    df_comb = pd.concat([reference_df, new_row], ignore_index=True)

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_comb['tenure'] = pd.to_numeric(df_comb.get('tenure', 0), errors='coerce').fillna(0).astype(int)
    df_comb['tenure_group'] = pd.cut(df_comb.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

    if 'tenure' in df_comb.columns:
        df_comb.drop(columns=['tenure'], inplace=True)

    dummies = pd.get_dummies(df_comb[CATEGORICAL_COLS], dummy_na=False)

    numeric_part = pd.DataFrame({
        'SeniorCitizen': pd.to_numeric(df_comb.get('SeniorCitizen', 0), errors='coerce').fillna(0).astype(int),
        'MonthlyCharges': pd.to_numeric(df_comb.get('MonthlyCharges', 0), errors='coerce').fillna(0.0),
        'TotalCharges': pd.to_numeric(df_comb.get('TotalCharges', 0), errors='coerce').fillna(0.0)
    })

    final = pd.concat([numeric_part.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return final

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", query="")

@app.route("/", methods=["POST"])
def predict():
    # read form values query1..query19 into dict for returning to template
    form_data = {f"query{i}": request.form.get(f"query{i}", "") for i in range(1, 20)}

    # build input DataFrame
    try:
        inputs = [request.form.get(f"query{i}", "") for i in range(1, 20)]
        input_df = pd.DataFrame([inputs], columns=FORM_COLS)
    except Exception as e:
        msg = f"Failed to read form inputs: {e}"
        logger.exception(msg)
        return render_template("home.html", output1="Error", output2=msg, **form_data)

    # convert numeric columns
    try:
        input_df['SeniorCitizen'] = pd.to_numeric(input_df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
        input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'], errors='coerce').fillna(0.0)
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0.0)
        input_df['tenure'] = pd.to_numeric(input_df['tenure'], errors='coerce').fillna(0).astype(int)
    except Exception as e:
        logger.warning(f"Numeric conversion issue: {e}")

    # preprocessing & build final features
    try:
        final_processed = preprocess_single_input(input_df, df_ref)
        X = final_processed.tail(1)
        if MODEL_COLUMNS is not None:
            X = X.reindex(columns=MODEL_COLUMNS, fill_value=0)
    except Exception as e:
        msg = f"Preprocessing failed: {e}"
        logger.exception(msg)
        return render_template("home.html", output1="Error", output2=msg, **form_data)

    # prediction
    try:
        pred = model.predict(X)
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X)[:, 1]
            except Exception:
                prob = None

        label = pred[0]
        if label == 1 or str(label) == "1":
            o1 = "This customer is likely to be churned!!"
        else:
            o1 = "This customer is likely to continue!!"

        o2 = f"Churning Confidence: {prob[0]*100:.2f}%" if (prob is not None) else "Confidence: N/A"
        return render_template("home.html", output1=o1, output2=o2, **form_data)
    except Exception as e:
        msg = f"Prediction failed: {e}"
        logger.exception(msg)
        return render_template("home.html", output1="Error", output2=msg, **form_data)

# ---------- Run ----------
if __name__ == "__main__":
    # debug=True for development; change to False for production deployment
    app.run(host="127.0.0.1", port=5000, debug=True)
