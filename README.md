# Customer Churn Prediction

This project is a **Customer Churn Prediction** system built using Python, scikit-learn, and Flask. It predicts whether a customer is likely to churn based on telecom service usage and account information.

---

## Project Overview

The system allows a user to input customer information through a web interface and returns a prediction along with the confidence score. It is designed for **local deployment** and uses a pre-trained `RandomForestClassifier` for predictions.

---

## Repo Structure

```
customer-churn-prediction/
│
├── app/
│   ├── app.py               # Flask application
│   └── templates/
│       └── home.html        # Frontend HTML page
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original dataset
│   ├── processed/
│   │   └── tel_churn.csv    # Processed dataset
│   └── reference/
│       └── first_telc.csv   # Reference CSV used for feature alignment
│
├── model/
│   └── model.sav            # Trained RandomForestClassifier
│
└── notebooks/
    ├── customer-churn-EDA.ipynb  # Exploratory Data Analysis
    └── Model.ipynb               # Model training notebook
```

---

## Features Used for Prediction

* **Numerical Features:** `SeniorCitizen`, `MonthlyCharges`, `TotalCharges`, `tenure`
* **Categorical Features:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
* **Derived Features:** `tenure_group` (binned tenure)

---

## Data Preprocessing

* Categorical features are **one-hot encoded**.
* Missing values in numeric features are filled with `0` or appropriate defaults.
* To ensure consistent feature alignment, each new customer input is appended to a **reference dataset** (`first_telc.csv`) before encoding.
* Tenure is binned into groups of 12 months for better categorization.

---

## Model Architecture

* The project uses a **RandomForestClassifier** from scikit-learn.
* Trained on the processed Telco churn dataset.
* Produces a probability score along with the final prediction.

---

## How to Run Locally

*(Make sure `Flask`, `pandas`, and `scikit-learn` are installed.)*

### Run the Flask app:

   ```bash
   python app.py
   ```

### Open your browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```

### Input customer details and get predictions.

---

## Screenshots

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f608205c-7111-4943-bacb-826f8fc5fb84" />



