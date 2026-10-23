
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Forecaster")

# Load the trained pipeline (preprocessing + model) that was serialized in the notebook
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

# The raw features the pipeline expects (the pipeline encodes and scales them itself)
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

# Input validation rules. The encoder of the model silently ignores categories it has not seen, so without
# these checks a typo or an absurd value would return a number that looks like a real forecast.
# In a batch, valid rows are scored and invalid rows are reported (not guessed).
ALLOWED_VALUES = {
    "Product_Sugar_Content": ["Low Sugar", "Regular", "No Sugar"],
    "Store_Size": ["Small", "Medium", "High"],
    "Store_Location_City_Type": ["Tier 1", "Tier 2", "Tier 3"],
    "Store_Type": ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"],
    "Product_Id_char": ["FD", "DR", "NC"],
    "Product_Type_Category": ["Perishables", "Non Perishables"],
}
NUMERIC_RANGES = {  # (minimum, maximum) accepted for each numeric feature
    "Product_Weight": (1, 30),
    "Product_Allocated_Area": (0.001, 0.5),
    "Product_MRP": (1, 400),
    "Store_Age_Years": (0, 60),
}


def show(value):
    """Plain Python representation of a value (numpy scalars would print as np.int64(...))."""
    return value.item() if hasattr(value, "item") else value


def validate_frame(frame):
    """Return (clean dataframe, {row: [problems]}). Rows are numbered as in the input."""
    problems = {}
    clean = frame[FEATURES].copy()

    for column, (low, high) in NUMERIC_RANGES.items():
        numbers = pd.to_numeric(clean[column], errors="coerce")
        bad = numbers.isna() | (numbers < low) | (numbers > high)
        for row in clean.index[bad]:
            problems.setdefault(row, []).append(
                f"{column}={show(clean.at[row, column])!r} must be a number between {low} and {high}"
            )
        clean[column] = numbers

    for column, allowed in ALLOWED_VALUES.items():
        bad = ~clean[column].isin(allowed)
        for row in clean.index[bad]:
            problems.setdefault(row, []).append(
                f"{column}={show(clean.at[row, column])!r} is not a known value (expected one of {allowed})"
            )

    return clean, problems


# Home route: simple health check
@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecasting API!"


# Online inference: one product-store combination per request (JSON body)
@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with the product and store features."}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    sample, problems = validate_frame(pd.DataFrame([{f: data[f] for f in FEATURES}]))
    if problems:
        return jsonify({"error": "Invalid input", "details": problems[0]}), 400

    prediction = float(model.predict(sample)[0])

    # Sales cannot be negative
    return jsonify({"Predicted_Sales": round(max(prediction, 0.0), 2)})


# Batch inference: a CSV file with one row per product-store combination
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload a CSV file in the 'file' field."}), 400

    try:
        batch = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"The file is not a valid CSV: {exc}"}), 400

    missing = [f for f in FEATURES if f not in batch.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400
    if batch.empty:
        return jsonify({"error": "The file has no rows."}), 400

    clean, problems = validate_frame(batch)
    valid_rows = [row for row in clean.index if row not in problems]
    if not valid_rows:
        return jsonify({"error": "No valid rows in the file", "details": {str(r): p for r, p in problems.items()}}), 400

    # Score the valid rows; a row that cannot be scored is reported instead of guessed
    predictions = np.clip(model.predict(clean.loc[valid_rows]).astype(float), 0, None).round(2)
    result = {str(row): float(pred) for row, pred in zip(valid_rows, predictions)}
    for row, issues in problems.items():
        result[str(row)] = "rejected: " + "; ".join(issues)

    # {row index: predicted sales (or the reason why the row was rejected)}
    return jsonify(dict(sorted(result.items(), key=lambda item: int(item[0]))))


# Run the app locally (inside the containers gunicorn is used instead, see the Dockerfile)
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
