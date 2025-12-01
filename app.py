from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load saved pipelines
label_pipeline = joblib.load("best_label_pipeline.joblib")
attack_pipeline = joblib.load("best_attack_pipeline.joblib") 

# Columns expected by the pipeline (copy from X_train.columns)
FEATURE_COLS = [
    "fps",
    "fps_diff",
    "fps_roll_mean_10",
    "fps_roll_std_10",
    "resolution",
    "codec",
    "seconds_from_start",
    "hour",
    "minute",
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_cols=FEATURE_COLS)

@app.route("/predict", methods=["POST"])
def predict():
    # Read form values
    data = {}
    for col in FEATURE_COLS:
        value = request.form.get(col)

        # Very simple parsing: strings for categoricals, floats for numerics
        if col in ["resolution", "codec"]:
            data[col] = value
        else:
            try:
                data[col] = float(value)
            except:
                data[col] = 0.0

    # DataFrame with one row, columns matching training set
    X_input = pd.DataFrame([data])

    # Binary prediction (normal vs attack)
    y_label_pred = label_pipeline.predict(X_input)[0]
    y_label_proba = label_pipeline.predict_proba(X_input)[0]
    label_text = "Attack" if y_label_pred == 1 else "Normal"
    label_conf = float(np.max(y_label_proba))

    # Optional multi-class prediction
    y_attack_pred = attack_pipeline.predict(X_input)[0]
    y_attack_proba = attack_pipeline.predict_proba(X_input)[0]
    attack_conf = float(np.max(y_attack_proba))

    return render_template(
        "result.html",
        input_data=data,
        label_pred=label_text,
        label_conf=round(label_conf, 4),
        attack_pred=y_attack_pred,
        attack_conf=round(attack_conf, 4),
    )
    
if __name__ == "__main__":
    app.run(debug=True)
