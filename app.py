from column_selector import ColumnSelector
import os
import traceback
from flask import Flask, request, jsonify, make_response
from werkzeug.exceptions import BadRequest
import pandas as pd

from model_loader import get_pipeline, get_required_features, decode_label

app = Flask(__name__)

# Optional: enable CORS if you want (uncomment)
# from flask_cors import CORS
# CORS(app)

# Load model pipeline at startup (cached inside model_loader)
PIPELINE_PACKAGE = None
try:
    PIPELINE_PACKAGE = get_pipeline()  # returns the pipeline package dict or Pipeline object
    REQUIRED_FEATURES = get_required_features()
except Exception as e:
    # Keep startup going but mark pipeline as None and log error
    PIPELINE_PACKAGE = None
    REQUIRED_FEATURES = []
    app.logger.error("Failed to load model at startup: %s", str(e))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": PIPELINE_PACKAGE is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if PIPELINE_PACKAGE is None:
        return make_response(jsonify({"error": "Model not loaded"}), 500)

    # Validate JSON body
    if not request.is_json:
        return make_response(jsonify({"error": "Expected JSON body"}), 400)

    try:
        payload = request.get_json()
    except BadRequest:
        return make_response(jsonify({"error": "Invalid JSON body"}), 400)

    if not isinstance(payload, dict):
        return make_response(jsonify({"error": "Expected JSON object (dictionary)"}), 400)

    # Required fields check
    missing = [f for f in REQUIRED_FEATURES if f not in payload]
    if missing:
        return make_response(jsonify({"error": "Missing required fields", "missing_fields": missing}), 400)

    # Validate types / convertibility
    casted = {}
    type_errors = {}
    for k in REQUIRED_FEATURES:
        v = payload.get(k)
        # Allow numeric types and numeric strings convertible to float
        if isinstance(v, (int, float)):
            casted[k] = v
        else:
            # try to convert to float
            try:
                casted_val = float(v)
                casted[k] = casted_val
            except Exception:
                type_errors[k] = f"Value for '{k}' is not numeric and cannot be converted to float: {repr(v)}"

    if type_errors:
        return make_response(jsonify({"error": "Wrong types for fields", "details": type_errors}), 400)

    # Create DataFrame ordering columns exactly as required
    try:
        df = pd.DataFrame([casted], columns=REQUIRED_FEATURES)
    except Exception as e:
        app.logger.error("Failed to create DataFrame: %s\n%s", str(e), traceback.format_exc())
        return make_response(jsonify({"error": "Internal server error while preparing input"}), 500)

    # Run pipeline: predict and predict_proba
    try:
        # The model loader returns either:
        # - a dict with 'pipeline' key (recommended), or
        # - the pipeline object directly.
        pipeline_obj = PIPELINE_PACKAGE.get("pipeline") if isinstance(PIPELINE_PACKAGE, dict) else PIPELINE_PACKAGE

        # Predictions
        preds = pipeline_obj.predict(df)
        probs = pipeline_obj.predict_proba(df)

        # single sample expected
        pred_raw = preds[0]
        # probability of predicted class
        prob = float(probs[0].max())

        # decode label if needed
        decoded = decode_label(pred_raw)

        response = {
            "prediction": decoded,
            "confidence": round(prob, 4)
        }
        return jsonify(response)

    except Exception as e:
        app.logger.error("Prediction error: %s\n%s", str(e), traceback.format_exc())
        return make_response(jsonify({"error": "Internal server error during prediction", "message": str(e)}), 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # For local dev only; in production use gunicorn as described in README
    app.run(host="0.0.0.0", port=port, debug=False)
