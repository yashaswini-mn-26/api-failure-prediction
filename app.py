from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return jsonify({"message": "API Failure Predictor is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Validate input
        required_fields = ["response_time", "status_code", "cpu_usage", "memory_usage"]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = [[
            float(data["response_time"]),
            int(data["status_code"]),
            float(data["cpu_usage"]),
            float(data["memory_usage"])
        ]]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": "High Risk ⚠️" if prediction == 1 else "Stable ✅",
            "confidence": round(float(probability), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)