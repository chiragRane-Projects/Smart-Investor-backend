from flask import Flask, request, jsonify
from flask_cors import CORS
from services.stock_data import get_stock_data
from services.preprocessing import preprocess_stock_data, create_features_targets
from services.model import train_and_evaluate
from services.plot import plot_predictions
import os

app = Flask(__name__)
CORS(app)

@app.route("/api/stock", methods=["GET"])
def fetch_stock_data():
    ticker = request.args.get("ticker")
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d")

    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        data = get_stock_data(ticker, period, interval)
        return jsonify(data.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["GET"])
def predict_stock_price():
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "6mo")

    try:
        raw_data = get_stock_data(ticker, period)
        processed = preprocess_stock_data(raw_data)
        X, y = create_features_targets(processed)
        result = train_and_evaluate(X, y)

        return jsonify({
            "ticker": ticker,
            "mse": round(result["mse"], 4),
            "last_actual_price": round(result["last_actual"], 2),
            "last_predicted_price": round(result["last_predicted"], 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/predict/chart", methods=["GET"])
def predict_stock_chart():
    ticker = request.args.get("ticker", "AAPL")
    period = request.args.get("period", "6mo")

    try:
        raw_data = get_stock_data(ticker, period)
        processed = preprocess_stock_data(raw_data)
        X, y = create_features_targets(processed)
        result = train_and_evaluate(X, y)

        # ✅ FIX: Plot full y_test vs. predictions
        chart_path = plot_predictions(
            y_test=result["y_test"],
            predictions=result["predictions"],
            ticker=ticker
        )

        chart_url = f"/static/{os.path.basename(chart_path)}"
        return jsonify({
            "mse": round(result["mse"], 4),
            "chart_url": chart_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
