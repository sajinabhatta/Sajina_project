"""Flask dashboard for login risk detection project."""

from pathlib import Path
import json
import os
import sys
from flask import Flask, render_template, request, send_from_directory


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_login

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("home.html", result=None, error=None)


@app.post("/predict")
def predict():
    try:
        record = {
            "login_timestamp": request.form.get("login_timestamp"),
            "ip_address": request.form.get("ip_address"),
            "country": request.form.get("country"),
            "region": request.form.get("region"),
            "os": request.form.get("os"),
            "browser": request.form.get("browser"),
            "device_type": request.form.get("device_type"),
            "round_trip_ms": float(request.form.get("round_trip_ms", "0")),
            "login_successful": int(request.form.get("login_successful", "1")),
            "is_attack_ip": int(request.form.get("is_attack_ip", "0")),
        }
        result = predict_login(record)
        return render_template("home.html", result=result, error=None)
    except Exception as exc:  # pylint: disable=broad-except
        return render_template("home.html", result=None, error=str(exc))


@app.route("/performance")
def performance():
    metrics_file = ROOT / "reports/model_metrics.json"
    metrics = {}
    if metrics_file.exists():
        with metrics_file.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    return render_template("performance.html", metrics=metrics)


@app.route("/insights")
def insights():
    diagram_dir = ROOT / "diagrams"
    images = sorted([path.name for path in diagram_dir.glob("*.png")])
    return render_template("insights.html", images=images)


@app.route("/diagram/<path:filename>")
def diagram_image(filename: str):
    return send_from_directory(ROOT / "diagrams", filename)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # Default to 5050 so local system services using 5000 do not block startup.
    port = int(os.getenv("PORT", "5050"))
    app.run(debug=True, host="0.0.0.0", port=port)
