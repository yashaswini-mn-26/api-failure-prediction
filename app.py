"""
SentinelAPI — Flask Backend (PRODUCTION READY)
"""

import os
import time
import logging
import sqlite3
from contextlib import contextmanager

import psutil
import joblib
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# ── LOGGING ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sentinel")

# ── APP ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── PATH FIX (IMPORTANT) ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "metrics.db")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ── LOAD MODEL ───────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    logger.warning(f"⚠️ Model not found — using rule-based: {e}")

# ── DB ───────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB Error: {e}")
        raise
    finally:
        conn.close()


def column_exists(conn, table, column):
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(col["name"] == column for col in cols)


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_time REAL,
                status_code INTEGER,
                cpu_usage REAL,
                memory_usage REAL,
                risk_score INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Auto-migration
        if not column_exists(conn, "metrics", "prediction"):
            conn.execute("ALTER TABLE metrics ADD COLUMN prediction TEXT")

        if not column_exists(conn, "metrics", "confidence"):
            conn.execute("ALTER TABLE metrics ADD COLUMN confidence REAL")

    logger.info("✅ DB READY (auto-migrated)")

# ── TIMER ─────────────────────────────────────────────
@app.before_request
def start_timer():
    g.start = time.time()


@app.after_request
def after_request(response):
    duration = (time.time() - g.start) * 1000

    if request.path != "/ingest":
        try:
            _store_metric(
                response_time=round(duration, 2),
                status_code=response.status_code,
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
            )
        except Exception as e:
            logger.error(f"Auto-ingest failed: {e}")

    return response

# ── LOGIC ─────────────────────────────────────────────
def rule_risk(rt, sc, cpu, mem):
    risk = 0
    if rt > 1000: risk += 30
    if sc >= 500: risk += 40
    if cpu > 85: risk += 20
    if mem > 85: risk += 10
    return min(risk, 100)


def _store_metric(
    response_time,
    status_code,
    cpu_usage,
    memory_usage,
    risk_score=0,
    prediction=None,
    confidence=None
):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO metrics
            (response_time, status_code, cpu_usage, memory_usage, risk_score, prediction, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (response_time, status_code, cpu_usage, memory_usage, risk_score, prediction, confidence))

# ── ROUTES ─────────────────────────────────────────────
@app.get("/")
def home():
    return jsonify({"status": "running"})


@app.post("/predict")
def predict():
    try:
        data = request.json

        rt = float(data["response_time"])
        sc = int(data["status_code"])
        cpu = float(data["cpu_usage"])
        mem = float(data["memory_usage"])

        if model:
            pred = model.predict([[rt, sc, cpu, mem]])[0]
            prob = model.predict_proba([[rt, sc, cpu, mem]])[0][1]
            risk = int(prob * 100)
            text = "High Risk ⚠️" if pred else "Stable ✅"
        else:
            risk = rule_risk(rt, sc, cpu, mem)
            prob = risk / 100
            text = "High Risk ⚠️" if risk > 50 else "Stable ✅"

        _store_metric(rt, sc, cpu, mem, risk, text, prob)

        return jsonify({
            "prediction": text,
            "risk_score": risk,
            "confidence": prob
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Invalid input"}), 400


@app.get("/history")
def history():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT response_time, status_code, cpu_usage, memory_usage,
                   risk_score, prediction, confidence, created_at
            FROM metrics
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()

    return jsonify([dict(r) for r in rows])


# ── ERROR ─────────────────────────────────────────────
@app.errorhandler(500)
def err(e):
    logger.exception("Server error")
    return jsonify({"error": "Internal server error"}), 500


# ── INIT ─────────────────────────────────────────────
init_db()

# ── START (RENDER SAFE) ─────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)