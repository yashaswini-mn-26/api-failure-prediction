"""
SentinelAPI v3 — Complete Backend
Existing: /predict, /ingest, /history, /stats, /health
New:      /github/analyze, /proxy/test, /proxy/test-all, /endpoint-metrics/<project_id>
"""

import os, re, time, logging, sqlite3, uuid
from contextlib import contextmanager
import requests as req_lib
import psutil, joblib
from flask import Flask, request, jsonify, g
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sentinel")

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {"origins": "http://localhost:3000"},
    r"/proxy/*": {"origins": "http://localhost:3000"},
    r"/github/*": {"origins": "http://localhost:3000"},
    r"/history": {"origins": "http://localhost:3000"},
})

DATABASE  = os.environ.get("DATABASE_PATH", "metrics.db")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded: %s", MODEL_PATH)
except:
    model = None
    logger.warning("model.pkl not found — rule-based fallback active")

# ── DB ────────────────────────────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn; conn.commit()
    except:
        conn.rollback(); raise
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS metrics (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                response_time REAL    NOT NULL,
                status_code   INTEGER NOT NULL,
                cpu_usage     REAL    NOT NULL,
                memory_usage  REAL    NOT NULL,
                risk_score    INTEGER NOT NULL DEFAULT 0,
                prediction    TEXT,
                confidence    REAL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_created ON metrics(created_at DESC);

            CREATE TABLE IF NOT EXISTS endpoint_metrics (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id       TEXT    NOT NULL,
                endpoint_path    TEXT    NOT NULL,
                method           TEXT    NOT NULL,
                status_code      INTEGER NOT NULL,
                response_time_ms REAL    NOT NULL,
                risk_score       INTEGER NOT NULL DEFAULT 0,
                tested_at        TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ep_project ON endpoint_metrics(project_id, tested_at DESC);
        """)
    logger.info("DB ready at %s", DATABASE)

# ── MIDDLEWARE ────────────────────────────────────────────────────────────────

@app.before_request
def start_timer():
    g.start_time = time.perf_counter()

@app.after_request
def after_request(response):
    dur = (time.perf_counter() - g.start_time) * 1000
    response.headers["X-Response-Time"] = f"{dur:.2f}ms"
    response.headers["X-Content-Type-Options"] = "nosniff"
    if request.path not in ("/ingest", "/proxy/test", "/proxy/test-all"):
        try:
            _store_metric(round(dur, 2), response.status_code,
                          psutil.cpu_percent(interval=None),
                          psutil.virtual_memory().percent)
        except Exception as e:
            logger.error("Auto-ingest failed: %s", e)
    return response

# ── SCORING LOGIC ─────────────────────────────────────────────────────────────

def rule_based_risk(rt, sc, cpu, mem):
    risk = 0
    if rt > 1000:  risk += 30
    if sc >= 500:  risk += 40
    if cpu > 85:   risk += 20
    if mem > 85:   risk += 10
    return min(risk, 100)

def calculate_risk(prob, rt, sc, cpu):
    risk = prob * 100
    if rt > 1000: risk += 10
    if sc >= 500: risk += 20
    if cpu > 85:  risk += 10
    return min(round(risk), 100)

def generate_suggestion(rt, sc, cpu, mem):
    s = []
    if rt > 1000:  s.append("High response time — optimise queries")
    if sc >= 500:  s.append("Server errors — check backend logs")
    if cpu > 85:   s.append("High CPU — consider scaling")
    if mem > 85:   s.append("High memory — check for leaks")
    return " | ".join(s) if s else "System operating normally"

def _store_metric(rt, sc, cpu, mem, risk=0, pred=None, conf=None):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO metrics (response_time,status_code,cpu_usage,memory_usage,risk_score,prediction,confidence) VALUES (?,?,?,?,?,?,?)",
            (rt, sc, cpu, mem, risk, pred, conf)
        )

# ── EXISTING ROUTES ───────────────────────────────────────────────────────────

@app.get("/")
def home():
    return jsonify({"service": "SentinelAPI", "version": "3.0.0", "model_loaded": model is not None})

@app.get("/health")
def health():
    return jsonify({"status": "ok", "timestamp": time.time()})

@app.post("/ingest")
def ingest():
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "JSON required"}), 400
    required = ["response_time","status_code","cpu_usage","memory_usage"]
    missing = [f for f in required if f not in data]
    if missing: return jsonify({"error": f"Missing: {missing}"}), 400
    try:
        rt, sc, cpu, mem = float(data["response_time"]), int(data["status_code"]), float(data["cpu_usage"]), float(data["memory_usage"])
    except: return jsonify({"error": "Invalid types"}), 400
    risk = rule_based_risk(rt, sc, cpu, mem)
    _store_metric(rt, sc, cpu, mem, risk_score=risk)
    return jsonify({"message": "stored", "risk_score": risk}), 201

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "JSON required"}), 400
    try:
        rt, sc, cpu, mem = float(data["response_time"]), int(data["status_code"]), float(data["cpu_usage"]), float(data["memory_usage"])
    except: return jsonify({"error": "Invalid types"}), 400

    if model:
        pred_label = model.predict([[rt,sc,cpu,mem]])[0]
        prob = float(model.predict_proba([[rt,sc,cpu,mem]])[0][1])
        risk = calculate_risk(prob, rt, sc, cpu)
        pred_text = "High Risk ⚠️" if pred_label == 1 else "Stable ✅"
    else:
        risk = rule_based_risk(rt, sc, cpu, mem)
        prob = risk / 100.0 
        pred_text = "High Risk ⚠️" if risk > 50 else "Stable ✅"

    suggestion = generate_suggestion(rt, sc, cpu, mem)
    _store_metric(rt, sc, cpu, mem, risk=risk, pred=pred_text, conf=round(prob, 4))
    return jsonify({"prediction": pred_text, "confidence": round(prob,4), "risk_score": risk, "suggestion": suggestion})

@app.get("/history")
def history():
    try:
        limit  = min(int(request.args.get("limit",  50)), 500)
        offset = max(int(request.args.get("offset",  0)),   0)
    except: return jsonify({"error": "limit/offset must be integers"}), 400
    with get_db() as conn:
        rows = conn.execute(
            "SELECT response_time,status_code,cpu_usage,memory_usage,risk_score,prediction,confidence,created_at FROM metrics ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.get("/stats")
def stats():
    n = min(int(request.args.get("n", 100)), 1000)
    with get_db() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS total, ROUND(AVG(response_time),2) AS avg_response_time,
                   ROUND(AVG(cpu_usage),2) AS avg_cpu, ROUND(AVG(memory_usage),2) AS avg_memory,
                   ROUND(AVG(risk_score),2) AS avg_risk, MAX(risk_score) AS max_risk,
                   SUM(CASE WHEN risk_score > 60 THEN 1 ELSE 0 END) AS high_risk_count
            FROM (SELECT response_time,cpu_usage,memory_usage,risk_score FROM metrics ORDER BY id DESC LIMIT ?)
        """, (n,)).fetchone()
    return jsonify(dict(row) if row else {})

# ── GITHUB ANALYZER ───────────────────────────────────────────────────────────

# Regex patterns per framework
ROUTE_PATTERNS = {
    "flask":   [
        r'@\w+\.route\(["\']([^"\']+)["\'][^)]*\)',                           # @app.route('/path')
        r'@\w+\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',            # @app.get('/path')
    ],
    "fastapi": [
        r'@\w+\.(get|post|put|delete|patch|options)\(["\']([^"\']+)["\']',
    ],
    "django":  [
        r'path\(["\']([^"\']+)["\']',                                          # path('endpoint/', view)
        r'url\(["\']([^"\']+)["\']',                                           # url(r'^endpoint/')
        r're_path\(["\']([^"\']+)["\']',
    ],
    "express": [
        r'(?:app|router)\.(get|post|put|delete|patch|use)\(["\`]([^"\'`]+)["\`]',
    ],
    "rails":   [
        r'(?:get|post|put|delete|patch)\s+["\']([^"\']+)["\']',
    ],
}

HTTP_METHODS_BY_FRAMEWORK = {
    "flask":   {"@app.get": "GET", "@app.post": "POST", "@app.put": "PUT", "@app.delete": "DELETE", "@app.patch": "PATCH"},
    "fastapi": {},
    "django":  {"GET": "GET"},
    "express": {},
    "rails":   {},
}

FRAMEWORK_FILES = {
    "flask":   ["app.py", "routes.py", "views.py", "api.py", "main.py"],
    "fastapi": ["main.py", "app.py", "router.py", "api.py"],
    "django":  ["urls.py"],
    "express": ["app.js", "routes.js", "index.js", "server.js", "app.ts", "routes.ts"],
    "rails":   ["routes.rb"],
}

FRAMEWORK_INDICATORS = {
    "flask":   ["flask", "Flask(", "from flask"],
    "fastapi": ["fastapi", "FastAPI(", "from fastapi"],
    "django":  ["django", "urlpatterns", "from django"],
    "express": ["express()", "require('express')", "require(\"express\")"],
    "rails":   ["Rails.application.routes"],
}

def detect_framework(files_map: dict) -> str:
    """Detect framework from file names and content."""
    for fw, indicators in FRAMEWORK_INDICATORS.items():
        for fname, content in files_map.items():
            for ind in indicators:
                if ind.lower() in content.lower():
                    return fw
    # Fallback: check filenames
    if any("urls.py" in f for f in files_map): return "django"
    if any("routes.rb" in f for f in files_map): return "rails"
    return "unknown"

def parse_routes_from_content(content: str, filename: str, framework: str) -> list:
    """Extract routes from file content using regex per framework."""
    routes = []
    patterns = ROUTE_PATTERNS.get(framework, [])

    for pat in patterns:
        for m in re.finditer(pat, content, re.MULTILINE):
            groups = m.groups()
            if len(groups) == 1:
                path = groups[0]
                method = "GET"
            elif len(groups) == 2:
                method_or_path, path_or_none = groups
                if method_or_path.upper() in ("GET","POST","PUT","DELETE","PATCH","OPTIONS"):
                    method = method_or_path.upper()
                    path = path_or_none
                else:
                    method = "GET"
                    path = method_or_path
            else:
                continue

            # Normalise path
            if not path.startswith("/"):
                path = "/" + path
            # Skip obviously invalid
            if any(c in path for c in ["*", "?", "[", "regex", "^$"]):
                continue
            if len(path) > 100:
                continue

            line_num = content[:m.start()].count("\n") + 1
            routes.append({"method": method, "path": path, "file": filename, "line": line_num, "framework": framework})

    return routes

def github_get(url: str, token: str | None = None) -> dict:
    """Make a GitHub API request."""
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "SentinelAPI/3.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = req_lib.get(url, headers=headers, timeout=15)
    if resp.status_code == 403:
        raise Exception("GitHub rate limit hit — add a Personal Access Token")
    if resp.status_code == 404:
        raise Exception("Repository not found — check the URL or token for private repos")
    resp.raise_for_status()
    return resp.json()

def fetch_file_content(url: str, token: str | None = None) -> str:
    """Fetch and decode a file from GitHub raw URL."""
    import base64
    data = github_get(url, token)
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    return data.get("content", "")

def fetch_repo_tree(owner: str, repo: str, token: str | None = None) -> list:
    """Get the full flat file tree of a repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    data = github_get(url, token)
    return data.get("tree", [])

@app.post("/github/analyze")
def github_analyze():
    """
    Accepts: { repo_url: str, token?: str } 
    Returns: RepoAnalysis JSON
    """
    data = request.get_json(silent=True)
    if not data or "repo_url" not in data:
        return jsonify({"error": "repo_url required"}), 400

    repo_url = data["repo_url"].strip().rstrip("/")
    token = data.get("token", "").strip() or None

    # Parse owner/repo from URL
    # Handles: https://github.com/owner/repo or github.com/owner/repo
    match = re.search(r"github\.com[/:]([^/]+)/([^/\s]+?)(?:\.git)?$", repo_url)
    if not match:
        return jsonify({"error": "Invalid GitHub URL — expected https://github.com/owner/repo"}), 400

    owner, repo = match.group(1), match.group(2)

    try:
        # 1. Get repo metadata
        repo_data = github_get(f"https://api.github.com/repos/{owner}/{repo}", token)

        # 2. Get file tree
        tree = fetch_repo_tree(owner, repo, token)

        # 3. Detect framework from file names first
        all_paths = [item["path"] for item in tree if item["type"] == "blob"]
        framework = "unknown"
        for fw, target_files in FRAMEWORK_FILES.items():
            if any(any(tf in p for p in all_paths) for tf in target_files):
                framework = fw
                break

        # 4. Find relevant files to scan (limit to 20 to avoid rate limits)
        target_names = FRAMEWORK_FILES.get(framework, ["app.py","routes.py","urls.py","app.js","routes.js","main.py"])
        route_files = [
            item for item in tree
            if item["type"] == "blob" and any(item["path"].endswith(tf) for tf in target_names)
        ][:20]

        # If framework still unknown, also scan .py and .js files for indicators
        if framework == "unknown":
            route_files = [
                item for item in tree
                if item["type"] == "blob" and item["path"].endswith((".py", ".js", ".ts", ".rb"))
            ][:15]

        # 5. Fetch file contents and detect framework more accurately
        files_map = {}
        for item in route_files:
            try:
                content = fetch_file_content(item["url"], token)
                files_map[item["path"]] = content
            except:
                pass

        if framework == "unknown" and files_map:
            framework = detect_framework(files_map)

        # 6. Parse routes from all fetched files
        all_routes = []
        seen = set()
        for fname, content in files_map.items():
            for r in parse_routes_from_content(content, fname, framework):
                key = f"{r['method']}:{r['path']}"
                if key not in seen:
                    seen.add(key)
                    all_routes.append(r)

        # 7. README summary
        readme_summary = ""
        readme_items = [i for i in tree if i["type"] == "blob" and "readme" in i["path"].lower()]
        if readme_items:
            try:
                readme_content = fetch_file_content(readme_items[0]["url"], token)
                # Take first non-empty paragraph
                lines = [l.strip() for l in readme_content.splitlines() if l.strip() and not l.startswith("#") and not l.startswith("!")]
                readme_summary = " ".join(lines[:3])[:300]
            except:
                pass

        # 8. Detect language
        lang_data = github_get(f"https://api.github.com/repos/{owner}/{repo}/languages", token)
        language = list(lang_data.keys())[0] if lang_data else repo_data.get("language", "Unknown")

        return jsonify({
            "repo_url": repo_url,
            "owner": owner,
            "repo": repo,
            "framework": framework,
            "language": language,
            "routes": all_routes,
            "files_scanned": len(files_map),
            "readme_summary": readme_summary,
            "stars": repo_data.get("stargazers_count", 0),
            "description": repo_data.get("description", ""),
        })

    except Exception as e:
        logger.error("GitHub analyze failed: %s", e)
        return jsonify({"error": str(e)}), 422

# ── PROXY TESTER ──────────────────────────────────────────────────────────────

def _compute_endpoint_risk(rt: float, sc: int) -> int:
    risk = 0
    if rt > 2000:  risk += 40
    elif rt > 1000: risk += 20
    elif rt > 500:  risk += 10
    if sc >= 500:  risk += 50
    elif sc >= 400: risk += 20
    elif sc == 0:  risk += 60  # network error
    return min(risk, 100)

def _do_proxy_request(base_url: str, method: str, path: str, headers: dict, body: any, project_id: str) -> dict:
    """Make one proxied HTTP request and return EndpointTestResult dict."""
    full_url = base_url.rstrip("/") + path
    start = time.perf_counter()
    error = None
    status_code = 0
    response_preview = ""
    response_headers = {}
    size = 0

    try:
        resp = req_lib.request(
            method=method.upper(),
            url=full_url,
            headers={**headers, "User-Agent": "SentinelAPI-Probe/3.0"},
            json=body if body else None,
            timeout=10,
            allow_redirects=True,
        )
        status_code = resp.status_code
        size = len(resp.content)
        response_headers = dict(resp.headers)
        try:
            response_preview = resp.text[:2000]
        except:
            response_preview = "<binary content>"
    except req_lib.exceptions.ConnectionError:
        error = f"Connection refused — is the app running at {base_url}?"
    except req_lib.exceptions.Timeout:
        error = "Request timed out (>10s)"
    except req_lib.exceptions.RequestException as e:
        error = str(e)

    rt_ms = round((time.perf_counter() - start) * 1000, 2)
    risk = _compute_endpoint_risk(rt_ms, status_code)

    # Persist to endpoint_metrics table
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO endpoint_metrics (project_id,endpoint_path,method,status_code,response_time_ms,risk_score) VALUES (?,?,?,?,?,?)",
                (project_id, path, method.upper(), status_code, rt_ms, risk)
            )
    except Exception as e:
        logger.error("Failed to store endpoint metric: %s", e)

    return {
        "endpoint_id": str(uuid.uuid4()),
        "project_id": project_id,
        "method": method.upper(),
        "path": path,
        "full_url": full_url,
        "status_code": status_code,
        "response_time_ms": rt_ms,
        "response_size_bytes": size,
        "response_preview": response_preview,
        "headers": response_headers,
        "error": error,
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "risk_score": risk,
    }

@app.post("/proxy/test")
def proxy_test():
    """Test a single endpoint. Body: { project_id, base_url, method, path, headers?, body? }"""
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "JSON required"}), 400
    required = ["project_id", "base_url", "method", "path"]
    missing = [f for f in required if f not in data]
    if missing: return jsonify({"error": f"Missing: {missing}"}), 400

    result = _do_proxy_request(
        base_url=data["base_url"],
        method=data["method"],
        path=data["path"],
        headers=data.get("headers", {}),
        body=data.get("body"),
        project_id=data["project_id"],
    )
    return jsonify(result)

@app.post("/proxy/test-all")
def proxy_test_all():
    """Test all routes for a project. Body: { project_id, base_url, routes: [{method,path}] }"""
    data = request.get_json(silent=True)
    if not data: return jsonify({"error": "JSON required"}), 400

    results = []
    for route in data.get("routes", []):
        result = _do_proxy_request(
            base_url=data["base_url"],
            method=route["method"],
            path=route["path"],
            headers={},
            body=None,
            project_id=data["project_id"],
        )
        results.append(result)
        time.sleep(0.05)  # small delay — don't hammer their server

    return jsonify(results)

@app.get("/endpoint-metrics/<project_id>")
def endpoint_metrics(project_id: str):
    """Get endpoint test history for a project."""
    path = request.args.get("path")
    with get_db() as conn:
        if path:
            rows = conn.execute(
                "SELECT * FROM endpoint_metrics WHERE project_id=? AND endpoint_path=? ORDER BY id DESC LIMIT 100",
                (project_id, path)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM endpoint_metrics WHERE project_id=? ORDER BY id DESC LIMIT 200",
                (project_id,)
            ).fetchall()
    return jsonify([dict(r) for r in rows])

# ── ERROR HANDLERS ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception("Server error")
    return jsonify({"error": "Internal server error"}), 500

# ── START ─────────────────────────────────────────────────────────────────────

init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info("SentinelAPI v3 starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)