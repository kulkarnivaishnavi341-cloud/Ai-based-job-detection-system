from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import closing
from functools import wraps
import hashlib
import os
import re
import secrets
import sqlite3

import joblib
import pandas as pd
from flask import (
    Flask, render_template, render_template_string, request, redirect,
    url_for, flash, session, make_response, abort, get_flashed_messages
)
from flask_mail import Mail, Message
from markupsafe import escape
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------
# Application and Paths Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)

# Vercel/Render compatibility
application = app

_secret = os.environ.get("AI_JOBGUARD_SECRET_KEY")
if not _secret:
    print("WARNING: AI_JOBGUARD_SECRET_KEY is not set. Using a temporary random key; "
          "users will be logged out on every restart. Set this variable in production.")
    _secret = secrets.token_hex(32)
app.secret_key = _secret
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request size

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
VECTORIZER_PATH = BASE_DIR / "rf_tfidf_vectorizer.pkl"
DATABASE_PATH = BASE_DIR / "jobs.db"
DATASET_PATH = BASE_DIR / "dataset" / "fake_job_postings.csv"

ADMIN_EMAIL = "admin@jobdetect.com"
MAX_OTP_ATTEMPTS = 5

# ---------------------------------------------------------------------
# Flask-Mail Configuration (Direct Credentials Added)
# ---------------------------------------------------------------------
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True

app.config["MAIL_USERNAME"] = "kulkarnivaishnavi341@gmail.com"
app.config["MAIL_PASSWORD"] = "desrkwudwnhwuqvf"

app.config["MAIL_DEFAULT_SENDER"] = app.config["MAIL_USERNAME"]
mail = Mail(app)

# ---------------------------------------------------------------------
# Machine Learning Model Loading
# ---------------------------------------------------------------------
def load_pickle(path, label):
    try:
        if not path.is_file():
            print(f"WARNING: {label} file not found: {path}")
            return None
        obj = joblib.load(path)
        print(f"{label} loaded: {path}")
        return obj
    except Exception as exc:
        print(f"ERROR loading {label} ({path}): {exc}")
        return None


model = load_pickle(MODEL_PATH, "Random Forest model")
vectorizer = load_pickle(VECTORIZER_PATH, "TF-IDF vectorizer")

try:
    dataset = pd.read_csv(DATASET_PATH) if DATASET_PATH.is_file() else None
    if dataset is None:
        print(f"INFO: Optional dataset not found: {DATASET_PATH}")
    else:
        print(f"Dataset loaded: {dataset.shape}")
except Exception as exc:
    print(f"WARNING: Dataset could not be loaded: {exc}")
    dataset = None

# ---------------------------------------------------------------------
# Database Helpers
# ---------------------------------------------------------------------
def get_db():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def query_all(sql, params=()):
    with closing(get_db()) as connection:
        return connection.execute(sql, params).fetchall()


def query_one(sql, params=()):
    with closing(get_db()) as connection:
        return connection.execute(sql, params).fetchone()


def execute(sql, params=()):
    with closing(get_db()) as connection:
        cursor = connection.execute(sql, params)
        connection.commit()
        return cursor.rowcount


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


DEFAULT_SETTINGS = {"site_name": "AI JobGuard", "maintenance_mode": "OFF"}


def get_settings():
    settings = dict(DEFAULT_SETTINGS)
    try:
        for row in query_all("SELECT key, value FROM app_settings"):
            settings[row["key"]] = row["value"]
    except sqlite3.Error:
        pass
    return settings


def set_setting(key, value):
    execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (key, value))


def set_admin_password(connection, password):
    hashed = generate_password_hash(password)
    cursor = connection.execute(
        "UPDATE users SET password = ? WHERE email = ?", (hashed, ADMIN_EMAIL)
    )
    if cursor.rowcount == 0:
        connection.execute(
            "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
            ("Admin", ADMIN_EMAIL, hashed, now_str()),
        )


def init_database():
    try:
        with closing(get_db()) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TEXT
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS job_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    job_title TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    job_description TEXT,
                    requirements TEXT,
                    salary TEXT,
                    prediction TEXT NOT NULL,
                    fake_probability REAL,
                    risk_score REAL,
                    risk_level TEXT,
                    suspicious_indicators TEXT,
                    analysis_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    message TEXT NOT NULL,
                    submitted_at TEXT
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)", (key, value)
                )
            connection.execute("""
                CREATE TABLE IF NOT EXISTS password_resets (
                    email TEXT PRIMARY KEY,
                    otp_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
            """)

            admin_password = os.environ.get("ADMIN_PASSWORD")
            if admin_password:
                set_admin_password(connection, admin_password)
            connection.commit()
        print(f"Database ready: {DATABASE_PATH}")
    except Exception as e:
        print(f"Database Initialization Error: {e}")


init_database()

# ---------------------------------------------------------------------
# Risk Analysis Helpers
# ---------------------------------------------------------------------
SUSPICIOUS_PATTERNS = {
    "Registration fee requested": ("registration fee", "registration fees", "registration charge", "registration payment"),
    "Application fee requested": ("application fee", "application fees", "application charge"),
    "Processing fee requested": ("processing fee", "processing fees", "processing charge"),
    "Security deposit requested": ("security deposit", "security money"),
    "Deposit requested": ("deposit money", "pay deposit", "deposit required"),
    "Payment requested": ("payment required", "payment requested", "pay money", "pay a fee", "make a payment", "send money", "pay registration"),
    "Guaranteed job claim": ("guaranteed job", "job guaranteed", "guaranteed employment", "100% job guarantee", "100% guaranteed job", "guaranteed placement"),
    "No interview required": ("no interview", "without interview", "interview not required", "no interview required"),
    "Unrealistically high salary claim": ("earn 1 lakh", "earn ₹1 lakh", "earn rs 100000", "earn 100000", "earn thousands per day", "huge salary", "earn lakhs per month", "earn lakhs daily", "high salary", "unlimited earning"),
    "Bank account information requested": ("bank account", "bank details", "bank information", "account number"),
    "Credit/debit card information requested": ("credit card", "card details", "debit card"),
    "OTP requested": ("otp", "one time password", "one-time password"),
    "Password requested": ("password", "login password", "account password"),
    "Urgent hiring claim": ("urgent hiring", "apply immediately", "limited vacancy", "limited vacancies", "apply now", "join immediately", "immediate joining"),
    "Sensitive personal information requested": ("aadhaar number", "aadhar number", "pan card", "pan number", "passport number", "personal identification"),
}


def _contains_keyword(text, keyword):
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text) is not None


def detect_suspicious_indicators(text):
    text = (text or "").casefold()
    return [
        reason for reason, keywords in SUSPICIOUS_PATTERNS.items()
        if any(_contains_keyword(text, k) for k in keywords)
    ]


def calculate_risk_score(fake_probability, suspicious_reasons):
    adjustment = min(len(suspicious_reasons), 4) * 5
    return round(max(0.0, min(100.0, float(fake_probability) + adjustment)), 2)


def get_risk_level(risk_score):
    if risk_score >= 70:
        return "HIGH RISK"
    if risk_score >= 40:
        return "MEDIUM RISK"
    return "LOW RISK"


def get_recommendations(risk_level):
    if risk_level == "HIGH RISK":
        return [
            "Do NOT pay any fee, deposit or registration charge.",
            "Do not share Aadhaar, PAN, bank details, OTP or passwords.",
            "Verify the company on its official website and LinkedIn page.",
            "Report the posting on the job platform and at cybercrime.gov.in.",
        ]
    if risk_level == "MEDIUM RISK":
        return [
            "Verify the company on its official website and LinkedIn page.",
            "Contact HR using an official company email address.",
            "Never pay money to get a job or interview.",
        ]
    return [
        "Looks low risk, but still verify the company before sharing personal documents.",
        "Apply only through the official careers page when possible.",
    ]


def get_dataset_information(job_title, company_name):
    if dataset is None:
        return {"found": False, "message": "Reference dataset is not available."}
    try:
        title_match = pd.Series(False, index=dataset.index)
        company_match = pd.Series(False, index=dataset.index)
        if "title" in dataset.columns and job_title:
            title_match = (
                dataset["title"].fillna("").astype(str).str.casefold()
                .str.contains(job_title.casefold(), regex=False)
            )
        if "company_profile" in dataset.columns and company_name:
            company_match = (
                dataset["company_profile"].fillna("").astype(str).str.casefold()
                .str.contains(company_name.casefold(), regex=False)
            )
        matches = dataset[title_match | company_match]
        if matches.empty:
            return {"found": False, "message": "No similar record found in the reference dataset."}
        fraud_count = 0
        if "fraudulent" in matches.columns:
            fraud_count = int(pd.to_numeric(matches["fraudulent"], errors="coerce").fillna(0).sum())
        return {
            "found": True,
            "total_matches": int(len(matches)),
            "fraudulent_matches": fraud_count,
            "message": f"{len(matches)} similar dataset record(s) found; {fraud_count} marked fraudulent.",
        }
    except Exception as exc:
        print(f"Dataset comparison error: {exc}")
        return {"found": False, "message": "Dataset comparison could not be completed."}


def build_chart_data(user_id=None):
    conditions, params = [], []
    if user_id is not None:
        conditions.append("user_id = ?")
        params.append(user_id)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    t = query_one(f"""
        SELECT SUM(CASE WHEN prediction = 'GENUINE' THEN 1 ELSE 0 END) AS genuine,
               SUM(CASE WHEN prediction LIKE 'FAKE%' THEN 1 ELSE 0 END) AS fake,
               SUM(CASE WHEN risk_level = 'HIGH RISK' THEN 1 ELSE 0 END) AS high,
               SUM(CASE WHEN risk_level = 'MEDIUM RISK' THEN 1 ELSE 0 END) AS medium,
               SUM(CASE WHEN risk_level = 'LOW RISK' THEN 1 ELSE 0 END) AS low
        FROM job_analysis {where}
    """, params)

    days = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(13, -1, -1)]
    day_conditions = conditions + ["substr(analysis_date, 1, 10) >= ?"]
    per_day = {
        r["d"]: r["n"] for r in query_all(
            "SELECT substr(analysis_date, 1, 10) AS d, COUNT(*) AS n FROM job_analysis "
            "WHERE " + " AND ".join(day_conditions) + " GROUP BY d",
            params + [days[0]])
    }

    ind_conditions = conditions + ["suspicious_indicators IS NOT NULL", "suspicious_indicators != ''"]
    indicator_counter = Counter()
    for r in query_all("SELECT suspicious_indicators FROM job_analysis WHERE "
                       + " AND ".join(ind_conditions), params):
        for item in r["suspicious_indicators"].split(","):
            item = item.strip()
            if item:
                indicator_counter[item] += 1
    top_indicators = indicator_counter.most_common(6)

    return {
        "prediction": {"labels": ["Genuine", "Fake / Suspicious"],
                       "values": [t["genuine"] or 0, t["fake"] or 0]},
        "risk": {"labels": ["High Risk", "Medium Risk", "Low Risk"],
                 "values": [t["high"] or 0, t["medium"] or 0, t["low"] or 0]},
        "daily": {"labels": [d[5:] for d in days],
                  "values": [per_day.get(d, 0) for d in days]},
        "indicators": {"labels": [k for k, _ in top_indicators],
                       "values": [v for _, v in top_indicators]},
    }


def save_analysis(job_title, company_name, job_description, requirements, salary,
                  prediction, fake_probability, risk_score, risk_level, suspicious_reasons):
    try:
        execute("""
            INSERT INTO job_analysis (
                user_id, job_title, company_name, job_description, requirements,
                salary, prediction, fake_probability, risk_score, risk_level,
                suspicious_indicators, analysis_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.get("user_id"), job_title, company_name, job_description,
            requirements, salary, prediction, fake_probability, risk_score,
            risk_level, ", ".join(suspicious_reasons), now_str(),
        ))
    except sqlite3.Error as exc:
        app.logger.exception("Could not save analysis: %s", exc)


def hash_otp(otp):
    return hashlib.sha256((otp + app.secret_key).encode("utf-8")).hexdigest()


def result_page(**context):
    defaults = {
        "result": "ERROR", "fake_probability": 0, "risk_score": 0, "risk_level": "UNKNOWN",
        "reasons": [], "recommendation": [], "job_title": "", "company_name": "",
        "job_description": "", "requirements": "", "salary": "", "dataset_information": None,
    }
    defaults.update(context)
    return render_template("result.html", **defaults)

# ---------------------------------------------------------------------
# Authentication decorators and template context
# ---------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        if session.get("user") != ADMIN_EMAIL:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    return {
        "current_user_name": session.get("user_name"),
        "current_user_email": session.get("user"),
        "is_logged_in": "user_id" in session,
        "is_admin": session.get("user") == ADMIN_EMAIL,
    }


@app.context_processor
def inject_site_settings():
    settings = get_settings()
    return {"site_name": settings["site_name"], "maintenance_mode": settings["maintenance_mode"]}


@app.after_request
def add_maintenance_banner(response):
    try:
        if (response.mimetype == "text/html" and response.status_code == 200
                and not request.path.startswith("/admin")):
            settings = get_settings()
            if settings["maintenance_mode"] == "ON":
                banner = (
                    '<div style="background:#f59e0b;color:#1f2937;text-align:center;padding:10px;'
                    'font-family:Arial,sans-serif;font-weight:bold;">'
                    f'🚧 {escape(settings["site_name"])} is under maintenance. '
                    'Some features may be unavailable.</div>'
                )
                html = response.get_data(as_text=True)
                html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + banner, html, count=1, flags=re.I)
                response.set_data(html)
    except Exception as exc:
        print("Banner error:", exc)
    return response


@app.after_request
def no_cache_for_private_pages(response):
    if "user_id" in session:
        response.headers["Cache-Control"] = "no-store"
    return response

# ---------------------------------------------------------------------
# Public Pages
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            flash("Please fill all required fields.")
            return redirect(url_for("contact"))

        try:
            execute(
                "INSERT INTO contacts (name, email, message, submitted_at) VALUES (?, ?, ?, ?)",
                (name, email, message, now_str()),
            )
            flash("Thank you! Your message has been sent successfully.")
        except Exception as e:
            print("Contact error:", e)
            flash("Something went wrong. Please try again.")
        return redirect(url_for("contact"))

    return render_template("contact.html")

# ---------------------------------------------------------------------
# Register / Login / Logout
# ---------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill in all required fields.", "error")
            return redirect(url_for("register"))
        if email == ADMIN_EMAIL:
            flash("This email is reserved. Please use a different email.", "error")
            return redirect(url_for("register"))
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "error")
            return redirect(url_for("register"))

        try:
            execute(
                "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
                (name, email, generate_password_hash(password), now_str()),
            )
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("This email is already registered.", "error")
        except sqlite3.Error:
            flash("Registration failed. Please try again.", "error")
        return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    saved_email = request.cookies.get("remembered_email", "")
    error = None
    success = None

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember")

        if not email or not password:
            error = "Please enter email and password."
        else:
            try:
                user = query_one("SELECT * FROM users WHERE email = ?", (email,))
                if user and check_password_hash(user["password"], password):
                    session.clear()
                    session["user_id"] = user["id"]
                    session["user"] = user["email"]
                    session["user_name"] = user["name"]

                    if user["email"] == ADMIN_EMAIL:
                        response = make_response(redirect(url_for("admin_dashboard")))
                    else:
                        response = make_response(redirect(url_for("dashboard")))

                    if remember:
                        response.set_cookie(
                            "remembered_email", email,
                            max_age=30 * 24 * 60 * 60, httponly=True, samesite="Lax",
                        )
                    else:
                        response.delete_cookie("remembered_email")
                    return response

                error = "Invalid email or password."
            except Exception as e:
                print("Login error:", e)
                error = "Login failed. Please try again."
    else:
        for category, message in get_flashed_messages(with_categories=True):
            if category == "error" and error is None:
                error = message
            elif category != "error" and success is None:
                success = message

    return render_template("login.html", saved_email=saved_email, error=error, success=success)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------------------------------------------------
# Forgot Password (OTP by email)
# ---------------------------------------------------------------------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = query_one("SELECT id FROM users WHERE email = ?", (email,))

        if not user:
            flash("This email is not registered!", "error")
            return render_template("forgot_password.html")

        if not app.config["MAIL_USERNAME"] or not app.config["MAIL_PASSWORD"]:
            flash("Email service is not configured. Please contact the admin.", "error")
            return render_template("forgot_password.html")

        otp = str(secrets.randbelow(900000) + 100000)
        expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()
        execute(
            "INSERT OR REPLACE INTO password_resets (email, otp_hash, expires_at, attempts) "
            "VALUES (?, ?, ?, 0)",
            (email, hash_otp(otp), expires_at),
        )

        msg = Message(
            "Password Reset OTP - AI JobGuard",
            sender=app.config["MAIL_USERNAME"],
            recipients=[email],
        )
        msg.body = (
            f"Your OTP for password reset is: {otp}\n"
            "It is valid for 10 minutes. Please do not share this code with anyone."
        )
        try:
            mail.send(msg)
            session["reset_email"] = email
            flash("OTP sent to your registered email!", "success")
            return redirect(url_for("verify_otp"))
        except Exception as e:
            print("Mail error:", e)
            flash("Error occurred while sending email.", "error")

    return render_template("forgot_password.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    email = session.get("reset_email")
    if not email:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        user_otp = request.form.get("otp", "").strip()
        new_password = request.form.get("password", "")

        row = query_one("SELECT * FROM password_resets WHERE email = ?", (email,))
        if not row:
            session.pop("reset_email", None)
            flash("Please request a new OTP.", "error")
            return redirect(url_for("forgot_password"))

        try:
            expired = datetime.now() > datetime.fromisoformat(row["expires_at"])
        except ValueError:
            expired = True

        if expired or row["attempts"] >= MAX_OTP_ATTEMPTS:
            execute("DELETE FROM password_resets WHERE email = ?", (email,))
            session.pop("reset_email", None)
            flash("OTP expired or too many attempts. Please request a new one.", "error")
            return redirect(url_for("forgot_password"))

        execute("UPDATE password_resets SET attempts = attempts + 1 WHERE email = ?", (email,))

        if not secrets.compare_digest(hash_otp(user_otp), row["otp_hash"]):
            flash("Invalid OTP entered!", "error")
            return render_template("verify_otp.html")

        if len(new_password) < 6:
            flash("Minimum 6 characters required for the password.", "error")
            return render_template("verify_otp.html")

        try:
            execute(
                "UPDATE users SET password = ? WHERE email = ?",
                (generate_password_hash(new_password), email),
            )
            execute("DELETE FROM password_resets WHERE email = ?", (email,))
            session.pop("reset_email", None)
            flash("Password changed successfully!", "success")
            return redirect(url_for("login"))
        except Exception as e:
            print("Password reset error:", e)
            flash("An error occurred while changing the password.", "error")

    return render_template("verify_otp.html")

# ---------------------------------------------------------------------
# User Dashboard
# ---------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]

    totals = query_one("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN prediction = 'GENUINE' THEN 1 ELSE 0 END) AS genuine,
               SUM(CASE WHEN prediction LIKE 'FAKE%' THEN 1 ELSE 0 END) AS fake,
               SUM(CASE WHEN risk_level = 'HIGH RISK' THEN 1 ELSE 0 END) AS high,
               SUM(CASE WHEN risk_level = 'MEDIUM RISK' THEN 1 ELSE 0 END) AS medium,
               SUM(CASE WHEN risk_level = 'LOW RISK' THEN 1 ELSE 0 END) AS low,
               AVG(fake_probability) AS avg_prob
        FROM job_analysis WHERE user_id = ?
    """, (user_id,))

    recent_analyses = query_all("""
        SELECT id, job_title, company_name, prediction, fake_probability,
               risk_score, risk_level, analysis_date
        FROM job_analysis WHERE user_id = ? ORDER BY id DESC LIMIT 5
    """, (user_id,))

    return render_template(
        "dashboard.html",
        total_jobs=totals["total"] or 0,
        genuine_jobs=totals["genuine"] or 0,
        fake_jobs=totals["fake"] or 0,
        high_risk=totals["high"] or 0,
        medium_risk=totals["medium"] or 0,
        low_risk=totals["low"] or 0,
        average_probability=round(totals["avg_prob"] or 0, 1),
        recent_analyses=recent_analyses,
        chart_data=build_chart_data(user_id),
    )

# ---------------------------------------------------------------------
# Analyze + Result (Updated with 98.5% / 99.9% Risk Display)
# ---------------------------------------------------------------------
@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    job_title = request.form.get("job_title", "").strip()
    company_name = request.form.get("company_name", "").strip()
    job_description = request.form.get("job_description", "").strip()
    requirements = request.form.get("requirements", "").strip()
    salary = request.form.get("salary", "").strip()

    common = {
        "job_title": job_title, "company_name": company_name,
        "job_description": job_description, "requirements": requirements, "salary": salary,
    }

    if not job_title or not company_name or not job_description:
        return result_page(**common, result="INVALID INPUT", reasons=["Required details missing."])

    if model is None or vectorizer is None:
        return result_page(**common, result="MODEL NOT READY", reasons=["Model files missing."])

    job_text = " ".join(
        part for part in (job_title, company_name, job_description, requirements, salary) if part
    ).strip()

    try:
        job_vector = vectorizer.transform([job_text])
        predicted_class = int(model.predict(job_vector)[0])
    except Exception as exc:
        return result_page(**common, result="PREDICTION ERROR", reasons=[str(exc)])

    fake_probability = 100.0 if predicted_class == 1 else 0.0
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(job_vector)
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                fake_probability = float(probabilities[0][classes.index(1)]) * 100.0
    except Exception:
        pass
    fake_probability = round(max(0.0, min(100.0, fake_probability)), 2)

    suspicious_reasons = detect_suspicious_indicators(job_text)
    
    # ---------------------------------------------------------------------
    # CRITICAL SECURITY FIX: STRICT 98.5% OR 99.9% RISK OVERRIDE
    # ---------------------------------------------------------------------
    severe_indicators = {
        "OTP requested", 
        "Registration fee requested", 
        "Processing fee requested", 
        "Bank account information requested",
        "Sensitive personal information requested"
    }
    has_severe_threat = any(ind in suspicious_reasons for ind in severe_indicators)

    if has_severe_threat:
        predicted_class = 1
        result = "FAKE / SUSPICIOUS"
        # Force a highly suspicious specific fractional display instead of exactly 100%
        fake_probability = 98.50 if "OTP requested" in suspicious_reasons else 99.90
        risk_score = 98.50 if "OTP requested" in suspicious_reasons else 99.90
        risk_level = "HIGH RISK"
    else:
        risk_score = calculate_risk_score(fake_probability, suspicious_reasons)
        # Cap safe sub-scores below absolute 100 for better realism
        if predicted_class == 1 or risk_score >= 70:
            risk_score = min(risk_score, 98.50)
        risk_level = get_risk_level(risk_score)
        result = "FAKE / SUSPICIOUS" if (predicted_class == 1 or risk_score >= 40) else "GENUINE"

    dataset_information = get_dataset_information(job_title, company_name)

    save_analysis(
        job_title, company_name, job_description, requirements, salary,
        result, fake_probability, risk_score, risk_level, suspicious_reasons,
    )

    return result_page(
        **common,
        result=result,
        fake_probability=fake_probability,
        risk_score=risk_score,
        risk_level=risk_level,
        reasons=suspicious_reasons,
        recommendation=get_recommendations(risk_level),
        dataset_information=dataset_information,
    )

# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------
@app.route("/history")
@login_required
def history():
    analyses = query_all("""
        SELECT id, job_title, company_name, prediction, fake_probability,
               risk_score, risk_level, suspicious_indicators, analysis_date
        FROM job_analysis WHERE user_id = ? ORDER BY id DESC LIMIT 500
    """, (session["user_id"],))
    return render_template("history.html", analyses=analyses)

# ---------------------------------------------------------------------
# Admin Panel
# ---------------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_dashboard():
    counts = {
        "users_count": query_one("SELECT COUNT(*) AS n FROM users WHERE email != ?", (ADMIN_EMAIL,))["n"],
        "logs_count": query_one("SELECT COUNT(*) AS n FROM job_analysis")["n"],
        "messages_count": query_one("SELECT COUNT(*) AS n FROM contacts")["n"],
    }

    chart_data = build_chart_data()
    return render_template("admin.html", counts=counts, chart_data=chart_data)


@app.route("/admin/users")
@admin_required
def admin_users():
    users = query_all(
        "SELECT id, name, email, created_at FROM users WHERE email != ? ORDER BY id DESC",
        (ADMIN_EMAIL,),
    )
    return render_template("admin_users.html", users=users)


@app.route("/admin/delete_user/<int:user_id>", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    user = query_one("SELECT email FROM users WHERE id = ?", (user_id,))
    if user is None:
        flash("User not found.")
    elif user["email"] == ADMIN_EMAIL:
        flash("The admin account cannot be deleted.")
    else:
        execute("DELETE FROM job_analysis WHERE user_id = ?", (user_id,))
        execute("DELETE FROM users WHERE id = ?", (user_id,))
        flash(f"User #{user_id} and all their analysis history were deleted.")
    return redirect(url_for("admin_users"))


@app.route("/admin/job-logs")
@admin_required
def admin_job_logs():
    analyses = query_all("SELECT * FROM job_analysis ORDER BY id DESC LIMIT 1000")
    return render_template("admin_logs.html", analyses=analyses)


@app.route("/admin/delete_log/<int:log_id>", methods=["POST"])
@admin_required
def admin_delete_log(log_id):
    execute("DELETE FROM job_analysis WHERE id = ?", (log_id,))
    flash(f"Analysis log #{log_id} deleted.")
    return redirect(url_for("admin_job_logs"))


@app.route("/admin/messages")
@admin_required
def admin_messages():
    contacts = query_all("SELECT * FROM contacts ORDER BY id DESC LIMIT 1000")
    return render_template("admin_messages.html", contacts=contacts)


@app.route("/admin/delete_message/<int:message_id>", methods=["POST"])
@admin_required
def admin_delete_message(message_id):
    execute("DELETE FROM contacts WHERE id = ?", (message_id,))
    flash("Enquiry deleted.")
    return redirect(url_for("admin_messages"))


@app.route("/admin/settings", methods=["GET", "POST"])
@admin_required
def admin_settings():
    if request.method == "POST":
        site_name = request.form.get("site_name", "").strip()[:80] or DEFAULT_SETTINGS["site_name"]
        maintenance = request.form.get("maintenance_mode", "OFF")
        if maintenance not in ("ON", "OFF"):
            maintenance = "OFF"
        set_setting("site_name", site_name)
        set_setting("maintenance_mode", maintenance)
        flash("Settings saved successfully.")
        return redirect(url_for("admin_settings"))
    return render_template("admin_settings.html", settings=get_settings())


@app.route("/admin/reset_database", methods=["POST"])
@admin_required
def admin_reset_database():
    execute("DELETE FROM job_analysis")
    execute("DELETE FROM contacts")
    flash("All job analysis logs and contact enquiries were deleted. User accounts were kept.")
    return redirect(url_for("admin_settings"))

# ---------------------------------------------------------------------
# Error Pages
# ---------------------------------------------------------------------
ERROR_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>{{ code }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font-family:Arial,sans-serif;background:#f4f7fb;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0}.box{text-align:center}h1{font-size:72px;margin:0;color:#182848}
a{color:#182848}</style></head><body><div class="box"><h1>{{ code }}</h1><p>{{ message }}</p>
<a href="{{ url_for('home') }}">Go to Home</a></div></body></html>
"""


@app.errorhandler(403)
def forbidden(_e):
    return render_template_string(ERROR_HTML, code=403, message="You do not have access to this page."), 403


@app.errorhandler(404)
def not_found(_e):
    return render_template_string(ERROR_HTML, code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template_string(ERROR_HTML, code=500, message="Something went wrong on our side."), 500


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "reset-admin":
        import getpass
        new_password = getpass.getpass(f"New password for {ADMIN_EMAIL}: ").strip()
        if len(new_password) < 6:
            print("Password must be at least 6 characters.")
        else:
            with closing(get_db()) as connection:
                set_admin_password(connection, new_password)
                connection.commit()
            print(f"Password set for {ADMIN_EMAIL}. Now run: python app.py")
        sys.exit(0)

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
