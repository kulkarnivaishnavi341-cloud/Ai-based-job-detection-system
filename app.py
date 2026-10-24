from datetime import datetime, timedelta
from pathlib import Path
from contextlib import closing
import hashlib
import os
import re
import secrets
import shutil
import sqlite3

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, make_response
)
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------
# Application and Paths Configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
IS_VERCEL = bool(os.environ.get("VERCEL"))

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
app.config["SESSION_COOKIE_SECURE"] = IS_VERCEL  # Vercel always serves HTTPS

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
VECTORIZER_PATH = BASE_DIR / "rf_tfidf_vectorizer.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "fake_job_postings.csv"

# FIX 1: Vercel's file system is read-only except /tmp.
# On Vercel the database lives in /tmp (temporary: it is wiped on cold starts).
if IS_VERCEL:
    DATABASE_PATH = Path("/tmp") / "jobs.db"
    _seed_db = BASE_DIR / "jobs.db"
    if not DATABASE_PATH.exists() and _seed_db.is_file():
        try:
            shutil.copy(_seed_db, DATABASE_PATH)
        except OSError as exc:
            print(f"WARNING: Could not copy seed database: {exc}")
else:
    DATABASE_PATH = BASE_DIR / "jobs.db"

ADMIN_EMAIL = "admin@jobdetect.com"
MAX_OTP_ATTEMPTS = 5

# ---------------------------------------------------------------------
# Flask-Mail Configuration (Gmail SMTP)
# Set these as environment variables, never hardcode them in the code.
# ---------------------------------------------------------------------
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD", "")
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

# FIX 2: Load the reference dataset lazily (only when first needed) and only
# the columns that are used. This keeps cold starts fast and memory low.
_dataset_cache = {"loaded": False, "df": None}
_DATASET_COLUMNS = {"title", "company_profile", "fraudulent"}


def get_dataset():
    if not _dataset_cache["loaded"]:
        _dataset_cache["loaded"] = True
        try:
            if DATASET_PATH.is_file():
                _dataset_cache["df"] = pd.read_csv(
                    DATASET_PATH, usecols=lambda c: c in _DATASET_COLUMNS
                )
                print(f"Dataset loaded: {_dataset_cache['df'].shape}")
            else:
                print(f"INFO: Optional dataset not found: {DATASET_PATH}")
        except Exception as exc:
            print(f"WARNING: Dataset could not be loaded: {exc}")
            _dataset_cache["df"] = None
    return _dataset_cache["df"]


# ---------------------------------------------------------------------
# Database Control & Initialization
# ---------------------------------------------------------------------
def get_db():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    return connection


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
                CREATE TABLE IF NOT EXISTS password_resets (
                    email TEXT PRIMARY KEY,
                    otp_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Create the admin account from an environment variable (optional)
            admin_password = os.environ.get("ADMIN_PASSWORD")
            if admin_password:
                exists = connection.execute(
                    "SELECT 1 FROM users WHERE email = ?", (ADMIN_EMAIL,)
                ).fetchone()
                if not exists:
                    connection.execute(
                        "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
                        ("Admin", ADMIN_EMAIL, generate_password_hash(admin_password),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    )
            connection.commit()
        print(f"Database ready: {DATABASE_PATH}")
    except Exception as e:
        print(f"Database Initialization Error: {e}")


# Create tables once when the app starts (safe to run repeatedly)
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
    """Whole-word / whole-phrase match so 'otp' does not match 'hotpot'."""
    pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
    return re.search(pattern, text) is not None


def detect_suspicious_indicators(text):
    text = (text or "").casefold()
    found = []
    for reason, keywords in SUSPICIOUS_PATTERNS.items():
        if any(_contains_keyword(text, keyword) for keyword in keywords):
            found.append(reason)
    return found


def calculate_risk_score(fake_probability, suspicious_reasons):
    adjustment = min(len(suspicious_reasons), 4) * 5
    return round(max(0.0, min(100.0, float(fake_probability) + adjustment)), 2)


def get_risk_level(risk_score):
    if risk_score >= 70:
        return "HIGH RISK"
    if risk_score >= 40:
        return "MEDIUM RISK"
    return "LOW RISK"


def get_dataset_information(job_title, company_name):
    dataset = get_dataset()
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


def save_analysis(job_title, company_name, job_description, requirements, salary,
                  prediction, fake_probability, risk_score, risk_level, suspicious_reasons):
    try:
        with closing(get_db()) as connection:
            connection.execute("""
                INSERT INTO job_analysis (
                    user_id, job_title, company_name, job_description, requirements,
                    salary, prediction, fake_probability, risk_score, risk_level,
                    suspicious_indicators, analysis_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.get("user_id"), job_title, company_name, job_description,
                requirements, salary, prediction, fake_probability, risk_score,
                risk_level, ", ".join(suspicious_reasons),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
            connection.commit()
    except sqlite3.Error as exc:
        app.logger.exception("Could not save analysis: %s", exc)


def login_required():
    return "user_id" in session


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
# Web Routes
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


# FIX 3: /dashboard used to return "Not Found". It now sends each visitor
# to the right page.
@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))
    if session.get("user") == ADMIN_EMAIL:
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("analyze"))


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
            with closing(get_db()) as connection:
                connection.execute(
                    "INSERT INTO contacts (name, email, message, submitted_at) VALUES (?, ?, ?, ?)",
                    (name, email, message, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                )
                connection.commit()
            flash("Thank you! Your message has been sent successfully.")
        except Exception as e:
            print("Contact error:", e)
            flash("Something went wrong. Please try again.")
        return redirect(url_for("contact"))

    return render_template("contact.html")


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
            with closing(get_db()) as connection:
                connection.execute(
                    "INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
                    (name, email, generate_password_hash(password),
                     datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                )
                connection.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("This email is already registered.", "error")
        except sqlite3.Error:
            flash("Registration failed. Please try again.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    saved_email = request.cookies.get("remembered_email", "")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember")

        if not email or not password:
            flash("Please enter email and password.")
            return redirect(url_for("login"))

        try:
            with closing(get_db()) as connection:
                user = connection.execute(
                    "SELECT * FROM users WHERE email = ?", (email,)
                ).fetchone()

            if user and check_password_hash(user["password"], password):
                session.clear()
                session["user_id"] = user["id"]
                session["user"] = user["email"]
                session["user_name"] = user["name"]
                flash("Login successful!")

                if user["email"] == ADMIN_EMAIL:
                    response = make_response(redirect(url_for("admin_dashboard")))
                else:
                    response = make_response(redirect(url_for("analyze")))

                if remember:
                    response.set_cookie(
                        "remembered_email", email,
                        max_age=30 * 24 * 60 * 60, httponly=True, samesite="Lax",
                        secure=IS_VERCEL,
                    )
                else:
                    response.delete_cookie("remembered_email")
                return response

            flash("Invalid email or password.")
            return redirect(url_for("login"))

        except Exception as e:
            print("Login error:", e)
            flash("Login failed. Please try again.")
            return redirect(url_for("login"))

    return render_template("login.html", saved_email=saved_email)


@app.route("/admin")
def admin_dashboard():
    """Minimal admin page so the login redirect to 'admin_dashboard' works."""
    if session.get("user") != ADMIN_EMAIL:
        return redirect(url_for("login"))
    try:
        with closing(get_db()) as connection:
            analyses = connection.execute(
                "SELECT * FROM job_analysis ORDER BY id DESC LIMIT 100"
            ).fetchall()
    except sqlite3.Error:
        analyses = []
    try:
        return render_template("admin_dashboard.html", analyses=analyses)
    except Exception:
        # If you do not have an admin template yet, fall back to the analyzer
        return redirect(url_for("analyze"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        with closing(get_db()) as connection:
            user = connection.execute(
                "SELECT id FROM users WHERE email = ?", (email,)
            ).fetchone()

            if user:
                otp = str(secrets.randbelow(900000) + 100000)
                expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()
                connection.execute(
                    "INSERT OR REPLACE INTO password_resets (email, otp_hash, expires_at, attempts) "
                    "VALUES (?, ?, ?, 0)",
                    (email, hash_otp(otp), expires_at),
                )
                connection.commit()

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
            else:
                flash("This email is not registered!", "error")

    return render_template("forgot_password.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    email = session.get("reset_email")
    if not email:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        user_otp = request.form.get("otp", "").strip()
        new_password = request.form.get("password", "")

        with closing(get_db()) as connection:
            row = connection.execute(
                "SELECT * FROM password_resets WHERE email = ?", (email,)
            ).fetchone()

            if not row:
                session.pop("reset_email", None)
                flash("Please request a new OTP.", "error")
                return redirect(url_for("forgot_password"))

            try:
                expired = datetime.now() > datetime.fromisoformat(row["expires_at"])
            except ValueError:
                expired = True

            if expired or row["attempts"] >= MAX_OTP_ATTEMPTS:
                connection.execute("DELETE FROM password_resets WHERE email = ?", (email,))
                connection.commit()
                session.pop("reset_email", None)
                flash("OTP expired or too many attempts. Please request a new one.", "error")
                return redirect(url_for("forgot_password"))

            connection.execute(
                "UPDATE password_resets SET attempts = attempts + 1 WHERE email = ?", (email,)
            )
            connection.commit()

            if not secrets.compare_digest(hash_otp(user_otp), row["otp_hash"]):
                flash("Invalid OTP entered!", "error")
                return render_template("verify_otp.html")

            if len(new_password) < 6:
                flash("Minimum 6 characters required for the password.", "error")
                return render_template("verify_otp.html")

            try:
                connection.execute(
                    "UPDATE users SET password = ? WHERE email = ?",
                    (generate_password_hash(new_password), email),
                )
                connection.execute("DELETE FROM password_resets WHERE email = ?", (email,))
                connection.commit()
                session.pop("reset_email", None)
                flash("Password changed successfully!", "success")
                return redirect(url_for("login"))
            except Exception as e:
                print("Password reset error:", e)
                flash("An error occurred while changing the password.", "error")

    return render_template("verify_otp.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if not login_required():
        return redirect(url_for("login"))

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
    risk_score = calculate_risk_score(fake_probability, suspicious_reasons)
    risk_level = get_risk_level(risk_score)
    result = "FAKE / SUSPICIOUS" if predicted_class == 1 else "GENUINE"
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
        recommendation=["Verify details."],
        dataset_information=dataset_information,
    )


# ---------------------------------------------------------------------
# Local and Cloud Execution Engine
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )