from datetime import datetime
from pathlib import Path
import os
import sqlite3
import random
from flask import flash, session
from flask_mail import Mail, Message
from flask import make_response

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for
)
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------
# Application and paths
# ---------------------------------------------------------------------
app = Flask(__name__)

app.secret_key = os.environ.get("AI_JOBGUARD_SECRET_KEY", "change-this-secret-key")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
VECTORIZER_PATH = BASE_DIR / "rf_tfidf_vectorizer.pkl"
DATABASE_PATH = BASE_DIR / "jobs.db"
DATASET_PATH = BASE_DIR / "dataset" / "fake_job_postings.csv"

# Flask-Mail Config (Gmail SMTP)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kulkarnivaishnavi341@gmail.com'  # Enter your actual email id
app.config['MAIL_PASSWORD'] = 'nyhm tnqx ucgc zava'    # Enter your actual Google App Password
mail = Mail(app)



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
# Database
# ---------------------------------------------------------------------
def get_db():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    return connection

def init_database():
    with get_db() as connection:
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
    print(f"Database ready: {DATABASE_PATH}")

init_database()

# ---------------------------------------------------------------------
# Risk analysis helpers
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

def detect_suspicious_indicators(text):
    text = (text or "").casefold()
    found = []
    for reason, keywords in SUSPICIOUS_PATTERNS.items():
        if any(keyword in text for keyword in keywords):
            found.append(reason)
    return found

def calculate_risk_score(fake_probability, suspicious_reasons):
    adjustment = min(len(suspicious_reasons), 4) * 5
    return round(max(0.0, min(100.0, float(fake_probability) + adjustment)), 2)

def get_risk_level(risk_score):
    if risk_score >= 70: return "HIGH RISK"
    if risk_score >= 40: return "MEDIUM RISK"
    return "LOW RISK"

def get_dataset_information(job_title, company_name):
    if dataset is None:
        return {"found": False, "message": "Reference dataset is not available."}
    try:
        title_match = pd.Series(False, index=dataset.index)
        company_match = pd.Series(False, index=dataset.index)
        if "title" in dataset.columns and job_title:
            title_match = dataset["title"].fillna("").astype(str).str.casefold().str.contains(job_title.casefold(), regex=False)
        if "company_profile" in dataset.columns and company_name:
            company_match = dataset["company_profile"].fillna("").astype(str).str.casefold().str.contains(company_name.casefold(), regex=False)
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
            "message": f"{len(matches)} similar dataset record(s) found; {fraud_count} marked fraudulent."
        }
    except Exception as exc:
        print(f"Dataset comparison error: {exc}")
        return {"found": False, "message": "Dataset comparison could not be completed."}

def save_analysis(job_title, company_name, job_description, requirements, salary,
                  prediction, fake_probability, risk_score, risk_level, suspicious_reasons):
    try:
        with get_db() as connection:
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
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
    except sqlite3.Error as exc:
        app.logger.exception("Could not save analysis: %s", exc)

def login_required():
    return "user_id" in session

def result_page(**context):
    defaults = {
        "result": "ERROR", "fake_probability": 0, "risk_score": 0, "risk_level": "UNKNOWN",
        "reasons": [], "recommendation": [], "job_title": "", "company_name": "",
        "job_description": "", "requirements": "", "salary": "", "dataset_information": None,
    }
    defaults.update(context)
    return render_template("result.html", **defaults)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def home(): return render_template("index.html")

@app.route("/about")
def about(): return render_template("about.html")

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
            connection = sqlite3.connect(DATABASE_PATH)
            cursor = connection.cursor()
            submitted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO contacts (name, email, message, submitted_at) VALUES (?, ?, ?, ?)
            """, (name, email, message, submitted_at))
            connection.commit()
            connection.close()
            flash("Thank you! Your message has been sent successfully.")
        except Exception as e:
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
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "error")
            return redirect(url_for("register"))
        try:
            with get_db() as connection:
                connection.execute("INSERT INTO users (name, email, password, created_at) VALUES (?, ?, ?, ?)",
                                   (name, email, generate_password_hash(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("This email is already registered.", "error")
        except sqlite3.Error:
            flash("Registration failed. Please try again.", "error")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        remember = request.form.get("remember")  # 👈 'Remember Me' चा चेकबॉक्स डेटा

        if not email or not password:
            flash("Please enter email and password.")
            return redirect(url_for("login"))

        try:
            connection = sqlite3.connect(DATABASE_PATH)
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            connection.close()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["user"] = user["email"]
                session["user_name"] = user["name"]
                flash("Login successful!")

                # --- ⚡ ADMIN AUTO-REDIRECT ⚡ ---
                if user["email"] == "admin@jobdetect.com":
                    response = make_response(redirect(url_for("admin_dashboard")))
                else:
                    response = make_response(redirect(url_for("analyze")))

                # --- 🍪 REMEMBER ME कुकीज लॉजिक 🍪 ---
                if remember:
                    # ३० दिवसांचा वेळ सेकंदात (३० दिवस * २४ तास * ६० मिनिटे * ६० सेकंद)
                    max_age = 30 * 24 * 60 * 60
                    response.set_cookie("remember_user", user["email"], max_age=max_age)
                else:
                    # जर युझरने टिक केले नसेल, तर जुनी कुकी डिलीट करा
                    response.set_cookie("remember_user", "", max_age=0)

                return response
            else:
                flash("Invalid email or password.")
                return redirect(url_for("login"))

        except Exception as e:
            print("Login error:", e)
            flash("Login failed. Please try again.")
            return redirect(url_for("login"))

    # GET रिक्वेस्ट असल्यास: आधी कुकीमध्ये ईमेल साठवला आहे का ते तपासा
    saved_email = request.cookies.get("remember_user", "")
    return render_template("login.html", saved_email=saved_email)

# ⚡ OTP Forgot Password Route
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        connection.close()
        if user:
            otp = str(random.randint(100000, 999999))
            session['reset_email'] = email
            session['reset_otp'] = otp
            msg = Message('Password Reset OTP - AI JobGuard', sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f"Your OTP for password reset is: {otp}\nPlease do not share this code with anyone."
            try:
                mail.send(msg)
                flash('OTP sent to your registered email!', 'success')
                return redirect(url_for('verify_otp'))
            except Exception as e:
                flash('Error occurred while sending email.', 'error')
        else:
            flash('This email is not registered!', 'error')
    return render_template("forgot_password.html")

# ⚡ OTP Verify Route
@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if 'reset_otp' not in session:
        return redirect(url_for('forgot_password'))
    if request.method == "POST":
        user_otp = request.form.get("otp", "").strip()
        new_password = request.form.get("password", "")
        if user_otp == session['reset_otp']:
            if len(new_password) < 6:
                flash("Minimum 6 characters required for the password.", "error")
                return render_template("verify_otp.html")
            try:
                hashed = generate_password_hash(new_password)
                connection = sqlite3.connect(DATABASE_PATH)
                cursor = connection.cursor()
                cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed, session['reset_email']))
                connection.commit()
                connection.close()
                session.pop('reset_otp', None)
                session.pop('reset_email', None)
                flash('Password changed successfully!', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                flash('An error occurred while changing the password.', 'error')
        else:
            flash('Invalid OTP entered!', 'error')
    return render_template("verify_otp.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if not login_required(): return redirect(url_for("login"))
    if request.method == "GET": return render_template("analyze.html")
    job_title = request.form.get("job_title", "").strip()
    company_name = request.form.get("company_name", "").strip()
    job_description = request.form.get("job_description", "").strip()
    requirements = request.form.get("requirements", "").strip()
    salary = request.form.get("salary", "").strip()
    common = {"job_title": job_title, "company_name": company_name, "job_description": job_description, "requirements": requirements, "salary": salary}
    if not job_title or not company_name or not job_description:
        return result_page(**common, result="INVALID INPUT", reasons=["Required details missing."])
    if model is None or vectorizer is None:
        return result_page(**common, result="MODEL NOT READY", reasons=["Model files missing."])
    job_text = " ".join(part for part in (job_title, company_name, job_description, requirements, salary) if part).strip()
    try:
        job_vector = vectorizer.transform([job_text])
        prediction = model.predict(job_vector)
    except Exception as exc:
        return result_page(**common, result="PREDICTION ERROR", reasons=[str(exc)])
    fake_probability = 100.0 if int(prediction) == 1 else 0.0
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(job_vector)
            for index, class_label in enumerate(model.classes_):
                if str(class_label) == "1":
                    fake_probability = float(probabilities[index]) * 100.0
                    break
    except Exception: pass
    fake_probability = round(max(0.0, min(100.0, fake_probability)), 2)
    suspicious_reasons = detect_suspicious_indicators(job_text)
    risk_score = calculate_risk_score(fake_probability, suspicious_reasons)
    risk_level = get_risk_level(risk_score)
    result = "FAKE / SUSPICIOUS" if int(prediction) == 1 else "GENUINE"
    dataset_information = get_dataset_information(job_title, company_name)
    save_analysis(job_title, company_name, job_description, requirements, salary, result, fake_probability, risk_score, risk_level, suspicious_reasons)
    return result_page(**common, result=result, fake_probability=fake_probability, risk_score=risk_score, risk_level=risk_level, reasons=suspicious_reasons, recommendation=["Verify details."], dataset_information=dataset_information)

# ---------------------------------------------------------------------
# Run locally
# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
