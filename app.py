from flask import Flask, render_template, request, redirect, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import sqlite3
import os
import uuid
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
app.secret_key = "crop_price_ai_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
GRAPH_DIR = os.path.join(STATIC_DIR, "graphs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

PRICE_MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
PRICE_ENCODER_PATH = os.path.join(MODEL_DIR, "crop_label_encoder.pkl")

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            crop TEXT NOT NULL,
            year INTEGER NOT NULL,
            month TEXT NOT NULL,
            past_price REAL NOT NULL,
            present_price REAL NOT NULL,
            predicted_price REAL NOT NULL,
            average_price REAL NOT NULL,
            change_amount REAL NOT NULL,
            change_percent REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin")
        )

    conn.commit()
    conn.close()


init_db()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["Crop"] = df["Crop"].astype(str).str.strip()
    df["Month"] = df["Month"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Convert dataset price index to rupees per quintal
    df["Price"] = df["Price"] * 10

    df = df.dropna(subset=["Year", "Month", "Crop", "Price"])
    df["Year"] = df["Year"].astype(int)
    df = df[df["Month"].isin(MONTHS)].copy()
    df["MonthNo"] = df["Month"].map(MONTH_TO_NUM)

    return df.sort_values(["Crop", "Year", "MonthNo"]).reset_index(drop=True)


def build_training_features(df):
    work = df.copy()

    work["PrevPrice1"] = work.groupby("Crop")["Price"].shift(1)
    work["PrevPrice2"] = work.groupby("Crop")["Price"].shift(2)
    work["RollingMean3"] = work.groupby("Crop")["Price"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )

    work["MonthSin"] = np.sin(2 * np.pi * work["MonthNo"] / 12)
    work["MonthCos"] = np.cos(2 * np.pi * work["MonthNo"] / 12)
    work["TimeIndex"] = work["Year"] * 12 + work["MonthNo"]

    return work.dropna()


def train_price_model():
    df = load_data()
    work = build_training_features(df)

    if len(work) < 8:
        return False

    encoder = LabelEncoder()
    work["CropEncoded"] = encoder.fit_transform(work["Crop"])

    X = work[[
        "CropEncoded", "Year", "MonthNo", "TimeIndex",
        "PrevPrice1", "PrevPrice2", "RollingMean3",
        "MonthSin", "MonthCos"
    ]]
    y = work["Price"]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        random_state=42
    )

    model.fit(X, y)

    joblib.dump(model, PRICE_MODEL_PATH)
    joblib.dump(encoder, PRICE_ENCODER_PATH)

    return True


def model_needs_retrain():
    if not os.path.exists(PRICE_MODEL_PATH):
        return True

    if not os.path.exists(PRICE_ENCODER_PATH):
        return True

    if os.path.getmtime(DATA_PATH) > os.path.getmtime(PRICE_MODEL_PATH):
        return True

    return False


def load_price_model():
    if model_needs_retrain():
        if not train_price_model():
            return None, None

    return joblib.load(PRICE_MODEL_PATH), joblib.load(PRICE_ENCODER_PATH)


def save_price_graph(crop_df, crop, year, month, selected_price):
    crop_df = crop_df.tail(12).reset_index(drop=True)

    x_labels = [f"{m[:3]}-{y}" for m, y in zip(crop_df["Month"], crop_df["Year"])]
    y_vals = crop_df["Price"].tolist()
    selected_label = f"{month[:3]}-{year}"

    plt.figure(figsize=(12, 6))
    plt.plot(x_labels, y_vals, marker="o", linewidth=3, label="Historical Price")
    plt.scatter([selected_label], [selected_price], marker="*", s=350, label="Selected / Predicted Price")

    if len(x_labels) > 0:
        plt.plot(
            [x_labels[-1], selected_label],
            [y_vals[-1], selected_price],
            linestyle="--"
        )

    plt.title(f"{crop} Price Trend")
    plt.xlabel("Month / Year")
    plt.ylabel("Price ₹ / Quintal")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"graph_{uuid.uuid4().hex}.png"
    path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(path, dpi=220)
    plt.close()

    return f"graphs/{filename}"


def predict_price(crop, year, month):
    df = load_data()

    crop_df = df[df["Crop"].str.lower() == crop.lower()].copy()

    if crop_df.empty:
        return None, "Crop not found"

    crop_df = crop_df.sort_values(["Year", "MonthNo"]).reset_index(drop=True)

    month_no = MONTH_TO_NUM[month]

    exact_row = crop_df[
        (crop_df["Year"] == int(year)) &
        (crop_df["Month"] == month)
    ]

    last_prices = crop_df["Price"].tolist()

    past = float(last_prices[-2])
    present = float(last_prices[-1])
    avg = float(np.mean(last_prices))

    if not exact_row.empty:
        future = float(exact_row.iloc[0]["Price"])
        price_type = "CSV Exact Price"
    else:
        model, encoder = load_price_model()

        if model is None or encoder is None:
            return None, "Model not trained"

        matched_crop = None
        for c in encoder.classes_:
            if str(c).lower() == crop.lower():
                matched_crop = c
                break

        if matched_crop is None:
            return None, "Crop not found in trained model"

        row = pd.DataFrame([{
            "CropEncoded": int(encoder.transform([matched_crop])[0]),
            "Year": int(year),
            "MonthNo": month_no,
            "TimeIndex": int(year) * 12 + month_no,
            "PrevPrice1": float(last_prices[-1]),
            "PrevPrice2": float(last_prices[-2]),
            "RollingMean3": float(np.mean(last_prices[-3:])),
            "MonthSin": np.sin(2 * np.pi * month_no / 12),
            "MonthCos": np.cos(2 * np.pi * month_no / 12)
        }])

        future = max(0, float(model.predict(row)[0]))
        price_type = "AI Predicted Price"

    change = future - present
    change_percent = (change / present * 100) if present else 0

    graph_file = save_price_graph(crop_df, crop, int(year), month, future)

    return {
        "crop": crop,
        "past": round(past, 2),
        "present": round(present, 2),
        "future": round(future, 2),
        "avg": round(avg, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2),
        "trend": "increase" if change > 0 else "decrease" if change < 0 else "stable",
        "graph_file": graph_file,
        "price_type": price_type
    }, None


def save_prediction_history(username, result, year, month):
    conn = get_conn()
    conn.execute("""
        INSERT INTO prediction_history
        (username, crop, year, month, past_price, present_price, predicted_price, average_price, change_amount, change_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        result["crop"],
        year,
        month,
        result["past"],
        result["present"],
        result["future"],
        result["avg"],
        result["change"],
        result["change_percent"]
    ))
    conn.commit()
    conn.close()


def create_prediction_pdf(username, result):
    filename = f"prediction_report_{uuid.uuid4().hex}.pdf"
    path = os.path.join(REPORT_DIR, filename)

    c = canvas.Canvas(path, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Crop Price Prediction Report")

    c.setFont("Helvetica", 12)
    y -= 40

    lines = [
        f"User: {username}",
        f"Crop: {result['crop']}",
        f"Past Price: {result['past']}",
        f"Present Price: {result['present']}",
        f"Selected/Future Price: {result['future']}",
        f"Average Price: {result['avg']}",
        f"Change: {result['change']}",
        f"Change Percent: {result['change_percent']}%",
        f"Trend: {result['trend']}",
        f"Price Type: {result.get('price_type', '')}",
        "Unit: Rupees per Quintal"
    ]

    for line in lines:
        c.drawString(50, y, line)
        y -= 25

    graph_path = os.path.join(STATIC_DIR, result["graph_file"])

    if os.path.exists(graph_path):
        c.drawImage(
            ImageReader(graph_path),
            50,
            230,
            width=500,
            height=250,
            preserveAspectRatio=True
        )

    c.save()

    return path


@app.route("/")
def home():
    if "user" in session:
        if session.get("role") == "admin":
            return redirect("/admin")
        return redirect("/dashboard")

    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_conn()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = user["username"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect("/admin")

            return redirect("/dashboard")

        error = "Invalid username or password"

    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = ""

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if len(username) < 3:
            error = "Username must be at least 3 characters"
            return render_template("signup.html", error=error)

        if len(password) < 6:
            error = "Password must be at least 6 characters"
            return render_template("signup.html", error=error)

        try:
            conn = get_conn()
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, generate_password_hash(password), "user")
            )
            conn.commit()
            conn.close()
            return redirect("/login")

        except sqlite3.IntegrityError:
            error = "Username already exists"

    return render_template("signup.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/login")

    df = load_data()
    crops = sorted(df["Crop"].dropna().unique().tolist())

    result = None
    error = ""

    if request.method == "POST":
        crop = request.form.get("crop", "").strip()
        year = request.form.get("year", "").strip()
        month = request.form.get("month", "").strip()

        if not crop or not year or not month:
            error = "Please fill all fields"

        elif not year.isdigit():
            error = "Year must be a number"

        elif month not in MONTHS:
            error = "Invalid month"

        else:
            result, error = predict_price(crop, int(year), month)

            if result:
                save_prediction_history(session["user"], result, int(year), month)
                session["last_prediction"] = result

    return render_template(
        "dashboard.html",
        result=result,
        error=error,
        crops=crops,
        months=MONTHS,
        username=session.get("user")
    )


@app.route("/download-prediction-report")
def download_prediction_report():
    if "user" not in session:
        return redirect("/login")

    result = session.get("last_prediction")

    if not result:
        flash("No prediction available")
        return redirect("/dashboard")

    path = create_prediction_pdf(session["user"], result)
    return send_file(path, as_attachment=True)


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM prediction_history
        WHERE username = ?
        ORDER BY id DESC
    """, (session["user"],)).fetchall()
    conn.close()

    return render_template("history.html", history_rows=rows, username=session["user"])


@app.route("/quality", methods=["GET", "POST"])
def quality():
    if "user" not in session:
        return redirect("/login")

    result = None
    crop_name = ""
    damage_percent = ""
    confidence = ""
    reason = ""
    suggestion = ""
    image_file = ""

    if request.method == "POST":
        selected_crop = request.form.get("crop_name", "").strip()
        file = request.files.get("image")

        if not selected_crop:
            result = "Please select crop type"

        elif not file or file.filename == "":
            result = "Please upload an image"

        elif not allowed_file(file.filename):
            result = "Only png, jpg, jpeg, webp allowed"

        else:
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)

            try:
                img = cv2.imread(save_path)

                if img is None:
                    raise ValueError("Image not readable")

                img = cv2.resize(img, (500, 500))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if selected_crop == "Auto Detect Crop":
                    mean_hue = np.mean(hsv[:, :, 0])
                    mean_sat = np.mean(hsv[:, :, 1])
                    mean_val = np.mean(hsv[:, :, 2])

                    if mean_val > 170 and mean_sat < 70:
                        crop_name = "Cotton"
                    elif 15 <= mean_hue <= 35:
                        crop_name = "Wheat"
                    elif 35 < mean_hue <= 75:
                        crop_name = "Rice"
                    else:
                        crop_name = "Maize"
                else:
                    crop_name = selected_crop

                green_mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
                crop_mask = cv2.bitwise_not(green_mask)

                light_bg = cv2.inRange(hsv, (0, 0, 210), (180, 60, 255))
                crop_mask = cv2.bitwise_and(crop_mask, cv2.bitwise_not(light_bg))

                kernel = np.ones((5, 5), np.uint8)
                crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_OPEN, kernel)
                crop_mask = cv2.morphologyEx(crop_mask, cv2.MORPH_CLOSE, kernel)

                crop_pixels = np.sum(crop_mask > 0)

                if crop_pixels < 1000:
                    result = "❌ Crop area not clear"
                    reason = "Please upload closer image of main crop/grain."
                    damage_percent = "0%"
                    confidence = "0%"
                    suggestion = "Capture image with clear crop area."
                    image_file = f"uploads/{filename}"

                else:
                    brown_mask = cv2.inRange(hsv, (5, 50, 20), (25, 255, 180))
                    black_mask = cv2.inRange(gray, 0, 55)

                    damage_mask = cv2.bitwise_or(brown_mask, black_mask)
                    damage_mask = cv2.bitwise_and(damage_mask, crop_mask)

                    damaged_pixels = np.sum(damage_mask > 0)
                    damage = min(100, (damaged_pixels / crop_pixels) * 100)
                    damage_percent = f"{round(damage, 2)}%"

                    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        if cv2.contourArea(cnt) > 250:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    result_filename = f"result_{filename}"
                    result_path = os.path.join(UPLOAD_DIR, result_filename)
                    cv2.imwrite(result_path, img)

                    image_file = f"uploads/{result_filename}"

                    if damage < 15:
                        result = "✅ Good Quality Crop"
                        confidence = "95%"
                        suggestion = "Ready for selling and storage."
                        reason = """
English:
Main crop area looks healthy with very low damage.
Leaves/background were ignored during analysis.

Kannada:
ಮುಖ್ಯ ಬೆಳೆ ಭಾಗ ಆರೋಗ್ಯಕರವಾಗಿದೆ ಮತ್ತು ಹಾನಿ ಬಹಳ ಕಡಿಮೆ ಇದೆ.
ಎಲೆಗಳು/ಹಿನ್ನೆಲೆಯನ್ನು ವಿಶ್ಲೇಷಣೆಯಲ್ಲಿ ಪರಿಗಣಿಸಲಾಗಿಲ್ಲ.
"""

                    elif damage < 40:
                        result = "⚠ Medium Quality Crop"
                        confidence = "82%"
                        suggestion = "Dry properly and store carefully."
                        reason = """
English:
Some damage is detected only in the main crop area.
Crop quality is average.

Kannada:
ಮುಖ್ಯ ಬೆಳೆ ಭಾಗದಲ್ಲಿ ಸ್ವಲ್ಪ ಹಾನಿ ಕಂಡುಬಂದಿದೆ.
ಬೆಳೆ ಗುಣಮಟ್ಟ ಮಧ್ಯಮವಾಗಿದೆ.
"""

                    else:
                        result = "❌ Poor Quality Crop"
                        confidence = "74%"
                        suggestion = "Avoid long storage and inspect crop carefully."
                        reason = """
English:
High damage is detected in the main crop area.
Crop may be affected by disease, rot, or poor storage.

Kannada:
ಮುಖ್ಯ ಬೆಳೆ ಭಾಗದಲ್ಲಿ ಹೆಚ್ಚು ಹಾನಿ ಕಂಡುಬಂದಿದೆ.
ಬೆಳೆ ರೋಗ, ಕೊಳೆತ ಅಥವಾ ಕೆಟ್ಟ ಸಂಗ್ರಹಣೆಯಿಂದ ಹಾನಿಗೊಂಡಿರಬಹುದು.
"""

                    session["last_quality"] = {
                        "result": result,
                        "crop_name": crop_name,
                        "damage_percent": damage_percent,
                        "confidence": confidence,
                        "reason": reason,
                        "suggestion": suggestion,
                        "image_file": image_file
                    }

            except Exception as e:
                result = "Detection failed"
                reason = f"Quality detection error: {str(e)}"

    return render_template(
        "quality.html",
        result=result,
        crop_name=crop_name,
        damage_percent=damage_percent,
        confidence=confidence,
        reason=reason,
        suggestion=suggestion,
        image_file=image_file
    )


@app.route("/admin")
def admin():
    if "user" not in session or session.get("role") != "admin":
        return redirect("/login")

    conn = get_conn()

    users = conn.execute(
        "SELECT id, username, role FROM users ORDER BY id DESC"
    ).fetchall()

    total_predictions = conn.execute(
        "SELECT COUNT(*) AS total FROM prediction_history"
    ).fetchone()["total"]

    recent_predictions = conn.execute("""
        SELECT username, crop, year, month, predicted_price, created_at
        FROM prediction_history
        ORDER BY id DESC
        LIMIT 5
    """).fetchall()

    conn.close()

    df = load_data()
    crop_count = len(df["Crop"].unique()) if not df.empty else 0
    record_count = len(df)

    return render_template(
        "admin.html",
        users=users,
        crop_count=crop_count,
        record_count=record_count,
        total_predictions=total_predictions,
        recent_predictions=recent_predictions
    )


@app.route("/delete-user/<int:user_id>")
def delete_user(user_id):
    if "user" not in session or session.get("role") != "admin":
        return redirect("/login")

    conn = get_conn()
    user_row = conn.execute(
        "SELECT username FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    if user_row and user_row["username"] != "admin":
        conn.execute(
            "DELETE FROM prediction_history WHERE username = ?",
            (user_row["username"],)
        )
        conn.execute(
            "DELETE FROM users WHERE id = ?",
            (user_id,)
        )

    conn.commit()
    conn.close()

    return redirect("/admin")


if __name__ == "__main__":
    app.run(debug=True)