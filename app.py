from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ================= LOAD MODEL =================

model = joblib.load("placement_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# ================= HOME =================

@app.route("/")
def home():
    return render_template("index.html")


# ================= PREDICT =================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # ---------- Numerical Inputs ----------
        ssc_p = float(request.form["ssc_p"])
        hsc_p = float(request.form["hsc_p"])
        degree_p = float(request.form["degree_p"])
        etest_p = float(request.form["etest_p"])
        mba_p = float(request.form["mba_p"])

        # ---------- Categorical Inputs ----------
        gender = request.form.get("gender")
        workex = request.form.get("workex")
        specialisation = request.form.get("specialisation")
        hsc_s = request.form.get("hsc_s")
        degree_t = request.form.get("degree_t")

        # ---------- Feature Engineering ----------
        academic_average = (ssc_p + hsc_p + degree_p) / 3

        # Create dictionary with all model columns
        input_data = {}

        for col in model_columns:
            input_data[col] = 0

        # Numerical Features
        input_data["ssc_p"] = ssc_p
        input_data["hsc_p"] = hsc_p
        input_data["degree_p"] = degree_p
        input_data["etest_p"] = etest_p
        input_data["mba_p"] = mba_p
        input_data["academic_average"] = academic_average

        # One Hot Encoding

        if gender == "M":
            input_data["gender_M"] = 1

        if hsc_s == "Commerce":
            input_data["hsc_s_Commerce"] = 1

        elif hsc_s == "Science":
            input_data["hsc_s_Science"] = 1

        if degree_t == "Others":
            input_data["degree_t_Others"] = 1

        elif degree_t == "Sci&Tech":
            input_data["degree_t_Sci&Tech"] = 1

        if workex == "Yes":
            input_data["workex_Yes"] = 1

        if specialisation == "Mkt&HR":
            input_data["specialisation_Mkt&HR"] = 1

        # ---------- Convert to DataFrame ----------
        input_df = pd.DataFrame([input_data])

        # ---------- Prediction ----------
        prediction = model.predict(input_df)[0]

        # ---------- Prediction Probability ----------
        probability = model.predict_proba(input_df)[0][1] * 100

        # ---------- Confidence ----------
        if probability >= 85:
            confidence = "Very High"

        elif probability >= 70:
            confidence = "High"

        elif probability >= 50:
            confidence = "Medium"

        else:
            confidence = "Low"

        # ---------- Final Result ----------
        if prediction == 1:

            result = "Likely to be Placed"
            card_color = "success"
            icon = "fa-circle-check"

        else:

            result = "Less Likely to be Placed"
            card_color = "danger"
            icon = "fa-circle-xmark"

        return render_template(
            "predict.html",
            prediction=result,
            probability=probability,
            confidence=confidence,
            academic_average=academic_average,
            card_color=card_color,
            icon=icon
        )

    return render_template("predict.html")


# ================= ANALYTICS =================

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


# ================= ABOUT =================

@app.route("/about")
def about():
    return render_template("about.html")


# ================= RUN APP =================

if __name__ == "__main__":
    app.run(debug=True)