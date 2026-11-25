"""Local web interface for the Pima diabetes classifier and GlucoAI RAG."""

import math
import random

import pandas as pd
from flask import Flask, render_template, request, session

from model import DATA_PATH, FEATURES, TARGET, add_features, build_model, load_data
from rag_model import answer as rag_answer

app = Flask(__name__)
app.config["SECRET_KEY"] = "glucoscope-local-session-key-change-before-hosting"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
_data = load_data()
_model = build_model()
_model.fit(add_features(_data[FEATURES]), _data[TARGET].astype(int))

ADVICE = {
    "lower": [
        "Keep routine checkups on your calendar; a clinician can advise when glucose screening is appropriate.",
        "Build movement into your week in ways that feel sustainable, such as walks or cycling.",
        "Choose a varied eating pattern with vegetables, high-fibre foods, and fewer sugary drinks.",
        "Bring questions about your family history and these measurements to your next health visit.",
        "Protect sleep and make changes gradually; small habits are easier to maintain.",
        "If you have concerns, ask a clinician whether a validated blood test is appropriate.",
        "Use this result as a prompt for a conversation, not as a reason to change medication or diet abruptly.",
    ],
    "middle": [
        "Consider discussing a validated glucose test with a healthcare professional; this model cannot diagnose diabetes.",
        "Ask your clinician how often screening makes sense for your personal history and health measurements.",
        "Try one manageable activity goal this week, such as a regular walk, and build from there.",
        "A clinician or registered dietitian can help tailor food choices to your needs and preferences.",
        "Write down your questions and family history before a checkup so you can review them together.",
        "If you have past lab results, bring them to a clinician; lab tests are more informative than this estimate.",
        "Choose one sustainable habit to work on rather than making several restrictive changes at once.",
    ],
    "higher": [
        "Please arrange a conversation with a healthcare professional about proper diabetes screening and confirmatory lab tests.",
        "Contact a clinician to interpret these measurements in context; this estimate is not a diagnosis.",
        "If you feel well, this is still a good reason to schedule routine care and ask about glucose testing.",
        "Ask about evidence-based prevention support if a clinician finds your lab results indicate increased risk.",
        "Bring your measurements and family history to an appointment; avoid starting or stopping medicines based on this result.",
        "A qualified clinician can explain whether repeat testing or follow-up is appropriate for you.",
        "Take a calm next step: book a routine appointment and discuss validated screening options.",
    ],
}
ADVICE_FORMATS = [
    ("A good next step", "{}"), ("Try this", "{}"), ("Consider", "{}"),
    ("For your next check-in", "{}"), ("One useful move", "{}"),
    ("Keep in mind", "{}"), ("A practical idea", "{}"),
]


def recommendations(band):
    picks = random.SystemRandom().sample(ADVICE[band], 5)
    formats = random.SystemRandom().sample(ADVICE_FORMATS, 5)
    return [{"title": title, "text": template.format(item)}
            for (title, template), item in zip(formats, picks)]


def probability_band(probability):
    """Map model output to app display groups, not clinical risk categories."""
    return "lower" if probability < 0.3 else "middle" if probability < 0.6 else "higher"


def page(**kwargs):
    result = session.get("glucoscope_result")
    if result and "probability" not in kwargs:
        kwargs["probability"] = result["percent"] / 100
    return render_template("index.html", features=FEATURES, dataset_name=DATA_PATH.name,
                           chat_history=session.get("glucoai_chat", []), **kwargs)


@app.get("/")
def index():
    return page()


@app.route("/glucoai", methods=["GET", "POST"])
def glucoai():
    question = request.form.get("question", "").strip()[:500] if request.method == "POST" else ""
    history = session.get("glucoai_chat", [])
    if question:
        history = history[-8:] + [
            {"role": "user", "text": question},
            {"role": "assistant", "text": rag_answer(
                question, context=session.get("glucoscope_result"))},
        ]
        session["glucoai_chat"] = history
    return render_template("glucoai.html", chat_history=history)


@app.get("/analyze-report")
def analyze_report():
    result = session.get("glucoscope_result")
    if not result:
        return render_template("report.html", result=None, explanation=None)

    band_ranges = {
        "lower": "below 30%",
        "middle": "30% to below 60%",
        "higher": "60% or above",
    }
    explanation = (
        f"The model returned {result['percent']:.1f}% for a positive label "
        f"({result['band']} app display band: {band_ranges[result['band']]}). "
        "This is a model output for the values entered "
        "and its historical training data. It is not your measured blood-glucose percentage, "
        "a diagnosis, a laboratory result, or a clinically validated personal risk. The app's "
        "lower, middle, and higher bands are display groups only, not medical thresholds. "
        "This estimate may be inaccurate or may not generalize to you. A qualified healthcare "
        "professional can review your history and advise whether validated tests, such as HbA1c "
        "or plasma glucose, are appropriate. Do not change medicines or treatment based on it."
    )
    next_steps = recommendations(result["band"])
    explanation += "\n\nGeneral next steps for this app display band:\n" + "\n".join(
        f"- {item['title']}: {item['text']}" for item in next_steps
    )
    return render_template("report.html", result=result, explanation=explanation)

@app.route("/bmi", methods=["GET", "POST"])
def bmi_calculator():
    result = None
    error = None
    values = request.form if request.method == "POST" else {}
    if request.method == "POST":
        try:
            weight = float(request.form.get("weight_kg", ""))
            height = float(request.form.get("height_cm", ""))
            if not (2 <= weight <= 500 and 50 <= height <= 260):
                raise ValueError("Enter weight from 2-500 kg and height from 50-260 cm.")
            bmi = weight / (height / 100) ** 2
            category = ("Underweight" if bmi < 18.5 else
                        "Healthy-weight range" if bmi < 25 else
                        "Overweight" if bmi < 30 else "Obesity range")
            result = {"value": round(bmi, 1), "category": category}
        except (TypeError, ValueError) as exc:
            error = str(exc) or "Enter valid weight and height values."
    return render_template("bmi.html", result=result, error=error, values=values)


@app.post("/predict")
def predict():
    try:
        values = {name: float(request.form[name]) for name in FEATURES}
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError(
                "Enter finite values greater than zero, using the units shown for each field."
            )
        row = add_features(pd.DataFrame([values], columns=FEATURES))
        probability = float(_model.predict_proba(row)[0, 1])
        band = probability_band(probability)
        session["glucoscope_result"] = {"percent": round(probability * 100, 1), "band": band}
        return page(values=request.form, probability=probability,
                    advice=recommendations(band))
    except (KeyError, TypeError, ValueError) as error:
        return page(values=request.form, error=str(error) or "Please enter valid numeric values."), 400


@app.post("/ragbot/clear")
def clear_chat():
    session.pop("glucoai_chat", None)
    session.pop("glucoscope_result", None)
    return page()


if __name__ == "__main__":
    app.run(debug=False)
