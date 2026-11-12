"""Local web interface for the Pima diabetes classifier."""

import random

import pandas as pd
from flask import Flask, render_template, request

from model import DATA_PATH, FEATURES, TARGET, add_features, build_model, load_data

app = Flask(__name__)
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
    ("A good next step", "{}"),
    ("Try this", "{}"),
    ("Consider", "{}"),
    ("For your next check-in", "{}"),
    ("One useful move", "{}"),
    ("Keep in mind", "{}"),
    ("A practical idea", "{}"),
]


def recommendations(probability):
    band = "lower" if probability < 0.3 else "middle" if probability < 0.6 else "higher"
    picks = random.SystemRandom().sample(ADVICE[band], 5)
    formats = random.SystemRandom().sample(ADVICE_FORMATS, 5)
    return [{"title": title, "text": template.format(item)}
            for (title, template), item in zip(formats, picks)]


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES, dataset_name=DATA_PATH.name)


@app.post("/predict")
def predict():
    try:
        values = {name: float(request.form[name]) for name in FEATURES}
        if any(value < 0 for value in values.values()):
            raise ValueError("Enter values of zero or greater.")
        row = add_features(pd.DataFrame([values], columns=FEATURES))
        probability = float(_model.predict_proba(row)[0, 1])
        return render_template("index.html", features=FEATURES, values=request.form,
                               probability=probability, advice=recommendations(probability),
                               dataset_name=DATA_PATH.name)
    except (KeyError, TypeError, ValueError) as error:
        return render_template("index.html", features=FEATURES, values=request.form,
                               error=str(error) or "Please enter valid numeric values.",
                               dataset_name=DATA_PATH.name), 400


if __name__ == "__main__":
    app.run(debug=False)
