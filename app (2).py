"""Flask app for the exported from-scratch thyroid Perceptron model."""
import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from flask import Flask, jsonify, render_template, request

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Windows: use the standard Tesseract installation when it exists.
if pytesseract is not None:
    _win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _win_tesseract

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

with open(os.path.join(HERE, "model.json"), encoding="utf-8") as f:
    M = json.load(f)

FEATS = M["features"]
W = np.asarray(M["weights"], dtype=float)
B = float(M["bias"])
MU = np.asarray(M["mean"], dtype=float)
SD = np.asarray(M["std"], dtype=float)
LOG_FEATURES = set(M.get("log_features", []))

FORM_TO_MODEL = {
    "age": "age",
    "sex": "sex",
    "on_thyroxine": "on thyroxine",
    "query_on_thyroxine": "query on thyroxine",
    "on_antithyroid_medication": "on antithyroid medication",
    "sick": "sick",
    "pregnant": "pregnant",
    "thyroid_surgery": "thyroid surgery",
    "i131_treatment": "I131 treatment",
    "query_hypothyroid": "query hypothyroid",
    "query_hyperthyroid": "query hyperthyroid",
    "lithium": "lithium",
    "goitre": "goitre",
    "tumor": "tumor",
    "hypopituitary": "hypopituitary",
    "psych": "psych",
    "tsh_measured": "TSH measured",
    "tsh": "TSH",
    "t3_measured": "T3 measured",
    "tt4_measured": "TT4 measured",
    "tt4": "TT4",
    "t4u_measured": "T4U measured",
    "t4u": "T4U",
    "fti_measured": "FTI measured",
    "fti": "FTI",
}

BINARY_FEATURES = {
    "sex",
    "on thyroxine",
    "query on thyroxine",
    "on antithyroid medication",
    "sick",
    "pregnant",
    "thyroid surgery",
    "I131 treatment",
    "query hypothyroid",
    "query hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH measured",
    "T3 measured",
    "TT4 measured",
    "T4U measured",
    "FTI measured",
}


def _to_numeric_value(feature: str, value):
    """Convert browser/API values safely. Blank strings are treated as missing."""
    if value is None:
        return None

    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return None

        token = raw.lower()

        if feature == "sex":
            if token in {"m", "male"}:
                return 1.0
            if token in {"f", "female"}:
                return 0.0

        if token in {"t", "true", "yes", "y"}:
            return 1.0
        if token in {"f", "false", "no", "n"}:
            return 0.0

        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid value for '{feature}': {value!r}") from exc

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{feature}': {value!r}") from exc


def normalize_input(values: dict) -> dict:
    normalized = {}
    for key, value in values.items():
        model_key = FORM_TO_MODEL.get(key, key)
        normalized[model_key] = _to_numeric_value(model_key, value)
    return normalized


def _neutral_raw_value(feature: str, index: int) -> float:
    """Use a safe neutral fallback for a blank/missing form field.

    Binary flags default to 0. Continuous values use the training mean, which
    becomes 0 after standardization. For log-transformed values, convert the
    stored transformed mean back to raw scale first.
    """
    if feature in BINARY_FEATURES:
        return 0.0

    mean_value = float(MU[index])
    if feature in LOG_FEATURES:
        return max(math.expm1(mean_value), 0.0)
    return mean_value


def score(values: dict) -> float:
    values = normalize_input(values)
    row = []

    for i, name in enumerate(FEATS):
        x = values.get(name)

        # Important fix: an empty HTML input arrives as "". Do not call
        # float(""); use a neutral fallback instead.
        if x is None:
            x = _neutral_raw_value(name, i)

        if not math.isfinite(x):
            raise ValueError(f"{name} must be a finite number")

        if name in LOG_FEATURES:
            x = math.log1p(max(x, 0.0))

        row.append(x)

    z = (np.asarray(row, dtype=float) - MU) / SD
    return float(z @ W + B)


def sigmoid_confidence(s: float) -> float:
    x = s / 5.0
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def make_result(values: dict) -> dict:
    s = score(values)
    positive = s > 0
    return {
        "prediction": int(positive),
        "score": round(s, 6),
        "confidence": round(sigmoid_confidence(s), 6),
        "label": "Positive" if positive else "Negative",
    }




def _clean_ocr_text(text: str) -> str:
    """Normalize common OCR punctuation/spacing without changing numeric values."""
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("–", "-").replace("—", "-")
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _find_lab_line(patterns, text):
    """Return (value, line) for the first plausible result on a matching lab row."""
    lines = _clean_ocr_text(text).splitlines()
    for i, line in enumerate(lines):
        for pat in patterns:
            if not re.search(pat, line, re.I):
                continue

            # Prefer numbers after the label; this avoids picking numbers inside names such as T4.
            mlabel = re.search(pat, line, re.I)
            tail = line[mlabel.end():] if mlabel else line
            nums = re.findall(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)(?![A-Za-z])", tail)
            if nums:
                return float(nums[0]), line

            # Some OCR engines put the result on the next row.
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                nums = re.findall(r"(?<![A-Za-z0-9])([0-9]+(?:\.[0-9]+)?)(?![A-Za-z])", nxt)
                if nums:
                    return float(nums[0]), line + " " + nxt
    return None, ""


def _first_number_after(label_patterns, text):
    value, _ = _find_lab_line(label_patterns, text)
    return value


def _t4_to_tt4_nmol(value: float, source_line: str) -> tuple[float, str]:
    """Convert total T4 to the nmol/L scale used by this model when needed."""
    line = source_line.lower().replace(" ", "")
    # Total T4 conversion: 1 ug/dL ≈ 12.87 nmol/L.
    if any(unit in line for unit in ("ug/dl", "mcg/dl", "ugdl", "mcgdl")):
        return round(value * 12.87, 3), "ug/dL"
    return value, "model-scale"


def extract_report_values(text: str) -> dict:
    """Extract supported patient/lab values from common thyroid-report wording."""
    out = {}
    clean = _clean_ocr_text(text)

    # Header fields.
    age = _first_number_after([r"\bAGE\b"], clean)
    if age is not None and 1 <= age <= 100:
        out["age"] = int(age)

    # Conservative sex extraction: only explicit SEX/GENDER labels.
    m = re.search(r"\b(?:SEX|GENDER)\b\s*[:\-]?\s*(MALE|FEMALE|M|F)\b", clean, re.I)
    if m:
        token = m.group(1).upper()
        out["sex"] = "F" if token.startswith("F") else "M"

    # TSH can appear as an abbreviation or as the full test name.
    tsh, _ = _find_lab_line([
        r"\bTSH\b",
        r"THYROID\s+STIMULAT(?:ING|ION)\s+HORMONE",
        r"THYROTROPIN",
    ], clean)
    if tsh is not None and 0 <= tsh <= 100:
        out["tsh"] = tsh

    # T3 has no numeric feature in the current model, but its presence maps to T3 measured.
    t3, _ = _find_lab_line([
        r"\b(?:TOTAL\s+)?T3\b",
        r"TRI[-\s]?IODO\s*THYRONIN(?:E)?\s*[,\-:]?\s*\(?T3\)?",
        r"TRI[-\s]?IODOTHYRONIN(?:E)?\s*[,\-:]?\s*\(?T3\)?",
        r"TRIIODOTHYRONINE\s*\(?T3\)?",
    ], clean)
    if t3 is not None and 0 <= t3 <= 1000:
        out["t3"] = t3
        out["t3_measured"] = "t"

    # Reports often print total T4 as THYROXIN (T4) / THYROXINE (T4), not as 'TT4'.
    t4, t4_line = _find_lab_line([
        r"\bTT4\b",
        r"\bTOTAL\s*T4\b",
        r"\bT4\s*TOTAL\b",
        r"TOTAL\s+THYROXINE",
        r"THYROXIN(?:E)?\s*[,\-:]?\s*\(?T4\)?",
    ], clean)
    if t4 is not None:
        tt4, source_unit = _t4_to_tt4_nmol(t4, t4_line)
        if 0 <= tt4 <= 300:
            out["tt4"] = tt4
            out["t4_report_value"] = t4
            out["t4_report_unit"] = source_unit

    t4u, _ = _find_lab_line([
        r"\bT4U\b",
        r"T4\s*UPTAKE",
        r"T[-\s]?UPTAKE",
    ], clean)
    if t4u is not None and 0.3 <= t4u <= 2.0:
        out["t4u"] = t4u

    fti, _ = _find_lab_line([
        r"\bFTI\b",
        r"FREE\s+THYROXINE\s+INDEX",
        r"FT4\s*INDEX",
    ], clean)
    if fti is not None and 0 <= fti <= 300:
        out["fti"] = fti

    return out


def _ocr_report_image(image: Image.Image) -> str:
    """Run OCR in two useful page-segmentation modes and combine the text."""
    gray = ImageOps.grayscale(image)
    # Upscaling helps Tesseract with small lab-table text.
    if gray.width < 1800:
        scale = 1800 / max(gray.width, 1)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)))
    gray = ImageEnhance.Contrast(gray).enhance(1.5)

    texts = []
    for config in ("--oem 3 --psm 6", "--oem 3 --psm 11"):
        try:
            t = pytesseract.image_to_string(gray, config=config)
            if t.strip():
                texts.append(t)
        except Exception:
            if not texts:
                raise
    return "\n".join(texts)


@app.route("/extract-report", methods=["POST"])
def extract_report():
    upload = request.files.get("report")
    if not upload or not upload.filename:
        return jsonify(error="Choose a report image first."), 400

    ext = os.path.splitext(upload.filename.lower())[1]
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        return jsonify(error="V1 supports PNG, JPG, JPEG and WEBP report images."), 400

    if pytesseract is None:
        return jsonify(error="OCR dependency is missing. Install pytesseract and Tesseract OCR, then restart Flask."), 500

    try:
        image = Image.open(upload.stream).convert("RGB")
        text = _ocr_report_image(image)
        extracted = extract_report_values(text)
        return jsonify(extracted=extracted, detected_fields=list(extracted.keys()))
    except Exception as exc:
        # TesseractNotFoundError subclasses RuntimeError; keep a useful message for local setup.
        msg = str(exc)
        if "tesseract" in msg.lower() and ("not installed" in msg.lower() or "not in your path" in msg.lower()):
            msg = "Tesseract OCR is not installed or not available in PATH. Install it, then restart Flask."
        return jsonify(error=msg or "Could not read this report image."), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        result = make_result(request.form.to_dict())
        return render_template("index.html", prediction=result["label"])
    except (TypeError, ValueError) as exc:
        return render_template("index.html", error=str(exc)), 400


@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object of feature values."), 400

    try:
        return jsonify(make_result(data))
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/model", methods=["GET"])
def model_info():
    return jsonify(train=M.get("train"), test=M.get("test"), features=FEATS)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
