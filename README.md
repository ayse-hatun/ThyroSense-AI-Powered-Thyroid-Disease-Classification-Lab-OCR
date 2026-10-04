
# ThyroSense – AI-Powered Thyroid Disease Classification & Lab OCR

**ThyroSense** is a Flask web application designed for interactive thyroid disease prediction. Powered by a custom Perceptron model trained from scratch, the system evaluates clinical history, demographic details, and lab values to predict thyroid condition status. It includes full OCR support for extracting lab parameters directly from medical report images.

---

## Key Features

* **Interactive Medical UI**: Modern chat-style web interface ("Dr. Thyro") for guided user input and single-click lab report analysis.
* **Automatic OCR Extraction**: Uses Tesseract OCR via `pytesseract` to parse lab values (`TSH`, `TT4`, `FTI`, `T4U`, `T3`, `Age`, `Sex`) directly from uploaded images (`PNG`, `JPG`, `WEBP`).
* **From-Scratch Perceptron Engine**: Uses standard feature standardization and log transformations powered by pre-calculated training weights (`model.json`).
* **REST API Endpoints**: Provides standalone routes for automated JSON-based predictions, report parsing, and model statistics inspection.

---

## Directory Structure

```text
thyroidSense/
│
├── app.py              # Flask server, OCR pipelines, input pre-processing, prediction backend
├── model.json          # Pre-calculated weights, biases, normalization metrics, and training stats
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── templates/
    └── index.html      # Frontend chat layout and dynamic patient form

```

---

## Model Performance Summary

The underlying custom Perceptron classifier achieves high sensitivity across training and testing splits:

* **Test Accuracy**: **97.75%**
* **Test Recall**: **98.31%**
* **Test Precision**: **78.38%**
* **Test F1-Score**: **0.8722**

---

## Setup & Installation

### Prerequisites

* Python 3.9+ installed on your system.
* **Tesseract OCR engine** (required only for image upload parsing).
* **Windows**: Download and run the installer from UB-Mannheim Tesseract OCR. Ensure it is installed in the default path (`C:\Program Files\Tesseract-OCR\tesseract.exe`).
* **Linux**: Run `sudo apt-get install tesseract-ocr`.
* **macOS**: Run `brew install tesseract`.



### Installation Steps

1. **Clone or Download the Repository**:
```bash
git clone [https://github.com/your-username/thyroidSense.git](https://github.com/your-username/thyroidSense.git)
cd thyroidSense

```


2. **Create and Activate a Virtual Environment** *(Optional but recommended)*:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


*(Optional for OCR support)*:
```bash
pip install pytesseract pillow

```


4. **Run the Application**:
```bash
python app.py

```


5. **Access the Web Interface**:
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## API Reference

### 1. Direct Web UI Form Prediction

* **Endpoint**: `POST /predict`
* **Form Parameters**: Standard web form inputs (`age`, `sex`, `tsh`, `tt4`, `fti`, `on_thyroxine`, etc.)
* **Response**: Renders `index.html` with prediction outcome label.

---

### 2. JSON Prediction Endpoint

* **Endpoint**: `POST /api/predict`
* **Content-Type**: `application/json`
* **Payload Example**:
```json
{
  "age": 45,
  "sex": "F",
  "TSH": 12.5,
  "TT4": 85.0,
  "FTI": 90.0,
  "on thyroxine": 0,
  "query hypothyroid": 1
}

```


* **Response Example**:
```json
{
  "confidence": 0.984123,
  "label": "Positive",
  "prediction": 1,
  "score": 20.628145
}

```



---

### 3. Parse Report Image (OCR)

* **Endpoint**: `POST /extract-report`
* **Body**: `multipart/form-data` with `report` file key attached.
* **Response Example**:
```json
{
  "detected_fields": ["age", "sex", "tsh", "tt4"],
  "extracted": {
    "age": 52,
    "sex": "F",
    "tsh": 4.8,
    "tt4": 110.0
  }
}

```



---

### 4. Fetch Model Information

* **Endpoint**: `GET /api/model`
* **Response**: Returns JSON object containing feature listing, training history metrics, and model evaluations.

---

## Disclaimer

> **Notice**: This project is for educational and experimental purposes only. It is **not** a certified medical diagnosis tool. All prediction outputs must be evaluated by a qualified medical professional.

```

```
