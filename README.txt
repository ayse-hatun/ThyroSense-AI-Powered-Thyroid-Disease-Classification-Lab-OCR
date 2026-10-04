THYROID PREDICTION - FIXED PROJECT

Folder structure:
thyroid_fixed/
  app.py
  model.json
  requirements.txt
  templates/
    index.html

Run on Windows:
1. Open CMD/PowerShell inside this folder.
2. python -m pip install -r requirements.txt
3. python app.py
4. Open http://127.0.0.1:5000

Important:
- This app uses model.json (the from-scratch Perceptron export).
- Do not mix thyroid_model.pkl/scaler.pkl with this app unless you intentionally rewrite the backend for the sklearn pipeline.
