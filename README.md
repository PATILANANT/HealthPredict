# HealthPredict: Mental Health, Brain Stroke and Diabetes Risk Prediction

Final-year project: *Detection and Prediction of Mental Health Disorders from Social Media Data and
Prediction of Brain Stroke and Diabetes using Machine Learning.*
Guide: Prof. R. P. Bagawade. Team: Vivek Sonawane (23121004), Sangram Virkar (23121013), Anant Patil (23121029).

> **Screening aid only. Not a medical diagnosis.**

## Portable Windows build (no installation)

If this folder contains a **`python`** subfolder, it is the portable build. It includes a Python 3.12 runtime with every package pre-installed. To use it:
1. Extract the zip, ideally to a short path like `C:\HealthPredict`.
2. Double-click **`start.bat`**.

No Python installation and no internet are needed to start. The browser does still need internet for fonts and styles. The portable runtime ignores any Python already installed on the PC, so the two cannot conflict. It works on 64-bit Windows 10 and 11 only. On macOS or Linux, use the normal build below.

## Quick start (one command)

You need **Python 3.10, 3.11 or 3.12** ([download 3.12](https://www.python.org/downloads/release/python-3128/); on Windows tick *"Add python.exe to PATH"*) and an internet connection.

| OS | Command |
|---|---|
| **Windows** | double-click **`start.bat`** (or run `start.bat` in a terminal) |
| **macOS / Linux** | `./start.sh` |

The first run installs the packages, which takes a few minutes. After that it starts in a few seconds and opens **http://127.0.0.1:5000** in your browser. The trained models are included, so no training is needed.

| Demo login | Email | Password |
|---|---|---|
| Patient | `ganesh@example.com` | `ganesh12345` |
| Doctor | `doctor@healthpredict.local` | `doctor12345` |
| Admin | `admin@healthpredict.local` | `admin12345` |

`test_inputs.txt` has sample inputs for every module.

Options:
- `start.bat --port 8080`: use a different port.
- `start.bat --lan`: let phones and laptops on the same Wi-Fi open the app.
- `start.bat --retrain`: retrain all models first (about 8 minutes).

Use `./start.sh` with the same options on macOS and Linux. Press **Ctrl+C** to stop the server.

## Features

| Requirement | Where |
|---|---|
| FR-01 Register / login, roles user, doctor, admin | `app/auth.py`, `app/models.py` |
| FR-02, 05, 07 Input forms for the 3 modules | `ml/schemas.py` (one schema drives the form, validation and model) |
| FR-03, 08, 17 Preprocessing: imputation, scaling, encoding, SMOTE, text cleaning, stemming, TF-IDF | `ml/*/train.py`, `ml/common/text.py` |
| FR-04, 06, 09 Predictions with probability, risk level and explanation | `ml/predict.py` |
| FR-11, 12 Comparison of all classifiers, automatic best-model selection | `ml/common/tabular.py`, `/metrics/<module>` |
| FR-13 Combined report and PDF download | `app/report.py`, `app/pdf.py` |
| FR-14 Input validation with clear messages | `ml/schemas.py::validate` |
| FR-15 Prediction log (timestamp, module, model version) | `Prediction` table |
| FR-16 Admin retraining and dataset upload, model promoted only if better | `app/admin.py`, `ml/common/store.py` |
| Doctor view of patients | `app/doctor.py` |
| REST API with API keys | `app/api.py` |
| Fairness check (recall by gender / age group) | `/metrics/<module>`, Fairness tab |
| Privacy: mental-health text is never stored | `Prediction.from_result` |

## Project layout

```
data/raw/            diabetes.csv, stroke.csv, mental_health.csv
ml/                  config, schemas, training scripts, inference (predict.py), figures
  common/            evaluation metrics, model zoo, leak-free tabular trainer, text cleaning
  diabetes/ stroke/ mental_health/   train.py per module
models/              <module>.joblib (live model), <module>_metrics.json, archive/
app/                 Flask app: blueprints, templates, static
tests/               pytest suite
reports/             figures/*.png and results.md (from `python -m ml.figures`)
```

## Manual setup (step by step, without start.bat)

You need **Python 3.10, 3.11 or 3.12** and an internet connection for step 2.

### Windows (PowerShell or Command Prompt, inside the extracted `HealthPredict` folder)

```powershell
# 1. Create a virtual environment (use -3.11 or -3.10 if you don't have 3.12)
py -3.12 -m venv .venv

# 2. Install the packages (a few minutes)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

# 3. Check that the database library loads
.venv\Scripts\python -c "import sqlalchemy.orm; print('OK')"
#    Only if step 3 fails with "DLL load failed ... Application Control policy" (locked-down PCs):
#    PowerShell:      $env:DISABLE_SQLALCHEMY_CEXT=1
#    Command Prompt:  set DISABLE_SQLALCHEMY_CEXT=1
#    then:            .venv\Scripts\python -m pip install --force-reinstall --no-deps --no-binary sqlalchemy sqlalchemy==2.1.9

# 4. Check the trained models load (skip to 5 if it prints {})
.venv\Scripts\python -c "from ml.predict import Registry; print(Registry().load().errors)"
#    If it lists errors, retrain (~8 min):  .venv\Scripts\python -m ml.train_all --force

# 5. Create the demo accounts (safe to run again, never changes existing accounts)
.venv\Scripts\python -m flask --app run seed-demo

# 6. Start the app, then open http://127.0.0.1:5000
.venv\Scripts\python run.py
```

Always use `.venv\Scripts\python -m ...`. Don't use `.venv\Scripts\flask.exe` or `activate`, because some locked-down PCs block those launchers. Press **Ctrl+C** to stop the server. Next time, only step 6 is needed.

### macOS / Linux (Terminal, inside the extracted `HealthPredict` folder)

```bash
python3.12 -m venv .venv                     # or python3.11 / python3.10
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "from ml.predict import Registry; print(Registry().load().errors)"   # {} = models OK
.venv/bin/python -m flask --app run seed-demo
.venv/bin/python run.py                      # http://127.0.0.1:5000
```

On Ubuntu or Debian, install venv support first with `sudo apt install python3.12-venv`.

### Useful extras

| What | Windows | macOS / Linux |
|---|---|---|
| Other port | `set PORT=8080` then `.venv\Scripts\python run.py` | `PORT=8080 .venv/bin/python run.py` |
| Allow phones on the same Wi-Fi | `set HOST=0.0.0.0` then run, and open `http://<your-PC-IP>:5000` | `HOST=0.0.0.0 .venv/bin/python run.py` |
| Create your own admin | `.venv\Scripts\python -m flask --app run create-user --name "Your Name" --email you@example.com --password yourpass123 --role admin` | same with `.venv/bin/python` |
| Retrain all models | `.venv\Scripts\python -m ml.train_all --force` | `.venv/bin/python -m ml.train_all --force` |
| Retrain one model | `.venv\Scripts\python -m ml.stroke.train` (or `ml.diabetes.train`, `ml.mental_health.train`) | same |
| Regenerate report charts | `.venv\Scripts\python -m ml.figures` | `.venv/bin/python -m ml.figures` |
| Reset the database | delete the `instance` folder, then run `seed-demo` again | same |

In PowerShell, set variables with `$env:PORT=8080` instead of `set PORT=8080`. Set a real `SECRET_KEY` environment variable before deploying anywhere public.

### Troubleshooting

| Problem | Fix |
|---|---|
| `py` / `python3.12` not found | Install Python 3.12 from python.org. On Windows, tick *Add python.exe to PATH*. |
| `No matching distribution found for numpy==1.26.4` | You are on Python 3.13 or newer. Use 3.10 to 3.12. |
| `DLL load failed ... Application Control policy` | Step 3 above. |
| `Model not trained yet` on a module card | `python -m ml.train_all --force` (step 4). |
| `Address already in use` | Another app uses port 5000. Set `PORT=5001`. |
| Page looks unstyled | Styles and fonts load from a CDN, so check your internet connection. |

### Datasets
The CSVs are already in `data/raw/`. The originals are on Kaggle:
- `diabetes.csv`: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- `stroke.csv`: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset (`healthcare-dataset-stroke-data.csv`)
- `mental_health.csv`: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health (`Combined Data.csv`)

A retrained model replaces the live one only if it scores at least as well on the selection metric. `--force` replaces it anyway.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

## REST API

Send the `X-API-Key` header with every request. Each user's key is shown on their Profile page.

```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict/stroke -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"gender":"Male","age":67,"hypertension":0,"heart_disease":1,"ever_married":"Yes","work_type":"Private",
       "Residence_type":"Urban","avg_glucose_level":228.7,"bmi":36.6,"smoking_status":"formerly smoked"}'
```
Endpoints:
- `GET /api/v1/health`
- `GET /api/v1/modules`
- `GET /api/v1/models`
- `POST /api/v1/predict/<diabetes|stroke|mental_health>`
- `GET /api/v1/predictions`

The mental health endpoint expects `{"text": "...", "consent": true}`.

## Methodology notes (useful for the viva)

- **No data leakage.**
  - Imputation, scaling, encoding and SMOTE run inside an `imblearn` pipeline, so they are re-fitted on each cross-validation fold.
  - The decision threshold and the best model are chosen from 5-fold out-of-fold predictions on the training set.
  - The 20% test set is used only for the final scores.
  - Many papers report 97–99% on Pima or the stroke data because they apply SMOTE or scaling before splitting. Our numbers are lower but honest.
- **Diabetes.** A 0 in Glucose, BloodPressure, SkinThickness, Insulin or BMI means "not measured", so it is treated as missing and imputed with the median. The threshold is tuned for F1.
- **Stroke.** About 5% of records are positive. SMOTE is applied to training folds and the threshold is tuned for **F2**, because missing a stroke is worse than a false alarm. Look at recall and ROC-AUC rather than accuracy.
- **Mental health.**
  - Negations and first-person pronouns are kept during cleaning.
  - Features are Snowball stems plus TF-IDF word 1–2-grams.
  - The model predicts 7 classes, plus a binary Distress vs Normal view.
  - The helpline message appears when "Suicidal" is predicted or has a probability of 30% or more.
- **Safety rules (`ml/safety.py`).** The dataset has no category for threats against other people, and "kill" is common slang in Normal posts. A rule layer therefore checks every text for self-harm phrases and harm-to-others phrases. It also catches misspelled and stretched forms ("kill himm", "worsttt"). The rules can only **raise** the risk: on a match the label becomes Suicidal or "Anger / violent thoughts", the risk becomes High, and a helpline message is shown. The result page also shows what the ML model alone predicted. Slang like "this traffic is killing me" is deliberately not matched (see `tests/test_safety.py`).
- **Explanations.**
  - Clinical modules: each input is swapped for a typical value and the change in predicted risk is shown.
  - Text module: the words that pushed the Logistic Regression explainer towards the predicted class are shown.

## Live end-to-end checks

With the server running (`python run.py`):

```powershell
.venv\Scripts\python scripts\e2e_live.py         # 36 browser-style checks with CSRF on: auth, XSS, access control, PDFs, API, privacy
.venv\Scripts\python scripts\e2e_admin_live.py   # the same checks, then dataset upload + retraining through the admin panel (~2 min)
```

The live checks create throw-away `e2e_*@example.com` accounts.
