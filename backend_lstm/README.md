# Patient Wait Time Prediction — Queueing Theory + LSTM

FastAPI backend + 4 HTML pages. FastAPI serves everything (API and pages) as one app.

## Architecture
Patient check-in -> Database -> live queue info -> queueing theory (M/M/c: lambda, mu, rho,
queue length) -> queueing features + recent wait-time sequence -> LSTM -> predicted wait
-> FastAPI -> patient status page / doctor dashboard.

- lambda = live arrival rate (patients/hour; falls back to historical baseline until enough live arrivals)
- mu = service rate per server (60 / rolling avg consult minutes)
- rho = utilization = lambda / (c * mu), c = providers on shift
- LSTM input: sequence of the last 5 actual wait times in the department + static features
  (including QueueLengthAhead and the M/M/c estimate)

## Run locally
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1        (Windows)
    pip install -r requirements.txt
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/  (landing page with links to every screen).

## Retrain / re-export the model (optional, artifacts are included)
    python export_artifacts.py path/to/Hospital_Wait__TIme_Data.csv
LSTM training is random, so test MAE varies run to run (observed ~28 to ~33 min).

## Important: category values must match the dataset exactly
AgeGroup, AppointmentType, InsuranceType, ArrivalMethod, DayOfWeek (Mon..Sun), BookingType,
ProviderID (DR1..DR50) use the dataset's real values (see encoders.json). The check-in form
already uses them. Unknown values silently fall back to index 0 in ml_model.encode_category.

## Endpoints
POST /checkin, POST /start_consult/{id}, POST /consult_complete, GET /queue/{dept},
GET /stats/{dept}, GET /queueing_metrics/{dept}, GET /health, /docs

## Note
Authentication (auth.py, doctor login) is NOT in this package — merge it in from your own copy.
