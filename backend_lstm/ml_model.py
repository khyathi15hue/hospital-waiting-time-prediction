
"""
Loads the exported hybrid LSTM model and computes live wait-time predictions.
Combines the LSTM prediction with an M/M/c queueing-theory estimate.
"""

import json
import math
from collections import deque
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session
from tensorflow import keras

from models import QueueEntry, VisitHistory

WINDOW = 5
ARRIVAL_WINDOW = 15
MIN_SPAN_HOURS = 0.1

CAT_FEATURES = [
    "AgeGroup", "Department", "AppointmentType", "InsuranceType",
    "ArrivalMethod", "TriageCategory", "ProviderID", "DayOfWeek",
    "BookingType"
]

NUM_FEATURES = [
    "FacilityOccupancyRate", "ProvidersOnShift", "NursesOnShift",
    "StaffToPatientRatio", "IsOnlineBooking", "ArrivalHour",
    "ArrivalDayOfWeek", "IsWeekend", "QueueLengthAhead", "MMc_estimate"
]

STATIC_COLS = CAT_FEATURES + NUM_FEATURES

_model = None
_encoders = None
_dept_baselines = None
_static_mean = None
_static_scale = None
_seq_mean = None
_seq_scale = None

_rolling_consult_cache = {}
_wait_history_cache = {}
_arrival_time_cache = {}


def load_artifacts(
    model_path="lstm_hybrid_model.keras",
    encoders_path="encoders.json",
    baselines_path="department_baselines.json",
    static_scaler_path="static_scaler.json",
    seq_scaler_path="seq_scaler.json",
    wait_seed_path="department_wait_seed.json",
):
    global _model, _encoders, _dept_baselines
    global _static_mean, _static_scale, _seq_mean, _seq_scale

    _model = keras.models.load_model(model_path)

    with open(encoders_path, encoding="utf-8") as f:
        _encoders = json.load(f)

    with open(baselines_path, encoding="utf-8") as f:
        _dept_baselines = json.load(f)

    with open(static_scaler_path, encoding="utf-8") as f:
        scaler = json.load(f)
        _static_mean = np.array(scaler["mean"], dtype=float)
        _static_scale = np.array(scaler["scale"], dtype=float)

    with open(seq_scaler_path, encoding="utf-8") as f:
        scaler = json.load(f)
        _seq_mean = np.array(scaler["mean"], dtype=float)
        _seq_scale = np.array(scaler["scale"], dtype=float)

    with open(wait_seed_path, encoding="utf-8") as f:
        wait_seed = json.load(f)

    _rolling_consult_cache.clear()
    _wait_history_cache.clear()
    _arrival_time_cache.clear()

    for dept, info in _dept_baselines.items():
        avg_consult = max(float(info["avg_consult_minutes"]), 0.1)
        _rolling_consult_cache[dept] = {
            "sum": avg_consult * 20,
            "count": 20,
        }

    for dept, seed_values in wait_seed.items():
        _wait_history_cache[dept] = deque(
            seed_values, maxlen=WINDOW
        )

    print(
        f"Loaded LSTM model, {len(_encoders)} encoders, "
        f"{len(_dept_baselines)} department baselines"
    )


def encode_category(feature: str, value: str) -> int:
    classes = _encoders.get(feature, [])
    return classes.index(value) if value in classes else 0


def get_avg_consult_minutes(department: str) -> float:
    cache = _rolling_consult_cache.get(department)

    if not cache or cache["count"] == 0:
        baseline = _dept_baselines.get(department, {})
        return max(float(baseline.get("avg_consult_minutes", 20.0)), 0.1)

    return max(cache["sum"] / cache["count"], 0.1)


def update_rolling_average(
    department: str, actual_consult_minutes: float
):
    """Update the rolling consultation-duration average."""
    cache = _rolling_consult_cache.setdefault(
        department, {"sum": 0.0, "count": 0}
    )

    cache["sum"] += max(float(actual_consult_minutes), 0.1)
    cache["count"] += 1


def record_arrival(department: str, checkin_time: datetime):
    """Record a check-in timestamp for the live arrival-rate estimate."""
    cache = _arrival_time_cache.setdefault(
        department, deque(maxlen=ARRIVAL_WINDOW)
    )
    cache.append(checkin_time)


def compute_queueing_metrics(
    department: str, providers_on_shift: int
) -> dict:
    """Return arrival rate, service rate per provider, server count and rho."""
    baseline = _dept_baselines.get(department, {})
    baseline_lambda = max(
        float(baseline.get("avg_arrival_rate_per_hour", 1.0)),
        0.0,
    )

    cache = _arrival_time_cache.get(department)
    lam = baseline_lambda

    if cache and len(cache) >= 5:
        span_hours = (
            cache[-1] - cache[0]
        ).total_seconds() / 3600.0

        if span_hours >= MIN_SPAN_HOURS:
            live_lambda = (len(cache) - 1) / span_hours
            lam = max(live_lambda, 0.0)

    avg_consult = get_avg_consult_minutes(department)
    mu = 60.0 / avg_consult
    servers = max(int(providers_on_shift), 1)
    rho = lam / (servers * mu) if mu > 0 else 0.0

    return {
        "lambda_per_hour": round(lam, 3),
        "mu_per_hour_per_server": round(mu, 3),
        "servers": servers,
        "rho": round(rho, 3),
    }


def mmc_wait_minutes(
    lambda_per_hour: float,
    mu_per_hour_per_server: float,
    servers: int,
) -> float:
    """
    Calculate expected waiting time in the queue using the M/M/c
    Erlang-C formula. This is queue waiting time, not consultation time.

    If arrival rate reaches/exceeds total service capacity, the stable
    M/M/c expected-wait formula is not defined; return 0 here and let
    the queue-position fallback in predict_wait_minutes handle it.
    """
    lam = max(float(lambda_per_hour), 0.0)
    mu = max(float(mu_per_hour_per_server), 0.0)
    c = max(int(servers), 1)

    if lam == 0.0 or mu == 0.0 or lam >= c * mu:
        return 0.0

    offered_load = lam / mu
    rho = lam / (c * mu)

    # P0 denominator: sum(a^n / n!, n=0..c-1)
    # plus a^c / (c! * (1-rho)).
    series_sum = sum(
        (offered_load ** n) / math.factorial(n)
        for n in range(c)
    )

    last_term = (
        (offered_load ** c)
        / (math.factorial(c) * (1.0 - rho))
    )

    p0 = 1.0 / (series_sum + last_term)
    probability_wait = last_term * p0

    # Erlang-C: Wq (hours) = P(wait) / (c*mu - lambda).
    wait_hours = probability_wait / (c * mu - lam)
    return max(wait_hours * 60.0, 0.0)


def record_actual_wait(
    department: str, actual_wait_minutes: float
):
    """Record a patient's actual wait when their consultation starts."""
    history = _wait_history_cache.setdefault(
        department,
        deque([0.0] * WINDOW, maxlen=WINDOW),
    )
    history.append(max(float(actual_wait_minutes), 0.0))


def compute_queue_length_ahead(
    db: Session, department: str, checkin_time: datetime
) -> int:
    """Count earlier patients who are waiting or in consultation."""
    return (
        db.query(QueueEntry)
        .filter(
            QueueEntry.department == department,
            QueueEntry.status.in_(["waiting", "in_progress"]),
            QueueEntry.checkin_time < checkin_time,
        )
        .count()
    )


def predict_wait_minutes(
    db: Session,
    entry_data: dict,
    queue_length_ahead: int,
) -> tuple[float, float]:
    """Return (LSTM predicted wait, M/M/c estimated queue wait)."""
    department = entry_data["department"]
    providers = max(int(entry_data["providers_on_shift"]), 1)
    avg_consult = get_avg_consult_minutes(department)

    # Use queueing metrics and Erlang-C for the theoretical queue wait.
    metrics = compute_queueing_metrics(department, providers)
    lam = metrics["lambda_per_hour"]
    mu = metrics["mu_per_hour_per_server"]

    mmc_estimate = mmc_wait_minutes(lam, mu, providers)

    # When the system is overloaded, a stable M/M/c wait is undefined.
    # Use a simple queue-position estimate as a fallback in that case.
    if lam >= providers * mu and queue_length_ahead > 0:
        mmc_estimate = queue_length_ahead * avg_consult / providers

    staff_ratio = _dept_baselines.get(
        department, {}
    ).get("avg_staff_to_patient_ratio", 0.3)

    row = {
        "AgeGroup": encode_category("AgeGroup", entry_data["age_group"]),
        "Department": encode_category("Department", department),
        "AppointmentType": encode_category(
            "AppointmentType", entry_data["appointment_type"]
        ),
        "InsuranceType": encode_category(
            "InsuranceType", entry_data["insurance_type"]
        ),
        "ArrivalMethod": encode_category(
            "ArrivalMethod", entry_data["arrival_method"]
        ),
        "TriageCategory": encode_category(
            "TriageCategory", entry_data["triage_category"]
        ),
        "ProviderID": encode_category(
            "ProviderID", entry_data["provider_id"]
        ),
        "DayOfWeek": encode_category(
            "DayOfWeek", entry_data["day_of_week"]
        ),
        "BookingType": encode_category(
            "BookingType", entry_data["booking_type"]
        ),
        "FacilityOccupancyRate": entry_data["facility_occupancy_rate"],
        "ProvidersOnShift": providers,
        "NursesOnShift": entry_data["nurses_on_shift"],
        "StaffToPatientRatio": staff_ratio,
        "IsOnlineBooking": entry_data["is_online_booking"],
        "ArrivalHour": entry_data["arrival_hour"],
        "ArrivalDayOfWeek": entry_data["arrival_day_of_week"],
        "IsWeekend": entry_data["is_weekend"],
        "QueueLengthAhead": queue_length_ahead,
        "MMc_estimate": mmc_estimate,
    }

    static_vec = np.array(
        [[row[col] for col in STATIC_COLS]], dtype=float
    )

    # Protect against division by zero in exported scaler values.
    safe_static_scale = np.where(_static_scale == 0, 1.0, _static_scale)
    static_scaled = (static_vec - _static_mean) / safe_static_scale

    history = _wait_history_cache.get(
        department, deque([0.0] * WINDOW, maxlen=WINDOW)
    )
    seq_raw = np.array(
        list(history), dtype=float
    ).reshape(1, WINDOW, 1)

    safe_seq_scale = np.where(_seq_scale == 0, 1.0, _seq_scale)
    seq_scaled = (seq_raw - _seq_mean) / safe_seq_scale

    predicted = float(
        _model.predict(
            [seq_scaled, static_scaled], verbose=0
        )[0][0]
    )

    return max(predicted, 0.0), max(mmc_estimate, 0.0)