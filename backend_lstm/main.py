from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, get_db, Base
import models
import ml_model

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Patient Wait Time Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    ml_model.load_artifacts()


class CheckInRequest(BaseModel):
    patient_id: str
    department: str
    triage_category: str
    provider_id: str
    age_group: str
    appointment_type: str
    insurance_type: str
    arrival_method: str
    day_of_week: str
    booking_type: str
    providers_on_shift: int
    nurses_on_shift: int
    facility_occupancy_rate: float
    is_online_booking: int
    arrival_hour: int
    arrival_day_of_week: int
    is_weekend: int


class CheckInResponse(BaseModel):
    patient_id: str
    queue_position: int
    predicted_wait_minutes: float
    mmc_estimate_minutes: float
    lambda_per_hour: float
    mu_per_hour_per_server: float
    rho: float


class ConsultCompleteRequest(BaseModel):
    patient_id: str


class QueueItem(BaseModel):
    patient_id: str
    department: str
    triage_category: str
    status: str
    predicted_wait_minutes: Optional[float]
    checkin_time: datetime


@app.post("/checkin", response_model=CheckInResponse)
def checkin(payload: CheckInRequest, db: Session = Depends(get_db)):
    existing = db.query(models.QueueEntry).filter_by(patient_id=payload.patient_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Patient already checked in")

    now = datetime.utcnow()
    queue_length_ahead = ml_model.compute_queue_length_ahead(db, payload.department, now)
    ml_model.record_arrival(payload.department, now)

    entry_data = payload.dict()
    predicted_wait, mmc_estimate = ml_model.predict_wait_minutes(db, entry_data, queue_length_ahead)
    queueing_metrics = ml_model.compute_queueing_metrics(payload.department, payload.providers_on_shift)

    entry = models.QueueEntry(
        patient_id=payload.patient_id,
        department=payload.department,
        triage_category=payload.triage_category,
        provider_id=payload.provider_id,
        age_group=payload.age_group,
        appointment_type=payload.appointment_type,
        insurance_type=payload.insurance_type,
        arrival_method=payload.arrival_method,
        day_of_week=payload.day_of_week,
        booking_type=payload.booking_type,
        providers_on_shift=payload.providers_on_shift,
        nurses_on_shift=payload.nurses_on_shift,
        facility_occupancy_rate=payload.facility_occupancy_rate,
        is_online_booking=payload.is_online_booking,
        arrival_hour=payload.arrival_hour,
        arrival_day_of_week=payload.arrival_day_of_week,
        is_weekend=payload.is_weekend,
        checkin_time=now,
        status="waiting",
        predicted_wait_minutes=predicted_wait,
        queue_length_ahead=queue_length_ahead,
    )
    db.add(entry)
    db.commit()

    return CheckInResponse(
        patient_id=payload.patient_id,
        queue_position=queue_length_ahead + 1,
        predicted_wait_minutes=round(predicted_wait, 1),
        mmc_estimate_minutes=round(mmc_estimate, 1),
        lambda_per_hour=queueing_metrics["lambda_per_hour"],
        mu_per_hour_per_server=queueing_metrics["mu_per_hour_per_server"],
        rho=queueing_metrics["rho"],
    )


@app.post("/start_consult/{patient_id}")
def start_consult(patient_id: str, db: Session = Depends(get_db)):
    entry = db.query(models.QueueEntry).filter_by(patient_id=patient_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found in queue")

    now = datetime.utcnow()
    entry.status = "in_progress"
    entry.provider_start_time = now

    # The patient's REAL wait is now known -- feed it to the LSTM's rolling
    # wait-time history for this department, so the next patients' sequence
    # input reflects what actually just happened.
    actual_wait = max((now - entry.checkin_time).total_seconds() / 60.0, 0.0)
    ml_model.record_actual_wait(entry.department, actual_wait)

    db.commit()
    return {"patient_id": patient_id, "status": "in_progress"}


@app.post("/consult_complete")
def consult_complete(payload: ConsultCompleteRequest, db: Session = Depends(get_db)):
    entry = db.query(models.QueueEntry).filter_by(patient_id=payload.patient_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Patient not found in queue")

    now = datetime.utcnow()
    entry.provider_end_time = now
    entry.status = "done"

    start = entry.provider_start_time or entry.checkin_time
    actual_minutes = max((now - start).total_seconds() / 60.0, 0.1)

    db.add(models.VisitHistory(department=entry.department, consult_duration_minutes=actual_minutes))
    ml_model.update_rolling_average(entry.department, actual_minutes)
    db.commit()

    updated = _recompute_department_queue(db, entry.department)

    return {
        "patient_id": payload.patient_id,
        "actual_consult_minutes": round(actual_minutes, 1),
        "updated_queue": updated,
    }


@app.get("/queue/{department}", response_model=list[QueueItem])
def get_queue(department: str, db: Session = Depends(get_db)):
    entries = (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department == department, models.QueueEntry.status != "done")
        .order_by(models.QueueEntry.checkin_time)
        .all()
    )
    return entries


def _recompute_department_queue(db: Session, department: str):
    waiting = (
        db.query(models.QueueEntry)
        .filter(models.QueueEntry.department == department, models.QueueEntry.status == "waiting")
        .order_by(models.QueueEntry.checkin_time)
        .all()
    )
    results = []
    for e in waiting:
        queue_length_ahead = ml_model.compute_queue_length_ahead(db, department, e.checkin_time)
        entry_data = {
            "department": e.department,
            "triage_category": e.triage_category,
            "provider_id": e.provider_id,
            "age_group": e.age_group,
            "appointment_type": e.appointment_type,
            "insurance_type": e.insurance_type,
            "arrival_method": e.arrival_method,
            "day_of_week": e.day_of_week,
            "booking_type": e.booking_type,
            "providers_on_shift": e.providers_on_shift,
            "nurses_on_shift": e.nurses_on_shift,
            "facility_occupancy_rate": e.facility_occupancy_rate,
            "is_online_booking": e.is_online_booking,
            "arrival_hour": e.arrival_hour,
            "arrival_day_of_week": e.arrival_day_of_week,
            "is_weekend": e.is_weekend,
        }
        predicted_wait, _ = ml_model.predict_wait_minutes(db, entry_data, queue_length_ahead)
        e.predicted_wait_minutes = predicted_wait
        e.queue_length_ahead = queue_length_ahead
        results.append({"patient_id": e.patient_id, "predicted_wait_minutes": round(predicted_wait, 1)})
    db.commit()
    return results


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats/{department}")
def get_stats(department: str, db: Session = Depends(get_db)):
    from sqlalchemy import func
    total_waiting = (
        db.query(models.QueueEntry)
        .filter_by(department=department, status="waiting")
        .count()
    )
    in_consultation = (
        db.query(models.QueueEntry)
        .filter_by(department=department, status="in_progress")
        .count()
    )
    today = datetime.utcnow().date()
    completed_today = (
        db.query(models.VisitHistory)
        .filter(
            models.VisitHistory.department == department,
            func.date(models.VisitHistory.completed_at) == today,
        )
        .count()
    )
    return {
        "total_waiting": total_waiting,
        "in_consultation": in_consultation,
        "completed_today": completed_today,
    }


@app.get("/queueing_metrics/{department}")
def get_queueing_metrics(department: str, providers_on_shift: int = 3):
    """Explicit M/M/c parameters for this department, computed live:
    lambda = arrival rate (patients/hour), mu = service rate per server
    (patients/hour), rho = utilization (lambda / (c * mu))."""
    return ml_model.compute_queueing_metrics(department, providers_on_shift)


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
