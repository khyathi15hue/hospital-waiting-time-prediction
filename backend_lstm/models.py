from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base


class QueueEntry(Base):
    __tablename__ = "queue_today"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)

    department = Column(String, index=True)
    triage_category = Column(String)
    provider_id = Column(String)
    age_group = Column(String)
    appointment_type = Column(String)
    insurance_type = Column(String)
    arrival_method = Column(String)
    day_of_week = Column(String)
    booking_type = Column(String)

    providers_on_shift = Column(Integer)
    nurses_on_shift = Column(Integer)
    facility_occupancy_rate = Column(Float)
    is_online_booking = Column(Integer)
    arrival_hour = Column(Integer)
    arrival_day_of_week = Column(Integer)
    is_weekend = Column(Integer)

    checkin_time = Column(DateTime, default=datetime.utcnow)
    provider_start_time = Column(DateTime, nullable=True)
    provider_end_time = Column(DateTime, nullable=True)

    status = Column(String, default="waiting", index=True)

    predicted_wait_minutes = Column(Float, nullable=True)
    queue_length_ahead = Column(Integer, nullable=True)


class VisitHistory(Base):
    __tablename__ = "visit_history"

    id = Column(Integer, primary_key=True, index=True)
    department = Column(String, index=True)
    consult_duration_minutes = Column(Float)
    completed_at = Column(DateTime, default=datetime.utcnow)
