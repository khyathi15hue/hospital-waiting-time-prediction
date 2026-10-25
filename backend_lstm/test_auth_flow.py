"""
Smoke test for the clinic queue API with auth enabled.

Usage (server must be running):
    pip install requests
    python test_auth_flow.py <doctor_username> <doctor_password>

IMPORTANT: fill in CHECKIN below with the same values your old passing test
used -- category values like age_group / insurance_type must match your encoders.
"""
import sys
import time
import requests

BASE = "http://127.0.0.1:8000"
DEPT = "Cardiology"
PID = f"TEST_{int(time.time())}"

CHECKIN = {
    "patient_id": PID,
    "department": DEPT,
    "triage_category": "Urgent",          # <- copy from your old test
    "provider_id": "P1",                  # <- copy from your old test
    "age_group": "Adult",                 # <- copy from your old test
    "appointment_type": "Walk-in",        # <- copy from your old test
    "insurance_type": "Private",          # <- copy from your old test
    "arrival_method": "Walk-in",          # <- copy from your old test
    "day_of_week": "Monday",
    "booking_type": "Walk-in",            # <- copy from your old test
    "providers_on_shift": 3,
    "nurses_on_shift": 5,
    "facility_occupancy_rate": 0.5,
    "is_online_booking": 0,
    "arrival_hour": 10,
    "arrival_day_of_week": 0,
    "is_weekend": 0,
}

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not ok else ""))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    user, pw = sys.argv[1], sys.argv[2]

    # --- Auth gate checks -------------------------------------------------
    r = requests.get(f"{BASE}/queue/{DEPT}")
    check("Queue without token is rejected (401)", r.status_code == 401, r.text)

    r = requests.post(f"{BASE}/login", data={"username": user, "password": "wrong-password-xyz"})
    check("Login with wrong password is rejected (401)", r.status_code == 401, r.text)

    r = requests.post(f"{BASE}/login", data={"username": user, "password": pw})
    check("Login with correct password works", r.status_code == 200, r.text)
    if r.status_code != 200:
        sys.exit(1)
    H = {"Authorization": f"Bearer {r.json()['access_token']}"}

    # --- 1. Patient check-in (public, no token) ---------------------------
    r = requests.post(f"{BASE}/checkin", json=CHECKIN)
    check("1. Patient check-in + prediction (no login needed)", r.status_code == 200, r.text)
    if r.status_code != 200:
        sys.exit(1)

    # --- Patient-facing status (public, anonymized) -----------------------
    r = requests.get(f"{BASE}/my_queue_status", params={"patient_id": PID, "department": DEPT})
    ok = r.status_code == 200 and r.json().get("status") == "waiting"
    check("Patient status page endpoint works without login", ok, r.text)
    if r.status_code == 200:
        leaked = any("patient_id" in row for row in r.json().get("queue", []))
        check("Patient status does NOT expose other patients' IDs", not leaked)

    # --- 2. Queue tracking (doctor) ---------------------------------------
    r = requests.get(f"{BASE}/queue/{DEPT}", headers=H)
    ok = r.status_code == 200 and any(e["patient_id"] == PID for e in r.json())
    check("2. Patient appears in doctor queue", ok, r.text)

    # --- 3. Consultation lifecycle (doctor) -------------------------------
    r = requests.post(f"{BASE}/start_consult/{PID}", headers=H)
    check("3a. Start consultation", r.status_code == 200, r.text)
    r = requests.post(f"{BASE}/consult_complete", headers=H, json={"patient_id": PID})
    check("3b. Complete consultation", r.status_code == 200, r.text)

    # --- 4. Stats (doctor) ------------------------------------------------
    r = requests.get(f"{BASE}/stats/{DEPT}", headers=H)
    ok = r.status_code == 200 and r.json().get("completed_today", 0) >= 1
    check("4. Stats show completed visit", ok, r.text)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()