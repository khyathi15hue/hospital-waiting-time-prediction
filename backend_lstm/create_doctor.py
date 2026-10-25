"""
One-off CLI helper for creating (or resetting the password of) a doctor login.
There's no public /register endpoint on purpose -- staff accounts should be
provisioned by whoever runs the clinic's server, not created from the internet.

Usage:
    python create_doctor.py
"""
import getpass

from database import SessionLocal, engine, Base
import models
import auth

Base.metadata.create_all(bind=engine)


def main():
    db = SessionLocal()
    try:
        username = input("Username: ").strip()
        if not username:
            print("Username can't be empty.")
            return

        full_name = input("Full name (optional): ").strip() or None
        department = input("Department (optional): ").strip() or None
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")

        if not password:
            print("Password can't be empty.")
            return
        if password != confirm:
            print("Passwords don't match.")
            return

        existing = db.query(models.Doctor).filter_by(username=username).first()
        if existing:
            existing.hashed_password = auth.hash_password(password)
            existing.full_name = full_name
            existing.department = department
            existing.is_active = True
            print(f"Updated existing account for '{username}'.")
        else:
            db.add(models.Doctor(
                username=username,
                hashed_password=auth.hash_password(password),
                full_name=full_name,
                department=department,
                is_active=True,
            ))
            print(f"Created new account for '{username}'.")
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    main()