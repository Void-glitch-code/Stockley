"""
Run this once to create all tables in your Postgres database.
Usage: python init_db.py
"""
from app.database import engine, Base
from app import models  # noqa: F401 -- imported so models register with Base

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")
