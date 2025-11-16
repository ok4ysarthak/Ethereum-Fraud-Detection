# backend/db.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load .env for DATABASE_URL
load_dotenv()

# Read DATABASE_URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://devuser:devpass@localhost:5432/detrust_db"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,        # Set True to debug SQL queries
    future=True
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# ----------- IMPORTANT FUNCTION -------------
def init_db():
    """
    Creates all tables in the database using SQLAlchemy models.
    Must be called once when server starts.
    """
    # IMPORT HERE to avoid circular imports
    from models_db import Base

    print("📌 Creating database tables if not exist...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized!")
