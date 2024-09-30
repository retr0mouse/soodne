from sqlalchemy.orm import Session

from app.database.database import SessionLocal

def get_db():
    with SessionLocal() as db:
        yield db
