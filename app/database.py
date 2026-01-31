import os
import uuid
import struct
from sqlalchemy import create_engine, Column, String, Text, JSON, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlite_vec
from typing import List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./solvify.db")

# Create engine
# check_same_thread=False is needed for SQLite with FastAPI
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

@event.listens_for(engine, "connect")
def load_vec_extension(dbapi_connection, connection_record):
    dbapi_connection.enable_load_extension(True)
    sqlite_vec.load(dbapi_connection)
    dbapi_connection.enable_load_extension(False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_name = Column(String, index=True)
    content = Column(Text)
    # Store vector as BLOB for sqlite-vec compatibility
    vector_embeddings = Column(JSON, nullable=True) 
    processed_output = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)
    # We will use a separate virtual table for vector search if needed, 
    # but for now we'll keep it simple and use JSON or BLOB.
    # sqlite-vec can search on JSON arrays too, but virtual tables are faster.

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def serialize_vector(vector: List[float]) -> bytes:
    """Convert float list to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)
