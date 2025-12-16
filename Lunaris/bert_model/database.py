import os
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

# Get DB URL from environment variable, default to SQLite file
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./soluna.db")

# SQLite specific connection arguments
# check_same_thread=False is required for SQLite in multithreaded environments (like FastAPI)
connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}

# Create the engine
# echo=True will log all SQL statements, useful for debugging
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def init_db():
    """
    Creates the database tables based on the SQLModel metadata.
    Should be called on startup.
    """
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get a database session.
    Yields a session and closes it after the request is done.
    """
    with Session(engine) as session:
        yield session
