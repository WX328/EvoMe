from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import EVOME_DB_URL


class Base(DeclarativeBase):
    pass


connect_args = (
    {"check_same_thread": False}
    if EVOME_DB_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    EVOME_DB_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)
