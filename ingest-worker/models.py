import datetime
from typing import Optional
import uuid
from sqlalchemy import Column, DateTime, func
from sqlmodel import Field, SQLModel

# ------------------------------
# SQLAlchemy ORM
# ------------------------------

class IngestionStatus(SQLModel, table=True):
    __tablename__ = 'ingestion_status'
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    status: str = Field()
    error_message: str = Field(nullable=True)
    updated_at: Optional[datetime.datetime] = Field(
        sa_column=Column(DateTime(), onupdate=func.now()),  # Updates on modification
    )
    num_chunks: int = Field(default=0)
    completed_chunks: int = Field(default=0)

class ChunkIngestionStatus(SQLModel, table=True):
    __tablename__ = 'chunk_ingestion_status'
    chunk_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    doc_id: uuid.UUID = Field(default=None, foreign_key='ingestion_status.id', ondelete='CASCADE')
    status: str = Field()
    error_message: str = Field(nullable=True)
    updated_at: Optional[datetime.datetime] = Field(
        sa_column=Column(DateTime(), onupdate=func.now()),  # Updates on modification
    )
    chunk_index: int = Field()