import datetime
from typing import List, Literal, Optional
import uuid
from pydantic import BaseModel
from sqlalchemy import UUID, Column, DateTime, Text, func, Integer, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlmodel import Field, SQLModel
# ------------------------------
# Request & Response Models
# ------------------------------

class IngestRequest(BaseModel):
    text: str

class IngestResponse(BaseModel):
    token: str

class StatusRequest(BaseModel):
    token: str

class StatusResponse(BaseModel):
    status: Literal["IN_PROGRESS", "FAILED", "COMPLETED"]
    error_message: Optional[str] = None


# ------------------------------
# CHUNK STRUCTURE
# ------------------------------

class ChunkMetadata(BaseModel):
    chunk_index: int
    source: Optional[str] = "unknown"
    doc_id: str


class Chunk(BaseModel):
    id: str  # Each chunk has a unique UUID (as str)
    text: str
    metadata: ChunkMetadata

    def to_qdrant(self, vector: List[float]) -> dict:
        """Helper to convert to Qdrant-compatible format."""
        return {
            "id": self.id,
            "vector": vector,
            "payload": self.metadata.model_dump(),
        }
    
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