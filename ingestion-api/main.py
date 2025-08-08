import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

import pika
import pika.exceptions
import psycopg2
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session
from chonk import chunk_document

from models import IngestRequest, IngestResponse, StatusRequest, StatusResponse, IngestionStatus, ChunkIngestionStatus


# ------------------------------
# Configuration
# ------------------------------

class AppConfig:
    RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
    QUEUE_NAME = "ingestion_queue"
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
    POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "mysecretpassword")


# ------------------------------
# Application Context
# ------------------------------

class AppContext:
    def __init__(self, config: AppConfig):
        self.config = config
        self.rabbit_conn: Optional[pika.BlockingConnection] = None
        self.rabbit_channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self.pg_conn: Optional[psycopg2.extensions.connection] = None
        self.pg_cursor: Optional[psycopg2.extensions.cursor] = None

    def publish(self, message: dict) -> None:
        """Open a short-lived connection, publish, close - avoids heartbeat timeouts."""
        try:
            with pika.BlockingConnection(
                pika.ConnectionParameters(
                    self.config.RABBITMQ_HOST,
                    heartbeat=0              # disable heartbeats for the short-lived conn
                )
            ) as conn:
                channel = conn.channel()
                channel.queue_declare(queue=self.config.QUEUE_NAME)
                channel.basic_publish(
                    exchange='',
                    routing_key=self.config.QUEUE_NAME,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2      # make message persistent
                    )
                )
        except pika.exceptions.AMQPError as e:
            raise RuntimeError(f"RabbitMQ publish failed: {e}") from e

    def connect_postgres(self):
        self.engine = create_engine(f'postgresql://{self.config.POSTGRES_USER}:{self.config.POSTGRES_PASSWORD}@{self.config.POSTGRES_HOST}:{self.config.POSTGRES_PORT}/{self.config.POSTGRES_DB}')
        

    def _create_postgres_table_if_not_exists(self):
        Base.metadata.create_all(self.engine)
        print("Connected to PostgreSQL and ensured table exists.")

    def close(self):
        if self.rabbit_conn and self.rabbit_conn.is_open:
            self.rabbit_conn.close()
            print("RabbitMQ connection closed.")

        if self.pg_cursor:
            self.pg_cursor.close()
        if self.pg_conn:
            self.pg_conn.close()
            print("PostgreSQL connection closed.")

# ------------------------------
# Global Instances
# ------------------------------

app_config = AppConfig()
app_context = AppContext(app_config)
postgres_url = f'postgresql://{app_config.POSTGRES_USER}:{app_config.POSTGRES_PASSWORD}@{app_config.POSTGRES_HOST}:{app_config.POSTGRES_PORT}/{app_config.POSTGRES_DB}'
engine = create_engine(postgres_url)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

# ------------------------------
# FastAPI with Lifespan
# ------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


# ------------------------------
# Route
# ------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, db_session: SessionDep):
    doc_id = str(uuid.uuid4())
    chunks = chunk_document(request.text, doc_id)
    db_session.add(IngestionStatus(id=doc_id, status='IN_PROGRESS', num_chunks=len(chunks)))
    db_session.commit()
    for chunk in chunks:
        try:
            chunk_id = str(uuid.uuid4())
            message = {"token":chunk_id, "doc":chunk.text}
            app_context.publish(message)
            print(f"Published message: {message}")
            # Insert into DB with status IN_PROGRESS
            db_session.add(ChunkIngestionStatus(chunk_id=chunk_id, status='IN_PROGRESS', chunk_index=chunk.metadata.chunk_index, doc_id=chunk.metadata.doc_id))
            db_session.commit()
            print(f"Inserted status IN_PROGRESS for chunk_id: {chunk_id} and doc_id: {doc_id}")

        except Exception as e:
            print(f"Publishing failed: {e}")
            db_session.add(ChunkIngestionStatus(chunk_id=chunk_id, status='FAILED', error_message=str(e), doc_id=doc_id))
            db_session.add(IngestionStatus(id=doc_id, status='FAILED', error_message=str(e)))
            db_session.commit()
            print(f"Inserted status FAILED for chunk_id: {chunk_id} and doc_id: {doc_id}")

    return IngestResponse(token=doc_id)

@app.post("/status", response_model=StatusResponse)
async def status(request: StatusRequest, db_session: SessionDep):
    # take the token from request and check for it exists in postgres and return the status in response
    try:
        result = db_session.get(IngestionStatus, request.token)
        # select returned empty then return wrong token error message to user
        if not result:
            raise HTTPException(status_code=404, detail="Ingestion token not found")
        
        return StatusResponse(status=result.status)
    except Exception as e:
        # if select threw an exception? That means table does not exist?
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
