import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import trace
from opentelemetry.trace import get_tracer

tracer = trace.get_tracer(__name__)

# Configure resource for your service
resource = Resource.create({"service.name": "orchestrator"})

# Set up the TracerProvider
provider = TracerProvider(resource=resource)

# Configure the OTLP exporter to send data to Jaeger
otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces") # Adjust endpoint if Jaeger is elsewhere

# Add the exporter to a BatchSpanProcessor
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

# Set the global tracer provider
trace.set_tracer_provider(provider)

QDRANT_URL = os.environ.get('QDRANT_URL', "http://localhost:6333")
QDRANT_COLLECTION_NAME = "test"
ENCODER_URL = os.environ.get('ENCODER_URL', "http://localhost:8001")
OLLAMA_URL = os.environ.get('OLLAMA_URL', "http://localhost:11434")

qdrant_client: QdrantClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client
    qdrant_client = QdrantClient(url=QDRANT_URL)

    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    yield
    qdrant_client.close()

class SubmitDocumentRequest(BaseModel):
    document: str

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
async def ask(request_body: AskRequest) -> AskResponse:
    with tracer.start_as_current_span("Call Encoder Service [ask]"):
        encoder_response = httpx.post(f'{ENCODER_URL}/encode', 
            json={"text": request_body.query}, 
        ).json()

    with tracer.start_as_current_span("Qdrant Query"):
        search_result = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=encoder_response['embedding'],
            with_payload=True,
            limit=3,
            score_threshold=0.3,
        ).points

    print(search_result)
    
    # Extract the 'text' from each payload
    texts = [p.payload['text'] for p in search_result]

    # Optional: remove duplicates while preserving order
    texts = list(dict.fromkeys(texts))

    # Combine into a single string for smolLM context
    context = " ".join(texts)

    print(context)

    payload = {
        "model": "smollm",
        "prompt":
        f"""Use context below to answer the question at the very end
        Context:
        {context}
        Question:
        {request_body.query}
        """
    }

    answer = ""

    with tracer.start_as_current_span("Call Ollama Service"):
        with httpx.stream(method="POST", url=f"{OLLAMA_URL}/api/generate", json=payload) as response:
            for line in response.iter_lines():
                if line:
                    answer += json.loads(line)['response']

    return {"answer": answer}


@app.post("/ask/stream")
async def ask_stream(request_body: AskRequest):
    """
    Server-Sent-Events stream that sends one JSON chunk per token.
    Media-type 'text/event-stream' keeps the connection open.

    curl -N -s -H "Content-Type: application/json" -H "Accept: text/event-stream" -X POST http://localhost:8000/ask/stream -d '{"query":"What does summer in Seattle look like?"}'
    """
    with tracer.start_as_current_span("Call Encoder Service [ask/stream]"):
        encoder_response = httpx.post(f'{ENCODER_URL}/encode', 
            json={"text": request_body.query}, 
        ).json()

    with tracer.start_as_current_span("Qdrant Query [stream]"):
        search_result = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=encoder_response['embedding'],
            with_payload=True,
            limit=3,
            score_threshold=0.3,
        ).points

    print(search_result)
    

    # Extract the 'text' from each payload
    texts = [p.payload['text'] for p in search_result]

    # Optional: remove duplicates while preserving order
    texts = list(dict.fromkeys(texts))

    # Combine into a single string for smolLM context
    context = " ".join(texts)

    async def token_generator():
    
        payload = {
            "model": "smollm",
            "prompt":
            f"""Use context below to answer the question at the very end
            Context:
            {context}
            Question:
            {request_body.query}
            """,
            "stream": True
        }
        with tracer.start_as_current_span("Call Ollama Service [stream]"):
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/generate",
                                        json=payload, timeout=None) as r:
                    async for line in r.aiter_lines():
                        yield json.loads(line)['response']
                        await asyncio.sleep(0.05)  # give event‑loop a chance

    return StreamingResponse(token_generator(),
                             media_type="text/event-stream")

FastAPIInstrumentor.instrument_app(app)