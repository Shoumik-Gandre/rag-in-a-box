from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from typing import List, Dict
import uuid

from models import Chunk, ChunkMetadata

# HuggingFace tokenizer for token-aware splitting
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def token_length(text: str) -> int:
    """Compute number of tokens using the encoder's tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_document(doc_text: str, doc_id: uuid = None, source: str = None) -> List[Chunk]:
    """Split the document into token-aware, semantically coherent chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=64,
        length_function=token_length,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(doc_text)

    return [
        Chunk(
            id=str(uuid.uuid4()),
            text=chunk,
            metadata=ChunkMetadata(
                chunk_index=i,
                doc_id=doc_id,
                source="api"
            )
        )
        for i, chunk in enumerate(chunks)
    ]
