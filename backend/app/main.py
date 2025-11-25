# backend/app/main.py

import os
from uuid import uuid4
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openai import OpenAI

from PIL import Image

from .router import Router
from .memory import MemoryStore
from .utils import decode_base64_image, extract_text_field

# --------------------
# Load .env + OpenAI
# --------------------
load_dotenv()
client = OpenAI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# --------------------
# FastAPI app
# --------------------
app = FastAPI(title="Sap — Nepal Plant Multimodal RAG")


# --------------------
# Request / Response Models
# --------------------
class QueryRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None
    session_id: Optional[str] = None


class RetrievedItem(BaseModel):
    id: str
    source: Optional[str] = None
    text: str
    score: Optional[float] = None
    rrf_score: Optional[float] = None
    extra: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    session_id: str
    mode: str
    caption: Optional[str]
    answer: str
    retrieved: List[RetrievedItem]


# --------------------
# Initialize global singletons
# --------------------
print("=== Loading Router, Memory, Models, FAISS... ===")
router = Router()
memory = MemoryStore()
print("=== Initialization complete ===\n")


# --------------------
# GPT Wrapper
# --------------------
def call_gpt(messages: List[Dict[str, str]]) -> str:
    """
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        ...
    ]
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content


# --------------------
# API Endpoint
# --------------------
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    # -------- Validate Input --------
    if not req.text and not req.image_base64:
        raise HTTPException(400, "Provide text and/or image_base64")

    session_id = req.session_id or str(uuid4())
    user_text = req.text.strip() if req.text else None

    # -------- Decode Image --------
    pil_image: Optional[Image.Image] = None
    if req.image_base64:
        try:
            pil_image = decode_base64_image(req.image_base64)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 image: {e}")

    # -------- Run Router --------
    route_out = router.handle_query(
        text=user_text,
        image=pil_image,
        top_k=5,
    )

    mode = route_out.get("mode", "text")
    caption = route_out.get("generated_caption")
    fused_all = route_out.get("fused_ranked", []) or []
    identified_plant = route_out.get("identified_plant")

    # We still only send top-3 fused items as "retrieved" context to GPT
    fused = fused_all[:3]

    # -------- Try to determine plant candidate --------
    # Priority 1: explicit identified_plant from router
    plant_candidate: Optional[Dict[str, Any]] = None
    if identified_plant:
        plant_candidate = identified_plant
    else:
        # Priority 2: first fused item from plant_metadata
        for item in fused_all:
            if item.get("source") == "plant_metadata":
                plant_candidate = item
                break

    # -------- Build Context for GPT --------
    context_blocks: List[str] = []

    # If we have a plant candidate, include its metadata explicitly
    if plant_candidate:
        pname = (
            plant_candidate.get("plant_name")
            or plant_candidate.get("name")
            or plant_candidate.get("plant_id")
        )
        ptext = plant_candidate.get("text") or extract_text_field(plant_candidate)
        context_blocks.append(
            f"Identified plant candidate: {pname or 'Unknown'}\n"
            f"Plant details: {ptext}"
        )

    # Then add fused context snippets (e.g., additional metadata entries)
    for idx, item in enumerate(fused):
        # Avoid duplicating the primary plant_candidate block if same object
        if plant_candidate is not None and item is plant_candidate:
            continue
        textval = extract_text_field(item)
        context_blocks.append(f"[{idx+1}] ({item.get('source')}) {textval}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "No retrieved context."

    # -------- User Query for GPT --------
    if user_text:
        question = user_text
    elif caption:
        question = "Identify and describe this plant based on the image caption."
    else:
        question = "Help the user with their plant-related request."

    system_msg = {
        "role": "system",
        "content": (
            "You are Sap, an assistant focused on plant identification and ethnobotanical knowledge "
            "from Nepal. Use the retrieved context carefully, avoid hallucination, and be concise. "
            "If you are not confident about the plant identity or its uses, clearly say you are unsure."
        ),
    }

    # Pull memory
    past = memory.get(session_id)

    # Build final GPT messages
    user_msg = {
        "role": "user",
        "content": (
            f"User question: {question}\n\n"
            f"Image caption (from vision model): {caption or 'N/A'}\n\n"
            f"Retrieved context:\n{context_str}\n\n"
            "Use the plant candidate and the retrieved context to answer. "
            "If the context does not support a claim, do NOT invent facts. "
            "Answer clearly and accurately. If unsure, say so."
        ),
    }

    messages = [system_msg] + past + [user_msg]

    # -------- GPT Call --------
    gpt_answer = call_gpt(messages)

    # Update memory
    memory.append(session_id, "user", question)
    memory.append(session_id, "assistant", gpt_answer)

    # -------- Build Output Format --------
    retrieved_clean: List[RetrievedItem] = []
    for item in fused:
        textval = extract_text_field(item)
        extra = {
            k: v
            for k, v in item.items()
            if k not in {"id", "source", "faiss_distance", "rrf_score"}
        }
        retrieved_clean.append(
            RetrievedItem(
                id=str(item.get("id")),
                source=item.get("source"),
                text=textval,
                score=float(item.get("faiss_distance", 0.0)),
                rrf_score=float(item.get("rrf_score", 0.0)),
                extra=extra,
            )
        )

    return QueryResponse(
        session_id=session_id,
        mode=mode,
        caption=caption,
        answer=gpt_answer,
        retrieved=retrieved_clean,
    )
