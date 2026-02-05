# Main loop for LLM Interaction
# Handles queries by routing to respective files (DinoV2 and LLaVA-Next for images)


import os
from uuid import uuid4
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from openai import OpenAI

from .router import Router
from .memory import MemoryStore
from .utils import decode_base64_image, extract_text_field


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1-mini")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set in .env")



app = FastAPI(title="Sap — Nepal Plant Multimodal RAG")



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


print("-- Loading Router, Memory, Models, FAISS... --")
router = Router()
memory = MemoryStore()
print("-- Initialization complete --\n")



def call_gpt(messages):
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content


# Endpoint for frontend to hit
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):

    if not req.text and not req.image_base64:
        raise HTTPException(400, "Provide text and/or image_base64")

    session_id = req.session_id or str(uuid4())
    user_text = req.text.strip() if req.text else None

    pil_image: Optional[Image.Image] = None
    if req.image_base64:
        try:
            pil_image = decode_base64_image(req.image_base64)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 image: {e}")

    route_out = router.handle_query(
        text=user_text,
        image=pil_image,
        top_k=5
    )

    mode = route_out["mode"]
    caption = route_out.get("generated_caption")

    fused = route_out.get("fused_ranked", [])
    fused = fused[:3]   

    if fused:
        context_blocks = [
            f"[{i+1}] ({item.get('source')}) {extract_text_field(item)}"
            for i, item in enumerate(fused)
        ]
        context_str = "\n\n".join(context_blocks)
    else:
        context_str = "No retrieved context."

    if user_text:
        question = user_text
    elif caption:
        question = "Identify and describe this plant based on the image caption."
    else:
        question = "Help the user with their plant-related request."

    system_msg = {
        "role": "system",
        "content": (
            "You are Sap, an assistant focused on plant identification and "
            "ethnobotanical knowledge from Nepal. Use retrieved context carefully, "
            "avoid hallucination, and be concise."
        ),
    }

    past_memory = memory.get(session_id)

    user_msg = {
        "role": "user",
        "content": (
            f"User question: {question}\n\n"
            f"Image caption: {caption or 'N/A'}\n\n"
            f"Retrieved context:\n{context_str}\n\n"
            "Answer clearly and accurately. If unsure, say so."
        ),
    }

    messages = [system_msg] + past_memory + [user_msg]

    gpt_answer = call_gpt(messages)

    memory.append(session_id, "user", question)
    memory.append(session_id, "assistant", gpt_answer)

    retrieved_clean: List[RetrievedItem] = []
    for item in fused:
        textval = extract_text_field(item)
        extra = {
            k: v for k, v in item.items()
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
