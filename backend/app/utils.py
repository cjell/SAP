import base64
import io
from typing import Any, Dict, Optional

from PIL import Image


def decode_base64_image(data: str) -> Image.Image:

    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def extract_text_field(meta: Dict[str, Any]) -> str:

    for key in ("text", "info", "caption", "chunk"):
        if key in meta and isinstance(meta[key], str):
            return meta[key]
    return str(meta)