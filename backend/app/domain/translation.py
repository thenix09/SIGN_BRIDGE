"""Domain model for translation history entries."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class Translation:
    """Represents a translation event between sign language and text."""

    id: int | None
    direction: Literal["sign_to_text", "text_to_sign"]
    language: str
    content: str
    created_at: datetime
