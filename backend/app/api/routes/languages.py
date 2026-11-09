"""Languages metadata route reporting status for available sign languages and modes."""

import json
from pathlib import Path
from typing import Literal, cast

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_app_settings
from backend.app.api.schemas import (
    LanguageItem,
    LanguagesResponse,
    ModeStatus,
)
from backend.app.core.config import Settings

router = APIRouter(tags=["languages"])

StatusLiteral = Literal["available", "experimental", "withheld", "unavailable"]


@router.get("/languages", response_model=LanguagesResponse)
async def get_languages(settings: Settings = Depends(get_app_settings)) -> LanguagesResponse:
    """Return available sign languages and mode statuses (available, experimental, withheld, unavailable)."""
    # 1. Check letter model status
    letter_model_path = Path(settings.model_path)
    spell_status: StatusLiteral = "available" if letter_model_path.is_file() else "withheld"

    # 2. Check word model status from gate file
    gate_path = Path(settings.word_gate_path)
    if gate_path.is_file():
        try:
            with open(gate_path, encoding="utf-8") as f:
                gate_data = json.load(f)
                raw_status = gate_data.get("status", "withheld")
                words_status = (
                    cast(StatusLiteral, raw_status)
                    if raw_status in {"available", "experimental", "withheld", "unavailable"}
                    else "withheld"
                )
        except Exception:
            words_status = "withheld"
    else:
        # Check if ONNX model exists directly
        word_model_path = Path(settings.word_model_path)
        words_status = "available" if word_model_path.is_file() else "withheld"

    return LanguagesResponse(
        languages=[
            LanguageItem(
                code="ASL",
                modes=[
                    ModeStatus(mode="spell", status=spell_status),
                    ModeStatus(mode="words", status=words_status),
                ],
            ),
            LanguageItem(
                code="ISL",
                modes=[
                    ModeStatus(mode="spell", status="unavailable"),
                    ModeStatus(mode="words", status="unavailable"),
                ],
            ),
        ]
    )
