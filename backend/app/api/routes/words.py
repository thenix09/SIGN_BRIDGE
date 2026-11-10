"""Word recognition API routes and WebSocket streaming endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.app.api.deps import get_app_settings, get_word_sign_service
from backend.app.api.schemas import (
    RecognizeWordsRequest,
    RecognizeWordsResponse,
    VocabularyResponse,
    VocabularyWord,
    WordCandidate,
    WordFrameInput,
)
from backend.app.core.config import Settings
from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.domain.sequence import LandmarkFrame, LandmarkWindow
from backend.app.services.word_sign import WordResult, WordSignService

router = APIRouter()


def _convert_frame_input(frame_input: WordFrameInput | None) -> LandmarkFrame | None:
    """Convert API WordFrameInput into domain LandmarkFrame."""
    if frame_input is None or frame_input.pose is None:
        return None
    return LandmarkFrame.from_detections(pose=frame_input.pose, hands=frame_input.hands)


def _to_response(result: WordResult) -> RecognizeWordsResponse:
    """Convert domain WordResult into API response schema."""
    return RecognizeWordsResponse(
        state=result.state,
        top_k=[
            WordCandidate(label=candidate.label, confidence=candidate.confidence)
            for candidate in result.top_k
        ],
        accepted_word=result.accepted_word,
        sentence=result.sentence,
        message=result.message,
    )


@router.post("/recognize-words", response_model=RecognizeWordsResponse)
def recognize_words(
    request: RecognizeWordsRequest,
    service: WordSignService = Depends(get_word_sign_service),
) -> RecognizeWordsResponse:
    """Process a single frame or pre-accumulated window for word sequence recognition.

    Primarily used for testing, offline evaluation, and debugging.
    """
    # If a full window is supplied (e.g. in test suites), evaluate directly
    if request.window is not None:
        frames: list[LandmarkFrame] = []
        for f in request.window:
            frame_obj = _convert_frame_input(f)
            if frame_obj is not None:
                frames.append(frame_obj)
        if not frames:
            raise InvalidLandmarkError("LandmarkWindow requires at least one detected frame")
        window = LandmarkWindow(frames)
        result = service.recognize_window(request.session_id, request.language, window)
        return _to_response(result)

    # Streaming single frame
    frame_obj = _convert_frame_input(request.frame)
    result = service.process_frame(request.session_id, request.language, frame_obj)
    return _to_response(result)


@router.get("/vocabulary", response_model=VocabularyResponse)
def get_vocabulary(
    language: str = Query("ASL", description="Sign language code"),
    settings: Settings = Depends(get_app_settings),
) -> VocabularyResponse:
    """Return closed vocabulary list and current availability status."""
    vocab_path = Path(settings.vocabulary_path)
    words: list[VocabularyWord] = []

    if vocab_path.is_file():
        try:
            with vocab_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_words = data.get("words", []) if isinstance(data, dict) else []
            for item in raw_words:
                if isinstance(item, dict) and "label" in item and "category" in item:
                    words.append(VocabularyWord(label=item["label"], category=item["category"]))
        except Exception:
            words = []

    # Check whether the word model is available and operational
    model_path = Path(settings.word_model_path)
    gate_path = Path(settings.word_gate_path)

    # Status is 'available' if model file exists, else 'withheld'
    if model_path.is_file():
        # Check gate if present
        status: Any = "available"
        if gate_path.is_file():
            try:
                with gate_path.open("r", encoding="utf-8") as gf:
                    gate_data = json.load(gf)
                if not gate_data.get("passed", True):
                    status = "withheld"
            except Exception:
                pass
    else:
        status = "withheld"

    return VocabularyResponse(
        size=len(words),
        status=status,
        words=words,
    )


@router.websocket("/ws/recognize-words")
async def websocket_recognize_words(
    websocket: WebSocket,
    service: WordSignService = Depends(get_word_sign_service),
) -> None:
    """Stream landmark frames over WebSocket for real-time word sequence recognition."""
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            try:
                payload = json.loads(text)
                req = RecognizeWordsRequest.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as e:
                await websocket.send_json({"error": "invalid_payload", "detail": str(e)})
                continue

            frame_obj = _convert_frame_input(req.frame)
            result = service.process_frame(req.session_id, req.language, frame_obj)
            response = _to_response(result)
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        pass
