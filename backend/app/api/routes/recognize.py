"""Recognition and session API routes for fingerspelling Spell mode."""

import json

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.app.api.deps import get_sign_to_text_service
from backend.app.api.schemas import (
    OkResponse,
    RecognizeRequest,
    RecognizeResponse,
    SessionActionRequest,
    SessionResetRequest,
)
from backend.app.core.exceptions import SignBridgeError
from backend.app.services.sign_to_text import SignToTextService

router = APIRouter(tags=["recognize"])


@router.post("/recognize", response_model=RecognizeResponse)
async def recognize_letter(
    request: RecognizeRequest,
    service: SignToTextService = Depends(get_sign_to_text_service),
) -> RecognizeResponse:
    """Test endpoint for single-frame letter recognition."""
    res = service.process_frame(
        session_id=request.session_id,
        language=request.language,
        raw_landmarks=request.landmarks,
        handedness=request.handedness,
    )
    return RecognizeResponse(
        label=res.label,
        confidence=res.confidence,
        state=res.state,
        confirmed_letter=res.confirmed_letter,
        current_word=res.current_word,
        sentence=res.sentence,
        suggestions=res.suggestions,
    )


@router.post("/session/reset", response_model=OkResponse)
async def reset_session(
    request: SessionResetRequest,
    service: SignToTextService = Depends(get_sign_to_text_service),
) -> OkResponse:
    """Reset filter state and in-progress spelled word for a session."""
    service.reset_session(request.session_id)
    return OkResponse(ok=True)


@router.post("/session/backspace", response_model=RecognizeResponse)
async def session_backspace(
    request: SessionActionRequest,
    service: SignToTextService = Depends(get_sign_to_text_service),
) -> RecognizeResponse:
    """Delete last character from current word or reopen last committed word."""
    res = service.backspace(request.session_id)
    return RecognizeResponse(
        label=res.label,
        confidence=res.confidence,
        state=res.state,
        confirmed_letter=res.confirmed_letter,
        current_word=res.current_word,
        sentence=res.sentence,
        suggestions=res.suggestions,
    )


@router.post("/session/commit", response_model=RecognizeResponse)
async def session_commit(
    request: SessionActionRequest,
    service: SignToTextService = Depends(get_sign_to_text_service),
) -> RecognizeResponse:
    """Commit current spelled word or selected suggestion to the sentence."""
    res = service.commit_word(request.session_id, request.word)
    return RecognizeResponse(
        label=res.label,
        confidence=res.confidence,
        state=res.state,
        confirmed_letter=res.confirmed_letter,
        current_word=res.current_word,
        sentence=res.sentence,
        suggestions=res.suggestions,
    )


@router.websocket("/ws/recognize")
async def websocket_recognize(
    websocket: WebSocket,
    service: SignToTextService = Depends(get_sign_to_text_service),
) -> None:
    """Streaming WebSocket endpoint for real-time video frame fingerspelling.

    Keeps connection open on invalid payloads, returning an error response.
    """
    await websocket.accept()

    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                payload = json.loads(text_data)
                req = RecognizeRequest(**payload)
                res = service.process_frame(
                    session_id=req.session_id,
                    language=req.language,
                    raw_landmarks=req.landmarks,
                    handedness=req.handedness,
                )
                response_dict = {
                    "label": res.label,
                    "confidence": res.confidence,
                    "state": res.state,
                    "confirmed_letter": res.confirmed_letter,
                    "current_word": res.current_word,
                    "sentence": res.sentence,
                    "suggestions": res.suggestions,
                }
                await websocket.send_json(response_dict)
            except ValidationError as ve:
                await websocket.send_json({"error": "VALIDATION_ERROR", "detail": str(ve)})
            except SignBridgeError as se:
                await websocket.send_json({"error": se.error_code, "detail": se.message})
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"error": "INVALID_JSON", "detail": "Invalid JSON text received"}
                )
    except WebSocketDisconnect:
        pass
