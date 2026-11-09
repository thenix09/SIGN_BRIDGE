"""Text-to-sign animation rendering route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_text_to_sign_service
from backend.app.api.schemas import (
    SignFrameResponse,
    SignSegment,
    TextToSignRequest,
    TextToSignResponse,
)
from backend.app.services.text_to_sign import TextToSignService

router = APIRouter()


@router.post("/text-to-sign", response_model=TextToSignResponse)
def text_to_sign(
    request: TextToSignRequest,
    service: TextToSignService = Depends(get_text_to_sign_service),
) -> TextToSignResponse:
    """Convert input text string into animated sign playback frames and segments."""
    result = service.convert(language=request.language, text=request.text)

    frame_responses = [
        SignFrameResponse(
            label=f.label,
            kind=f.kind,
            pose=f.pose,
            left_hand=f.left_hand,
            right_hand=f.right_hand,
            duration_ms=f.duration_ms,
        )
        for f in result.frames
    ]

    segment_responses = [
        SignSegment(
            token=s["token"],
            kind=s["kind"],
            start_index=s["start_index"],
            end_index=s["end_index"],
        )
        for s in result.segments
    ]

    return TextToSignResponse(
        frames=frame_responses,
        segments=segment_responses,
        fallback_tokens=result.fallback_tokens,
    )
