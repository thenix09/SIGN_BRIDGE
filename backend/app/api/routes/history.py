"""History API endpoints for auditing and managing translation records."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import get_history_service
from backend.app.api.schemas import HistoryItemResponse, HistoryListResponse, OkResponse
from backend.app.services.history import HistoryService

router = APIRouter()


@router.get("/history", response_model=HistoryListResponse)
def get_history(
    limit: int = Query(50, ge=1, le=500, description="Maximum history records to retrieve"),
    service: HistoryService = Depends(get_history_service),
) -> HistoryListResponse:
    """Retrieve recent translation history records ordered most recent first."""
    translations = service.list_recent(limit=limit)
    items = [
        HistoryItemResponse(
            id=t.id,
            direction=t.direction,
            language=t.language,
            content=t.content,
            created_at=t.created_at,
        )
        for t in translations
    ]
    return HistoryListResponse(items=items)


@router.delete("/history", response_model=OkResponse)
def clear_history(
    service: HistoryService = Depends(get_history_service),
) -> OkResponse:
    """Purge all recorded translation history records."""
    service.clear()
    return OkResponse(ok=True)
