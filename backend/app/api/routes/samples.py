"""Sample collection API routes for letter training data."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, Response

from backend.app.api.deps import get_sample_repository
from backend.app.api.schemas import (
    CreateSampleRequest,
    CreateSampleResponse,
    SampleStatsResponse,
)
from backend.app.domain.hand_frame import HandFrame, canonicalise_landmarks
from backend.app.domain.sample import Sample
from backend.app.repositories.sample_repo import SQLiteSampleRepository

router = APIRouter()


@router.post("/samples", response_model=CreateSampleResponse)
def create_sample(
    request: CreateSampleRequest,
    repo: SQLiteSampleRepository = Depends(get_sample_repository),
) -> CreateSampleResponse:
    """Store raw collected hand landmarks as a normalized, scale-invariant 63-feature sample."""
    # 1. Parse raw landmarks and canonicalize handedness to reference ("Right")
    canonical_raw = canonicalise_landmarks(
        raw=request.landmarks,
        handedness=request.handedness,
        canonical="Right",
    )
    hand_frame = HandFrame.from_raw(canonical_raw)

    # 2. Extract invariant 63-feature vector
    features = hand_frame.to_feature_vector()

    # 3. Persist to sample repository
    domain_sample = Sample(
        label=request.label,
        group=request.group,
        features=tuple(features),
    )
    sample_id, _ = repo.add_sample(domain_sample)
    count = repo.count_for_label(request.label)

    return CreateSampleResponse(id=sample_id, count_for_label=count)


@router.get("/samples/stats", response_model=SampleStatsResponse)
def get_sample_stats(
    repo: SQLiteSampleRepository = Depends(get_sample_repository),
) -> SampleStatsResponse:
    """Return collected sample counts grouped by letter label and overall total."""
    counts, total = repo.get_stats()
    return SampleStatsResponse(counts=counts, total=total)


@router.get("/samples/export")
def export_samples(
    repo: SQLiteSampleRepository = Depends(get_sample_repository),
) -> Response:
    """Export all recorded letter samples as a downloadable CSV dataset."""
    output = io.StringIO()
    writer = csv.writer(output)

    header = ["label", "group"] + [f"f{i}" for i in range(63)]
    writer.writerow(header)

    samples = repo.list(limit=100_000)
    for s in samples:
        writer.writerow([s.label, s.group] + list(s.features))

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=samples.csv"},
    )
