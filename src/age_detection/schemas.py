"""Public API models (also the source of the generated OpenAPI document)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from age_detection.age import AGE_GROUPS, AgeEstimate
from age_detection.detector import Face
from age_detection.jobs import JobStatus


class Box(BaseModel):
    x: float = Field(description="Left edge in pixels of the (EXIF-upright) image")
    y: float = Field(description="Top edge in pixels")
    width: float
    height: float


class Point(BaseModel):
    x: float
    y: float


class Landmarks(BaseModel):
    right_eye: Point
    left_eye: Point
    nose: Point
    right_mouth: Point
    left_mouth: Point


class AgeOut(BaseModel):
    """Age *group* estimate. Not an exact age, and not a reliable one (see the model card)."""

    group: str = Field(description="One of 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100")
    age_low: int
    age_high: int
    confidence: float = Field(ge=0, le=1, description="Mean probability of the top group over the views")
    agreement: float = Field(ge=0, le=1, description="Fraction of views that chose the same group")
    views: int
    uncertain: bool = Field(description="confidence is below the configured threshold: treat the group as unreliable")
    probabilities: dict[str, float] = Field(description="Mean probability per age group")

    @classmethod
    def from_estimate(cls, e: AgeEstimate) -> AgeOut:
        return cls(
            group=e.group,
            age_low=e.age_low,
            age_high=e.age_high,
            confidence=round(e.confidence, 4),
            agreement=round(e.agreement, 3),
            views=e.views,
            uncertain=e.uncertain,
            probabilities={g: round(p, 4) for g, p in zip(AGE_GROUPS, e.probabilities, strict=True)},
        )


class FaceOut(BaseModel):
    box: Box
    score: float = Field(ge=0, le=1, description="Face detection confidence")
    landmarks: Landmarks
    age: AgeOut

    @classmethod
    def from_face(cls, face: Face, estimate: AgeEstimate) -> FaceOut:
        re, le, nose, rm, lm = (Point(x=x, y=y) for x, y in face.landmarks)
        return cls(
            box=Box(x=face.x, y=face.y, width=face.width, height=face.height),
            score=face.score,
            landmarks=Landmarks(right_eye=re, left_eye=le, nose=nose, right_mouth=rm, left_mouth=lm),
            age=AgeOut.from_estimate(estimate),
        )


class ImageInfo(BaseModel):
    width: int
    height: int


class ModelInfo(BaseModel):
    name: str
    sha256: str
    execution_providers: list[str]


class ModelsInfo(BaseModel):
    face_detector: ModelInfo
    age_classifier: ModelInfo
    age_views: int


class EstimateResult(BaseModel):
    image: ImageInfo
    faces: list[FaceOut]
    inference_ms: float = Field(description="Server-side time for detection and age estimation (no network/queueing)")
    models: ModelsInfo


class EstimateResponse(EstimateResult):
    request_id: str


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    attempts: int
    result: EstimateResult | None = None
    error: str | None = None


class ProblemDetails(BaseModel):
    """RFC 9457 problem document."""

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None
    request_id: str | None = None
    errors: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    version: str | None = None
