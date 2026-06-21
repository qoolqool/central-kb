"""Pydantic models for the Central KB API."""
from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional


EXPECTED_VEC_DIM = 1024


class EntrySubmission(BaseModel):
    """A single entry being submitted to the central KB (legacy format)."""
    namespace: str = Field(..., pattern=r"^(decisions|patterns|sessions)$")
    key: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: Optional[list[float]] = None
    simhash: int = 0

    @field_validator("vector")
    @classmethod
    def check_vector_dimension(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != EXPECTED_VEC_DIM:
            raise ValueError(
                f"Expected {EXPECTED_VEC_DIM}-dim vector from bge-large-en-v1.5, "
                f"got {len(v)}-dim. All embeddings must be {EXPECTED_VEC_DIM} dimensions."
            )
        return v


class OKFEntrySubmission(BaseModel):
    """A single entry submitted as OKF markdown (frontmatter + body).

    The markdown is parsed server-side to extract frontmatter fields.
    """
    markdown: str = Field(..., description="Full OKF markdown with YAML frontmatter")
    namespace: Optional[str] = Field(None, description="Override namespace (auto-detected from type if omitted)")
    key: Optional[str] = Field(None, description="Override key (auto-detected from title if omitted)")
    vector: Optional[list[float]] = None
    simhash: int = 0

    @field_validator("vector")
    @classmethod
    def check_vector_dimension(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != EXPECTED_VEC_DIM:
            raise ValueError(
                f"Expected {EXPECTED_VEC_DIM}-dim vector from bge-large-en-v1.5, "
                f"got {len(v)}-dim. All embeddings must be {EXPECTED_VEC_DIM} dimensions."
            )
        return v


class SubmitRequest(BaseModel):
    project: str
    source: str = "local:unknown"
    entries: list[EntrySubmission] = Field(default_factory=list)
    okf_entries: list[OKFEntrySubmission] = Field(default_factory=list,
                                                    description="OKF markdown entries")


class SubmitDetail(BaseModel):
    fqn: str
    status: str
    version: Optional[int] = None
    superseded_by: Optional[str] = None
    conflict_id: Optional[int] = None


class SubmitResponse(BaseModel):
    accepted: int
    duplicates: int
    conflicted: int
    conflict_ids: list[int] = Field(default_factory=list)
    details: list[SubmitDetail] = Field(default_factory=list)


class PullEntry(BaseModel):
    fqn: str
    namespace: str
    scope: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: Optional[list[float]] = None
    version: int
    source: str


class DriftWarning(BaseModel):
    your_entry: str
    your_conclusion: str
    other_entry: str
    other_conclusion: str
    topic_similarity: float


class PullResponse(BaseModel):
    entries: list[PullEntry] = Field(default_factory=list)
    drift_warnings: list[DriftWarning] = Field(default_factory=list)
    next_cursor: int


class SearchResult(BaseModel):
    fqn: str
    scope: str
    namespace: str
    title: str
    content: str
    score: float
    cosine_score: float
    fts_score: float
    okf_type: Optional[str] = None
    okf_tags: Optional[list[str]] = None
    okf_description: Optional[str] = None
    okf_timestamp: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult] = Field(default_factory=list)


class Conflict(BaseModel):
    id: int
    existing_fqn: str
    proposed_fqn: str
    proposed_content: str
    similarity: Optional[float] = None
    status: str = "pending"
    created_at: str


class ConflictListResponse(BaseModel):
    conflicts: list[Conflict] = Field(default_factory=list)


class ConflictResolveRequest(BaseModel):
    resolution: str


class Candidate(BaseModel):
    id: int
    candidate_fqn: str
    match_fqns: list[str] = Field(default_factory=list)
    avg_similarity: float
    project_count: int
    status: str = "candidate"


class CandidateListResponse(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)


class PromoteRequest(BaseModel):
    candidate_id: int
    action: str
    verdict_by: str
