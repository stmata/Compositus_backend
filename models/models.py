from pydantic import BaseModel
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class LoginRequest(BaseModel):
    email: str

class CandidateSource(str, Enum):
    internal = "internal"
    external = "external"
    all = "all"

class Department(str, Enum):
    All = "All"
    EAP_Professors = "EAP_Professors"
    EAP_Administratif = "EAP_Administratif"
    EAP_Professional_Interviews = "EAP_Professional_Interviews"

class Candidate(BaseModel):
    source: str
    collab_key: str
    name: str
    profile_text: str
    vector: Optional[list[float]] = None
    vector_blob: Optional[str] = None

class JobInfo(BaseModel):
    en: dict = Field(default_factory=dict)
    fr: dict = Field(default_factory=dict)
    react_icon_import: str | None = None

class RankedItem(BaseModel):
    name: str
    profile_text: str
    embedding_score: float

class ExplainItem(BaseModel):
    name: str
    embedding_score: float
    combined_score: float
    llm: dict