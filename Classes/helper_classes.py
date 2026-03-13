import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator, validator


@dataclass
class SQLLineageResult:
    """Data class for SQL lineage extraction result"""
    target: str
    sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {"target": self.target, "sources": self.sources}

    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def source_count(self) -> int:
        """Get number of source tables"""
        return len(self.sources)

    def add_source(self, source: str) -> None:
        """Add a source table if not already present"""
        if source and source not in self.sources:
            self.sources.append(source)

    def __str__(self) -> str:
        """String representation"""
        return f"Target: {self.target}, Sources: {', '.join(self.sources)}"


class SQLDependencies(BaseModel):
    """Pydantic model for SQL lineage output"""
    target: str = Field(description="The main object being created or modified (fully qualified name)")
    sources: List[str] = Field(description="List of DISTINCT base tables/views (fully qualified names)")

    @field_validator('target')
    def normalize_target(cls, v):
        """Normalize target name"""
        if v:
            # Remove quotes and normalize
            v = v.replace('"', '').replace("'", "")
        return v.lower() if v else v

    @validator('sources', each_item=True)
    def normalize_source(cls, v):
        """Normalize source names"""
        if v:
            # Remove quotes and normalize
            v = v.replace('"', '').replace("'", "")
        return v.lower() if v else v

    def to_lineage_result(self) -> 'SQLLineageResult':
        """Convert to SQLLineageResult"""
        return SQLLineageResult(target=self.target, sources=self.sources)

