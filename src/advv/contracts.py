from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Source:
    source_id: str
    image_path: str
    original_path: str
    file_sha256: str
    pixel_sha256: str
    width: int
    height: int
    split: str = "source"
    group_id: str = ""
    domain: str = "unspecified"
    preserve_hint: str | None = None
    label: dict | None = None
    group_inferred: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EditRequest:
    source_id: str
    edit_id: str
    attempt_index: int
    seed: int
    sampler_version: str
    operation: str
    operation_params: dict
    region_mask_path: str
    source_point: list[float]
    target_point: list[float]
    anchor_point: list[float] | None
    source_prompt: str
    target_prompt: str
    mask_sha256: str
    schema_version: str = "1.1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RawResponse:
    text: str
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generated:
    path: Path
    effective: dict
    info: dict = field(default_factory=dict)


class Backend(Protocol):
    provenance: str

    def complete(
        self, images: list[Path], prompt: str, max_new_tokens: int, *, receipt=None
    ) -> RawResponse: ...

    def generate(self, source: Source, plan: EditRequest, run_dir: Path) -> Generated: ...

    def close(self) -> None: ...
