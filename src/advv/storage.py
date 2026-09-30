from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from .errors import DataError


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise DataError(f"Animated/multiframe image is unsupported: {path}")
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def pixel_hash(image_or_path) -> str:
    image = load_rgb(image_or_path) if isinstance(image_or_path, Path) else image_or_path.convert("RGB")
    h = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_image(path: Path, image: Image.Image) -> None:
    output = io.BytesIO()
    image.save(output, format="PNG")
    atomic_bytes(path, output.getvalue())


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise DataError(f"Invalid ID: {value!r}")
    return value


def within(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise DataError(f"Expected a relative path within {root}: {relative}")
    root = root.resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise DataError(f"Path escapes root: {relative}")
    if must_exist and not path.is_file():
        raise DataError(f"Missing file: {path}")
    return path


def delete_generated(run_dir: Path, candidate_id: str, relative: str) -> None:
    """Delete only known generated artifacts for this candidate, never source/accepted files."""
    safe_id(candidate_id)
    allowed = {
        f"candidates/{candidate_id}/generated.png",
        f"quarantine/{candidate_id}/generated.png",
        f"visualizations/{candidate_id}/comparison.png",
        f"visualizations/{candidate_id}/thumbnail.png",
    }
    p = Path(relative)
    # Interrupted atomic PNG writes can leave mkstemp files in the same owned directory.
    temp_owned = any(
        p.parent == Path(name).parent
        and re.fullmatch(re.escape("." + Path(name).name + ".") + r"[A-Za-z0-9_\-]+", p.name)
        for name in allowed
    )
    if relative not in allowed and not temp_owned:
        raise DataError(f"Unowned deletion request: {relative}")
    path = within(run_dir, relative)
    current = path
    while current != run_dir.resolve():
        if current.is_symlink():
            raise DataError(f"Refusing symlink deletion: {path}")
        current = current.parent
    if path.exists():
        if not path.is_file() or path.stat().st_nlink > 1:
            raise DataError(f"Refusing non-regular or hard-linked artifact: {path}")
        path.unlink()


def generated_temporary_paths(run_dir: Path, candidate_id: str) -> list[str]:
    safe_id(candidate_id)
    paths = []
    for relative, stem in (
        (f"candidates/{candidate_id}", "generated.png"),
        (f"quarantine/{candidate_id}", "generated.png"),
        (f"visualizations/{candidate_id}", "comparison.png"),
        (f"visualizations/{candidate_id}", "thumbnail.png"),
    ):
        folder = within(run_dir, relative)
        paths.extend(str(p.relative_to(run_dir)) for p in folder.glob(f".{stem}.*"))
    return paths


@contextlib.contextmanager
def run_lock(run_dir: Path):
    with (run_dir / ".writer.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DataError(f"Another writer holds this run: {run_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
