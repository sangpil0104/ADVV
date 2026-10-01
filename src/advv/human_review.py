"""Final human pass/fail review of Qwen-accepted images, driven by terminal arrow keys."""

from __future__ import annotations

import getpass
import logging
import os
import select
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from .errors import DataError
from .storage import atomic_image, atomic_json, delete_generated, file_hash, load_rgb, now, read_json, within

LOG = logging.getLogger(__name__)
PASS, FAIL, QUIT = "pass", "fail", "quit"
# Normal and application cursor modes; left passes, right fails.
KEYS = {b"\x1b[D": PASS, b"\x1bOD": PASS, b"\x1b[C": FAIL, b"\x1bOC": FAIL, b"q": QUIT, b"Q": QUIT}
REVIEW_IMAGE = "review/current.png"
PANEL_SIDE = 768


def enabled(cfg: dict) -> bool:
    """Optional stage fixed per run; runs without the key keep the Qwen-only pipeline."""
    return bool(cfg.get("human_review", {}).get("enabled", False))


def records_for(run_dir: Path) -> list[dict]:
    return sorted([read_json(p) for p in (run_dir / "records").glob("*.json")], key=lambda r: r["sequence"])


def save_record(run_dir: Path, record: dict) -> None:
    record["updated_at"] = now()
    atomic_json(run_dir / f"records/{record['candidate_id']}.json", record)


def pending(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("export_status") == "eligible" and "human_review" not in r]


def summary(records: list[dict]) -> dict:
    decisions = [r["human_review"]["decision"] for r in records if "human_review" in r]
    return {
        "pass": decisions.count(PASS),
        "fail": decisions.count(FAIL),
        "pending": len(pending(records)),
    }


def human_rejected_paths(candidate_id: str) -> list[str]:
    return [
        f"accepted/{candidate_id}.png",
        f"visualizations/{candidate_id}/comparison.png",
        f"visualizations/{candidate_id}/thumbnail.png",
    ]


def cleanup(run_dir: Path, record: dict) -> None:
    """Delete a human-rejected image after its decision is durable; retried on the next review."""
    if record.get("export_status") != "human_rejected" or not record.get("cleanup_pending"):
        return
    cid = record["candidate_id"]
    try:
        for relative in human_rejected_paths(cid):
            delete_generated(run_dir, cid, relative, human_rejected=True)
    except (OSError, DataError) as exc:
        record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        save_record(run_dir, record)
        LOG.error("Could not delete human-rejected artifacts for %s: %s", cid, exc)
        return
    record.update(image_path=None, cleanup_pending=False)
    record.pop("cleanup_error", None)
    save_record(run_dir, record)


def apply_decision(run_dir: Path, record: dict, decision: str, reviewer: str) -> None:
    if decision not in (PASS, FAIL):
        raise ValueError(f"Unknown human review decision: {decision}")
    record["human_review"] = {
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": now(),
        "file_sha256": record["file_sha256"],
    }
    if decision == FAIL:
        record.update(export_status="human_rejected", cleanup_pending=True)
    # The decision is durable before any deletion.
    save_record(run_dir, record)
    cleanup(run_dir, record)


def render(run_dir: Path, source: dict, record: dict, position: int, total: int) -> Path:
    panels = []
    for label, relative in (("ORIGINAL", source["image_path"]), ("CANDIDATE", record["image_path"])):
        image = load_rgb(within(run_dir, relative, must_exist=True))
        image.thumbnail((PANEL_SIDE, PANEL_SIDE), Image.Resampling.LANCZOS)
        panels.append((label, image))
    header, gap = 44, 16
    width = sum(im.width for _, im in panels) + gap * (len(panels) + 1)
    height = max(im.height for _, im in panels) + header + gap
    canvas = Image.new("RGB", (width, height), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 6),
        f"[{position}/{total}] {record['candidate_id']}   LEFT = PASS   RIGHT = FAIL   q = quit",
        fill=(255, 255, 255),
    )
    x = gap
    for label, image in panels:
        draw.text((x, header - 18), label, fill=(200, 200, 200))
        canvas.paste(image, (x, header))
        x += image.width + gap
    path = run_dir / REVIEW_IMAGE
    atomic_image(path, canvas)
    return path


def read_key(fd: int) -> str | None:
    """Read one keypress from a cbreak-mode terminal; unknown keys return None."""
    data = os.read(fd, 1)
    if data == b"\x1b":
        # Arrow keys arrive as a short escape sequence.
        while len(data) < 3 and select.select([fd], [], [], 0.05)[0]:
            data += os.read(fd, 1)
    return KEYS.get(data)


def terminal_decisions():
    import termios
    import tty

    if not sys.stdin.isatty():
        raise DataError("Human review needs an interactive terminal")
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # Ctrl+C still raises KeyboardInterrupt.
    try:
        while True:
            key = read_key(fd)
            if key is not None:
                yield key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def review_run(run_dir: Path, *, decisions=None, reviewer: str | None = None, out=sys.stdout) -> dict:
    """Show each Qwen-accepted image once; LEFT keeps it, RIGHT discards it. No regeneration follows."""
    run_dir = run_dir.resolve()
    if not enabled(read_json(run_dir / "config.json")):
        raise DataError("Human review is disabled for this run; create a new run with --human-review")
    state = read_json(run_dir / "state.json")
    if state["status"] != "completed":
        raise DataError("Human review starts after the Qwen stage completes; resume the run first")
    sources = {s["source_id"]: s for s in read_json(run_dir / "sources.json")}
    records = records_for(run_dir)
    for record in records:
        cleanup(run_dir, record)
    queue = pending(records)
    reviewer = reviewer or getpass.getuser()
    if queue:
        print(f"Open {run_dir / REVIEW_IMAGE} in VS Code; it refreshes after each decision.", file=out)
    keys = iter(decisions) if decisions is not None else terminal_decisions()
    done = sum("human_review" in r for r in records)
    total = done + len(queue)
    try:
        for offset, record in enumerate(queue, 1):
            path = within(run_dir, record["image_path"], must_exist=True)
            if file_hash(path) != record["file_sha256"]:
                raise DataError(f"Accepted artifact changed before review: {record['candidate_id']}")
            render(run_dir, sources[record["source_id"]], record, done + offset, total)
            print(f"[{done + offset}/{total}] {record['candidate_id']}  <- pass / -> fail / q quit", file=out)
            decision = next(keys, QUIT)
            if decision == QUIT:
                print("Review paused; run the same command to continue.", file=out)
                break
            apply_decision(run_dir, record, decision, reviewer)
            print(f"  {decision}", file=out)
    finally:
        close = getattr(keys, "close", None)
        if close:
            close()  # Restore the terminal mode.
    result = summary(records_for(run_dir))
    print(f"Human review: pass={result['pass']} fail={result['fail']} pending={result['pending']}", file=out)
    return result
