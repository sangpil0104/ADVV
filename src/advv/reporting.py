from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

from .errors import DataError
from .storage import atomic_bytes, atomic_json, file_hash, pixel_hash, read_json, within
from .verifiers.decision import decide


def records_for(run_dir):
    return sorted([read_json(p) for p in (run_dir / "records").glob("*.json")], key=lambda r: r["sequence"])


def export_run(run_dir: Path) -> Path:
    import json

    manifest = read_json(run_dir / "manifest.json")
    if manifest["backend"] != "dragflow+qwen_local":
        raise DataError("Production export refuses fake or unrecognized backends")
    sources = {s["source_id"]: s for s in read_json(run_dir / "sources.json")}
    seen = {s["pixel_sha256"] for s in sources.values()}
    rows = []
    for r in records_for(run_dir):
        if r.get("export_status") != "eligible":
            continue
        source = sources[r["source_id"]]
        if (
            r["backend"] != manifest["backend"]
            or decide(r["checks"]) != "accepted"
            or source["split"] not in ("source", "train")
        ):
            raise DataError("Ineligible record in export")
        path = within(run_dir, r["image_path"], must_exist=True)
        pixels = pixel_hash(path)
        if pixels != r["pixel_sha256"] or file_hash(path) != r["file_sha256"] or pixels in seen:
            raise DataError("Export artifact failed hash/dedup verification")
        seen.add(pixels)
        rows.append(
            {
                "schema_version": "1.1",
                "candidate_id": r["candidate_id"],
                "source_id": r["source_id"],
                "image_path": r["image_path"],
                "source_image_path": source["image_path"],
                "split": source["split"],
                "group_id": source["group_id"],
                "domain": source["domain"],
                "label": None,
                "task": "image_only",
                "pixel_sha256": pixels,
                "file_sha256": r["file_sha256"],
                "backend": r["backend"],
                "record_path": f"records/{r['candidate_id']}.json",
                "visualization_artifacts": r.get("visualization", {}).get("artifacts", {}),
            }
        )
    destination = run_dir / "exports/images.jsonl"
    atomic_bytes(destination, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    return destination


def report_run(run_dir: Path, *, final_status=None) -> Path:
    state = read_json(run_dir / "state.json")
    if final_status is not None:
        state["status"] = final_status
    records = records_for(run_dir)
    counts = dict(Counter(r["status"] for r in records))
    sources = read_json(run_dir / "sources.json")
    per_source = {}
    for source in sources:
        subset = [r for r in records if r["source_id"] == source["source_id"]]
        per_source[source["source_id"]] = {
            "attempts": len(subset),
            "accepted": sum(r.get("export_status") == "eligible" for r in subset),
        }
    accepted = sum(r.get("export_status") == "eligible" for r in records)
    physical_no = sum(
        r.get("checks", {}).get("physical", {}).get("response", {}).get("answer") == "NO"
        for r in records
        if r.get("checks", {}).get("physical", {}).get("response")
    )
    semantic_no = sum(
        r.get("checks", {}).get("semantic", {}).get("response", {}).get("answer") == "NO"
        for r in records
        if r.get("checks", {}).get("semantic", {}).get("response")
    )
    cfg = read_json(run_dir / "config.json")
    result = {
        "run_id": run_dir.name,
        "backend": read_json(run_dir / "manifest.json")["backend"],
        "status": state["status"],
        "accepted": accepted,
        "target": cfg["run"]["target_count"],
        "attempts": len(records),
        "status_counts": counts,
        "physical_no": physical_no,
        "semantic_no": semantic_no,
        "duplicates": sum(r.get("export_status") == "duplicate" for r in records),
        "acceptance_rate": accepted / len(records) if records else None,
        "per_source": per_source,
        "last_error": state.get("last_error"),
        "cleanup_pending": [r["candidate_id"] for r in records if r.get("cleanup_pending")],
        "visualization_errors": [
            r["candidate_id"] for r in records if r.get("visualization", {}).get("status") == "error"
        ],
    }
    profiles = [read_json(p) for p in (run_dir / "profiles").glob("*.json")]
    stage_seconds = {
        "generation": sum(
            a.get("elapsed_seconds", 0) for r in records for a in r.get("generation_attempts", [])
        )
    }
    for stage, owners in [("profile", profiles), ("physical", records), ("semantic", records)]:
        stage_seconds[stage] = sum(
            a.get("elapsed_seconds", 0)
            for owner in owners
            for a in owner.get("checks", {}).get(stage, {}).get("attempts", [])
        )
    result["stage_elapsed_seconds"] = stage_seconds
    result["peak_vram_bytes"] = {
        "generator_max_single_logical_device": max(
            [v for r in records for v in r.get("generation_info", {}).get("peak_vram_bytes", [])] or [0]
        ),
        "qwen_max_single_logical_device": max(
            [
                a.get("info", {}).get("peak_vram_bytes", 0)
                for owner in records + profiles
                for c in owner.get("checks", {}).values()
                for a in c.get("attempts", [])
            ]
            or [0]
        ),
    }
    atomic_json(run_dir / "report.json", result)
    lines = [
        f"# ADVV {run_dir.name}",
        "",
        f"Status: **{state['status']}** | Backend: `{result['backend']}`",
        "",
        f"Accepted: **{accepted}/{result['target']}** | Attempts: {len(records)} | Duplicates: {result['duplicates']}",
        "",
        "Qwen judgments are visual estimates, not physical proof or supervised ground truth.",
        "",
        "| Candidate | Final status | Export | Preview |",
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        links = []
        for name, relative in record.get("visualization", {}).get("artifacts", {}).items():
            if within(run_dir, relative).is_file():
                links.append(f"[{name}]({relative})")
        lines.append(
            f"| {html.escape(record['candidate_id'])} | {record['status']} | {record.get('export_status', 'pending')} | {' / '.join(links)} |"
        )
    if state.get("last_error"):
        lines.extend(["", "Last error: " + html.escape(str(state["last_error"]))])
    path = run_dir / "report.md"
    atomic_bytes(path, ("\n".join(lines) + "\n").encode())
    return path
