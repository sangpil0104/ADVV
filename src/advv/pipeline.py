from __future__ import annotations

import copy
import json
import logging
import platform
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .config import recipe_hash
from .contracts import EditRequest, Source
from .errors import BackendError, DataError, FatalBackendError, ResponseError
from .ingest import scan_sources, snapshot_sources
from .sampler import sample_edit, validate_plan
from .storage import (
    atomic_image,
    atomic_json,
    delete_generated,
    digest,
    file_hash,
    generated_temporary_paths,
    load_rgb,
    now,
    pixel_hash,
    read_json,
    run_lock,
    safe_id,
    within,
)
from .verifiers.decision import decide
from .verifiers.parser import parse_response
from .visualization import render

LOG = logging.getLogger(__name__)
TERMINAL = {"accepted", "rejected", "uncertain", "verification_error", "generation_error"}


def implementation_hash() -> str:
    base = Path(__file__).parent
    return digest({str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob("*.py"))})


def create_run(cfg: dict, run_id: str, *, provenance: str) -> Path:
    safe_id(run_id)
    sources, stats = scan_sources(cfg)
    root = Path(cfg["run"]["output_root"]).resolve()
    # A run cannot nest inside an input image tree that would be ingested on future runs.
    data_root = Path(cfg["dataset"]["root"]).resolve()
    if root.is_relative_to(data_root):
        raise DataError("output_root must be outside dataset root")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=False)
    cfg = copy.deepcopy(cfg)
    atomic_json(run_dir / "config.json", cfg)
    sources = snapshot_sources(sources, run_dir)
    atomic_json(run_dir / "sources.json", [s.to_dict() for s in sources])
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "1.1",
            "run_id": run_id,
            "created_at": now(),
            "backend": provenance,
            "recipe_sha256": recipe_hash(cfg),
            "implementation_sha256": implementation_hash(),
            "ingest": stats,
            "input_manifest_sha256": digest([s.to_dict() for s in sources]),
        },
    )
    atomic_json(
        run_dir / "state.json",
        {
            "status": "running",
            "accepted_count": 0,
            "attempts": 0,
            "cursor": {},
            "next_source": 0,
            "segments": [],
        },
    )
    return run_dir


class Pipeline:
    def __init__(self, run_dir: Path, backend, *, gpu_ids=None):
        self.root = run_dir.resolve()
        self.cfg = read_json(self.root / "config.json")
        self.manifest = read_json(self.root / "manifest.json")
        if recipe_hash(self.cfg) != self.manifest["recipe_sha256"]:
            raise DataError("Run configuration changed; start a new run")
        if implementation_hash() != self.manifest["implementation_sha256"]:
            raise DataError("ADVV implementation changed; use the original code to resume or start a new run")
        self.sources = [Source(**s) for s in read_json(self.root / "sources.json")]
        if digest([s.to_dict() for s in self.sources]) != self.manifest["input_manifest_sha256"]:
            raise DataError("Source snapshot manifest changed")
        for source in self.sources:
            if pixel_hash(within(self.root, source.image_path, must_exist=True)) != source.pixel_sha256:
                raise DataError(f"Input snapshot changed: {source.source_id}")
        self.by_id = {s.source_id: s for s in self.sources}
        self.backend = backend
        if backend.provenance != self.manifest["backend"]:
            raise DataError("Backend provenance changed")
        self.state = read_json(self.root / "state.json")
        self.gpu_ids = gpu_ids or self.cfg["execution"]["selected_gpu_ids"]
        self.records = sorted(
            [read_json(p) for p in (self.root / "records").glob("*.json")], key=lambda r: r["sequence"]
        )
        self.profiles = {}
        self.seen = {s.pixel_sha256 for s in self.sources}
        self.accepted = []

    def save_state(self):
        self.state.update(accepted_count=len(self.accepted), attempts=len(self.records), updated_at=now())
        atomic_json(self.root / "state.json", self.state)

    def save_record(self, record):
        record["updated_at"] = now()
        atomic_json(self.root / f"records/{record['candidate_id']}.json", record)

    def _complete(self, owner, check, images, prompt, schema, max_tokens, save, receipt_base):
        checks = owner.setdefault("checks", {})
        entry = checks.setdefault(check, {"status": "pending", "response": None, "attempts": []})
        if entry["status"] in ("completed", "error", "not_run"):
            return entry
        identity = {
            "check": check,
            "image_hashes_in_order": [pixel_hash(p) for p in images],
            "prompt_hash": digest(prompt),
            "verifier": self.cfg["verifier"],
            "max_new_tokens": max_tokens,
        }
        key = digest(identity)
        if entry.get("cache_key", key) != key:
            raise DataError("Verification inputs changed during resume")
        entry.update(cache_key=key, prompt_sha256=digest(prompt), input_identity=identity)
        max_attempts = 1 + self.cfg["execution"]["max_technical_retries"]
        while len(entry["attempts"]) < max_attempts or (
            entry["attempts"] and entry["attempts"][-1]["status"] == "pending"
        ):
            if not entry["attempts"] or entry["attempts"][-1]["status"] != "pending":
                attempt = {"index": len(entry["attempts"]), "status": "pending", "started_at": now()}
                entry["attempts"].append(attempt)
            else:
                attempt = entry["attempts"][-1]
            receipt = f"{receipt_base}/{check}_{attempt['index']}.json"
            attempt["receipt"] = receipt
            save()
            started = time.monotonic()
            try:
                path = within(self.root, receipt)
                if path.exists():
                    result = read_json(path)
                    if not result["ok"]:
                        cls = FatalBackendError if result["error"]["fatal"] else BackendError
                        raise cls(result["error"]["message"])
                    text, info = result["result"]["text"], result["result"]["info"]
                else:
                    response = self.backend.complete(images, prompt, max_tokens, receipt=receipt)
                    text, info = response.text, response.info
                    # Test backends also have a durable receipt, never silently branded as real.
                    atomic_json(path, {"ok": True, "result": {"text": text, "info": info}})
                attempt.update(raw_completion=text, info=info)
                parsed = parse_response(text, schema)
                attempt.update(status="completed", elapsed_seconds=time.monotonic() - started)
                entry.update(status="completed", response=parsed)
                save()
                return entry
            except (ResponseError, BackendError, FatalBackendError) as exc:
                attempt.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    elapsed_seconds=time.monotonic() - started,
                )
                if isinstance(exc, FatalBackendError):
                    entry.update(status="error", fatal=True)
                    save()
                    raise
                save()
        entry["status"] = "error"
        save()
        return entry

    def profile_sources(self):
        assets = self.cfg["_assets"]
        for source in self.sources:
            if source.split not in self.cfg["dataset"]["eligible_splits"]:
                continue
            path = self.root / f"profiles/{source.source_id}.json"
            profile = (
                read_json(path)
                if path.exists()
                else {
                    "source_id": source.source_id,
                    "source_pixel_sha256": source.pixel_sha256,
                    "preserve_hint": source.preserve_hint,
                    "status": "pending",
                    "checks": {},
                }
            )
            if profile["status"] == "pending":
                prompt = assets["profile_prompt"].replace(
                    "{{user_hint}}", json.dumps(source.preserve_hint, ensure_ascii=False)
                )
                entry = self._complete(
                    profile,
                    "profile",
                    [within(self.root, source.image_path)],
                    prompt,
                    assets["profile_schema"],
                    self.cfg["source_profile"]["max_new_tokens"],
                    lambda: atomic_json(path, profile),
                    f"profiles/{source.source_id}_receipts",
                )
                profile["response"] = entry["response"]
                profile["status"] = (
                    "source_error"
                    if entry["status"] == "error"
                    else "source_uncertain"
                    if entry["response"]["uncertain"]
                    else "ready"
                )
                profile["frozen_sha256"] = digest(
                    {"response": profile["response"], "hint": source.preserve_hint}
                )
                atomic_json(path, profile)
            if profile.get("response") and profile.get("frozen_sha256") != digest(
                {"response": profile["response"], "hint": source.preserve_hint}
            ):
                raise DataError(f"Frozen profile changed: {source.source_id}")
            self.profiles[source.source_id] = profile
        self.eligible = [
            s for s in self.sources if self.profiles.get(s.source_id, {}).get("status") == "ready"
        ]
        if not self.eligible:
            raise DataError("no_eligible_sources: all source profiles are uncertain or invalid")

    def recover(self):
        self.seen = {s.pixel_sha256 for s in self.sources}
        self.accepted = []
        for record in self.records:
            source = self.by_id[record["source_id"]]
            if digest(record["plan"]) != record["plan_sha256"]:
                raise DataError("Stored edit plan changed")
            validate_plan(EditRequest(**record["plan"]), source, self.root)
            if record.get("export_status") == "eligible":
                if decide(record["checks"]) != "accepted" or record["backend"] != self.manifest["backend"]:
                    raise DataError("Invalid accepted record")
                path = within(self.root, record["image_path"], must_exist=True)
                if file_hash(path) != record["file_sha256"] or pixel_hash(path) != record["pixel_sha256"]:
                    raise DataError(f"Accepted artifact changed: {record['candidate_id']}")
                if record["pixel_sha256"] in self.seen:
                    raise DataError("Duplicate in committed accepted artifacts")
                self.seen.add(record["pixel_sha256"])
                self.accepted.append(record)
            sid = record["source_id"]
            self.state["cursor"][sid] = max(
                self.state["cursor"].get(sid, 0), record["plan"]["attempt_index"] + 1
            )
        if len(self.accepted) > self.cfg["run"]["target_count"]:
            raise DataError("Accepted count exceeds quota")
        self.save_state()

    def visualize(self, record):
        try:
            record["visualization"] = render(
                self.root, self.by_id[record["source_id"]].to_dict(), record, self.cfg["visualization"]
            )
        except (OSError, ValueError, DataError) as exc:
            record["visualization"] = {"status": "error", "error": str(exc), "error_type": type(exc).__name__}
            LOG.exception("Visualization failed for %s", record["candidate_id"])
        self.save_record(record)

    def finalize(self, record):
        cid = record["candidate_id"]
        if not record.get("finalized"):
            decision = record["status"]
            for check in ("physical", "semantic"):
                record["checks"].setdefault(check, {"status": "not_run", "response": None, "attempts": []})
            if decision == "accepted":
                if record["pixel_sha256"] in self.seen:
                    record["export_status"] = "duplicate"
                else:
                    destination = f"accepted/{cid}.png"
                    atomic_image(
                        within(self.root, destination),
                        load_rgb(within(self.root, record["image_path"], must_exist=True)),
                    )
                    record.update(
                        image_path=destination,
                        export_status="eligible",
                        file_sha256=file_hash(within(self.root, destination)),
                    )
                    self.seen.add(record["pixel_sha256"])
                    self.accepted.append(record)
            else:
                record["export_status"] = "excluded"
            settings = self.cfg["storage"]
            delete = (
                (decision == "rejected" and settings["delete_rejected_images"])
                or (decision == "uncertain" and not settings["keep_uncertain_images"])
                or (
                    decision in ("verification_error", "generation_error")
                    and not settings["keep_error_images"]
                )
            )
            record["delete_image"] = delete
            if record.get("image_path") and record["export_status"] != "eligible" and not delete:
                destination = f"quarantine/{cid}/generated.png"
                atomic_image(
                    within(self.root, destination),
                    load_rgb(within(self.root, record["image_path"], must_exist=True)),
                )
                record["image_path"] = destination
            record.update(finalized=True, cleanup_pending=True)
            # Decisions and hashes are durable before deletion or cleanup.
            self.save_record(record)
        self.cleanup(record)
        self.visualize(record)

    def cleanup(self, record):
        if not record.get("cleanup_pending"):
            return
        cid = record["candidate_id"]
        paths = [f"candidates/{cid}/generated.png"]
        paths += generated_temporary_paths(self.root, cid)
        if record.get("delete_image"):
            paths += [
                f"quarantine/{cid}/generated.png",
                f"visualizations/{cid}/comparison.png",
                f"visualizations/{cid}/thumbnail.png",
            ]
        # A generation error may have no committed candidate; clean any unfinished output too.
        try:
            for relative in paths:
                delete_generated(self.root, cid, relative)
            if record.get("delete_image"):
                record["image_path"] = None
            record["cleanup_pending"] = False
            record.pop("cleanup_error", None)
        except (OSError, DataError) as exc:
            record["cleanup_error"] = str(exc)
        self.save_record(record)

    def process_candidate(self, record):
        if record.get("source_profile_sha256") != self.profiles[record["source_id"]]["frozen_sha256"]:
            raise DataError("Candidate source profile changed")
        if record.get("effective") and record.get("effective_sha256") != digest(record["effective"]):
            raise DataError("Effective backend instruction changed")
        if record["status"] in TERMINAL:
            self.finalize(record)
            return
        plan = EditRequest(**record["plan"])
        source = self.by_id[record["source_id"]]
        if record["status"] == "planned":
            self.visualize(record)
            attempts = record.setdefault("generation_attempts", [])
            limit = 1 + self.cfg["execution"]["max_technical_retries"]
            while len(attempts) < limit or (attempts and attempts[-1]["status"] == "pending"):
                if not attempts or attempts[-1]["status"] != "pending":
                    attempts.append({"status": "pending", "started_at": now()})
                self.save_record(record)
                started = time.monotonic()
                try:
                    result = self.backend.generate(source, plan, self.root)
                    path = result.path.resolve()
                    expected = within(self.root, f"candidates/{plan.edit_id}/generated.png").resolve()
                    if path != expected:
                        raise FatalBackendError("Generator wrote an unexpected output path")
                    record.update(
                        image_path=str(path.relative_to(self.root)),
                        pixel_sha256=pixel_hash(path),
                        file_sha256=file_hash(path),
                        effective=result.effective,
                        effective_sha256=digest(result.effective),
                        generation_info=result.info,
                        status="generated",
                    )
                    attempts[-1].update(status="completed", elapsed_seconds=time.monotonic() - started)
                    self.save_record(record)
                    break
                except (BackendError, FatalBackendError) as exc:
                    attempts[-1].update(
                        status="error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    if isinstance(exc, FatalBackendError):
                        record["status"] = "generation_error"
                        record["fatal"] = True
                        self.save_record(record)
                        self.finalize(record)
                        raise
                    self.save_record(record)
            if record["status"] != "generated":
                record["status"] = "generation_error"
                self.save_record(record)
                self.finalize(record)
                return
        candidate = within(self.root, record["image_path"], must_exist=True)
        if pixel_hash(candidate) != record["pixel_sha256"]:
            raise DataError("Candidate changed during resume")
        self.visualize(record)
        a = self.cfg["_assets"]

        def save():
            self.save_record(record)

        receipt_base = f"candidates/{record['candidate_id']}/vqa"
        try:
            physical = self._complete(
                record,
                "physical",
                [candidate],
                a["physical_prompt"],
                a["vqa_schema"],
                self.cfg["verifier"]["max_new_tokens"],
                save,
                receipt_base,
            )
            if physical.get("response", {}) and physical["response"]["answer"] == "YES":
                profile = self.profiles[source.source_id]
                context = json.dumps(
                    {"source_profile": profile["response"], "preserve_hint": source.preserve_hint},
                    ensure_ascii=False,
                )
                prompt = a["semantic_prompt"].replace("{{preservation_context}}", context)
                self._complete(
                    record,
                    "semantic",
                    [within(self.root, source.image_path), candidate],
                    prompt,
                    a["vqa_schema"],
                    self.cfg["verifier"]["max_new_tokens"],
                    save,
                    receipt_base,
                )
            else:
                record["checks"]["semantic"] = {"status": "not_run", "response": None, "attempts": []}
            record["status"] = decide(record["checks"])
        except FatalBackendError:
            record["status"] = "verification_error"
            record["fatal"] = True
            self.save_record(record)
            self.finalize(record)
            raise
        self.save_record(record)
        self.finalize(record)

    def run(self):
        with run_lock(self.root):
            # Reload mutable state under the writer lock, not from a pre-lock snapshot.
            self.state = read_json(self.root / "state.json")
            self.records = sorted(
                [read_json(p) for p in (self.root / "records").glob("*.json")], key=lambda r: r["sequence"]
            )
            self.state.update(status="running", last_error=None)
            self.state["segments"].append(
                {
                    "started_at": now(),
                    "selected_gpu_ids": self.gpu_ids,
                    "generator_gpu_ids": self.gpu_ids[:2] if self.gpu_ids else [],
                    "verifier_gpu_ids": self.gpu_ids[:1] if self.gpu_ids else [],
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                }
            )
            self.save_state()
            try:
                self.recover()
                self.profile_sources()
                # Reconstruct round-robin position from the last committed reservation.
                if self.records:
                    last = self.records[-1]["source_id"]
                    self.state["next_source"] = ([s.source_id for s in self.eligible].index(last) + 1) % len(
                        self.eligible
                    )
                for record in self.records:
                    self.process_candidate(record)
                target = self.cfg["run"]["target_count"]
                consecutive_failures = 0
                with tqdm(
                    total=target,
                    initial=len(self.accepted),
                    desc="accepted",
                    disable=not self.cfg["report"]["terminal_progress"],
                ) as bar:
                    while len(self.accepted) < target:
                        index = self.state["next_source"]
                        source = self.eligible[index]
                        attempt_index = self.state["cursor"].get(source.source_id, 0)
                        plan = sample_edit(
                            source,
                            self.profiles[source.source_id]["response"],
                            self.cfg,
                            attempt_index,
                            self.root,
                        )
                        record = {
                            "schema_version": "1.1",
                            "candidate_id": plan.edit_id,
                            "source_id": source.source_id,
                            "sequence": len(self.records),
                            "backend": self.backend.provenance,
                            "plan": plan.to_dict(),
                            "plan_sha256": digest(plan.to_dict()),
                            "source_profile_sha256": self.profiles[source.source_id]["frozen_sha256"],
                            "status": "planned",
                            "checks": {},
                            "created_at": now(),
                            "export_status": "pending",
                        }
                        self.save_record(record)
                        self.records.append(record)
                        self.state["cursor"][source.source_id] = attempt_index + 1
                        self.state["next_source"] = (index + 1) % len(self.eligible)
                        self.save_state()  # Reserve before any inference.
                        self.process_candidate(record)
                        self.save_state()
                        consecutive_failures = (
                            consecutive_failures + 1
                            if record["status"] in ("generation_error", "verification_error")
                            else 0
                        )
                        if (
                            consecutive_failures
                            >= self.cfg["execution"]["max_consecutive_technical_failures"]
                        ):
                            raise FatalBackendError(
                                "Consecutive technical failures; inspect logs before resuming"
                            )
                        bar.update(len(self.accepted) - bar.n)
                        counts = Counter(r["status"] for r in self.records)
                        bar.set_postfix(
                            attempts=len(self.records),
                            rejected=counts["rejected"],
                            uncertain=counts["uncertain"],
                            duplicates=sum(r.get("export_status") == "duplicate" for r in self.records),
                        )
                if any(
                    r.get("cleanup_pending") or r.get("visualization", {}).get("status") == "error"
                    for r in self.records
                ):
                    raise DataError(
                        "Artifacts need recovery; resume repairs cleanup/visualization without extra generation"
                    )
                self.state["status"] = "completed"
            except KeyboardInterrupt:
                self.state["status"] = "interrupted"
            except Exception as exc:
                self.state.update(
                    status="failed", last_error={"type": type(exc).__name__, "message": str(exc)}
                )
                LOG.exception("Run failed; partial results remain in %s", self.root)
            finally:
                self.backend.close()
                self.state["segments"][-1]["ended_at"] = now()
                final_status = self.state["status"]
                if final_status == "completed":
                    self.state["status"] = "running"
                self.save_state()
                from .reporting import export_run, report_run

                try:
                    if self.backend.provenance != "fake":
                        export_run(self.root)
                    report_run(self.root, final_status=final_status)
                    self.state["status"] = final_status
                    self.save_state()
                except (OSError, DataError) as exc:
                    self.state.update(
                        status="failed", last_error={"type": type(exc).__name__, "message": str(exc)}
                    )
                    self.save_state()
                    LOG.exception("Could not finalize export/report")
                    raise
        return self.state
