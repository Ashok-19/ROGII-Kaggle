#!/usr/bin/env python3
"""ROGII project ledger and local dashboard.

The durable sources of truth are tracking/seed.json, experiment manifests, and
tracking/inbox events. tracking/rogii.sqlite is generated and may be deleted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_metric_arg(text: str) -> dict[str, Any]:
    # Accepted: cv:rmse=7.5 or rmse=7.5 (defaults to cv).
    try:
        left, raw = text.rsplit("=", 1)
        value = float(raw)
    except Exception as exc:
        raise argparse.ArgumentTypeError("metric must be [split:]name=value") from exc
    if ":" in left:
        split, name = left.split(":", 1)
    else:
        split, name = "cv", left
    if not split or not name:
        raise argparse.ArgumentTypeError("metric must be [split:]name=value")
    return {"split": split, "name": name, "value": value}


class Tracker:
    def __init__(self, root: Path, db_path: Path | None = None) -> None:
        self.root = root.resolve()
        self.db_path = (db_path or self.root / "tracking" / "rogii.sqlite").resolve()
        self.schema_path = self.root / "tracking" / "schema.sql"
        self.seed_path = self.root / "tracking" / "seed.json"
        self._last_signature = ""
        self._last_sync_monotonic = 0.0
        self._learning_well_cache: dict[str, dict[str, Any]] = {}

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def init(self, reset: bool = False) -> None:
        if reset:
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(self.db_path) + suffix)
                if candidate.exists():
                    candidate.unlink()
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Missing schema: {self.schema_path}")
        with self.connect() as con:
            con.executescript(self.schema_path.read_text(encoding="utf-8"))
        self.seed()

    @staticmethod
    def _upsert(con: sqlite3.Connection, table: str, row: dict[str, Any], key: str) -> None:
        cols = list(row)
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != key)
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        if updates:
            sql += f" ON CONFLICT({key}) DO UPDATE SET {updates}"
        con.execute(sql, [row[c] for c in cols])

    def seed(self) -> None:
        if not self.seed_path.exists():
            raise FileNotFoundError(f"Missing seed: {self.seed_path}")
        seed = read_json(self.seed_path)
        now = utcnow()
        with self.connect() as con:
            for key, value in seed.get("meta", {}).items():
                con.execute(
                    "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value)),
                )
            for row in seed.get("sources", []):
                self._upsert(con, "sources", row, "source_id")
            for row in seed.get("claims", []):
                self._upsert(con, "claims", row, "claim_id")
            for row in seed.get("hypotheses", []):
                item = {**row, "created_at": row.get("created_at", now), "updated_at": row.get("updated_at", now)}
                self._upsert(con, "hypotheses", item, "hypothesis_id")
            for row in seed.get("decisions", []):
                self._upsert(con, "decisions", row, "decision_id")
            for row in seed.get("tasks", []):
                item = {**row, "created_at": row.get("created_at", now), "updated_at": row.get("updated_at", now)}
                self._upsert(con, "tasks", item, "task_id")
            for row in seed.get("submissions", []):
                self._upsert(con, "submissions", row, "submission_id")

    def source_signature(self) -> str:
        paths = [self.seed_path, self.schema_path]
        paths += sorted((self.root / "experiments").glob("**/manifest.json"))
        paths += sorted((self.root / "tracking" / "inbox").glob("*.json"))
        payload = []
        for path in paths:
            if path.exists():
                stat = path.stat()
                payload.append(f"{path.relative_to(self.root)}:{stat.st_mtime_ns}:{stat.st_size}")
        return hashlib.sha256("\n".join(payload).encode()).hexdigest()

    def sync_if_needed(self, force: bool = False) -> dict[str, Any]:
        signature = self.source_signature()
        stale_by_time = time.monotonic() - self._last_sync_monotonic > 10
        if force or signature != self._last_signature or stale_by_time or not self.db_path.exists():
            result = self.sync()
            self._last_signature = self.source_signature()
            self._last_sync_monotonic = time.monotonic()
            return result
        return {"status": "unchanged", "signature": signature}

    def sync(self) -> dict[str, Any]:
        self.init(reset=False)
        manifests = sorted((self.root / "experiments").glob("**/manifest.json"))
        inbox = sorted((self.root / "tracking" / "inbox").glob("*.json"))
        result = {"status": "ok", "manifests": 0, "runs": 0, "events": 0, "errors": []}
        for path in manifests:
            try:
                result["runs"] += self._sync_manifest(path)
                result["manifests"] += 1
            except Exception as exc:
                result["errors"].append(f"{path.relative_to(self.root)}: {exc}")
        for path in inbox:
            try:
                self._sync_event(path)
                result["events"] += 1
            except Exception as exc:
                result["errors"].append(f"{path.relative_to(self.root)}: {exc}")
        with self.connect() as con:
            con.execute(
                "INSERT INTO sync_events(synced_at,source_path,status,detail) VALUES(?,?,?,?)",
                (utcnow(), "project", "error" if result["errors"] else "ok", json_text(result)),
            )
        if result["errors"]:
            result["status"] = "partial"
        return result

    def _sync_manifest(self, path: Path) -> int:
        data = read_json(path)
        required = ["experiment_id", "title", "hypothesis", "status", "track", "created_at", "updated_at"]
        missing = [x for x in required if not data.get(x)]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        exp = {
            "experiment_id": data["experiment_id"],
            "title": data["title"],
            "hypothesis": data["hypothesis"],
            "status": data["status"],
            "track": data["track"],
            "priority": int(data.get("priority", 99)),
            "parent_id": data.get("parent_id"),
            "owner": data.get("owner"),
            "manifest_path": str(path.relative_to(self.root)),
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }
        run_count = 0
        with self.connect() as con:
            self._upsert(con, "experiments", exp, "experiment_id")
            for run in data.get("runs", []):
                run_id = run.get("run_id")
                if not run_id:
                    raise ValueError("run without run_id")
                item = {
                    "run_id": run_id,
                    "experiment_id": data["experiment_id"],
                    "started_at": run.get("started_at"),
                    "ended_at": run.get("ended_at"),
                    "status": run.get("status", "unknown"),
                    "git_sha": run.get("git_sha"),
                    "fold_map": run.get("fold_map"),
                    "config_path": run.get("config_path"),
                    "artifact_dir": run.get("artifact_dir"),
                    "controls_json": json_text(run.get("controls", {})),
                    "notes": run.get("notes"),
                    "updated_at": run.get("updated_at") or run.get("ended_at") or data["updated_at"],
                }
                self._upsert(con, "runs", item, "run_id")
                for metric in run.get("metrics", []):
                    con.execute(
                        """INSERT INTO metrics(run_id,split,name,value,unit,fold,well_id,created_at)
                           VALUES(?,?,?,?,?,?,?,?)
                           ON CONFLICT(run_id,split,name,fold,well_id)
                           DO UPDATE SET value=excluded.value,unit=excluded.unit,created_at=excluded.created_at""",
                        (
                            run_id,
                            metric.get("split", "cv"),
                            metric["name"],
                            float(metric["value"]),
                            metric.get("unit"),
                            str(metric.get("fold", "")),
                            str(metric.get("well_id", "")),
                            metric.get("created_at") or item["updated_at"],
                        ),
                    )
                self._sync_well_metrics(con, run_id, run.get("well_metrics_path"))
                for artifact in run.get("artifacts", []):
                    artifact_path = artifact.get("path")
                    artifact_id = artifact.get("artifact_id") or hashlib.sha256(
                        f"{run_id}:{artifact_path}".encode()
                    ).hexdigest()[:20]
                    self._upsert(
                        con,
                        "artifacts",
                        {
                            "artifact_id": artifact_id,
                            "run_id": run_id,
                            "path": artifact_path,
                            "kind": artifact.get("kind", "unknown"),
                            "sha256": artifact.get("sha256"),
                            "bytes": artifact.get("bytes"),
                            "created_at": artifact.get("created_at") or item["updated_at"],
                        },
                        "artifact_id",
                    )
                run_count += 1
        return run_count

    def _sync_well_metrics(self, con: sqlite3.Connection, run_id: str, relative: str | None) -> None:
        if not relative:
            return
        path = self.root / relative
        if not path.exists():
            raise FileNotFoundError(f"well_metrics_path does not exist: {relative}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                con.execute(
                    """INSERT INTO well_metrics(run_id,well_id,split,rows_scored,rmse,mean_error,sse,regime,uncertainty)
                       VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(run_id,well_id,split) DO UPDATE SET
                       rows_scored=excluded.rows_scored,rmse=excluded.rmse,mean_error=excluded.mean_error,
                       sse=excluded.sse,regime=excluded.regime,uncertainty=excluded.uncertainty""",
                    (
                        run_id,
                        row["well_id"],
                        row.get("split", "cv"),
                        int(row["rows_scored"]) if row.get("rows_scored") else None,
                        float(row["rmse"]) if row.get("rmse") else None,
                        float(row["mean_error"]) if row.get("mean_error") else None,
                        float(row["sse"]) if row.get("sse") else None,
                        row.get("regime"),
                        float(row["uncertainty"]) if row.get("uncertainty") else None,
                    ),
                )

    def _sync_event(self, path: Path) -> None:
        event = read_json(path)
        kind = event.get("type")
        payload = event.get("payload", {})
        now = event.get("recorded_at", utcnow())
        with self.connect() as con:
            if kind == "submission":
                self._upsert(con, "submissions", payload, "submission_id")
            elif kind == "task":
                payload = {**payload, "created_at": payload.get("created_at", now), "updated_at": now}
                self._upsert(con, "tasks", payload, "task_id")
            elif kind == "hypothesis":
                payload = {**payload, "created_at": payload.get("created_at", now), "updated_at": now}
                self._upsert(con, "hypotheses", payload, "hypothesis_id")
            elif kind == "decision":
                self._upsert(con, "decisions", payload, "decision_id")
            else:
                raise ValueError(f"unsupported event type: {kind!r}")

    def state(self) -> dict[str, Any]:
        self.sync_if_needed()
        with self.connect() as con:
            meta = {r["key"]: r["value"] for r in con.execute("SELECT key,value FROM meta")}
            submissions = [dict(r) for r in con.execute("SELECT * FROM submissions ORDER BY submitted_at")]
            experiments = [dict(r) for r in con.execute("SELECT * FROM v_experiment_summary ORDER BY priority,experiment_id")]
            runs = [dict(r) for r in con.execute(
                """SELECT r.*,v.cv_rmse,v.worst_5pct_sse_share,v.worst_10pct_sse_share,
                          v.median_well_rmse,v.p90_well_rmse
                   FROM runs r LEFT JOIN v_run_cv v ON v.run_id=r.run_id
                   ORDER BY COALESCE(r.ended_at,r.updated_at) DESC LIMIT 100"""
            )]
            hypotheses = [dict(r) for r in con.execute("SELECT * FROM hypotheses ORDER BY priority,hypothesis_id")]
            tasks = [dict(r) for r in con.execute("SELECT * FROM tasks ORDER BY CASE status WHEN 'next' THEN 0 WHEN 'active' THEN 1 WHEN 'queued' THEN 2 WHEN 'blocked' THEN 3 ELSE 4 END,priority,task_id")]
            decisions = [dict(r) for r in con.execute("SELECT * FROM decisions ORDER BY decided_at DESC")]
            claims = [dict(r) for r in con.execute("SELECT * FROM claims ORDER BY claim_id")]
            sources = [dict(r) for r in con.execute("SELECT * FROM sources ORDER BY source_id")]
            sync_events = [dict(r) for r in con.execute("SELECT * FROM sync_events ORDER BY sync_id DESC LIMIT 15")]
            best_cv_row = con.execute("SELECT MIN(value) AS value FROM metrics WHERE split IN ('cv','oof') AND name='rmse'").fetchone()
            integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
        scored = [x["public_score"] for x in submissions if x["public_score"] is not None]
        best_public = min(scored) if scored else None
        leader = float(meta.get("leader_score", "nan"))
        deadline = datetime.fromisoformat(meta["final_deadline_utc"].replace("Z", "+00:00"))
        days_left = max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds() / 86400)
        bands = self._leaderboard_bands()
        return {
            "generated_at": utcnow(),
            "meta": meta,
            "kpis": {
                "leader_score": leader,
                "best_verified_public": best_public,
                "user_reported_best": float(meta.get("user_reported_best_score", "nan")),
                "gap_to_leader": (best_public - leader) if best_public is not None else None,
                "best_cv_rmse": best_cv_row["value"] if best_cv_row else None,
                "days_left": days_left,
                "experiment_count": len(experiments),
                "run_count": len(runs),
                "submission_count": len(submissions),
                "active_hypotheses": sum(1 for x in hypotheses if x["status"] in ("active", "queued")),
                "open_tasks": sum(1 for x in tasks if x["status"] not in ("done", "rejected")),
                "db_integrity": integrity,
            },
            "leaderboard_bands": bands,
            "submissions": submissions,
            "experiments": experiments,
            "runs": runs,
            "hypotheses": hypotheses,
            "tasks": tasks,
            "decisions": decisions,
            "claims": claims,
            "sources": sources,
            "sync_events": sync_events,
            "validation_protocol": [
                {"name": "Whole-well suffix split", "required": True},
                {"name": "Pooled row RMSE", "required": True},
                {"name": "Positive control", "required": True},
                {"name": "No-op / duplicate controls", "required": True},
                {"name": "Shuffled-evidence control", "required": True},
                {"name": "Repeated fold maps", "required": True},
                {"name": "Worst-well SSE concentration", "required": True},
                {"name": "Harsher distribution-shift split", "required": True},
                {"name": "Residual-correlation report", "required": True},
                {"name": "Offline notebook parity under 8h target", "required": True},
            ],
        }

    def _leaderboard_bands(self) -> list[dict[str, Any]]:
        path = self.root / "archive" / "competition" / "leaderboard_2026-07-18.csv"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        out = []
        for rank in (1, 2, 3, 5, 10, 20, 25, 50, 75, 100):
            if len(rows) >= rank:
                row = rows[rank - 1]
                out.append({"rank": rank, "score": float(row["score"]), "team": row["team_name"]})
        return out

    @staticmethod
    def _sample_indices(length: int, max_points: int, must_include: Iterable[int] = ()) -> list[int]:
        if length <= 0:
            return []
        limit = max(2, int(max_points))
        if length <= limit:
            return list(range(length))
        indices = {0, length - 1}
        for value in must_include:
            if 0 <= value < length:
                indices.add(int(value))
        slots = max(2, limit - len(indices))
        for step in range(slots):
            indices.add(round(step * (length - 1) / max(1, slots - 1)))
        return sorted(indices)

    def learning_state(self) -> dict[str, Any]:
        content_path = self.root / "dashboard" / "learning_content.json"
        profile_path = self.root / "experiments" / "E001" / "results" / "data_profile.csv"
        metrics_path = self.root / "experiments" / "E001" / "results" / "well_metrics.csv"
        summary_path = self.root / "experiments" / "E001" / "results" / "summary.json"
        e003_summary_path = self.root / "experiments" / "E003" / "results" / "summary.json"
        e004_summary_path = self.root / "experiments" / "E004" / "results" / "summary.json"
        e005_summary_path = self.root / "experiments" / "E005" / "results" / "summary.json"
        e006_summary_path = self.root / "experiments" / "E006" / "results" / "summary.json"
        e007_summary_path = self.root / "experiments" / "E007" / "results" / "summary.json"
        e008_summary_path = self.root / "experiments" / "E008" / "results" / "summary.json"
        for path in (content_path, profile_path, metrics_path, summary_path):
            if not path.exists():
                raise FileNotFoundError(path)
        content = read_json(content_path)
        summary = read_json(summary_path)
        e003_summary = read_json(e003_summary_path) if e003_summary_path.exists() else None
        e004_summary = read_json(e004_summary_path) if e004_summary_path.exists() else None
        e005_summary = read_json(e005_summary_path) if e005_summary_path.exists() else None
        e006_summary = read_json(e006_summary_path) if e006_summary_path.exists() else None
        e007_summary = read_json(e007_summary_path) if e007_summary_path.exists() else None
        e008_summary = read_json(e008_summary_path) if e008_summary_path.exists() else None
        with profile_path.open(newline="", encoding="utf-8") as handle:
            profiles = {row["well_id"]: row for row in csv.DictReader(handle)}
        catalog: list[dict[str, Any]] = []
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                well_id = row["well_id"]
                profile = profiles.get(well_id, {})
                sse = float_or_none(row.get("sse")) or 0.0
                datum_sse = float_or_none(row.get("datum_sse")) or 0.0
                trend_sse = float_or_none(row.get("trend_sse")) or 0.0
                shape_sse = float_or_none(row.get("shape_sse")) or 0.0
                catalog.append(
                    {
                        "well_id": well_id,
                        "rmse": float_or_none(row.get("rmse")),
                        "mean_error": float_or_none(row.get("mean_error")),
                        "sse": sse,
                        "hidden_rows": int(profile.get("hidden_rows") or row.get("rows_scored") or 0),
                        "known_rows": int(profile.get("known_rows") or 0),
                        "total_rows": int(profile.get("total_rows") or 0),
                        "regime": row.get("regime") or "unknown",
                        "datum_share": datum_sse / sse if sse > 0 else 0.0,
                        "trend_share": trend_sse / sse if sse > 0 else 0.0,
                        "shape_share": shape_sse / sse if sse > 0 else 0.0,
                    }
                )
        baseline = summary["actual_target_metrics"]["last_known_tvt"]
        median_rmse = float(baseline["median_well_rmse"])
        selectors = [
            ("Typical well", min(catalog, key=lambda x: abs((x["rmse"] or 0.0) - median_rmse))),
            ("Largest SSE", max(catalog, key=lambda x: x["sse"])),
            ("Datum-dominated", max(catalog, key=lambda x: x["datum_share"])),
            ("Trend-dominated", max(catalog, key=lambda x: x["trend_share"])),
            ("Shape-dominated", max(catalog, key=lambda x: x["shape_share"])),
            ("Shortest hidden zone", min(catalog, key=lambda x: x["hidden_rows"])),
            ("Longest hidden zone", max(catalog, key=lambda x: x["hidden_rows"])),
        ]
        recommended: list[dict[str, Any]] = []
        seen: set[str] = set()
        for label, item in selectors:
            if item["well_id"] in seen:
                continue
            seen.add(item["well_id"])
            recommended.append({"label": label, **item})
        return {
            "generated_at": utcnow(),
            "content": content,
            "e001": {
                "baseline": baseline,
                "data": summary["data"],
                "controls": summary["controls"],
            },
            "e003": e003_summary,
            "e004": e004_summary,
            "e005": e005_summary,
            "e006": e006_summary,
            "e007": e007_summary,
            "e008": e008_summary,
            "recommended_wells": recommended,
            "well_catalog": sorted(catalog, key=lambda x: x["well_id"]),
        }

    def learning_well(self, well_id: str, max_points: int = 1600) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{8}", well_id):
            raise ValueError("well_id must be an eight-character lowercase hexadecimal ID")
        cache_key = f"{well_id}:{max_points}"
        if cache_key in self._learning_well_cache:
            return self._learning_well_cache[cache_key]
        horizontal_path = self.root / "data" / "train" / f"{well_id}__horizontal_well.csv"
        typewell_path = self.root / "data" / "train" / f"{well_id}__typewell.csv"
        if not horizontal_path.exists():
            raise FileNotFoundError(horizontal_path)
        formation_names = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
        rows: list[dict[str, float | None]] = []
        with horizontal_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            available_formations = [name for name in formation_names if name in (reader.fieldnames or [])]
            for raw in reader:
                item: dict[str, float | None] = {}
                for name in ["MD", "X", "Y", "Z", "TVT", "TVT_input", "GR", *available_formations]:
                    item[name] = float_or_none(raw.get(name))
                rows.append(item)
        if not rows:
            raise ValueError(f"well {well_id} contains no rows")
        hidden_start = len(rows)
        seen_hidden = False
        for index, row in enumerate(rows):
            visible = row.get("TVT_input") is not None
            if not visible and not seen_hidden:
                hidden_start = index
                seen_hidden = True
            elif visible and seen_hidden:
                raise ValueError(f"well {well_id} has non-contiguous TVT_input visibility")
        if hidden_start <= 0 or hidden_start >= len(rows):
            raise ValueError(f"well {well_id} does not contain a visible prefix and hidden suffix")
        indices = self._sample_indices(len(rows), max_points, (hidden_start - 1, hidden_start))
        series_names = ["MD", "X", "Y", "Z", "TVT", "TVT_input", "GR", *available_formations]
        series = {name: [rows[index].get(name) for index in indices] for name in series_names}
        typewell_rows: list[dict[str, float | None]] = []
        if typewell_path.exists():
            with typewell_path.open(newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    typewell_rows.append({"TVT": float_or_none(raw.get("TVT")), "GR": float_or_none(raw.get("GR"))})
        type_indices = self._sample_indices(len(typewell_rows), 1800)
        typewell = {
            "TVT": [typewell_rows[index]["TVT"] for index in type_indices],
            "GR": [typewell_rows[index]["GR"] for index in type_indices],
        }
        metrics: dict[str, Any] = {}
        metrics_path = self.root / "experiments" / "E001" / "results" / "well_metrics.csv"
        if metrics_path.exists():
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("well_id") == well_id:
                        metrics = {
                            "rmse": float_or_none(row.get("rmse")),
                            "mean_error": float_or_none(row.get("mean_error")),
                            "sse": float_or_none(row.get("sse")),
                            "datum_share": (float_or_none(row.get("datum_sse")) or 0.0) / (float_or_none(row.get("sse")) or 1.0),
                            "trend_share": (float_or_none(row.get("trend_sse")) or 0.0) / (float_or_none(row.get("sse")) or 1.0),
                            "shape_share": (float_or_none(row.get("shape_sse")) or 0.0) / (float_or_none(row.get("sse")) or 1.0),
                            "regime": row.get("regime") or "unknown",
                        }
                        break
        e003_metrics: dict[str, Any] = {}
        e003_path = self.root / "experiments" / "E003" / "results" / "selected_well_metrics.csv"
        if e003_path.exists():
            with e003_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("well_id") == well_id:
                        e003_metrics = {
                            "candidate": row.get("candidate"),
                            "predicted_datum": float_or_none(row.get("predicted_datum")),
                            "predicted_trend": float_or_none(row.get("predicted_trend")),
                            "actual_datum_oracle": float_or_none(row.get("actual_datum")),
                            "actual_trend_oracle": float_or_none(row.get("actual_trend")),
                            "baseline_rmse": float_or_none(row.get("baseline_rmse")),
                            "corrected_rmse": float_or_none(row.get("corrected_rmse")),
                            "evidence_label": "cross_fitted_oof",
                            "deployment_ready": False,
                        }
                        break
        last_visible = rows[hidden_start - 1]
        payload = {
            "well_id": well_id,
            "source": str(horizontal_path.relative_to(self.root)),
            "metadata": {
                "total_rows": len(rows),
                "sampled_rows": len(indices),
                "known_rows": hidden_start,
                "hidden_rows": len(rows) - hidden_start,
                "hidden_start_md": rows[hidden_start].get("MD"),
                "last_visible_md": last_visible.get("MD"),
                "last_visible_z": last_visible.get("Z"),
                "last_visible_tvt": last_visible.get("TVT"),
                "formation_names": available_formations,
            },
            "original_indices": indices,
            "series": series,
            "typewell": typewell,
            "e001_metrics": metrics,
            "e003_metrics": e003_metrics,
        }
        if len(self._learning_well_cache) >= 32:
            oldest_key = next(iter(self._learning_well_cache))
            self._learning_well_cache.pop(oldest_key, None)
        self._learning_well_cache[cache_key] = payload
        return payload

    def validate(self) -> dict[str, Any]:
        required = [
            "AGENTS.md", "MEMORY.md", "GOLD_ROADMAP.md", "PROJECT_WORKFLOW.md",
            "archive/README.md", "archive/claims.csv", "archive/sources.csv",
            "archive/discussions/index.csv", "archive/discussions/synthesis.md",
            "archive/writeups/synthesis.md", "tracking/schema.sql", "tracking/seed.json",
            "dashboard/index.html", "dashboard/learn.html", "dashboard/learning_content.json", "tools/rogii.py",
        ]
        errors: list[str] = []
        warnings: list[str] = []
        for relative in required:
            if not (self.root / relative).exists():
                errors.append(f"missing {relative}")
        try:
            self.init(reset=False)
            sync = self.sync()
            errors.extend(sync.get("errors", []))
        except Exception as exc:
            errors.append(f"database init/sync: {exc}")
        try:
            with (self.root / "archive/discussions/index.csv").open(encoding="utf-8") as handle:
                index_count = sum(1 for _ in csv.DictReader(handle))
            message_count = 0
            for path in sorted((self.root / "archive/discussions").glob("messages_*.jsonl")):
                with path.open(encoding="utf-8") as handle:
                    for line_no, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        json.loads(line)
                        message_count += 1
            if index_count != 132:
                errors.append(f"discussion topic count {index_count}, expected 132")
            if message_count != 981:
                errors.append(f"discussion message count {message_count}, expected 981")
        except Exception as exc:
            errors.append(f"discussion archive validation: {exc}")
        try:
            seed = read_json(self.seed_path)
            with (self.root / "archive/competition/notebook_dependencies.csv").open(encoding="utf-8") as handle:
                unknown = [x["source"] for x in csv.DictReader(handle) if x.get("license") == "Unknown"]
            if unknown:
                warnings.append("unknown-license notebook inputs remain quarantined: " + ", ".join(unknown))
            if seed.get("meta", {}).get("discussion_messages_archived") != "981":
                errors.append("seed discussion message count mismatch")
        except Exception as exc:
            errors.append(f"seed/dependency validation: {exc}")
        try:
            learning = read_json(self.root / "dashboard" / "learning_content.json")
            for field in ("schema_version", "updated_at", "title", "feature_groups", "breakthroughs"):
                if not learning.get(field):
                    errors.append(f"dashboard/learning_content.json missing {field}")
        except Exception as exc:
            errors.append(f"learning dashboard validation: {exc}")
        for path in sorted((self.root / "experiments").glob("**/manifest.json")):
            try:
                data = read_json(path)
                for field in ("experiment_id", "title", "hypothesis", "status", "track", "created_at", "updated_at"):
                    if not data.get(field):
                        errors.append(f"{path.relative_to(self.root)} missing {field}")
            except Exception as exc:
                errors.append(f"{path.relative_to(self.root)}: {exc}")
        try:
            with self.connect() as con:
                result = con.execute("PRAGMA integrity_check").fetchone()[0]
                if result != "ok":
                    errors.append(f"sqlite integrity: {result}")
        except Exception as exc:
            errors.append(f"sqlite integrity check: {exc}")
        return {"status": "ok" if not errors else "error", "errors": errors, "warnings": warnings, "checked_at": utcnow()}

    def write_event(self, event_type: str, payload: dict[str, Any], filename: str) -> Path:
        inbox = self.root / "tracking" / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        path = inbox / filename
        path.write_text(json.dumps({"type": event_type, "recorded_at": utcnow(), "payload": payload}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.sync_if_needed(force=True)
        return path


def make_manifest(args: argparse.Namespace, tracker: Tracker) -> Path:
    path = tracker.root / "experiments" / args.id / "manifest.json"
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} already exists; use --force to replace")
    now = utcnow()
    data = {
        "experiment_id": args.id,
        "title": args.title,
        "hypothesis": args.hypothesis or "TODO: state one falsifiable hypothesis",
        "status": "designed",
        "track": args.track,
        "priority": args.priority,
        "owner": args.owner,
        "parent_id": args.parent,
        "created_at": now,
        "updated_at": now,
        "design": {
            "baseline": "TODO",
            "legal_inputs": [],
            "fold_map": "TODO",
            "controls": ["positive", "noop", "duplicate", "shuffle"],
            "promotion": "TODO: pre-register pooled-RMSE and tail-risk thresholds",
            "runtime_budget_minutes": None,
        },
        "runs": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tracker.sync_if_needed(force=True)
    return path


def add_run(args: argparse.Namespace, tracker: Tracker) -> Path:
    path = tracker.root / "experiments" / args.experiment / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing manifest {path}")
    data = read_json(path)
    if any(x.get("run_id") == args.run for x in data.get("runs", [])) and not args.replace:
        raise ValueError(f"run {args.run} already exists; use --replace")
    now = utcnow()
    run = {
        "run_id": args.run,
        "status": args.status,
        "started_at": args.started_at or now,
        "ended_at": args.ended_at or (now if args.status in ("complete", "rejected", "failed") else None),
        "git_sha": args.git_sha,
        "fold_map": args.fold_map,
        "config_path": args.config,
        "artifact_dir": args.artifact_dir,
        "controls": {},
        "metrics": args.metric or [],
        "well_metrics_path": args.well_metrics,
        "notes": args.notes,
        "updated_at": now,
    }
    existing = [x for x in data.get("runs", []) if x.get("run_id") != args.run]
    data["runs"] = existing + [run]
    data["updated_at"] = now
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tracker.sync_if_needed(force=True)
    return path


class DashboardHandler(BaseHTTPRequestHandler):
    tracker: Tracker
    index_path: Path
    learning_path: Path

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("dashboard: " + fmt % args + "\n")

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/api/state":
                self._json(self.tracker.state())
                return
            if path == "/api/health":
                validation = self.tracker.validate()
                self._json(validation, 200 if validation["status"] == "ok" else 503)
                return
            if path == "/api/sync":
                self._json(self.tracker.sync_if_needed(force=True))
                return
            if path == "/api/learning":
                self._json(self.tracker.learning_state())
                return
            if path == "/api/learning/well":
                well_id = query.get("well_id", [""])[0]
                if not well_id:
                    self._json({"error": "well_id query parameter is required"}, 400)
                    return
                try:
                    self._json(self.tracker.learning_well(well_id))
                except ValueError as exc:
                    self._json({"error": str(exc)}, 400)
                except FileNotFoundError as exc:
                    self._json({"error": str(exc)}, 404)
                return
            if path in ("/", "/index.html"):
                body = self.index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path in ("/learn", "/learn.html"):
                body = self.learning_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)


def run_dashboard(tracker: Tracker, host: str, port: int, open_browser: bool) -> None:
    tracker.sync_if_needed(force=True)
    handler = type("ROGIIDashboardHandler", (DashboardHandler,), {})
    handler.tracker = tracker
    handler.index_path = tracker.root / "dashboard" / "index.html"
    handler.learning_path = tracker.root / "dashboard" / "learn.html"
    if not handler.index_path.exists():
        raise FileNotFoundError(handler.index_path)
    if not handler.learning_path.exists():
        raise FileNotFoundError(handler.learning_path)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"ROGII dashboard: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROGII experiment ledger and dashboard")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root")
    parser.add_argument("--db", type=Path, help="override generated SQLite path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="initialize the generated database")
    p.add_argument("--reset", action="store_true")
    sub.add_parser("sync", help="sync seeds, manifests and inbox events")
    sub.add_parser("validate", help="validate archive, manifests and database")
    sub.add_parser("state", help="print dashboard state JSON")

    p = sub.add_parser("dashboard", help="serve the local dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true")

    p = sub.add_parser("new-experiment", help="create a manifest template")
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--hypothesis")
    p.add_argument("--track", required=True)
    p.add_argument("--priority", type=int, default=99)
    p.add_argument("--owner", default=os.environ.get("USER", "unknown"))
    p.add_argument("--parent")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("record-run", help="append or replace a run in an experiment manifest")
    p.add_argument("--experiment", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--status", default="complete")
    p.add_argument("--started-at")
    p.add_argument("--ended-at")
    p.add_argument("--git-sha")
    p.add_argument("--fold-map")
    p.add_argument("--config")
    p.add_argument("--artifact-dir")
    p.add_argument("--well-metrics")
    p.add_argument("--metric", action="append", type=parse_metric_arg)
    p.add_argument("--notes", default="")
    p.add_argument("--replace", action="store_true")

    p = sub.add_parser("record-submission", help="record a durable submission event")
    p.add_argument("--id", required=True)
    p.add_argument("--run")
    p.add_argument("--ref")
    p.add_argument("--submitted-at", default=None)
    p.add_argument("--score", type=float)
    p.add_argument("--private-score", type=float)
    p.add_argument("--description", required=True)
    p.add_argument("--selected", action="store_true")

    p = sub.add_parser("set-task", help="create or update a durable task event")
    p.add_argument("--id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--priority", type=int, default=99)
    p.add_argument("--phase", required=True)
    p.add_argument("--depends-on", default="")
    p.add_argument("--due-at")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    db = args.db.resolve() if args.db else None
    tracker = Tracker(root, db)
    try:
        if args.command == "init":
            tracker.init(reset=args.reset)
            print(json.dumps({"status": "ok", "db": str(tracker.db_path)}, indent=2))
        elif args.command == "sync":
            print(json.dumps(tracker.sync_if_needed(force=True), indent=2))
        elif args.command == "validate":
            result = tracker.validate()
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "ok" else 1
        elif args.command == "state":
            print(json.dumps(tracker.state(), indent=2, ensure_ascii=False, allow_nan=False))
        elif args.command == "dashboard":
            run_dashboard(tracker, args.host, args.port, args.open)
        elif args.command == "new-experiment":
            print(make_manifest(args, tracker).relative_to(root))
        elif args.command == "record-run":
            print(add_run(args, tracker).relative_to(root))
        elif args.command == "record-submission":
            payload = {
                "submission_id": args.id,
                "run_id": args.run,
                "kaggle_ref": args.ref,
                "submitted_at": args.submitted_at or utcnow(),
                "public_score": args.score,
                "private_score": args.private_score,
                "description": args.description,
                "selected": 1 if args.selected else 0,
                "source": "cli",
            }
            print(tracker.write_event("submission", payload, f"submission_{args.id}.json").relative_to(root))
        elif args.command == "set-task":
            payload = {
                "task_id": args.id,
                "title": args.title,
                "status": args.status,
                "priority": args.priority,
                "phase": args.phase,
                "depends_on": args.depends_on,
                "due_at": args.due_at,
            }
            print(tracker.write_event("task", payload, f"task_{args.id}.json").relative_to(root))
        return 0
    except (ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
