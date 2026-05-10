"""UniProt ID mapping helpers for annotation integrations."""

from __future__ import annotations

import json
import re
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from biastracker.annotations._http import (
    ensure_cache_dir,
    read_json_cache,
    request_with_retries,
    safe_cache_key,
    write_json_cache,
)


UNIPROT_BASE_URL = "https://rest.uniprot.org"


def submit_id_mapping_job(
    ids: list[str],
    from_ns: str,
    to_ns: str,
    session=None,
    timeout: float = 60,
) -> str:
    """Submit a UniProt asynchronous ID mapping job and return its job ID."""
    clean_ids = _unique_nonempty(ids)
    if not clean_ids:
        raise ValueError("No IDs provided for UniProt ID mapping")

    response = request_with_retries(
        "POST",
        f"{UNIPROT_BASE_URL}/idmapping/run",
        session=session,
        timeout=timeout,
        data={
            "from": from_ns,
            "to": to_ns,
            "ids": ",".join(clean_ids),
        },
    )
    payload = response.json()
    job_id = payload.get("jobId")
    if not job_id:
        raise RuntimeError("UniProt ID mapping response did not include a jobId")
    return str(job_id)


def poll_id_mapping_job(
    job_id: str,
    session=None,
    poll_seconds: float = 2.0,
    timeout: float = 60,
    max_wait_seconds: float = 300,
) -> None:
    """Poll UniProt until an ID mapping job is complete."""
    deadline = time.monotonic() + max_wait_seconds
    url = f"{UNIPROT_BASE_URL}/idmapping/status/{job_id}"

    while True:
        response = request_with_retries("GET", url, session=session, timeout=timeout)
        payload = response.json()
        status = payload.get("jobStatus")

        if status is None:
            if "results" in payload or "redirectURL" in payload:
                return
            return

        normalized_status = str(status).upper()
        if normalized_status == "RUNNING":
            if time.monotonic() + poll_seconds > deadline:
                raise TimeoutError(
                    f"UniProt ID mapping job {job_id} did not complete within "
                    f"{max_wait_seconds} seconds"
                )
            time.sleep(poll_seconds)
            continue

        if normalized_status in {"FINISHED", "COMPLETED", "SUCCESS"}:
            return

        raise RuntimeError(
            f"UniProt ID mapping job {job_id} failed with status {status!r}"
        )


def fetch_id_mapping_results(
    job_id: str,
    session=None,
    timeout: float = 60,
) -> list[dict]:
    """Fetch all result pages for a completed UniProt ID mapping job."""
    url = f"{UNIPROT_BASE_URL}/idmapping/results/{job_id}"
    results: list[dict] = []

    while url:
        response = request_with_retries("GET", url, session=session, timeout=timeout)
        payload = response.json()
        page_results = payload.get("results", [])
        if not isinstance(page_results, list):
            raise RuntimeError("UniProt ID mapping results payload is malformed")
        results.extend(page_results)
        url = _next_link(response.headers.get("Link", ""))

    return results


def extract_uniprot_mapping(results: list[dict]) -> dict[str, list[str]]:
    """Convert raw UniProt ID mapping results into a one-to-many mapping."""
    mapping: dict[str, list[str]] = defaultdict(list)

    for item in results:
        source_id = item.get("from")
        target = _target_accession(item.get("to"))
        if not source_id or not target:
            continue

        source_key = str(source_id)
        if target not in mapping[source_key]:
            mapping[source_key].append(target)

    return dict(mapping)


def map_ids_to_uniprot(
    ids: list[str],
    from_ns: str = "Ensembl",
    to_ns: str = "UniProtKB",
    batch_size: int = 500,
    cache_dir: str | Path = ".cache/biastracker/uniprot_mapping",
    use_cache: bool = True,
    session=None,
) -> dict[str, list[str]]:
    """Map source IDs to UniProt accessions with batching and caching."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    clean_ids = _unique_nonempty(ids)
    if not clean_ids:
        raise ValueError("No IDs provided for UniProt ID mapping")

    cache_path = ensure_cache_dir(cache_dir) if use_cache else None
    merged: dict[str, list[str]] = defaultdict(list)

    for batch in _batched(clean_ids, batch_size):
        cached_mapping = None
        cache_file = None
        if cache_path is not None:
            cache_file = cache_path / f"{_batch_cache_key(batch, from_ns, to_ns)}.json"
            cached_mapping = read_json_cache(cache_file)

        if cached_mapping is None:
            job_id = submit_id_mapping_job(
                batch,
                from_ns=from_ns,
                to_ns=to_ns,
                session=session,
            )
            poll_id_mapping_job(job_id, session=session)
            results = fetch_id_mapping_results(job_id, session=session)
            cached_mapping = extract_uniprot_mapping(results)
            if cache_file is not None:
                write_json_cache(cache_file, cached_mapping)

        _merge_mapping(merged, cached_mapping)

    mapping = dict(merged)
    unmapped = [source_id for source_id in clean_ids if source_id not in mapping]
    if unmapped:
        warnings.warn(
            f"{len(unmapped)} IDs were not mapped to UniProt accessions",
            RuntimeWarning,
            stacklevel=2,
        )

    return mapping


def _unique_nonempty(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    clean_ids: list[str] = []
    for value in ids:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean_ids.append(text)
    return clean_ids


def _batched(ids: list[str], batch_size: int) -> list[list[str]]:
    return [ids[index : index + batch_size] for index in range(0, len(ids), batch_size)]


def _batch_cache_key(batch: list[str], from_ns: str, to_ns: str) -> str:
    value = json.dumps(
        {"from": from_ns, "to": to_ns, "ids": batch},
        sort_keys=True,
        separators=(",", ":"),
    )
    return safe_cache_key(value)


def _merge_mapping(
    merged: dict[str, list[str]],
    batch_mapping: dict[str, list[str]],
) -> None:
    for source_id, targets in batch_mapping.items():
        source_key = str(source_id)
        for target in targets:
            if target not in merged[source_key]:
                merged[source_key].append(target)


def _target_accession(target: Any) -> str | None:
    if isinstance(target, str):
        return target
    if isinstance(target, dict):
        for key in ("primaryAccession", "uniProtkbId", "id"):
            value = target.get(key)
            if value:
                return str(value)
    return None


def _next_link(link_header: str) -> str | None:
    if not link_header:
        return None

    for part in link_header.split(","):
        if 'rel="next"' not in part and "rel=next" not in part:
            continue
        match = re.search(r"<([^>]+)>", part)
        if match:
            return match.group(1)
    return None
