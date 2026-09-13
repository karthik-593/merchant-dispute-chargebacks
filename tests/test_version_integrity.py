"""The guard that would have caught the corpus pointer drift.

Positive tests assert the live records agree. Negative controls rebuild the exact drift that
existed — a pointer naming a different artifact from the version record — in a temporary tree and
assert the guard fails on it. A guard nobody has watched fail is not known to be a guard; that is
the same discipline as the leak-probe negative controls in `tests/test_leak_probes.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.evaluation.version_integrity import (
    CORPUS_DVC_FILE,
    CORPUS_VERSION_FILE,
    VERSIONS_FILE,
    check_corpus_version_integrity,
    dvc_pointer_hash,
    md5_of,
    summary,
)

HTRUE = "34a085ab04488672b9d8d79a582655ba"
# The artifact the pointer was stale at: corpus-v1.1, three content versions behind the freeze.
STALE = "a4ed9bf4a5fc894960f159e7b7194ee4"

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus" / "circulars.jsonl"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A copy of the three records, so a fixture can be drifted without touching the repo."""
    for relative in (CORPUS_VERSION_FILE, VERSIONS_FILE, CORPUS_DVC_FILE):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, target)
    return tmp_path


def _edit(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# -- the live repo --------------------------------------------------------------------------


def test_the_repository_records_agree():
    problems = check_corpus_version_integrity()
    assert not problems, summary(problems)


def test_the_recorded_hash_is_h_true():
    """Pinned. The hash was recovered by metric reproduction; a silent change to it is a finding."""
    data = yaml.safe_load((ROOT / CORPUS_VERSION_FILE).read_text(encoding="utf-8"))
    entry = next(e for e in data["history"] if e["version"] == data["version"])
    assert entry["content_md5"] == HTRUE
    assert dvc_pointer_hash(ROOT / CORPUS_DVC_FILE) == HTRUE


def test_the_recorded_hash_carries_its_provenance():
    """The hash is reconstructed, not captured. A reader must be able to see that."""
    data = yaml.safe_load((ROOT / CORPUS_VERSION_FILE).read_text(encoding="utf-8"))
    entry = next(e for e in data["history"] if e["version"] == data["version"])
    note = entry["content_md5_provenance"].lower()
    assert "not logged at freeze time" in note
    assert "metric reproduction" in note


@pytest.mark.skipif(not CORPUS.is_file(), reason="corpus not on disk (dvc pull)")
def test_the_corpus_on_disk_is_the_recorded_artifact():
    assert md5_of(CORPUS) == HTRUE
    assert not check_corpus_version_integrity(require_file=True)


# -- negative controls: the guard must fail on the drift that existed -------------------------


def test_guard_catches_the_exact_drift_that_occurred(tree: Path):
    """The real incident: pointer at corpus-v1.1 while the records claim v1.4."""
    _edit(tree / CORPUS_DVC_FILE, lambda d: d["outs"][0].update(md5=STALE, size=554913))
    problems = check_corpus_version_integrity(tree)
    assert problems, "the guard passed on the drift it exists to catch"
    assert any(STALE in p.detail and HTRUE in p.detail for p in problems), summary(problems)


def test_guard_catches_a_version_label_with_no_hash_behind_it(tree: Path):
    """The pre-incident state: a bare label, which certifies nothing and cannot be checked."""
    def strip(data):
        entry = next(e for e in data["history"] if e["version"] == data["version"])
        entry.pop("content_md5", None)

    _edit(tree / CORPUS_VERSION_FILE, strip)
    problems = check_corpus_version_integrity(tree)
    assert problems
    assert any("bare label" in p.detail for p in problems), summary(problems)


def test_guard_catches_the_freeze_naming_a_different_hash(tree: Path):
    """versions.yaml and the corpus record disagreeing about which artifact was frozen on."""
    _edit(tree / VERSIONS_FILE, lambda d: d["meta"].update(corpus_content_md5=STALE))
    problems = check_corpus_version_integrity(tree)
    assert problems
    assert any("versions.yaml records" in p.detail for p in problems), summary(problems)


def test_guard_catches_the_freeze_naming_a_different_version(tree: Path):
    _edit(tree / VERSIONS_FILE, lambda d: d["meta"].update(corpus_version="corpus-v1.2"))
    problems = check_corpus_version_integrity(tree)
    assert problems
    assert any("freezes on" in p.detail for p in problems), summary(problems)


def test_guard_catches_a_corpus_on_disk_that_is_not_the_recorded_artifact(tree: Path):
    """The working copy drifting away from what the records name."""
    corpus = tree / "data" / "corpus" / "circulars.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text('{"doc_id": "not the real corpus"}\n', encoding="utf-8")
    problems = check_corpus_version_integrity(tree, require_file=True)
    assert problems
    assert any("hashes to" in p.detail for p in problems), summary(problems)


def test_guard_is_not_self_referential(tree: Path):
    """A hash that is merely PRESENT must not be taken as a hash that is RIGHT.

    Every record is set to the same wrong artifact. The guard passes, and it should: consistency
    is all it can see. The hash's correctness is anchored by the metric reproduction recorded in
    `content_md5_provenance`, not by this check — and stating that here keeps the limit visible
    rather than implied.
    """
    def repoint(data):
        entry = next(e for e in data["history"] if e["version"] == data["version"])
        entry["content_md5"] = STALE

    _edit(tree / CORPUS_VERSION_FILE, repoint)
    _edit(tree / VERSIONS_FILE, lambda d: d["meta"].update(corpus_content_md5=STALE))
    _edit(tree / CORPUS_DVC_FILE, lambda d: d["outs"][0].update(md5=STALE, size=554913))
    assert not check_corpus_version_integrity(tree), (
        "consistency is what this guard checks; correctness comes from the provenance note"
    )
