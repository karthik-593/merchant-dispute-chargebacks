"""Does every version record name an artifact DVC can actually restore?

The incident this exists to prevent: `configs/versions.yaml` said every M7 number was measured on
corpus-v1.4, and `data/corpus/circulars.jsonl.dvc` pointed at a corpus three versions older. Both
files were internally consistent, both were committed, and nothing compared them — so a
`dvc checkout` from a clean clone would have restored a corpus that cannot reproduce the freeze,
and the freeze would still have claimed it could. The drift ran undetected for four days across
three content changes.

The root cause is that a version LABEL is uncheckable. "corpus-v1.4" is a string; it matches
whatever bytes happen to be on disk, so it certifies nothing. A content hash beside it turns the
claim into something a test can fail on, which is what this module provides.

WHAT THIS GUARD DOES AND DOES NOT ESTABLISH
-------------------------------------------
It checks that the recorded `content_md5`, the `.dvc` pointer and (optionally) the file on disk
all name the SAME artifact. That is a consistency check between records.

It does NOT establish that the recorded hash is the RIGHT one. Nothing here can: a wrong hash
written into all three places would pass. The correctness of corpus-v1.4's hash rests on the
metric reproduction recorded in `configs/corpus/corpus_version.yaml`
(`content_md5_provenance`) and `experiments/EXP-CORPUS-001.md` — the frozen rule_hit@5 of 0.640
reproduces on that artifact and on no other candidate. Keeping that distinction explicit matters:
a guard that treated "a hash is recorded" as proof the hash is correct would be self-referential
and would have passed on the very drift it exists to catch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config import PROJECT_ROOT

CORPUS_VERSION_FILE = Path("configs/corpus/corpus_version.yaml")
VERSIONS_FILE = Path("configs/versions.yaml")
CORPUS_DVC_FILE = Path("data/corpus/circulars.jsonl.dvc")


@dataclass(frozen=True)
class IntegrityProblem:
    """One disagreement between records that should name the same artifact."""

    version: str
    detail: str

    def __str__(self) -> str:
        """Render for an assertion message."""
        return f"{self.version}: {self.detail}"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dvc_pointer_hash(dvc_file: Path) -> str | None:
    """The md5 a `.dvc` pointer records for its single output."""
    if not dvc_file.is_file():
        return None
    outs = _read_yaml(dvc_file).get("outs") or []
    return str(outs[0]["md5"]) if outs else None


def md5_of(path: Path) -> str:
    """Content hash of a file, matching how DVC hashes a single file."""
    digest = hashlib.md5()  # noqa: S324 - matching DVC's own algorithm, not a security use
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_corpus_version_integrity(
    root: Path | None = None, *, require_file: bool = False
) -> list[IntegrityProblem]:
    """Check the corpus version records, the `.dvc` pointer and the file all name one artifact.

    `require_file` additionally checks the working copy. It is off by default because the corpus
    is DVC-tracked and legitimately absent on a fresh clone before `dvc pull`; the records must
    agree regardless.
    """
    base = root or PROJECT_ROOT
    problems: list[IntegrityProblem] = []

    corpus_versions = _read_yaml(base / CORPUS_VERSION_FILE)
    declared = corpus_versions["version"]
    history = {entry["version"]: entry for entry in corpus_versions.get("history", [])}

    entry = history.get(declared)
    if entry is None:
        problems.append(
            IntegrityProblem(declared, "declared version has no entry in `history`")
        )
        return problems

    recorded = entry.get("content_md5")
    if not recorded:
        problems.append(
            IntegrityProblem(
                declared,
                "no `content_md5` recorded - the version is a bare label and cannot be checked "
                "against the artifact it claims to name",
            )
        )
        return problems

    pointer = dvc_pointer_hash(base / CORPUS_DVC_FILE)
    if pointer is None:
        problems.append(IntegrityProblem(declared, f"no DVC pointer at {CORPUS_DVC_FILE}"))
    elif pointer != recorded:
        problems.append(
            IntegrityProblem(
                declared,
                f"DVC pointer records {pointer} but {CORPUS_VERSION_FILE.name} records "
                f"{recorded} - a `dvc checkout` would restore a corpus this version does not "
                f"describe",
            )
        )

    # The freeze restates the hash, so a reader of versions.yaml sees what it ran on. If the two
    # records disagree, the freeze is naming an artifact its own corpus record does not.
    meta = _read_yaml(base / VERSIONS_FILE)["meta"]
    frozen_label = meta.get("corpus_version")
    frozen_hash = meta.get("corpus_content_md5")
    if frozen_label != declared:
        problems.append(
            IntegrityProblem(
                declared,
                f"{VERSIONS_FILE.name} freezes on {frozen_label!r} but the corpus record declares "
                f"{declared!r}",
            )
        )
    elif frozen_hash and frozen_hash != recorded:
        problems.append(
            IntegrityProblem(
                declared,
                f"{VERSIONS_FILE.name} records {frozen_hash} against the corpus record's "
                f"{recorded}",
            )
        )
    elif not frozen_hash:
        problems.append(
            IntegrityProblem(
                declared, f"{VERSIONS_FILE.name} names a corpus version but records no hash for it"
            )
        )

    if require_file:
        corpus = base / "data" / "corpus" / "circulars.jsonl"
        if not corpus.is_file():
            problems.append(IntegrityProblem(declared, f"{corpus} is missing (run `dvc pull`)"))
        else:
            actual = md5_of(corpus)
            if actual != recorded:
                problems.append(
                    IntegrityProblem(
                        declared,
                        f"the corpus on disk hashes to {actual}, not the recorded {recorded}",
                    )
                )
    return problems


def summary(problems: list[IntegrityProblem]) -> str:
    """One block for a console or CI run."""
    if not problems:
        return "corpus version integrity: PASS - records, pointer and artifact agree"
    lines = ["corpus version integrity: FAIL"]
    lines += [f"  - {problem}" for problem in problems]
    return "\n".join(lines)


def main() -> int:
    """CI entry point: non-zero when a version record names an artifact DVC cannot restore."""
    problems = check_corpus_version_integrity(require_file=True)
    print(summary(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
