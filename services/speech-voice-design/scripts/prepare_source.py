"""Build-only extraction of hash-locked source, without running upstream installers."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import urllib.request


def prepare(lock_path, destination, archive_path=None):
    lock = json.loads(Path(lock_path).read_text())
    destination = Path(destination)
    if destination.exists():
        raise ValueError("source_destination_exists")
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(archive_path) if archive_path else Path(temp) / "source.tar.gz"
        if not archive_path:
            with urllib.request.urlopen(lock["url"], timeout=120) as source, archive.open("xb") as target:
                count = 0
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > lock["size"]:
                        raise ValueError("source_archive_oversized")
                    target.write(block)
        if archive.stat().st_size != lock["size"] or hashlib.sha256(archive.read_bytes()).hexdigest() != lock["sha256"]:
            raise ValueError("source_archive_integrity_failed")
        with tarfile.open(archive) as source:
            selected = []
            names = set()
            for member in source.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                    raise ValueError("unsafe_archive_member")
                parts = path.parts[1:]
                if not member.isfile() or not parts or parts[0] not in ("src", "LICENSE"):
                    continue
                relative = Path(*parts)
                if relative in names:
                    raise ValueError("duplicate_archive_member")
                names.add(relative)
                selected.append((member, relative))
            if Path("LICENSE") not in names or Path("src/voxcpm/core.py") not in names:
                raise ValueError("incomplete_upstream_source")
            destination.mkdir(parents=True)
            for member, relative in selected:
                output = destination / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("xb") as target:
                    target.write(source.extractfile(member).read())
    receipt = {"commit": lock["commit"], "archive_sha256": lock["sha256"], "files": {}}
    for path in sorted((destination / "src").rglob("*.py")):
        receipt["files"][path.relative_to(destination / "src").as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lock")
    parser.add_argument("destination")
    parser.add_argument("--archive")
    args = parser.parse_args()
    prepare(args.lock, args.destination, args.archive)
