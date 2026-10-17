"""Partition a built environment into independent OCI COPY layers without changing files."""

import os
from pathlib import Path
import stat
import sys


def partition(source, destination, count=16):
    source, destination = Path(source), Path(destination)
    if not source.is_dir() or source.is_symlink() or destination.exists() or count < 1:
        raise ValueError("invalid_partition_paths")
    entries, directories = [], []
    for base, dirs, files in os.walk(source, followlinks=False):
        base = Path(base)
        directories.append(base.relative_to(source))
        for name in dirs[:]:
            path = base / name
            if path.is_symlink():
                entries.append(path)
                dirs.remove(name)
        entries.extend(base / name for name in files)
    for path in entries:
        if not path.is_symlink() and not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("unsupported_environment_entry")
    roots = [destination / f"{index:02d}" for index in range(count)]
    for root in roots:
        root.mkdir(parents=True)
    # Preserve empty directories too. All layers reconstruct the original relative paths.
    for relative in directories:
        target = roots[0] / relative
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(stat.S_IMODE((source / relative).stat().st_mode))
    sizes = [0] * count
    entries.sort(key=lambda path: (-path.lstat().st_size, path.relative_to(source).as_posix()))
    for path in entries:
        index = min(range(count), key=lambda item: (sizes[item], item))
        size = path.lstat().st_size
        target = roots[index] / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        sizes[index] += size
    return {"files": len(entries), "layers": count, "layer_bytes": sizes}


if __name__ == "__main__":
    import json

    print(json.dumps(partition(sys.argv[1], sys.argv[2])))
