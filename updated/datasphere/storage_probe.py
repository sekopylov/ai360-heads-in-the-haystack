"""Read-only inventory of model locations visible on an attached project disk."""
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ["DS_PROJECT_HOME"])
    report = {"project_home": str(root), "root_entries": [], "checkpoints": []}
    report["root_entries"] = sorted(p.name for p in root.iterdir())
    pending = [(root, 0)]
    visited = 0
    while pending and visited < 2000:
        directory, depth = pending.pop()
        visited += 1
        try:
            entries = sorted(directory.iterdir())
        except (PermissionError, OSError):
            continue
        if (directory / "config.json").is_file():
            files = [{"name": p.name, "bytes": p.stat().st_size}
                     for p in entries if p.is_file()]
            report["checkpoints"].append({"path": str(directory), "files": files})
        if depth < 6:
            pending.extend((p, depth + 1) for p in entries
                           if p.is_dir() and not p.is_symlink()
                           and not p.name.startswith(".")
                           and p.name not in {"node_modules", "__pycache__", "site-packages"})
    report["directories_visited"] = visited
    report["scan_limit_reached"] = bool(pending)
    output = Path("datasphere-results/storage-probe/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
