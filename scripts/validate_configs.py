#!/usr/bin/env python3
"""Resolve every checked-in YAML and enforce unique result directories."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train import load_config  # noqa: E402


def main():
    paths = sorted(Path("configs").rglob("*.yaml"))
    destinations = {}
    for path in paths:
        config = load_config(path)
        if not isinstance(config, dict) or not config:
            raise ValueError(f"{path}: expected a non-empty mapping")
        destination = config.get("logging", {}).get("tensorboard_dir")
        if path.name != "base.yaml" and not destination:
            raise ValueError(f"{path}: missing logging.tensorboard_dir")
        if destination:
            previous = destinations.setdefault(destination, path)
            if previous != path:
                raise ValueError(
                    f"duplicate tensorboard_dir {destination}: {previous}, {path}"
                )
        print(f"OK {path}")
    print(f"validated {len(paths)} configs")


if __name__ == "__main__":
    main()
