#!/usr/bin/env python3
"""Translate a versioned YAML experiment into the canonical training CLI."""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def merge(left, right):
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(left.get(key), dict):
            merge(left[key], value)
        else:
            left[key] = value
    return left


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text()) or {}
    parent = config.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        return merge(base, config)
    return config


def set_value(config, expression):
    dotted, raw = expression.split("=", 1)
    value = yaml.safe_load(raw)
    cursor = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def cli_args(config):
    aliases = {"batch_size": "per-rank-batch", "chunk_size": "chunk"}
    args = []
    for section in ("model", "data", "training", "optimizer", "logging"):
        for key, value in config.get(section, {}).items():
            key = aliases.get(key, key).replace("_", "-")
            if value is True:
                args.append(f"--{key}")
            elif value is False or value is None:
                continue
            elif isinstance(value, list):
                args.extend((f"--{key}", ",".join(map(str, value))))
            else:
                args.extend((f"--{key}", str(value)))
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    parsed = parser.parse_args()
    config = load_config(parsed.config)
    for expression in parsed.set:
        set_value(config, expression)
    root = Path(__file__).resolve().parents[1]
    if parsed.prepare_only:
        command = [sys.executable, str(root / "experiments/prune_wordpiece_tokenizer.py")]
    else:
        tensorboard_dir = config.get("logging", {}).get("tensorboard_dir")
        if tensorboard_dir:
            run_dir = root / tensorboard_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir.parent / "config.resolved.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
        command = [sys.executable, str(root / "experiments/train_gradcache_qwen3.py"), *cli_args(config)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
