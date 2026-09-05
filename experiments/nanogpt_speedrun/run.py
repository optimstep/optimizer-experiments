#!/usr/bin/env python3
"""Prepare and launch a pinned modded-nanogpt baseline or patch overlay."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("speedrun config must be a mapping")
    return config


def upstream_path(config):
    return (ROOT / config["upstream"]["path"]).resolve()


def checked_revision(config):
    upstream = upstream_path(config)
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = str(config["upstream"]["revision"])
    if revision != expected:
        raise RuntimeError(f"submodule revision {revision} != configured {expected}")
    status = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("modded-nanogpt submodule is dirty")
    return revision


def command_for(config, action):
    upstream = upstream_path(config)
    if action == "install":
        return [sys.executable, "-m", "pip", "install", "-r", str(upstream / "requirements.txt")]
    if action == "data":
        return [
            sys.executable,
            str(upstream / config["data"]["cache_script"]),
            str(config["data"]["shards"]),
        ]
    if action == "run":
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={int(config['hardware']['nproc_per_node'])}",
            "train_gpt.py",
        ]
    raise ValueError(action)


def materialize_treatment(config):
    patch = config.get("runtime", {}).get("patch")
    if not patch:
        return upstream_path(config)
    patch_path = (ROOT / patch).resolve()
    if not patch_path.is_file():
        raise FileNotFoundError(patch_path)
    output = (ROOT / "artifacts/nanogpt_speedrun_worktrees" / config["name"]).resolve()
    if output.exists():
        raise FileExistsError(f"treatment worktree already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(upstream_path(config)), "worktree", "add", "--detach", str(output),
         config["upstream"]["revision"]],
        check=True,
    )
    subprocess.run(["git", "-C", str(output), "apply", "--check", str(patch_path)], check=True)
    subprocess.run(["git", "-C", str(output), "apply", str(patch_path)], check=True)
    return output


def launch(config, dry_run=False):
    checked_revision(config)
    worktree = materialize_treatment(config) if not dry_run else upstream_path(config)
    command = command_for(config, "run")
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in config.get("runtime", {}).get("environment", {}).items()})
    environment.setdefault("DATA_PATH", str(upstream_path(config)))
    output = (ROOT / config["logging"]["output_dir"]).resolve()
    resolved = dict(config)
    if dry_run:
        print(json.dumps({"cwd": str(worktree), "command": command,
                          "data_path": environment["DATA_PATH"]}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    with (output / "train.log").open("w") as log:
        print("+", shlex.join(command), flush=True)
        subprocess.run(command, cwd=worktree, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "data", "run"))
    parser.add_argument("--config", default="configs/nanogpt_speedrun/baseline.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checked_revision(config)
    if args.action == "run":
        launch(config, args.dry_run)
        return
    command = command_for(config, args.action)
    print("+", shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=upstream_path(config), check=True)


if __name__ == "__main__":
    main()
