"""Compare training and evaluation curves from multiple GradCache logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP = re.compile(r"^step\s+(\d+)\s+loss\s+([0-9.eE+-]+)")
EVAL = re.compile(
    r"^eval\s+step\s+(\d+)\s+nli-dev acc\s+([0-9.eE+-]+)\s+"
    r"stsb-dev spearman\s+([0-9.eE+-]+)"
)


def parse_log(path: Path):
    steps, losses = [], []
    eval_steps, nli, stsb = [], [], []
    for line in path.read_text(errors="replace").splitlines():
        if match := STEP.match(line):
            step, loss = match.groups()
            steps.append(int(step))
            losses.append(float(loss))
        elif match := EVAL.match(line):
            step, acc, rho = match.groups()
            eval_steps.append(int(step))
            nli.append(float(acc))
            stsb.append(float(rho))
    return steps, losses, eval_steps, nli, stsb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "runs", nargs="+", metavar="LABEL=LOG",
        help="legend label and log path, separated by =",
    )
    parser.add_argument("--title", default="GooAQ optimizer comparison")
    parser.add_argument(
        "--max-step", type=int,
        help="plot only points at or before this training step",
    )
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for run in args.runs:
        label, separator, raw_path = run.partition("=")
        if not separator:
            parser.error(f"run must be LABEL=LOG: {run}")
        steps, losses, eval_steps, nli, stsb = parse_log(Path(raw_path))
        if args.max_step is not None:
            train_points = [
                (step, loss) for step, loss in zip(steps, losses)
                if step <= args.max_step
            ]
            eval_points = [
                (step, acc, rho)
                for step, acc, rho in zip(eval_steps, nli, stsb)
                if step <= args.max_step
            ]
            steps = [point[0] for point in train_points]
            losses = [point[1] for point in train_points]
            eval_steps = [point[0] for point in eval_points]
            nli = [point[1] for point in eval_points]
            stsb = [point[2] for point in eval_points]
        axes[0].plot(steps, losses, marker="o", markersize=3, label=label)
        axes[1].plot(eval_steps, nli, marker="o", markersize=3, label=label)
        axes[2].plot(eval_steps, stsb, marker="o", markersize=3, label=label)

    axes[0].set(title="Training loss", ylabel="loss")
    axes[1].set(title="NLI development set", ylabel="accuracy")
    axes[2].set(title="STS-B development set", ylabel="Spearman")
    for axis in axes:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize="small")
    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

