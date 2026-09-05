# Optimstep optimizer experiments

Reproducible large-batch contrastive-learning experiments comparing AdamW,
Muon, inverse-half Muon, joint Muon, shared-interface SOAP-Muon, and controlled
weight-radius variants. The default benchmark is a random 6-layer BERT trained
on GooAQ with a pruned English WordPiece tokenizer and the memory-efficient
Qwen3 contrastive loss.

## Quick start

```bash
make bootstrap
make prepare
make smoke CONFIG=configs/muon.yaml
make run CONFIG=configs/muon.yaml
```

Results, checkpoints, and TensorBoard events are written below `results/`.
Configuration files in `configs/` are the source of truth for hyperparameters.
The launcher writes the fully inherited configuration beside each run before
training, so command-line overrides remain auditable.

## Vast.ai

Rent an A100 40 GB instance with a current PyTorch CUDA image, then:

```bash
export REMOTE=root@HOST
export VAST_INSTANCE_ID=12345678
export MAX_DOLLARS=8
make remote-setup
make remote-run CONFIG=configs/muon.yaml
make sync
make hard-stop
```

See `AGENTS.md` for operating rules, monitoring, and known failure modes.

## Next-token benchmark

The compact second setup trains a 6-layer decoder on TinyStories. It compares
AdamW and Muon with and without residual-activation feedback while holding the
model, schedule, update/weight ratio, validation, and seed fixed.

```bash
make next-token-smoke
make next-token-suite

# Width is an override, not another checked-in config.
WIDTH=1024 STEPS=4000 make next-token-suite
```

The five commented configs live in `configs/next_token/`. The larger exploratory
grid is intentionally not published; resolved configs are saved with every run.

## Modded-NanoGPT speedrun

`third_party/modded-nanogpt` pins the canonical 124M NanoGPT speedrun as a Git
submodule. This is the realistic FineWeb/H100 validation tier; the compact
TinyStories setup remains the cheap iteration tier.

```bash
git clone --recurse-submodules https://github.com/optimstep/optimizer-experiments.git
make bootstrap
make nanogpt-install
make nanogpt-data
make nanogpt-dry-run
make nanogpt-run
```

The pinned upstream baseline is described by
`configs/nanogpt_speedrun/baseline.yaml`. Official timing uses 8xH100 and the
first nine FineWeb10B cache shards. Future optimizer treatments should be small,
reviewable patch files referenced from a derived YAML. The harness applies each
patch in a disposable worktree, leaving the submodule pristine and making a
baseline/treatment diff auditable.
