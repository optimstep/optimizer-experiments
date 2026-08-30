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
