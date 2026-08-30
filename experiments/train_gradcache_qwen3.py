"""GradCache + memeff Qwen3-loss integration test / smoke train (2 x A100 box).

Trains a small text encoder on sentence-transformers/all-nli triplets: anchors are
queries, positives are documents, negatives ride as K=1 row-specific hard negatives
of DistributedMemoryEfficientQwen3Loss. A minimal gradient-cache (arXiv 2101.06983)
is vendored below -- chunked no-grad forward, loss backward on the cached reps,
RNG-replayed re-forward per chunk with a surrogate backward -- so the contrastive
batch is bounded by the loss (O(B * dim)), not by encoder activations.

All three texts of a triplet go through ONE encoder as a single (3B, L) batch, so
DDP gradient sync happens once, on the last chunk. The loss_fn splits the reps
back into q / d / h.

Modes (run under torchrun, one process per GPU; world=1 also works):
  equiv  -- correctness gate: param grads of the grad-cached step must match a
            single full-batch forward/backward through the same loss (dropout off,
            fp32, ieee matmuls). Run this FIRST.
  train  -- smoke train: bf16 autocast, large batch, logs loss / lr / step time /
            peak memory, plus downstream proxies (all-nli dev triplet accuracy,
            STS-B dev Spearman) every --eval-every steps. LR follows --schedule
            (wsd | cosine | const; const reproduces the old fixed-lr runs).
            --wd-anchor switches AdamW's decay target from 0 to the pretrained
            weights (decoupled L2-SP, arXiv 1802.01483). Scale --lr with the
            global batch: 8e-5 was right at 262k, so ~2e-5 at 65k.

  torchrun --nproc-per-node=2 train_gradcache_qwen3.py --mode equiv
  torchrun --nproc-per-node=2 train_gradcache_qwen3.py --mode train \
      --per-rank-batch 32768 --chunk 4096 --steps 30 --tau-plus 0.1

Or just `bash examples/run_train_gradcache.sh`, which installs the deps and
runs both modes. Validated on 2 x A100-PCIE-40GB (torch 2.5.1, transformers
5.14): equiv worst grad rel err 2.8e-6; 65k global batch trains at ~9-11
s/step, loss 0.25 -> 0.20 over 30 steps, peak 25.9 GiB/GPU.

Needs: pip install <repo root> transformers datasets  (plus torch + triton).
"""
import argparse
import contextlib
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "optimizers"))

if "equiv" in sys.argv:  # ieee matmuls for the grad comparison; must precede memeff import
    os.environ.setdefault("MEMEFF_INPUT_PRECISION", "ieee")

import torch
import torch.distributed as dist
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["equiv", "train"], default="train")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer name/path; defaults to --model")
    p.add_argument("--random-bert", action="store_true",
                   help="initialize a compact BERT encoder from scratch instead "
                        "of loading pretrained --model weights")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=6)
    p.add_argument("--num-attention-heads", type=int, default=8)
    p.add_argument("--intermediate-size", type=int, default=1024)
    p.add_argument("--dataset", default="sentence-transformers/all-nli")
    p.add_argument("--dataset-config", default=None,
                   help="dataset config name (all-nli needs 'triplet')")
    p.add_argument("--columns", default="anchor,positive,negative",
                   help="text columns: 2 = pairs (in-batch negatives only), "
                        "3 = triplets with one hard negative per row")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pretokenize", action="store_true",
                   help="tokenize the whole dataset once up front (datasets.map "
                        "cache); collate only pads. Removes the per-step "
                        "tokenization bottleneck at large per-rank batches")
    p.add_argument("--pretokenized-cache-file", default=None,
                   help="optional fixed Arrow cache path for the pretokenized "
                        "training dataset, reusable across separate runs")
    p.add_argument("--per-rank-batch", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--optimizer", choices=["adamw", "soap-eigh", "soap-muon-eigh", "soap-muon-independent", "soap-muon-interface-shared", "joint-muon-p0", "joint-muon-p05", "muon", "muon-pre-ns-cautious-sphere", "muon-attention-operator", "muon-stable", "muon-polished", "muon-progressive-polished", "muon-equivariant-polished", "muon-adaptive", "muon-spectral-sketch-polished", "muon-split-merge-one", "muon-split-merge-two", "muon-inverse-half", "muon-inverse-half-cubed-quarter",
                                            "muon-projected-inverse-half",
                                            "muon-gated-inverse-half",
                                            "muon-correlation-gated-inverse-half",
                                            "muon-randomized-svd-inverse-half",
                                            "muon-randomized-svd-pinv",
                                            "muon-ns-pinv",
                                            "muon-ema-gram-inverse-half",
                                            "muon-ema-gram-pinv"],
                   default="adamw", help="matrix optimizer; Muon variants keep "
                   "AdamW for embeddings, norms, and biases")
    p.add_argument("--muon-lr", type=float, default=None,
                   help="learning rate for hidden matrices (default: --lr)")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--pre-ns-sphere-strength", type=float, default=0.03,
                   help="dimensionless strength of cautious radial momentum bias")
    p.add_argument("--pre-ns-sphere-log-error-clip", type=float, default=0.5,
                   help="absolute clip for log(||W||_F / initial_radius)")
    p.add_argument("--spectral-eps", type=float, default=1e-3,
                   help="relative singular-value floor for muon-inverse-half")
    p.add_argument("--muon-rms-scale", type=float, default=None,
                   help="post-NS Frobenius renormalization to this elementwise "
                        "update RMS (e.g. 0.2 for Kimi-style Adam matching)")
    p.add_argument("--muon-projection-beta", type=float, default=0.25,
                   help="inverse-half residual retained after projection toward Muon")
    p.add_argument("--randomized-svd-niter", type=int, default=2,
                   help="power iterations for randomized-SVD inverse-half Muon")
    p.add_argument("--randomized-svd-seed", type=int, default=17,
                   help="rank-independent base seed for randomized-SVD sketches")
    p.add_argument("--pinv-ns-steps", type=int, default=8,
                   help="cubic Newton-Schulz iterations for damped Gram pinv")
    p.add_argument("--gram-beta2", type=float, default=0.99,
                   help="EMA coefficient for raw-gradient Gram statistics")
    p.add_argument("--interface-mix-lambda", type=float, default=0.5,
                   help="weight of the upstream normalized Gram in a shared "
                        "SOAP-Muon interface covariance")
    p.add_argument("--joint-mix-lambda", type=float, default=0.5,
                   help="relative energy assigned to the upstream block in "
                        "state-free joint Muon")
    p.add_argument("--no-joint-block-normalize", action="store_true",
                   help="disable per-block Frobenius normalization before "
                        "state-free joint Muon")
    p.add_argument("--joint-diagnostics-every", type=int, default=0,
                   help="compute exact small-Gram diagnostics for joint Muon "
                        "every N optimizer steps (0 disables)")
    p.add_argument("--spectral-diagnostics-every", type=int, default=0,
                   help="compare inverse-half NS against exact small-Gram eigh "
                        "every N optimizer steps (0 disables)")
    p.add_argument("--full-diagnostics-every", type=int, default=0,
                   help="record aggregate weights, gradients, applied updates, "
                        "RMS scaling, alignment, and matrix isotropy every N "
                        "optimizer steps (0 disables)")
    p.add_argument("--matrix-step-mode",
                   choices=["none", "trust-ratio", "fro-sphere",
                            "sphere-attraction", "cautious-sphere-attraction",
                            "hyperball"],
                   default="none",
                   help="post-process each Linear-weight displacement to have a "
                        "scheduled relative Frobenius norm; fro-sphere also keeps "
                        "the initial weight Frobenius norm exactly")
    p.add_argument("--matrix-step-ratio", type=float, default=None,
                   help="peak ||delta W||_F / ||W||_F for --matrix-step-mode; "
                        "scaled by the optimizer's current/base learning rate")
    p.add_argument("--matrix-sphere-attraction", type=float, default=0.01,
                   help="peak per-step fraction of radial error restored toward "
                        "the initial Frobenius sphere; scaled by current/base LR")
    p.add_argument("--tensorboard-dir", default=None,
                   help="write TensorBoard events to this directory")
    p.add_argument("--checkpoint-dir", default=None,
                   help="save model checkpoints to this directory")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="checkpoint interval in optimizer steps")
    p.add_argument("--soap-beta1", type=float, default=0.95)
    p.add_argument("--soap-beta2", type=float, default=0.95)
    p.add_argument("--soap-shampoo-beta", type=float, default=None)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--tau-plus", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--no-stable", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--schedule", choices=["const", "wsd", "cosine"], default="wsd",
                   help="const reproduces the old fixed-lr behaviour; wsd = "
                        "warmup / stable / linear-decay tail; cosine = warmup + "
                        "cosine to 0")
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--decay-frac", type=float, default=0.2,
                   help="fraction of steps in the WSD decay tail")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--wd-anchor", action="store_true",
                   help="decay weights toward the PRETRAINED values instead of 0 "
                        "(decoupled L2-SP, arXiv 1802.01483): "
                        "w <- (1 - lr*wd) * w + lr*wd * w_pretrained")
    p.add_argument("--eval-every", type=int, default=10,
                   help="run downstream proxies every N steps (0 = start/end only)")
    return p.parse_args()


def lr_lambda(args):
    """Multiplier on args.lr per optimizer step (LambdaLR)."""
    warm = max(1, round(args.steps * args.warmup_frac))
    decay_start = args.steps - max(1, round(args.steps * args.decay_frac))

    def f(step):
        if args.schedule == "const":
            return 1.0
        if step < warm:
            return (step + 1) / warm
        if args.schedule == "cosine":
            t = (step - warm) / max(1, args.steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
        if step >= decay_start:  # wsd tail: linear to 0
            return max(0.0, (args.steps - step) / max(1, args.steps - decay_start))
        return 1.0

    return f


class Encoder(nn.Module):
    """Backbone + masked mean pooling. Returns UNnormalized reps (the loss
    normalizes). bf16 autocast in train mode; pooling always accumulates in fp32."""

    def __init__(self, backbone, use_bf16):
        super().__init__()
        self.backbone = backbone
        self.use_bf16 = use_bf16

    def forward(self, input_ids=None, attention_mask=None):
        with torch.autocast("cuda", torch.bfloat16,
                            enabled=self.use_bf16 and input_ids.is_cuda):
            hidden = self.backbone(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return pooled.to(hidden.dtype)


def split_dict(inputs, chunk_size):
    keys = list(inputs.keys())
    chunked = [inputs[k].split(chunk_size, dim=0) for k in keys]
    return [dict(zip(keys, vals)) for vals in zip(*chunked)]


class RandContext:
    """Fork-and-replay RNG state so a chunk's dropout pattern matches between the
    rep pass and the gradient pass."""

    def __init__(self, device):
        self.cpu_state = torch.get_rng_state()
        self.device = device
        self.gpu_state = (torch.cuda.get_rng_state(device)
                          if device.type == "cuda" else None)

    def __enter__(self):
        devices = [self.device] if self.device.type == "cuda" else []
        self._fork = torch.random.fork_rng(devices=devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.cpu_state)
        if self.gpu_state is not None:
            torch.cuda.set_rng_state(self.gpu_state, self.device)

    def __exit__(self, *exc):
        self._fork.__exit__(*exc)
        self._fork = None


class GradCache:
    """Minimal gradient cache (arXiv 2101.06983) for one encoder. The loss sees
    the full batch of reps as a leaf tensor; its backward (which, for the memeff
    distributed losses, runs the ring + reduce-scatter collectives) produces the
    cached rep grads, which the chunked re-forwards then push through the encoder
    via a surrogate dot-product backward."""

    def __init__(self, model, chunk_size, loss_fn):
        self.model = model
        self.chunk_size = chunk_size
        self.loss_fn = loss_fn

    def __call__(self, inputs, no_sync_except_last=False):
        chunks = split_dict(inputs, self.chunk_size)
        reps, states = [], []
        for c in chunks:
            states.append(RandContext(c["input_ids"].device))
            with torch.no_grad():
                reps.append(self.model(**c))
        full = torch.cat(reps).detach().requires_grad_(True)
        loss = self.loss_fn(full)
        loss.backward()
        grads = full.grad.split([r.shape[0] for r in reps])
        for i, (c, state, g) in enumerate(zip(chunks, states, grads)):
            sync = (not no_sync_except_last) or i == len(chunks) - 1
            ctx = contextlib.nullcontext() if sync else self.model.no_sync()
            with ctx, state:
                out = self.model(**c)
                (out.float() * g.float()).sum().backward()
        return loss.detach()


def make_loss_fn(args):
    from memeff import DistributedMemoryEfficientQwen3Loss
    loss_mod = DistributedMemoryEfficientQwen3Loss(
        temperature=args.temperature, margin=args.margin,
        stable=not args.no_stable, tau_plus=args.tau_plus,
        label_smoothing=args.label_smoothing)
    n_parts = len(args.columns.split(","))
    assert n_parts in (2, 3), "--columns must name 2 (pair) or 3 (triplet) columns"

    def loss_fn(reps):
        b = reps.shape[0] // n_parts
        q, d = reps[:b], reps[b:2 * b]
        h = reps[2 * b:].unsqueeze(1) if n_parts == 3 else None
        return loss_mod(q, d, h)

    return loss_fn


def make_loader(args, world, rank, tokenizer):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, DistributedSampler

    cols = args.columns.split(",")
    config = args.dataset_config or (
        "triplet" if args.dataset == "sentence-transformers/all-nli" else None)
    ds = load_dataset(args.dataset, config, split="train")

    if args.pretokenize:
        def tok_fn(batch):
            return {f"{c}_ids": tokenizer(batch[c], truncation=True,
                                          max_length=args.max_len)["input_ids"]
                    for c in cols}
        map_kwargs = {}
        if args.pretokenized_cache_file:
            cache_path = pathlib.Path(args.pretokenized_cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            map_kwargs["cache_file_name"] = str(cache_path)
        ds = ds.map(tok_fn, batched=True, num_proc=max(1, args.num_workers),
                    remove_columns=ds.column_names, desc="pretokenize",
                    **map_kwargs)

        def collate(rows):
            ids = [r[f"{c}_ids"] for c in cols for r in rows]
            enc = tokenizer.pad({"input_ids": ids}, padding=True,
                                return_tensors="pt")
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}
    else:
        def collate(rows):
            texts = [r[c] for c in cols for r in rows]
            return dict(tokenizer(texts, padding=True, truncation=True,
                                  max_length=args.max_len, return_tensors="pt",
                                  return_token_type_ids=False))

    sampler = (DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed,
                                  drop_last=True) if world > 1 else None)
    # fork-safety: the Rust tokenizer's thread pool segfaults forked workers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return DataLoader(ds, batch_size=args.per_rank_batch, sampler=sampler,
                      shuffle=sampler is None, drop_last=True, collate_fn=collate,
                      num_workers=args.num_workers, pin_memory=True,
                      persistent_workers=args.num_workers > 0)


def build_eval(args, tokenizer):
    """Pre-tokenize the two downstream proxies: all-nli dev triplet accuracy
    (in-domain, saturates quickly) and STS-B dev Spearman (the standard
    sentence-embedding proxy). Small enough to run redundantly on every rank,
    which avoids any cross-rank synchronization."""
    from datasets import load_dataset

    def tok(texts):
        return dict(tokenizer(list(texts), padding=True, truncation=True,
                              max_length=args.max_len, return_tensors="pt",
                              return_token_type_ids=False))

    nli = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    sts = load_dataset("sentence-transformers/stsb", split="validation")
    return {"nli": (tok(nli["anchor"]), tok(nli["positive"]), tok(nli["negative"])),
            "sts": (tok(sts["sentence1"]), tok(sts["sentence2"]),
                    torch.tensor(sts["score"], dtype=torch.float32))}


def encode_eval(mod, enc, device, chunk):
    outs = []
    n = enc["input_ids"].shape[0]
    with torch.no_grad():
        for i in range(0, n, chunk):
            c = {k: v[i:i + chunk].to(device) for k, v in enc.items()}
            outs.append(nn.functional.normalize(mod(**c).float(), dim=-1))
    return torch.cat(outs)


def spearman(a, b):
    """Spearman rho, ties broken by order (fine for a training proxy)."""
    def rank(x):
        r = torch.empty_like(x)
        r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
        return r
    ra, rb = rank(a) - (len(a) - 1) / 2, rank(b) - (len(b) - 1) / 2
    return ((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12)).item()


def run_eval(model, evalpack, device, chunk, rank, step):
    mod = model.module if hasattr(model, "module") else model
    was_training = mod.training
    mod.eval()
    a, p, n = (encode_eval(mod, e, device, chunk) for e in evalpack["nli"])
    acc = ((a * p).sum(-1) > (a * n).sum(-1)).float().mean().item()
    s1, s2, gold = evalpack["sts"]
    cos = (encode_eval(mod, s1, device, chunk)
           * encode_eval(mod, s2, device, chunk)).sum(-1).cpu()
    rho = spearman(cos, gold)
    if was_training:
        mod.train()
    if rank == 0:
        print(f"eval  step {step:5d}  nli-dev acc {acc:.4f}  "
              f"stsb-dev spearman {rho:.4f}", flush=True)
    return acc, rho


def save_checkpoint(model, args, step, eval_acc, eval_rho):
    """Save analysis-oriented model weights and run metadata atomically."""
    if not args.checkpoint_dir:
        return
    directory = pathlib.Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"step_{step:05d}.pt"
    temporary = directory / f".{target.name}.tmp"
    module = model.module if hasattr(model, "module") else model
    torch.save({
        "step": step,
        "model": module.state_dict(),
        "eval_nli_accuracy": eval_acc,
        "eval_stsb_spearman": eval_rho,
        "arguments": vars(args),
    }, temporary)
    temporary.replace(target)


def run_equiv(model, gc_wrap, loss_fn, batch, world, rank):
    """Param grads: grad-cached step vs one full-batch forward/backward."""
    model.eval()  # dropout off: the two paths must see identical stochasticity

    model.zero_grad(set_to_none=True)
    loss_gc = gc_wrap(batch, no_sync_except_last=world > 1)
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}

    model.zero_grad(set_to_none=True)
    loss_fn(model(**batch)).backward()

    worst, worst_name, tiny_bad = 0.0, "?", []
    for n, p in model.named_parameters():
        if p.grad is None:
            assert n not in grads, f"{n}: grad only in the grad-cache path"
            continue
        ref = p.grad.detach()
        diff = (grads[n] - ref).norm().item()
        if ref.norm().item() < 1e-6:
            # theoretically-zero grads (softmax shift invariance makes the
            # attention key biases' grads exact zeros): compare absolutely
            if diff > 1e-6:
                tiny_bad.append(n)
            continue
        err = diff / ref.norm().item()
        if err > worst:
            worst, worst_name = err, n
    dl = (loss_gc - loss_fn(model(**batch)).detach()).abs().item()
    ok = worst < 1e-3 and not tiny_bad
    print(f"[rank {rank}] equiv: {'OK ' if ok else 'FAIL'} "
          f"worst grad rel err {worst:.2e} ({worst_name}), |dloss| {dl:.2e}"
          + (f", zero-grad params off: {tiny_bad}" if tiny_bad else ""))
    assert ok


def make_optimizer(args, model):
    """Route hidden Linear weights to Muon and everything else to AdamW."""
    decay = 0.0 if args.wd_anchor else args.wd
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=decay)

    from muon import MuonWithAuxAdam
    mod = model.module if hasattr(model, "module") else model
    muon_ids = {id(layer.weight) for layer in mod.modules()
                if isinstance(layer, nn.Linear)}
    muon_params, adam_params = [], []
    for parameter in model.parameters():
        (muon_params if id(parameter) in muon_ids else adam_params).append(parameter)
    attention_pairs = []
    if args.optimizer == "muon-attention-operator":
        for layer in mod.modules():
            query = getattr(layer, "query", None)
            key = getattr(layer, "key", None)
            heads = getattr(layer, "num_attention_heads", None)
            if (isinstance(query, nn.Linear) and isinstance(key, nn.Linear)
                    and heads is not None):
                attention_pairs.append((query.weight, key.weight, heads))
        if not attention_pairs:
            raise ValueError("no query/key attention pairs found in model")
    if args.optimizer == "soap-eigh":
        from soap import SOAPWithAuxAdam
        return SOAPWithAuxAdam(
            muon_params, adam_params, lr=args.lr, adam_lr=args.lr,
            betas=(args.soap_beta1, args.soap_beta2),
            shampoo_beta=args.soap_shampoo_beta,
            weight_decay=decay)
    variants = {
        "muon": "polar",
        "muon-pre-ns-cautious-sphere": "pre-ns-cautious-sphere",
        "muon-attention-operator": "attention-operator",
        "muon-stable": "stable",
        "muon-polished": "polished",
        "muon-progressive-polished": "progressive-polished",
        "muon-equivariant-polished": "equivariant-polished",
        "muon-adaptive": "adaptive",
        "muon-spectral-sketch-polished": "spectral-sketch-polished",
        "muon-split-merge-one": "split-merge-one",
        "muon-split-merge-two": "split-merge-two",
        "muon-inverse-half": "inverse-half",
        "muon-inverse-half-cubed-quarter": "inverse-half-cubed-quarter",
        "muon-projected-inverse-half": "projected-inverse-half",
        "muon-gated-inverse-half": "gated-inverse-half",
        "muon-correlation-gated-inverse-half": "correlation-gated-inverse-half",
        "muon-randomized-svd-inverse-half": "randomized-svd-inverse-half",
        "muon-randomized-svd-pinv": "randomized-svd-pinv",
        "muon-ns-pinv": "ns-pinv",
        "muon-ema-gram-inverse-half": "ema-gram-inverse-half",
        "muon-ema-gram-pinv": "ema-gram-pinv",
        "soap-muon-eigh": "soap-muon-eigh",
        "soap-muon-independent": "soap-muon-independent",
        "soap-muon-interface-shared": "soap-muon-interface-shared",
        "joint-muon-p0": "joint-muon-p0",
        "joint-muon-p05": "joint-muon-p05",
    }
    return MuonWithAuxAdam(
        muon_params, adam_params, lr=args.muon_lr or args.lr,
        adam_lr=args.lr, variant=variants[args.optimizer], momentum=args.muon_momentum,
        weight_decay=decay, spectral_eps=args.spectral_eps,
        rms_scale=args.muon_rms_scale,
        projection_beta=args.muon_projection_beta,
        randomized_svd_niter=args.randomized_svd_niter,
        randomized_svd_seed=args.randomized_svd_seed,
        pinv_ns_steps=args.pinv_ns_steps,
        gram_beta2=args.gram_beta2,
        interface_mix_lambda=args.interface_mix_lambda,
        joint_mix_lambda=args.joint_mix_lambda,
        joint_normalize_blocks=not args.no_joint_block_normalize,
        joint_diagnostics_every=args.joint_diagnostics_every,
        spectral_diagnostics_every=args.spectral_diagnostics_every,
        pre_ns_sphere_strength=args.pre_ns_sphere_strength,
        pre_ns_sphere_log_error_clip=args.pre_ns_sphere_log_error_clip,
        attention_pairs=attention_pairs)


@torch.no_grad()
def capture_optimizer_diagnostic_inputs(model, optimizer):
    """Snapshot parameters and optimizer-group metadata before an update."""
    group_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    snapshots = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = group_by_parameter[id(parameter)]
        snapshots.append((
            name, parameter, parameter.detach().float().clone(),
            parameter.grad.detach().float().clone(), float(group["lr"]),
            float(group.get("weight_decay", 0.0)),
            bool(group.get("use_muon", False)),
        ))
    return snapshots


def initialize_matrix_step_control(model, optimizer, mode, peak_ratio,
                                   sphere_attraction=0.01):
    """Build state for controlling only the Linear matrices used by Muon."""
    if mode == "none":
        return None
    if peak_ratio is None or not 0.0 < peak_ratio < 2.0:
        raise ValueError("--matrix-step-ratio must be in (0, 2) when enabled")
    if not 0.0 <= sphere_attraction <= 1.0:
        raise ValueError("--matrix-sphere-attraction must be in [0, 1]")
    mod = model.module if hasattr(model, "module") else model
    controlled_ids = {id(layer.weight) for layer in mod.modules()
                      if isinstance(layer, nn.Linear)}
    group_by_id = {id(parameter): group for group in optimizer.param_groups
                   for parameter in group["params"]}
    entries = []
    for parameter in model.parameters():
        if id(parameter) not in controlled_ids:
            continue
        group = group_by_id[id(parameter)]
        entries.append({"parameter": parameter, "group": group,
                        "radius": parameter.detach().float().norm().item()})
    return {"mode": mode, "peak_ratio": peak_ratio,
            "sphere_attraction": sphere_attraction, "entries": entries}


@torch.no_grad()
def capture_matrix_step_inputs(control):
    """Snapshot controlled matrices immediately before the optimizer update."""
    if control is None:
        return None
    return [(entry, entry["parameter"].detach().float().clone())
            for entry in control["entries"]]


@torch.no_grad()
def apply_matrix_step_control(control, snapshots, collect_diagnostics=False):
    """Apply a relative trust step or a geodesic step on a fixed Fro sphere."""
    if control is None:
        return {}
    totals = {"raw_sq": 0.0, "applied_sq": 0.0, "weight_sq": 0.0,
              "restoring_sq": 0.0, "mask_selected": 0, "mask_elements": 0,
              "radial_abs": 0.0, "raw_weight_product": 0.0,
              "radius_error_sq": 0.0, "count": 0, "nonfinite": 0}
    for entry, before in snapshots:
        parameter, group = entry["parameter"], entry["group"]
        raw = parameter.detach().float() - before
        raw_norm = raw.norm()
        before_norm = before.norm()
        base_lr = float(group.get("initial_lr", group["lr"]))
        schedule_scale = float(group["lr"]) / max(base_lr, 1e-30)
        ratio = control["peak_ratio"] * schedule_scale
        if raw_norm > 0 and before_norm > 0:
            if control["mode"] == "fro-sphere":
                radius = entry["radius"]
                tangent = raw - (raw * before).sum() / before.square().sum() * before
                tangent_norm = tangent.norm()
                if tangent_norm > 0:
                    theta = 2.0 * math.asin(min(ratio / 2.0, 1.0))
                    after = radius * (
                        math.cos(theta) * before / before_norm
                        + math.sin(theta) * tangent / tangent_norm)
                else:
                    after = before * (radius / before_norm)
            else:
                controlled = raw * (ratio * before_norm / raw_norm)
                after = before + controlled
                if control["mode"] == "hyperball":
                    after = after * (before_norm / after.norm())
                if control["mode"] in {
                        "sphere-attraction", "cautious-sphere-attraction"}:
                    radius = entry["radius"]
                    after_norm = after.norm()
                    attraction = control["sphere_attraction"] * schedule_scale
                    restoring = attraction * (radius / after_norm - 1.0) * after
                    if control["mode"] == "cautious-sphere-attraction":
                        mask = restoring * controlled >= 0
                        restoring = restoring * mask
                        if collect_diagnostics:
                            totals["mask_selected"] += mask.sum().item()
                            totals["mask_elements"] += mask.numel()
                    after = after + restoring
                    if collect_diagnostics:
                        totals["restoring_sq"] += restoring.square().sum().item()
            parameter.copy_(after.to(parameter.dtype))
        if collect_diagnostics:
            applied = parameter.detach().float() - before
            totals["raw_sq"] += raw.square().sum().item()
            totals["applied_sq"] += applied.square().sum().item()
            totals["weight_sq"] += before.square().sum().item()
            totals["radial_abs"] += abs((raw * before).sum().item())
            totals["raw_weight_product"] += raw_norm.item() * before_norm.item()
            totals["radius_error_sq"] += (
                parameter.detach().float().norm().item() - entry["radius"]) ** 2
            totals["count"] += 1
            totals["nonfinite"] += (~torch.isfinite(parameter)).sum().item()
    if not collect_diagnostics:
        return {}
    weight_norm = math.sqrt(totals["weight_sq"])
    return {
        "raw_relative_fro": math.sqrt(totals["raw_sq"]) / max(weight_norm, 1e-30),
        "applied_relative_fro": math.sqrt(totals["applied_sq"]) / max(weight_norm, 1e-30),
        "raw_radial_fraction": totals["radial_abs"] / max(
            totals["raw_weight_product"], 1e-30),
        "radius_error_rms": math.sqrt(totals["radius_error_sq"] /
                                      max(totals["count"], 1)),
        "restoring_relative_fro": math.sqrt(totals["restoring_sq"]) /
                                  max(weight_norm, 1e-30),
        "cautious_mask_fraction": totals["mask_selected"] /
                                  max(totals["mask_elements"], 1),
        "nonfinite": totals["nonfinite"],
    }


@torch.no_grad()
def summarize_optimizer_step(snapshots):
    """Aggregate scale, alignment, and isotropy statistics after an update."""
    all_adam = all(not row[-1] for row in snapshots)
    buckets = {}
    for _, parameter, before, gradient, learning_rate, decay, use_muon in snapshots:
        after = parameter.detach().float()
        displacement = before - after
        optimizer_update = displacement / max(learning_rate, 1e-30) - decay * before
        category = ("all_parameters" if all_adam else
                    "muon_matrices" if use_muon else "auxiliary_parameters")
        bucket = buckets.setdefault(category, {
            "count": 0, "elements": 0,
            "weight_sq": 0.0, "weight_abs": 0.0, "weight_max": 0.0,
            "gradient_sq": 0.0, "gradient_abs": 0.0, "gradient_max": 0.0,
            "update_sq": 0.0, "update_abs": 0.0, "update_max": 0.0,
            "displacement_sq": 0.0, "displacement_max": 0.0,
            "gradient_update_dot": 0.0, "weight_update_dot": 0.0,
            "nonfinite_weights": 0, "nonfinite_gradients": 0,
            "nonfinite_updates": 0, "matrix_count": 0,
            "isotropy_residual_sum": 0.0,
        })
        n = parameter.numel()
        bucket["count"] += 1
        bucket["elements"] += n
        for prefix, tensor in (
            ("weight", before), ("gradient", gradient),
            ("update", optimizer_update), ("displacement", displacement),
        ):
            finite = torch.nan_to_num(tensor)
            bucket[f"{prefix}_sq"] += finite.square().sum().item()
            if f"{prefix}_abs" in bucket:
                bucket[f"{prefix}_abs"] += finite.abs().sum().item()
            bucket[f"{prefix}_max"] = max(
                bucket[f"{prefix}_max"], finite.abs().max().item())
        bucket["gradient_update_dot"] += (
            torch.nan_to_num(gradient) * torch.nan_to_num(optimizer_update)
        ).sum().item()
        bucket["weight_update_dot"] += (
            torch.nan_to_num(before) * torch.nan_to_num(optimizer_update)
        ).sum().item()
        bucket["nonfinite_weights"] += (~torch.isfinite(before)).sum().item()
        bucket["nonfinite_gradients"] += (~torch.isfinite(gradient)).sum().item()
        bucket["nonfinite_updates"] += (~torch.isfinite(optimizer_update)).sum().item()
        if parameter.ndim == 2:
            matrix = torch.nan_to_num(optimizer_update)
            if matrix.shape[0] > matrix.shape[1]:
                matrix = matrix.mT
            gram = matrix @ matrix.mT
            dimension = gram.shape[0]
            mean_eigenvalue = gram.trace() / max(dimension, 1)
            if mean_eigenvalue > 0:
                identity = torch.eye(dimension, device=gram.device, dtype=gram.dtype)
                residual = (gram / mean_eigenvalue - identity).norm() / math.sqrt(dimension)
                bucket["isotropy_residual_sum"] += residual.item()
                bucket["matrix_count"] += 1

    metrics = {}
    for category, bucket in buckets.items():
        n = max(bucket["elements"], 1)
        weight_norm = math.sqrt(bucket["weight_sq"])
        gradient_norm = math.sqrt(bucket["gradient_sq"])
        update_norm = math.sqrt(bucket["update_sq"])
        displacement_norm = math.sqrt(bucket["displacement_sq"])
        values = {
            "tensor_count": bucket["count"], "element_count": bucket["elements"],
            "weight_rms": weight_norm / math.sqrt(n),
            "weight_mean_abs": bucket["weight_abs"] / n,
            "weight_max_abs": bucket["weight_max"],
            "gradient_rms": gradient_norm / math.sqrt(n),
            "gradient_mean_abs": bucket["gradient_abs"] / n,
            "gradient_max_abs": bucket["gradient_max"],
            "optimizer_update_rms": update_norm / math.sqrt(n),
            "optimizer_update_mean_abs": bucket["update_abs"] / n,
            "optimizer_update_max_abs": bucket["update_max"],
            "parameter_displacement_rms": displacement_norm / math.sqrt(n),
            "parameter_displacement_max_abs": bucket["displacement_max"],
            "update_to_gradient_rms_ratio": update_norm / max(gradient_norm, 1e-30),
            "update_to_weight_rms_ratio": update_norm / max(weight_norm, 1e-30),
            "displacement_to_weight_rms_ratio": displacement_norm / max(weight_norm, 1e-30),
            "gradient_update_cosine": bucket["gradient_update_dot"] / max(
                gradient_norm * update_norm, 1e-30),
            "weight_update_cosine": bucket["weight_update_dot"] / max(
                weight_norm * update_norm, 1e-30),
            "nonfinite_weights": bucket["nonfinite_weights"],
            "nonfinite_gradients": bucket["nonfinite_gradients"],
            "nonfinite_updates": bucket["nonfinite_updates"],
        }
        if bucket["matrix_count"]:
            values["matrix_isotropy_residual_mean"] = (
                bucket["isotropy_residual_sum"] / bucket["matrix_count"])
        metrics[category] = values
    return metrics


def main():
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    if world > 1 and not torch.cuda.is_available():
        raise RuntimeError("distributed training currently requires CUDA")
    device = (torch.device("cuda", rank if world > 1 else 0)
              if torch.cuda.is_available() else torch.device("cpu"))
    torch.manual_seed(args.seed + rank)

    equiv = args.mode == "equiv"
    if equiv:
        torch.backends.cuda.matmul.allow_tf32 = False
        args.per_rank_batch = min(args.per_rank_batch, 256)
        args.chunk = min(args.chunk, 64)

    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if args.random_bert:
        backbone = BertModel(BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=max(128, args.max_len + 2),
            pad_token_id=tokenizer.pad_token_id,
        ), add_pooling_layer=False)
    else:
        # no pooling head: it would receive no grad and trip DDP's reducer
        try:
            backbone = AutoModel.from_pretrained(args.model, add_pooling_layer=False)
        except TypeError:  # architectures without a pooler (e.g. distilbert)
            backbone = AutoModel.from_pretrained(args.model)
    model = Encoder(backbone, use_bf16=not equiv).to(device)
    if world > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss_fn = make_loss_fn(args)
    gc_wrap = GradCache(model, args.chunk, loss_fn)
    loader = make_loader(args, world, rank, tokenizer)
    if rank == 0:
        print(f"{args.mode}: world={world} global_batch={world * args.per_rank_batch} "
              f"chunk={args.chunk} stable={not args.no_stable} "
              f"tau_plus={args.tau_plus} ls={args.label_smoothing} "
              f"temp={args.temperature} lr={args.lr} sched={args.schedule} "
              f"optim={args.optimizer} "
              f"muon_rms={args.muon_rms_scale} "
              f"muon_projection_beta={args.muon_projection_beta} "
              f"gram_beta2={args.gram_beta2} "
              f"wd={args.wd}{'->pretrained' if args.wd_anchor else '->0'} "
              f"model={'random-bert' if args.random_bert else args.model} "
              f"tokenizer={tokenizer_source} "
              f"data={args.dataset}[{args.columns}]")

    if equiv:
        batch = {k: v.to(device) for k, v in next(iter(loader)).items()}
        run_equiv(model, gc_wrap, loss_fn, batch, world, rank)
        if world > 1:
            dist.destroy_process_group()
        return

    # anchors capture the PRETRAINED weights before any update; with
    # --wd-anchor AdamW's own decay (toward 0) is disabled and replaced by a
    # decoupled pull toward the anchor after each step (L2-SP style).
    anchors = None
    if args.wd_anchor:
        anchors = {p: p.detach().clone() for p in model.parameters()
                   if p.requires_grad}
    optim = make_optimizer(args, model)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda(args))
    matrix_step_control = initialize_matrix_step_control(
        model, optim, args.matrix_step_mode, args.matrix_step_ratio,
        args.matrix_sphere_attraction)
    evalpack = build_eval(args, tokenizer)
    writer = None
    if rank == 0 and args.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tensorboard_dir)
        writer.add_text("config/arguments", json.dumps(vars(args), indent=2), 0)
    initial_acc, initial_rho = run_eval(
        model, evalpack, device, args.chunk, rank, step=0)
    if rank == 0:
        save_checkpoint(model, args, 0, initial_acc, initial_rho)
    if writer is not None:
        writer.add_scalar("eval/nli_accuracy", initial_acc, 0)
        writer.add_scalar("eval/stsb_spearman", initial_rho, 0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step, t0 = 0, time.perf_counter()
    epoch = 0
    while step < args.steps:
        if world > 1:
            loader.sampler.set_epoch(epoch)  # reshuffle if the data wraps around
        epoch += 1
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            loss = gc_wrap(batch, no_sync_except_last=world > 1)
            next_step = step + 1
            collect_full_diagnostics = (
                rank == 0 and args.full_diagnostics_every > 0
                and next_step % args.full_diagnostics_every == 0)
            diagnostic_inputs = (
                capture_optimizer_diagnostic_inputs(model, optim)
                if collect_full_diagnostics else None)
            matrix_step_inputs = capture_matrix_step_inputs(matrix_step_control)
            optim.step()
            matrix_step_diagnostics = apply_matrix_step_control(
                matrix_step_control, matrix_step_inputs, collect_full_diagnostics)
            if anchors is not None:
                with torch.no_grad():
                    for group in optim.param_groups:
                        rate = group["lr"] * args.wd
                        for p in group["params"]:
                            if p in anchors:
                                # w <- (1-rate)*w + rate*w_pretrained
                                p.lerp_(anchors[p], rate)
            sched.step()
            step += 1
            full_diagnostics = (
                summarize_optimizer_step(diagnostic_inputs)
                if diagnostic_inputs is not None else {})
            if writer is not None:
                loss_for_writer = loss.detach().float().clone()
                if world > 1:
                    dist.all_reduce(loss_for_writer)
                writer.add_scalar("train/loss", loss_for_writer.item(), step)
                for category, values in full_diagnostics.items():
                    for key, value in values.items():
                        writer.add_scalar(
                            f"diagnostics/full/{category}/{key}", value, step)
                for key, value in matrix_step_diagnostics.items():
                    writer.add_scalar(f"diagnostics/matrix_step/{key}", value, step)
                writer.flush()
            if step % args.log_every == 0:
                total = loss.clone()
                if world > 1:
                    dist.all_reduce(total)  # partial losses sum to the global loss
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / args.log_every
                peak = (torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda" else 0.0)
                if rank == 0:
                    print(f"step {step:5d}  loss {total.item():.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  "
                          f"{dt:6.2f} s/step  peak {peak:5.2f} GiB")
                    diagnostics = getattr(optim, "last_joint_diagnostics", {})
                    if diagnostics.get("step") == step:
                        print(
                            f"joint_diag step {step:5d}  pairs {diagnostics['pairs']}  "
                            f"sigma_in [{diagnostics['input_sigma_min']:.3e},"
                            f"{diagnostics['input_sigma_max']:.3e}]  "
                            f"sigma_out [{diagnostics['output_sigma_min']:.3e},"
                            f"{diagnostics['output_sigma_max']:.3e}]  "
                            f"ns_err {diagnostics['ns_relative_error_max']:.3e}  "
                            f"energy_first [{diagnostics['first_energy_fraction_min']:.3f},"
                            f"{diagnostics['first_energy_fraction_max']:.3f}]  "
                            f"scale_amp_max {diagnostics['scale_amplification_max']:.2e}  "
                            f"nonfinite {diagnostics['nonfinite']}")
                    spectral = getattr(optim, "last_spectral_diagnostics", {})
                    if spectral.get("step") == step:
                        print(
                            f"spectral_diag step {step:5d}  matrices {spectral['matrices']}  "
                            f"sigma_in [{spectral['input_sigma_min']:.3e},"
                            f"{spectral['input_sigma_max']:.3e}]  "
                            f"sigma_out [{spectral['output_sigma_min']:.3e},"
                            f"{spectral['output_sigma_max']:.3e}]  "
                            f"ns_err {spectral['relative_error_max']:.3e}  "
                            f"raw_norm_min {spectral['raw_output_norm_min']:.3e}  "
                            f"scale_amp_max {spectral['scale_amplification_max']:.2e}  "
                            f"nonfinite {spectral['nonfinite']}")
                    pre_ns = getattr(optim, "last_pre_ns_diagnostics", {})
                    if writer is not None:
                        writer.add_scalar("train/learning_rate", sched.get_last_lr()[0], step)
                        writer.add_scalar("performance/seconds_per_step", dt, step)
                        writer.add_scalar("performance/peak_gib", peak, step)
                        for diagnostics_name, diagnostics_values in (
                            ("joint", diagnostics), ("spectral", spectral),
                            ("pre_ns_sphere", pre_ns)
                        ):
                            if diagnostics_values.get("step") == step:
                                for key, value in diagnostics_values.items():
                                    if key not in ("step",) and isinstance(value, (int, float)):
                                        writer.add_scalar(
                                            f"diagnostics/{diagnostics_name}/{key}",
                                            value, step)
                        writer.flush()
                t0 = time.perf_counter()
            if (args.eval_every and step % args.eval_every == 0) \
                    or step >= args.steps:
                eval_acc, eval_rho = run_eval(
                    model, evalpack, device, args.chunk, rank, step)
                if writer is not None:
                    writer.add_scalar("eval/nli_accuracy", eval_acc, step)
                    writer.add_scalar("eval/stsb_spearman", eval_rho, step)
                    writer.flush()
                if (rank == 0 and args.checkpoint_dir
                        and (step % args.checkpoint_every == 0
                             or step >= args.steps)):
                    save_checkpoint(model, args, step, eval_acc, eval_rho)
                t0 = time.perf_counter()  # keep eval time out of s/step
            if step >= args.steps:
                break
    if world > 1:
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
