#!/usr/bin/env python3
"""Small decoder-only next-token benchmark with final-residual control."""
import argparse
import contextlib
import math
import os
import pathlib
import random
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from experiments.train import load_config, set_value

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size, width, layers, heads, max_len,
                 abc_reference_width):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(max_len, width)
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, batch_first=True, norm_first=True,
            activation="gelu")
        self.blocks = nn.TransformerEncoder(block, layers)
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._init)
        scale = math.sqrt(abc_reference_width / width)
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear) and module is not self.head:
                    module.weight.mul_(scale)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        mask = torch.triu(torch.ones(length, length, device=tokens.device,
                                     dtype=torch.bool), diagonal=1)
        hidden = self.blocks(hidden, mask=mask, is_causal=True)
        return self.head(self.final_norm(hidden))


class FinalResidualL1Controller:
    """Rescale the residual stream immediately before its final LayerNorm."""
    def __init__(self, model, target, strength=1.0, log_clip=4.0, eps=1e-12):
        self.target = float(target)
        self.strength = float(strength)
        self.log_clip = float(log_clip)
        self.eps = float(eps)
        self.last = {}
        self.handle = model.final_norm.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        value = inputs[0]
        pre = value.detach().float().abs().mean().item()
        error = math.log(max(self.target, self.eps) / max(pre, self.eps))
        scale = math.exp(self.strength * max(-self.log_clip,
                                             min(self.log_clip, error)))
        post = pre * scale
        self.last = {
            "pre_l1": pre,
            "pre_relative_error": pre / self.target - 1.0,
            "scale": scale,
            "post_l1": post,
            "post_relative_error": post / self.target - 1.0,
        }
        return (value * scale, *inputs[1:])

    def close(self):
        self.handle.remove()


@torch.no_grad()
def residual_l1_probe(model, tokens):
    values = {}
    handles = []
    for index, layer in enumerate(model.blocks.layers):
        def record(_module, _inputs, output, name=f"layer_{index}"):
            values[name] = output.detach().float().abs().mean().item()
        handles.append(layer.register_forward_hook(record))
    model(tokens)
    for handle in handles:
        handle.remove()
    return values


class ResidualActivationWeightController:
    """Use residual L1 error to steer output weights within a fixed step norm."""
    def __init__(self, model, targets, ratio, strength, log_clip, hard=False,
                 source_strength=0.0):
        self.model = model
        self.targets = dict(targets)
        self.ratio = float(ratio)
        self.strength = float(strength)
        self.log_clip = float(log_clip)
        self.hard = bool(hard)
        self.source_strength = float(source_strength)
        self.last = {}
        self.handles = []
        self.parameter_stages = {}
        self.parameter_strengths = {}
        self.parameter_stages[id(model.token.weight)] = "layer_0"
        self.parameter_stages[id(model.position.weight)] = "layer_0"
        self.parameter_strengths[id(model.token.weight)] = self.source_strength
        self.parameter_strengths[id(model.position.weight)] = self.source_strength
        for index, layer in enumerate(model.blocks.layers):
            name = f"layer_{index}"
            self.handles.append(layer.register_forward_hook(self._record(name)))
            self.parameter_stages[id(layer.self_attn.out_proj.weight)] = name
            self.parameter_stages[id(layer.linear2.weight)] = name

    def _record(self, name):
        def record(_module, _inputs, output):
            current = output.detach().float().abs().mean().item()
            target = self.targets[name]
            self.last[name] = {
                "l1": current,
                "relative_error": current / target - 1.0,
                "log_error": math.log(target / max(current, 1e-30)),
            }
        return record

    def capture(self):
        return [(parameter, parameter.detach().float().clone())
                for parameter in self.model.parameters() if parameter.ndim == 2]

    def set_strength(self, strength):
        self.strength = float(strength)

    @torch.no_grad()
    def apply(self, snapshots, ratio_scale=1.0):
        ratio = self.ratio * float(ratio_scale)
        totals = {"raw_sq": 0.0, "applied_sq": 0.0, "weight_sq": 0.0,
                  "correction_sq": 0.0, "correction_base_dot": 0.0,
                  "correction_base_product": 0.0, "nonfinite": 0}
        for parameter, before in snapshots:
            raw = parameter.detach().float() - before
            before_norm, raw_norm = before.norm(), raw.norm()
            target_norm = ratio * before_norm
            base = (raw * (target_norm / raw_norm)
                    if raw_norm > 0 and before_norm > 0 else raw)
            correction = torch.zeros_like(base)
            stage = self.parameter_stages.get(id(parameter))
            if stage is not None and before_norm > 0:
                error = self.last[stage]["log_error"]
                clipped = max(-self.log_clip, min(self.log_clip, error))
                correction = (self.strength * clipped * target_norm
                              * self.parameter_strengths.get(id(parameter), 1.0)
                              * before / before_norm)
            combined = base + correction
            if self.hard and stage is not None and correction.norm() > 0:
                combined = correction
            combined_norm = combined.norm()
            applied = (combined * (target_norm / combined_norm)
                       if combined_norm > 0 and before_norm > 0 else combined)
            parameter.copy_((before + applied).to(parameter.dtype))
            totals["raw_sq"] += raw.square().sum().item()
            totals["applied_sq"] += applied.square().sum().item()
            totals["weight_sq"] += before.square().sum().item()
            totals["correction_sq"] += correction.square().sum().item()
            totals["correction_base_dot"] += (correction * base).sum().item()
            totals["correction_base_product"] += correction.norm().item() * base.norm().item()
            totals["nonfinite"] += (~torch.isfinite(parameter)).sum().item()
        weight_norm = math.sqrt(totals["weight_sq"])
        return {
            "raw_relative_fro": math.sqrt(totals["raw_sq"]) / max(weight_norm, 1e-30),
            "applied_relative_fro": math.sqrt(totals["applied_sq"]) / max(weight_norm, 1e-30),
            "correction_relative_fro": math.sqrt(totals["correction_sq"]) / max(weight_norm, 1e-30),
            "correction_base_cosine": totals["correction_base_dot"] / max(
                totals["correction_base_product"], 1e-30),
            "nonfinite": totals["nonfinite"],
        }

    def close(self):
        for handle in self.handles:
            handle.remove()


def synthetic_batches(vocab, batch, length, seed):
    generator = torch.Generator().manual_seed(seed)
    while True:
        yield torch.randint(0, vocab, (batch, length + 1), generator=generator)


def text_batches(config, tokenizer, split=None):
    from datasets import load_dataset
    data = config["data"]
    stream = load_dataset(data["dataset"], split=split or data.get("split", "train"),
                          streaming=True)
    buffer = []
    batch_size = config["training"]["batch_size"]
    length = config["training"]["sequence_length"]
    for row in stream:
        ids = tokenizer(row[data.get("text_column", "text")],
                        add_special_tokens=False)["input_ids"]
        buffer.extend(ids + [tokenizer.sep_token_id])
        while len(buffer) >= batch_size * (length + 1):
            count = batch_size * (length + 1)
            yield torch.tensor(buffer[:count]).view(batch_size, length + 1)
            del buffer[:count]


@torch.no_grad()
def calibrate(model, batch):
    value = {}
    def record(_module, inputs):
        value["l1"] = inputs[0].float().abs().mean().item()
    handle = model.final_norm.register_forward_pre_hook(record)
    model(batch[:, :-1])
    handle.remove()
    return value["l1"]


def run(config):
    training, model_cfg, logging = (config["training"], config["model"],
                                    config["logging"])
    seed = int(training["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDecoderLM(
        model_cfg["vocab_size"], model_cfg["hidden_size"],
        model_cfg["num_hidden_layers"], model_cfg["num_attention_heads"],
        training["sequence_length"], model_cfg["abc_reference_width"]).to(device)
    if config["data"].get("synthetic", False):
        batches = synthetic_batches(model_cfg["vocab_size"],
                                    training["batch_size"],
                                    training["sequence_length"], seed)
        eval_stream = synthetic_batches(model_cfg["vocab_size"],
                                        training["batch_size"],
                                        training["sequence_length"], seed + 1)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer"])
        batches = text_batches(config, tokenizer)
        eval_stream = text_batches(
            config, tokenizer, split=config["data"].get("validation_split", "validation"))
    eval_count = int(logging.get("validation_batches", 0))
    validation_batches = [next(eval_stream).to(device) for _ in range(eval_count)]
    probe = next(batches).to(device)
    targets = residual_l1_probe(model, probe[:, :-1])
    controller = ResidualActivationWeightController(
        model, targets, config["optimizer"]["matrix_step_ratio"],
        config["optimizer"].get("weight_activation_force", 0.0),
        config["optimizer"].get("weight_activation_log_clip", 1.0),
        config["optimizer"].get("weight_activation_hard", False),
        config["optimizer"].get("weight_activation_source_strength", 0.0))
    optimizer = make_optimizer(model, config)
    lr_multiplier = make_lr_multiplier(training)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    writer = SummaryWriter(logging["tensorboard_dir"])
    loss_fn = nn.CrossEntropyLoss()
    final_strength = float(config["optimizer"].get("weight_activation_force", 0.0))
    initial_strength = float(config["optimizer"].get(
        "weight_activation_force_initial", final_strength))
    decay_steps = int(config["optimizer"].get("weight_activation_force_decay_steps", 0))
    validation_every = int(logging.get("validation_every", 0))
    checkpoint_every = int(logging.get("checkpoint_every", 0))
    checkpoint_path = pathlib.Path(logging["tensorboard_dir"]).parent / "checkpoint_latest.pt"
    precision = training.get("precision", "fp32")
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unknown training precision: {precision}")
    for step in range(1, training["steps"] + 1):
        fraction = min(max((step - 1) / max(decay_steps, 1), 0.0), 1.0)
        strength = initial_strength + fraction * (final_strength - initial_strength)
        controller.set_strength(strength)
        batch = probe if step == 1 else next(batches).to(device)
        inputs, labels = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if precision == "bf16" and device.type == "cuda"
                    else contextlib.nullcontext())
        with autocast:
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        snapshots = controller.capture()
        optimizer.step()
        ratio_scale = lr_multiplier(step - 1)
        diagnostics = controller.apply(snapshots, ratio_scale)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("diagnostics/logit_rms", logits.float().square().mean().sqrt(), step)
        writer.add_scalar("diagnostics/weight_activation_force", strength, step)
        writer.add_scalar("diagnostics/matrix_step/target_ratio",
                          config["optimizer"]["matrix_step_ratio"] * ratio_scale,
                          step)
        writer.add_scalar("train/lr_scale", ratio_scale, step)
        for stage, values in controller.last.items():
            for key, value in values.items():
                writer.add_scalar(f"coord_check/{stage}/{key}", value, step)
        if step % logging["full_diagnostics_every"] == 0:
            for key, value in diagnostics.items():
                writer.add_scalar(f"diagnostics/matrix_step/{key}", value, step)
        if validation_every and step % validation_every == 0:
            writer.add_scalar("validation/loss",
                              evaluate(model, validation_batches, loss_fn), step)
        if checkpoint_every and step % checkpoint_every == 0:
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "config": config},
                       checkpoint_path)
        writer.flush()
        print(f"step {step:4d} loss {loss.item():.4f}", flush=True)
        scheduler.step()
    controller.close()
    writer.close()


def make_lr_multiplier(training):
    """Return the shared optimizer-LR and relative-step schedule."""
    schedule = training.get("schedule", "const")
    if schedule not in {"const", "wsd", "cosine"}:
        raise ValueError(f"unknown training schedule: {schedule}")
    steps = int(training["steps"])
    warm = max(1, round(steps * float(training.get("warmup_frac", 0.1))))
    decay_start = steps - max(
        1, round(steps * float(training.get("decay_frac", 0.2))))

    def multiplier(step):
        if schedule == "const":
            return 1.0
        if step < warm:
            return (step + 1) / warm
        if schedule == "cosine":
            progress = (step - warm) / max(1, steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        if step >= decay_start:
            return max(0.0, (steps - step) / max(1, steps - decay_start))
        return 1.0

    return multiplier


@torch.no_grad()
def evaluate(model, batches, loss_fn):
    was_training = model.training
    model.eval()
    total = 0.0
    for batch in batches:
        inputs, labels = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        total += loss_fn(logits.reshape(-1, logits.shape[-1]),
                         labels.reshape(-1)).item()
    model.train(was_training)
    return total / len(batches)


def make_optimizer(model, config):
    optimizer_cfg = config["optimizer"]
    if optimizer_cfg.get("algorithm", "adamw") == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        )
    if optimizer_cfg["algorithm"] != "muon":
        raise ValueError(f"unknown optimizer algorithm: {optimizer_cfg['algorithm']}")
    from optimizers.muon import MuonWithAuxAdam
    excluded = {id(model.token.weight), id(model.position.weight)}
    muon_params, adam_params = [], []
    for parameter in model.parameters():
        target = (muon_params if parameter.ndim == 2 and id(parameter) not in excluded
                  else adam_params)
        target.append(parameter)
    return MuonWithAuxAdam(
        muon_params, adam_params,
        lr=float(optimizer_cfg.get("muon_lr", 0.02)),
        adam_lr=float(optimizer_cfg["lr"]),
        variant=str(optimizer_cfg.get("muon_variant", "polar")),
        momentum=float(optimizer_cfg.get("muon_momentum", 0.95)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    for expression in args.set:
        set_value(config, expression)
    path = pathlib.Path(config["logging"]["tensorboard_dir"])
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    run(config)
    # Hugging Face streaming can retain HTTP worker threads after iteration.
    # All writers and hooks are closed above, so exit without waiting on them.
    os._exit(0)


if __name__ == "__main__":
    main()
