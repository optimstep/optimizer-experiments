"""Prune an existing English WordPiece tokenizer using corpus token frequency."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, BertTokenizerFast


SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def is_ascii_wordpiece(token: str) -> bool:
    if token in SPECIAL_TOKENS:
        return True
    piece = token[2:] if token.startswith("##") else token
    is_bracketed_metadata_token = (
        len(piece) > 2 and piece.startswith("[") and piece.endswith("]")
    )
    return bool(piece) and not is_bracketed_metadata_token and all(
        ord(character) < 128 and character.isprintable() for character in piece
    )


def is_required_fallback_wordpiece(token: str) -> bool:
    if token in SPECIAL_TOKENS:
        return True
    piece = token[2:] if token.startswith("##") else token
    return len(piece) == 1 and is_ascii_wordpiece(token)


def select_pruned_vocabulary(
    original_vocabulary: dict[str, int],
    token_frequencies: Counter,
    target_size: int,
) -> list[str]:
    eligible = {
        token for token in original_vocabulary if is_ascii_wordpiece(token)
    }
    required = {
        token for token in eligible if is_required_fallback_wordpiece(token)
    }
    if target_size < len(required):
        raise ValueError(
            f"target size {target_size} cannot hold {len(required)} required tokens"
        )
    ranked = sorted(
        eligible - required,
        key=lambda token: (
            -token_frequencies[original_vocabulary[token]],
            original_vocabulary[token],
        ),
    )
    selected = required | set(ranked[: target_size - len(required)])
    return sorted(selected, key=original_vocabulary.__getitem__)


def count_token_frequencies(tokenizer, texts, batch_size: int) -> Counter:
    frequencies = Counter()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        for token_ids in encoded:
            frequencies.update(token_ids)
    return frequencies


def percentile(sorted_values: list[int], fraction: float) -> int:
    index = min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def measure_tokenizer(tokenizer, texts, batch_size: int) -> dict[str, float]:
    lengths = []
    unknown_tokens = 0
    total_tokens = 0
    unknown_id = tokenizer.unk_token_id
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        for token_ids in encoded:
            lengths.append(len(token_ids))
            unknown_tokens += sum(token_id == unknown_id for token_id in token_ids)
            total_tokens += len(token_ids)
    ordered = sorted(lengths)
    return {
        "mean_tokens": sum(lengths) / len(lengths),
        "p50_tokens": percentile(ordered, 0.50),
        "p95_tokens": percentile(ordered, 0.95),
        "p99_tokens": percentile(ordered, 0.99),
        "unknown_rate": unknown_tokens / max(1, total_tokens),
        "over_64_rate": sum(length > 64 for length in lengths) / len(lengths),
        "over_96_rate": sum(length > 96 for length in lengths) / len(lengths),
    }


def build_pruned_tokenizer(
    base_tokenizer,
    selected_tokens: list[str],
    output_directory: Path,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    vocabulary_file = output_directory / "vocab.txt"
    vocabulary_file.write_text("\n".join(selected_tokens) + "\n")
    tokenizer = BertTokenizerFast(
        vocab_file=str(vocabulary_file),
        do_lower_case=True,
        tokenize_chinese_chars=False,
        strip_accents=True,
        model_max_length=base_tokenizer.model_max_length,
    )
    tokenizer.save_pretrained(output_directory)
    return AutoTokenizer.from_pretrained(output_directory, local_files_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tokenizer", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--dataset", default="sentence-transformers/gooaq")
    parser.add_argument("--dataset-config", default="pair")
    parser.add_argument("--columns", default="question,answer")
    parser.add_argument("--frequency-pairs", type=int, default=200_000)
    parser.add_argument("--evaluation-pairs", type=int, default=20_000)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8192, 12288])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-root", type=Path,
                        default=Path("artifacts/pruned_tokenizers"))
    args = parser.parse_args()

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split="train",
        download_mode="reuse_dataset_if_exists",
    )
    columns = args.columns.split(",")
    required_pairs = args.frequency_pairs + args.evaluation_pairs
    subset = dataset.select(range(min(required_pairs, len(dataset))))
    frequency_texts = []
    evaluation_texts = []
    for index, row in enumerate(subset):
        destination = (
            frequency_texts if index < args.frequency_pairs else evaluation_texts
        )
        destination.extend(str(row[column]) for column in columns)

    frequencies = count_token_frequencies(
        base_tokenizer, frequency_texts, args.batch_size
    )
    base_metrics = measure_tokenizer(
        base_tokenizer, evaluation_texts, args.batch_size
    )
    report = {
        "base_tokenizer": args.base_tokenizer,
        "base_vocab_size": len(base_tokenizer),
        "frequency_pairs": args.frequency_pairs,
        "evaluation_pairs": args.evaluation_pairs,
        "base_metrics": base_metrics,
        "candidates": {},
    }
    original_vocabulary = base_tokenizer.get_vocab()
    for vocabulary_size in args.vocab_sizes:
        selected_tokens = select_pruned_vocabulary(
            original_vocabulary, frequencies, vocabulary_size
        )
        output_directory = args.output_root / f"gooaq_wordpiece_{vocabulary_size}"
        tokenizer = build_pruned_tokenizer(
            base_tokenizer, selected_tokens, output_directory
        )
        metrics = measure_tokenizer(tokenizer, evaluation_texts, args.batch_size)
        metrics["vocab_size"] = len(tokenizer)
        metrics["mean_token_inflation"] = (
            metrics["mean_tokens"] / base_metrics["mean_tokens"] - 1.0
        )
        metrics["embedding_parameters_at_hidden_256"] = len(tokenizer) * 256
        report["candidates"][str(vocabulary_size)] = metrics

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "pruning_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

