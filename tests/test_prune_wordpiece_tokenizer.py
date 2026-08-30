import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from prune_wordpiece_tokenizer import (
    SPECIAL_TOKENS,
    is_ascii_wordpiece,
    is_required_fallback_wordpiece,
    select_pruned_vocabulary,
)


def test_language_filter_and_fallback_detection():
    assert is_ascii_wordpiece("hello")
    assert is_ascii_wordpiece("##ing")
    assert is_ascii_wordpiece("[")
    assert is_ascii_wordpiece("]")
    assert not is_ascii_wordpiece("東京")
    assert not is_ascii_wordpiece("[unused1]")
    assert is_required_fallback_wordpiece("a")
    assert is_required_fallback_wordpiece("##z")
    assert not is_required_fallback_wordpiece("word")


def test_selection_preserves_special_and_atomic_fallback_tokens():
    tokens = list(SPECIAL_TOKENS) + [
        "a", "b", "##a", "##b", "hello", "world", "rare", "東京"
    ]
    vocabulary = {token: index for index, token in enumerate(tokens)}
    frequencies = Counter({vocabulary["world"]: 100, vocabulary["hello"]: 50})
    required = {
        token for token in vocabulary if is_required_fallback_wordpiece(token)
    }
    selected = select_pruned_vocabulary(
        vocabulary, frequencies, target_size=len(required) + 1
    )

    assert required <= set(selected)
    assert "world" in selected
    assert "hello" not in selected
    assert "東京" not in selected

