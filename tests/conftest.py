"""Shared test fixtures.

Most tests are hermetic: they use :class:`WordCounter`, a deterministic
`TokenCounter` that needs no model, so policy/runner logic is tested without any
HuggingFace dependency. Tests that need the real tokenizer/model are marked
``slow`` and skipped unless ``CONTEXTBENCH_SLOW=1``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from contextbench.core.config_schema import DatasetConfig
from contextbench.core.types import Message

DATA_PATH = Path("data/locomo10.json")


class WordCounter:
    """A deterministic TokenCounter: ~1 token per whitespace word + overhead.

    ``count_messages`` adds one separator token between lines so the exact block
    exceeds the additive per-message sum — exercising the policy's exact-fit trim.
    """

    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_message(self, message: Message) -> int:
        speaker = message.speaker or message.role
        return len(speaker.split()) + len(message.content.split()) + 1

    def count_messages(self, messages) -> int:
        if not messages:
            return 0
        return sum(self.count_message(m) for m in messages) + (len(messages) - 1)


class FakeEmbedder:
    """A deterministic, model-free Embedder for hermetic ΔContext tests.

    Each word maps to a fixed pseudo-random unit vector (seeded by a stable hash),
    and a text embeds to the L2-normalized mean of its word vectors. This is
    deterministic across processes (unlike ``hash``) and gives semantic-ish
    behaviour: texts sharing words land closer together.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _word_vector(self, word: str) -> np.ndarray:
        digest = hashlib.md5(word.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dim).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        words = text.lower().split() or ["<empty>"]
        vector = np.mean([self._word_vector(w) for w in words], axis=0)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    def embed_batch(self, texts):
        return np.stack([self.embed(t) for t in texts])


@pytest.fixture
def counter() -> WordCounter:
    return WordCounter()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=16)


@pytest.fixture
def messages() -> list[Message]:
    """A 10-turn synthetic conversation with increasing message lengths."""
    out = []
    for i in range(10):
        out.append(
            Message(
                role="user" if i % 2 == 0 else "assistant",
                content=" ".join(["word"] * (i + 1)),
                turn_id=i,
                speaker=f"Speaker{i % 2}",
                session_id="session_1",
            )
        )
    return out


@pytest.fixture
def locomo_dataset():
    """The real LoCoMo dataset, or skip if the data file is absent."""
    if not DATA_PATH.is_file():
        pytest.skip(f"LoCoMo data not present at {DATA_PATH}")
    from contextbench.datasets.locomo import LoCoMoDataset

    return LoCoMoDataset(DatasetConfig(name="locomo", path=str(DATA_PATH)))


def pytest_collection_modifyitems(config, items):
    """Skip ``slow`` tests unless CONTEXTBENCH_SLOW=1."""
    if os.environ.get("CONTEXTBENCH_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow; set CONTEXTBENCH_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
