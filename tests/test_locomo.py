"""Tests for the LoCoMo loader (skipped if the data file is absent)."""

from __future__ import annotations

from contextbench.core.config_schema import DatasetConfig
from contextbench.datasets.base import ConversationSample
from contextbench.datasets.locomo import LoCoMoDataset


def test_loads_ten_samples(locomo_dataset):
    assert len(locomo_dataset) == 10


def test_sample_structure(locomo_dataset):
    s = next(iter(locomo_dataset))
    assert isinstance(s, ConversationSample)
    # turn ids contiguous and chronological
    assert [m.turn_id for m in s.messages] == list(range(s.num_turns))


def test_sessions_numeric_order(locomo_dataset):
    s = next(iter(locomo_dataset))
    order = []
    for m in s.messages:
        if not order or order[-1] != m.session_id:
            order.append(m.session_id)
    nums = [int(x.split("_")[1]) for x in order]
    assert nums == sorted(nums)


def test_role_mapping(locomo_dataset):
    s = next(iter(locomo_dataset))
    a, b = s.meta["speaker_a"], s.meta["speaker_b"]
    assert {m.role for m in s.messages if m.speaker == a} == {"user"}
    assert {m.role for m in s.messages if m.speaker == b} == {"assistant"}


def test_adversarial_answers_and_totals(locomo_dataset):
    all_qa = [q for s in locomo_dataset for q in s.qa_items]
    assert len(all_qa) == 1986
    adv = [q for q in all_qa if q.category == 5]
    assert len(adv) == 446
    assert all(isinstance(q.answer, str) and q.answer for q in adv)
    assert all(q.meta["adversarial"] for q in adv)
    assert all(isinstance(q.answer, str) for q in all_qa)  # ints coerced


def test_category_filter():
    ds = LoCoMoDataset(DatasetConfig(name="locomo", path="data/locomo10.json",
                                     qa_categories=[5]))
    qa = [q for s in ds for q in s.qa_items]
    assert qa and all(q.category == 5 for q in qa)


def test_num_samples_and_max_qa():
    ds = LoCoMoDataset(DatasetConfig(name="locomo", path="data/locomo10.json",
                                     num_samples=2, max_qa_per_sample=3))
    samples = list(ds)
    assert len(samples) == 2
    assert all(s.num_qa <= 3 for s in samples)


def test_image_caption_toggle():
    on = next(iter(LoCoMoDataset(DatasetConfig(name="locomo", path="data/locomo10.json"))))
    off = next(iter(LoCoMoDataset(DatasetConfig(name="locomo", path="data/locomo10.json",
                                                include_image_captions=False))))
    d15_on = next(m for m in on.messages if m.meta.get("dia_id") == "D1:5")
    d15_off = next(m for m in off.messages if m.meta.get("dia_id") == "D1:5")
    assert "[shared a photo:" in d15_on.content
    assert "[shared a photo:" not in d15_off.content
