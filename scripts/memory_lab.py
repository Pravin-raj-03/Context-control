"""Memory lab: SeCom-style memory baselines vs ΔContext topic retrieval (LongMemEval-S).

Every method fills the same token budget (Qwen2.5 tokens of the rendered turn
lines, as in the harness) from one history, and is scored by evidence recall:
the share of the question's labelled evidence turns that end up in the context.

Memory units:
* ``turn``      -- one message;
* ``exchange``  -- a user message and the assistant replies that follow it
                   (SeCom's and LongMemEval's "turn/round" level);
* ``session``   -- a whole chat session;
* ``segment``   -- ΔContext v3 topics, found online with no session information;
* ``segsess``   -- ΔContext topics cut at session starts as well (SeCom also
                   segments within given sessions).

Keys: the raw unit text, or its LLMLingua-2 compression (SeCom's denoising;
``scripts/compress_memory.py``). Retrievers: BM25 (Okapi, lowercased word
tokens), MPNet (multi-qa-mpnet-base-dot-v1, SeCom's dense retriever) and their
reciprocal-rank fusion (RRF, k=60).

Unit-level methods (SeCom) fill the budget with whole units in rank order; a
unit larger than what is left contributes the turns that still fit. ΔContext
methods rank turns using the topic structure (see ``ours``).

Usage:
    python scripts/memory_lab.py --split dev [--budget 2048] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
os.environ.setdefault("USE_TF", "0")

import numpy as np
import yaml

from contextbench.chatbot.formatting import TranscriptFormatter
from contextbench.context.state import SentenceTransformerEmbedder, text_key
from contextbench.core.config_schema import DatasetConfig
from contextbench.datasets.longmemeval import QUESTION_TYPES, LongMemEvalDataset
from contextbench.policies.precomputed import selection_key
from scripts.segmentation_report import _WordCounter, delta_boundaries, delta_policy_from_params, dev_test_split

TYPE_NAMES = {v: k for k, v in QUESTION_TYPES.items()}
MPNET = "data/models/multi-qa-mpnet-base-dot-v1"  # sentence-transformers/multi-qa-mpnet-base-dot-v1
RRF_K = 60
_WORD = re.compile(r"\w+")


# -- caches ------------------------------------------------------------------------------

class DenseCache:
    """Text -> fp16 vector, computed on demand on the GPU and kept on disk."""

    def __init__(self, model_name: str, path: Path, normalize: bool):
        self.model_name, self.path, self.normalize = model_name, path, normalize
        self.table: dict[str, np.ndarray] = {}
        if path.is_file():
            data = np.load(path, allow_pickle=False)
            self.table = dict(zip(data["keys"].tolist(), data["vectors"]))
        self._model = None
        self._dirty = False

    def ensure(self, texts: list[str], batch_size: int = 64) -> None:
        missing = sorted({t for t in texts if text_key(t) not in self.table})
        if not missing:
            return
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cuda")
            self._model.half()
        order = sorted(range(len(missing)), key=lambda i: len(missing[i]))  # length-bucketed batches
        vectors = self._model.encode([missing[i] for i in order], batch_size=batch_size,
                                     normalize_embeddings=self.normalize, convert_to_numpy=True,
                                     show_progress_bar=len(missing) > 5000)
        for i, v in zip(order, vectors):
            self.table[text_key(missing[i])] = v.astype(np.float16)
        self._dirty = True

    def matrix(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.table[text_key(t)] for t in texts]).astype(np.float32)

    def save(self) -> None:
        if self._dirty:
            keys = np.array(list(self.table), dtype=object).astype(str)
            np.savez(self.path, keys=keys, vectors=np.stack(list(self.table.values())))
            self._dirty = False


def load_compression(path: Path) -> dict[str, str]:
    table = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            table[row["k"]] = row["c"]
    return table


# -- ranking helpers ------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def slug(name: str) -> str:
    """Method name usable as a Hydra override value (export keys)."""
    return re.sub(r"[^0-9A-Za-z.]+", "_", name.replace("Δ", "D").replace("=", "")).strip("_")


def bm25_scores(docs: list[str], query: str) -> np.ndarray:
    from rank_bm25 import BM25Okapi
    return np.asarray(BM25Okapi([tokens(d) or ["_"] for d in docs]).get_scores(tokens(query)), dtype=np.float64)


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank position of each item (0 = best), ties broken by index."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(len(scores))
    return r


def rrf(*score_lists: np.ndarray) -> np.ndarray:
    return sum(1.0 / (RRF_K + 1 + ranks(s)) for s in score_lists)


def fill_units(unit_order, units, cost, budget):
    """Whole units in rank order; a unit that does not fit gives the turns that do."""
    used, chosen = 0, []
    for u in unit_order:
        for i in units[u]:
            if used + cost[i] <= budget:
                chosen.append(i)
                used += cost[i]
        if budget - used < cost.min():
            break
    return chosen


def fill_turns(turn_order, cost, budget):
    used, chosen = 0, []
    for i in turn_order:
        if used + cost[i] <= budget:
            chosen.append(i)
            used += cost[i]
    return chosen


# -- per-history memory --------------------------------------------------------------------

class History:
    def __init__(self, sample, policy, tokenizer, formatter, comp):
        self.msgs = list(sample.messages)
        self.qa = sample.qa_items[0]
        n = len(self.msgs)
        self.n = n
        self.cost = np.array([len(x) for x in tokenizer([formatter.format_line(m) for m in self.msgs],
                                                         add_special_tokens=False)["input_ids"]])
        self.evidence = {i for i, m in enumerate(self.msgs) if m.meta["dia_id"] in set(self.qa.evidence)}
        self.session_of = [m.session_id for m in self.msgs]
        self.qtype = TYPE_NAMES[self.qa.category]
        raw = [m.content for m in self.msgs]
        self.comp_missing = sum(text_key(c) not in comp for c in raw)
        compressed = [comp.get(text_key(c), c) for c in raw]
        self.line = {"raw": [f"{m.role}: {c}" for m, c in zip(self.msgs, raw)],
                     "comp": [f"{m.role}: {c}" for m, c in zip(self.msgs, compressed)]}
        self.minilm_text = [f"{m.speaker or m.role}: {m.content}" for m in self.msgs]

        session_starts = {i for i in range(1, n) if self.session_of[i] != self.session_of[i - 1]}
        exchange_starts = session_starts | {i for i in range(1, n) if self.msgs[i].role == "user"}
        seg_bounds = delta_boundaries(policy, self.msgs)  # last turn of each topic
        seg_starts = {b + 1 for b in seg_bounds}
        self.units = {
            "turn": [[i] for i in range(n)],
            "exchange": self._split(exchange_starts),
            "session": self._split(session_starts),
            "segment": self._split(seg_starts),
            "segsess": self._split(seg_starts | session_starts),
            "fixed10": self._split(set(range(10, n, 10))),
        }

    def _split(self, starts):
        cuts = [0] + sorted(s for s in starts if 0 < s < self.n) + [self.n]
        return [list(range(a, b)) for a, b in zip(cuts, cuts[1:]) if b > a]

    def unit_text(self, kind: str, key: str) -> list[str]:
        return ["\n".join(self.line[key][i] for i in unit) for unit in self.units[kind]]

    def unit_of_turn(self, kind: str) -> np.ndarray:
        out = np.empty(self.n, dtype=np.int64)
        for u, unit in enumerate(self.units[kind]):
            out[unit] = u
        return out


# -- methods -----------------------------------------------------------------------------

class Scorer:
    """Unit scores for (unit kind, key, retriever), memoised per history."""

    def __init__(self, h: History, mpnet: DenseCache, minilm, query: str):
        self.h, self.mpnet, self.minilm, self.query = h, mpnet, minilm, query
        self.memo = {}
        self.q_mpnet = None

    def __call__(self, kind: str, key: str, retriever: str) -> np.ndarray:
        name = (kind, key, retriever)
        if name in self.memo:
            return self.memo[name]
        texts = self.h.unit_text(kind, key)
        if retriever == "bm25":
            s = bm25_scores(texts, self.query)
        elif retriever == "mpnet":
            if self.q_mpnet is None:
                self.q_mpnet = self.mpnet.matrix([self.query])[0]
            s = self.mpnet.matrix(texts) @ self.q_mpnet
        elif retriever == "hybrid":
            s = rrf(self(kind, key, "bm25"), self(kind, key, "mpnet"))
        elif retriever == "hybrid_minilm":
            s = rrf(self(kind, key, "bm25"), self(kind, "raw", "minilm"))
        elif retriever == "minilm":  # ΔContext's own encoder: turn vectors, unit = centroid
            E = self.minilm["E"]
            C = np.stack([E[u].mean(0) for u in self.h.units[kind]])
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
            s = C @ self.minilm["q"]
        else:
            raise ValueError(retriever)
        self.memo[name] = s
        return s


def method_secom(h, sc, budget, kind, key, retriever):
    order = np.argsort(-sc(kind, key, retriever), kind="stable")
    return fill_units(order, h.units[kind], h.cost, budget)


def method_turns(h, sc, budget, key, retriever):
    return fill_turns(np.argsort(-sc("turn", key, retriever), kind="stable"), h.cost, budget)


def method_ours_user(h, sc, budget, w=0.25, boost=0.0, k_topics=3, reserve=0.0, topics="segment"):
    """ΔContext topic retrieval that reads the user's words inside relevant topics.

    Starts from the fused turn/topic ranking of ``method_ours`` (BM25+MiniLM,
    ΔContext segments). ``boost`` adds ``boost`` x (best RRF term) to every user
    turn inside the ``k_topics`` best topics. ``reserve`` keeps that fraction of
    the budget for the remaining user turns of the topics already read from,
    taken in fused order once the main fill stops. ``topics`` swaps the grouping
    (``session``: true sessions; ``fixed10``: 10-turn chunks) for controls.
    """
    turn_s = sc("turn", "raw", "hybrid_minilm")
    topic_of = h.unit_of_turn(topics)
    topic_s = sc(topics, "raw", "hybrid_minilm")
    topic_rank = ranks(topic_s)
    fused = 1.0 / (RRF_K + 1 + ranks(turn_s)) + w / (RRF_K + 1 + topic_rank)[topic_of]
    is_user = np.array([m.role == "user" for m in h.msgs])
    if boost:
        fused = fused + boost / (RRF_K + 1) * (is_user & (topic_rank[topic_of] < k_topics))
    order = np.argsort(-fused, kind="stable")
    main = fill_turns(order, h.cost, int(budget * (1 - reserve)))
    if not reserve:
        return main
    used = int(h.cost[main].sum())
    chosen, touched = list(main), {int(topic_of[i]) for i in main}
    taken = set(chosen)
    for i in order:  # the touched topics' user turns, then anything that still fits
        if i not in taken and is_user[i] and int(topic_of[i]) in touched and used + h.cost[i] <= budget:
            chosen.append(int(i)); taken.add(int(i)); used += h.cost[i]
    for i in order:
        if i not in taken and used + h.cost[i] <= budget:
            chosen.append(int(i)); taken.add(int(i)); used += h.cost[i]
    return chosen


def method_user_prior(h, sc, budget, boost=1.0, only=False):
    """Control without topics: turn-level BM25+MiniLM plus a global user-turn bonus
    (``only``: every user turn before any assistant turn)."""
    s = 1.0 / (RRF_K + 1 + ranks(sc("turn", "raw", "hybrid_minilm")))
    is_user = np.array([m.role == "user" for m in h.msgs])
    s = s + (1.0 if only else boost / (RRF_K + 1)) * is_user
    return fill_turns(np.argsort(-s, kind="stable"), h.cost, budget)


def method_density(h, sc, budget, alpha=0.5):
    """Control without topics or roles: turn-level BM25+MiniLM, ranked by RRF
    score per token^alpha, so short turns are favoured."""
    s = 1.0 / (RRF_K + 1 + ranks(sc("turn", "raw", "hybrid_minilm")))
    return fill_turns(np.argsort(-(s / np.power(h.cost, alpha)), kind="stable"), h.cost, budget)


def method_ours(h, sc, budget, topics="segment", turn_key="raw", topic_key="comp", retriever="hybrid",
                w=1.0, cap=None, partner=False, penalty=0.0):
    """ΔContext topic retrieval.

    Each turn's score fuses its own rank with its topic's rank (RRF; ``w`` weighs
    the topic), so a turn inside a relevant topic is preferred while every turn
    stays reachable. ``cap`` limits turns taken from one topic per round
    (topic-diverse filling, for evidence spread over several topics);
    ``partner`` adds the other half of the exchange when it lies in the same topic.
    ``penalty`` (> 0) fills greedily instead, lowering a turn's score by
    ``penalty`` x (best possible RRF term) for every turn already taken from its
    topic -- a soft topic-coverage rule.
    """
    turn_s = sc("turn", turn_key, retriever)
    topic_of = h.unit_of_turn(topics)
    topic_s = sc(topics, topic_key, retriever)
    fused = 1.0 / (RRF_K + 1 + ranks(turn_s)) + w / (RRF_K + 1 + ranks(topic_s))[topic_of]
    if penalty > 0:
        unit = penalty / (RRF_K + 1)
        taken = np.zeros(int(topic_of.max()) + 1)
        free = np.ones(h.n, bool)
        used, chosen = 0, []
        while True:
            fits = free & (h.cost <= budget - used)
            if not fits.any():
                return chosen
            i = int(np.argmax(np.where(fits, fused - unit * taken[topic_of], -np.inf)))
            chosen.append(i)
            used += h.cost[i]
            free[i] = False
            taken[topic_of[i]] += 1
    order = list(np.argsort(-fused, kind="stable"))
    if cap:
        buckets = defaultdict(list)
        for i in order:
            buckets[topic_of[i]].append(i)
        topic_order = list(dict.fromkeys(topic_of[i] for i in order))
        order, rnd = [], 0
        while len(order) < h.n:
            for t in topic_order:
                order.extend(buckets[t][rnd * cap:(rnd + 1) * cap])
            rnd += 1
    if partner:
        ex_of = h.unit_of_turn("exchange")
        ex_members = h.units["exchange"]
        expanded, seen = [], set()
        for i in order:
            for j in [i] + [j for j in ex_members[ex_of[i]] if j != i and topic_of[j] == topic_of[i]]:
                if j not in seen:
                    seen.add(j)
                    expanded.append(j)
        order = expanded
    return fill_turns(order, h.cost, budget)


def method_topic_first(h, sc, budget, k=3):
    """Current ΔContext store: top-k topics by MiniLM centroid, then turns by MiniLM."""
    topic_of = h.unit_of_turn("segment")
    top = np.argsort(-sc("segment", "raw", "minilm"), kind="stable")[:k]
    sims = sc("turn", "raw", "minilm")
    inside = np.isin(topic_of, top)
    order = list(np.flatnonzero(inside)[np.argsort(-sims[inside], kind="stable")]) + \
        list(np.flatnonzero(~inside)[np.argsort(-sims[~inside], kind="stable")])
    return fill_turns(order, h.cost, budget)


def methods():
    m = {}
    for kind in ("turn", "exchange", "session", "segsess", "segment"):
        for key in ("raw", "comp"):
            for r in ("bm25", "mpnet", "hybrid", "minilm", "hybrid_minilm"):
                if r == "minilm" and key == "comp":
                    continue  # MiniLM unit keys are centroids of the raw turn vectors
                m[f"SeCom-style | {kind:8s} | {key:4s} | {r}"] = (method_secom, dict(kind=kind, key=key, retriever=r))
    m["ΔContext topic-first (MiniLM, current)"] = (method_topic_first, {})
    for topics in ("segment", "segsess"):
        for topic_key in ("raw", "comp"):
            for r in ("hybrid", "minilm", "hybrid_minilm"):
                for w in (0.25, 0.5, 1.0):
                    for partner in (False, True):
                        m[f"ours | {topics} | topic {topic_key} | {r} | w={w} | partner={partner}"] = (
                            method_ours, dict(topics=topics, topic_key=topic_key, retriever=r, w=w,
                                              partner=partner))
    for boost in (0.25, 0.5, 1.0, 2.0):
        for k in (1, 2, 3, 5):
            m[f"ours+user | boost={boost} | k={k}"] = (method_ours_user, dict(boost=boost, k_topics=k))
    for reserve in (0.15, 0.25, 0.4):
        m[f"ours+user | reserve={reserve}"] = (method_ours_user, dict(reserve=reserve))
        for boost in (0.5, 1.0):
            m[f"ours+user | boost={boost} | k=3 | reserve={reserve}"] = (
                method_ours_user, dict(boost=boost, k_topics=3, reserve=reserve))
            for topics in ("session", "fixed10"):
                m[f"control: {topics} grouping | boost={boost} | k=3 | reserve={reserve}"] = (
                    method_ours_user, dict(boost=boost, k_topics=3, reserve=reserve, topics=topics))
    for boost in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        m[f"control: global user prior | boost={boost}"] = (method_user_prior, dict(boost=boost))
    m["control: user turns first"] = (method_user_prior, dict(only=True))
    for alpha in (0.25, 0.5, 0.75, 1.0):
        m[f"control: cost-aware fill | alpha={alpha}"] = (method_density, dict(alpha=alpha))
    for w in (0.25, 0.5):
        for penalty in (0.05, 0.1, 0.2, 0.4):
            m[f"ours+coverage | segment | topic raw | hybrid_minilm | w={w} | penalty={penalty}"] = (
                method_ours, dict(topics="segment", topic_key="raw", retriever="hybrid_minilm", w=w,
                                  penalty=penalty))
    return m


# -- main ----------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--budgets", default="2048,4096")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="Regex over method names.")
    parser.add_argument("--compression", type=Path, default=Path("data/cache/longmemeval_s_llmlingua2_r075.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--diagnose", default=None,
                        help="Exact method name: report where its missed evidence lies (first budget).")
    parser.add_argument("--export", type=Path, default=None,
                        help="Write chosen turn ids per method and question (for the QA harness).")
    args = parser.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]

    config = DatasetConfig(**yaml.safe_load(Path("configs/dataset/longmemeval_s.yaml").read_text()))
    dev, test = dev_test_split(list(LongMemEvalDataset(config)))
    samples = dev if args.split == "dev" else test
    samples = [s for s in samples if not s.qa_items[0].meta["adversarial"] and s.qa_items[0].evidence]
    if args.limit:
        samples = samples[: args.limit]

    minilm = SentenceTransformerEmbedder(device="cpu", cache_path="data/cache/longmemeval_s_minilm.npz")
    import contextbench.policies.delta_context as dc
    dc.SentenceTransformerEmbedder = lambda **_: minilm
    params = json.loads(Path("paper/results/params_v3_lme.json").read_text(encoding="utf-8"))
    policy = delta_policy_from_params(params, _WordCounter())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    formatter = TranscriptFormatter(include_dates=True)
    comp = load_compression(args.compression)
    print(f"{len(samples)} answerable {args.split} questions; {len(comp)} compressed turns cached", flush=True)

    histories = [History(s, policy, tokenizer, formatter, comp) for s in samples]
    missing = sum(h.comp_missing for h in histories)
    if missing:
        print(f"WARNING: {missing} turns have no compressed text (raw text used)", flush=True)
    mpnet = DenseCache(MPNET, Path("data/cache/longmemeval_s_mpnet_dot.npz"), normalize=False)
    pending = [h.qa.question for h in histories]
    for h in histories:
        for kind in h.units:
            for key in ("raw", "comp"):
                pending.extend(h.unit_text(kind, key))
    print(f"MPNet: {len(set(pending))} distinct texts needed", flush=True)
    mpnet.ensure(pending)
    mpnet.save()

    table = methods()
    if args.only:
        table = {k: v for k, v in table.items() if re.search(args.only, k)}
    acc = defaultdict(lambda: defaultdict(list))
    export = defaultdict(dict)
    diag = defaultdict(lambda: defaultdict(int))
    for n, h in enumerate(histories):
        E = minilm.embed_batch(h.minilm_text).astype(np.float64)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        q = minilm.embed(h.qa.question).astype(np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        sc = Scorer(h, mpnet, {"E": E, "q": q}, h.qa.question)
        evidence_sessions = {h.session_of[i] for i in h.evidence}
        for name, (fn, kw) in table.items():
            for b in budgets:
                chosen = fn(h, sc, b, **kw)
                hit = h.evidence & set(chosen)
                acc[(name, b)]["recall"].append(len(hit) / len(h.evidence))
                acc[(name, b)]["all"].append(float(len(hit) == len(h.evidence)))
                got_sessions = {h.session_of[i] for i in hit}
                acc[(name, b)]["sess"].append(len(got_sessions) / len(evidence_sessions))
                acc[(name, b)][h.qtype].append(len(hit) / len(h.evidence))
                if args.export:
                    export[slug(f"{name} @ {b}")][selection_key(h.qa.question, h.n)] = sorted(int(i) for i in chosen)
                if args.diagnose and name == args.diagnose and b == budgets[0]:
                    topic_of = h.unit_of_turn("segment")
                    touched = {int(topic_of[i]) for i in chosen}
                    missed = h.evidence - set(chosen)
                    diag[h.qtype]["evidence"] += len(h.evidence)
                    diag[h.qtype]["missed"] += len(missed)
                    diag[h.qtype]["missed_untouched"] += sum(int(topic_of[i]) not in touched for i in missed)
                    diag[h.qtype]["evidence_topics"] += len({int(topic_of[i]) for i in h.evidence})
                    diag[h.qtype]["questions"] += 1
                    diag[h.qtype]["topics_touched"] += len(touched)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(histories)}", flush=True)

    qtypes = ["single-session-user", "single-session-assistant", "single-session-preference",
              "multi-session", "knowledge-update", "temporal-reasoning"]
    short = {"single-session-user": "ss-user", "single-session-assistant": "ss-asst",
             "single-session-preference": "ss-pref", "multi-session": "multi", "knowledge-update": "k-upd",
             "temporal-reasoning": "temp"}
    report = {}
    for b in budgets:
        print(f"\n=== budget {b} ({len(histories)} questions): evidence recall / all-evidence / session recall")
        rows = sorted(((name, m) for (name, bb), m in acc.items() if bb == b), key=lambda r: -np.mean(r[1]["recall"]))
        print(f"{'method':72s} {'recall':>6s} {'all':>6s} {'sess':>6s} " + " ".join(f"{short[t]:>7s}" for t in qtypes))
        for name, m in rows:
            per = " ".join(f"{np.mean(m[t]) if m[t] else float('nan'):7.3f}" for t in qtypes)
            print(f"{name:72s} {np.mean(m['recall']):6.3f} {np.mean(m['all']):6.3f} {np.mean(m['sess']):6.3f} {per}")
            report[f"{name} @ {b}"] = {"recall": float(np.mean(m["recall"])), "all": float(np.mean(m["all"])),
                                       "sess": float(np.mean(m["sess"])),
                                       "per_question": [float(x) for x in m["recall"]],
                                       **{t: float(np.mean(m[t])) for t in qtypes if m[t]}}
    if diag:
        print(f"\ndiagnosis of {args.diagnose} @ {budgets[0]}: where missed evidence lies")
        print(f"{'type':26s} {'questions':>9s} {'ev/q':>5s} {'ev topics/q':>11s} {'touched/q':>9s} "
              f"{'missed':>7s} {'in untouched topics':>20s}")
        for t, d in sorted(diag.items()):
            q = d["questions"]
            print(f"{t:26s} {q:9d} {d['evidence'] / q:5.2f} {d['evidence_topics'] / q:11.2f} "
                  f"{d['topics_touched'] / q:9.2f} {d['missed']:7d} {d['missed_untouched']:20d}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"split": args.split, "questions": len(histories), "budgets": budgets,
                                        "results": report}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.export:
        args.export.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.export}")


if __name__ == "__main__":
    main()
