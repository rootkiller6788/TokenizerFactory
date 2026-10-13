"""编排层：把语料、设计空间、训练、度量、对比、Diff、选型串成一条链。

流水线的形状是 **扇形展开再收拢**：

    corpus ─┬─ norm:unicode ─┬─ train:bpe-v512      ── bench:bpe-v512      ─┐
            │                ├─ train:bpe-v1024     ── bench:bpe-v1024     │
            │                ├─ train:unigram-v512  ── bench:unigram-v512  ├─ sel ─ art
            │                └─ …                                          │
            ├─ norm:whitespace ─ …                                        │
            └─ norm:char ─ …                                             ─┘

三个刻意的做法
--------------

1. **词频只算一次，三个算法共用。** 同一个 ``(预分词规则, 是否小写)`` 下，
   ``word_counts`` 的结果被缓存并传给三个算法。各算各的会让「同一份语料」
   这个前提在实现层面悄悄失效 —— 而它正是整个对比的前提。

2. **分词器对象留在内存里，不落盘。** Diff 需要**真实编码**任意文本，
   只存词表不够（BPE/WordPiece 还要合并表、Unigram 还要概率）。
   所以 Diff 在这里现算、结果存进 ``result.diffs``；同时每个 spec 的
   分词器都导出成 ``artifacts/{spec_id}.json``，那是**能重新载入使用**的产物
   （有往返测试守着），不是快照。

3. **收拢点必须是真的多输入节点。** ``sel`` 节点把所有前沿候选的 bench 节点
   都接进来 —— 于是「哪些 spec 进了前沿」在一张图里就能看见，
   不必翻报告。这是本项目 DAG 与前几个项目最不一样的地方。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from . import compare, diff
from .config import load_text_file, resolve_path
from .corpus import (
    alphabet_of,
    corpus_stats,
    pre_tokenize,
    word_counts,
)
from .lineage import (
    CORPUS_NODE_ID,
    LineageDAG,
    LineageNode,
    artifact_node_id,
    bench_node_id,
    norm_node_id,
    select_node_id,
    train_node_id,
)
from .metrics import measure, point_of
from .normalize import normalize, normalize_words
from .schema import (
    CODE_VERSION,
    TokenizerResult,
    TokenizerRun,
    TokenizerSpec,
    Version,
    sha256_text,
    utcnow,
)
from .tokenizers import train_tokenizer
from .tokenizers.base import vocab_digest

SELECT_TAG = "main"


# ----------------------------------------------------------------------
# 设计空间
# ----------------------------------------------------------------------
def expand_specs(cfg: dict[str, Any]) -> list[TokenizerSpec]:
    """把配置里的设计空间展开成具体的 spec 列表。

    展开顺序（算法 → 词表规模 → 预分词规则 → 是否小写）与配置文件里的
    嵌套顺序一致，于是「改了配置里的哪一项」和「矩阵里多了哪些行」
    能直接对上。
    """
    s = cfg["specs"]
    lowers = [False, True] if s["lowercase_variants"] else [False]
    unigram_extra = {
        key: s["unigram"][key]
        for key in ("seed_max_len", "seed_min_count", "prune_rounds", "prune_ratio", "em_iters")
        if key in s["unigram"]
    }
    specs: list[TokenizerSpec] = []
    for algorithm in s["algorithms"]:
        for size in s["vocab_sizes"]:
            for rule in s["pre_tokenizers"]:
                for lower in lowers:
                    specs.append(
                        TokenizerSpec(
                            algorithm=str(algorithm),
                            vocab_size=int(size),
                            pre_tokenizer=str(rule),
                            lowercase=bool(lower),
                            extra=dict(unigram_extra) if algorithm == "unigram" else {},
                        )
                    )
    return specs


# ----------------------------------------------------------------------
# 语料准备
# ----------------------------------------------------------------------
class CorpusBundle:
    """归一化后的语料 + 按（规则, 是否小写）缓存的词频。

    缓存是**正确性**的一部分，不只是性能：三个算法必须拿到同一份词频，
    否则「同一份语料下谁更合适」这个问题在实现层面就已经不成立了。
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.form = str(cfg["normalize"]["form"])
        self.train_text = normalize(load_text_file(cfg, "train"), form=self.form)
        self.bench_text = normalize(load_text_file(cfg, "benchmark"), form=self.form)
        self.hard_text = normalize(load_text_file(cfg, "hard"), form=self.form)
        self.diff_lines = self._load_diff_lines(cfg)
        self._cache: dict[tuple[str, bool], tuple[list[str], dict[str, int]]] = {}

    def _load_diff_lines(self, cfg: dict[str, Any]) -> list[str]:
        """Diff 用的文本行。缺省时退回 benchmark 的前几行。"""
        rel = cfg["corpus"].get("diff_words")
        if rel:
            path = resolve_path(cfg["_root"], rel)
            if path.exists():
                lines = [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                return [line for line in lines if line and not line.startswith("#")]
        return [line for line in self.bench_text.splitlines() if line.strip()][:8]

    def words_and_counts(self, rule: str, lowercase: bool) -> tuple[list[str], dict[str, int]]:
        key = (rule, lowercase)
        if key not in self._cache:
            words = normalize_words(pre_tokenize(self.train_text, rule), lowercase)
            self._cache[key] = (words, word_counts(words))
        return self._cache[key]

    def bench_words(self, rule: str, lowercase: bool) -> list[str]:
        return normalize_words(pre_tokenize(self.bench_text, rule), lowercase)

    def hard_words(self, rule: str, lowercase: bool) -> list[str]:
        return normalize_words(pre_tokenize(self.hard_text, rule), lowercase)

    def alphabet(self, rules: list[str]) -> set[str]:
        out: set[str] = set()
        for rule in rules:
            out |= alphabet_of(pre_tokenize(self.train_text, rule))
        return out

    @property
    def bench_chars(self) -> int:
        """归一化后 benchmark 原文的字符数 —— ``tokens_per_char`` 的分母。"""
        return len(self.bench_text)

    @property
    def hard_chars(self) -> int:
        return len(self.hard_text)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_pipeline(cfg: dict[str, Any], *, verbose: bool = False) -> tuple[TokenizerResult, dict[str, Any]]:
    """跑完整条流水线。返回 ``(result, 训练好的分词器表)``。"""
    log = (lambda msg: print(msg, flush=True)) if verbose else (lambda msg: None)

    bundle = CorpusBundle(cfg)
    rules = [str(r) for r in cfg["specs"]["pre_tokenizers"]]
    specs = expand_specs(cfg)

    dag = LineageDAG()
    dag.add_node(
        LineageNode(
            node_id=CORPUS_NODE_ID,
            kind="corpus",
            label=Path(cfg["corpus"]["train"]).name,
            params={
                "train": cfg["corpus"]["train"],
                "benchmark": cfg["corpus"]["benchmark"],
                "hard": cfg["corpus"]["hard"],
                "normalize_form": bundle.form,
            },
            stats={
                "chars": bundle.bench_chars,
                "train_chars": len(bundle.train_text),
                "diff_lines": len(bundle.diff_lines),
            },
        )
    )

    train_stats = corpus_stats("train", bundle.train_text, rules[0])
    result = TokenizerResult(corpus=train_stats, cfg=cfg)

    tokenizers: dict[str, Any] = {}
    runs: list[TokenizerRun] = []
    norm_nodes: dict[tuple[str, bool], str] = {}

    for index, spec in enumerate(specs, 1):
        key = (spec.pre_tokenizer, spec.lowercase)
        if key not in norm_nodes:
            norm_id = norm_node_id(spec.pre_tokenizer, spec.lowercase)
            words, counts = bundle.words_and_counts(*key)
            norm_nodes[key] = dag.add_node(
                LineageNode(
                    node_id=norm_id,
                    kind="normalize",
                    label=f"归一化 + 预分词({spec.pre_tokenizer})"
                    + ("+lower" if spec.lowercase else ""),
                    inputs=[CORPUS_NODE_ID],
                    params={
                        "form": bundle.form,
                        "pre_tokenizer": spec.pre_tokenizer,
                        "lowercase": spec.lowercase,
                    },
                    stats={
                        "alphabet": len(alphabet_of(words)),
                        "words": len(words),
                        "unique_words": len(counts),
                    },
                )
            )

        words, counts = bundle.words_and_counts(*key)

        started = time.perf_counter()
        tokenizer = train_tokenizer(spec, counts)
        train_seconds = time.perf_counter() - started

        if len(tokenizer.vocab) > spec.vocab_size:
            raise AssertionError(
                f"{spec.spec_id} 实际词表 {len(tokenizer.vocab)} 超过目标 "
                f"{spec.vocab_size} —— 「同等词表预算下谁压得更狠」这个对比的前提被破坏了"
            )

        metrics = measure(
            tokenizer,
            bundle.bench_words(*key),
            bundle.hard_words(*key),
            bench_chars=bundle.bench_chars,
            hard_chars=bundle.hard_chars,
            cfg=cfg,
            tag=spec.spec_id,
        )
        bench_sample = bundle.bench_words(*key)[: int(cfg["metrics"]["sample_words"])]
        run = TokenizerRun(
            spec=spec,
            vocab=list(tokenizer.vocab),
            vocab_digest=tokenizer.vocab_digest,
            metrics=metrics,
            train_seconds=train_seconds,
            sample=[(w, tokenizer.pieces_of(w)) for w in bench_sample],
            meta=dict(tokenizer.meta),
        )
        runs.append(run)
        tokenizers[spec.spec_id] = tokenizer

        train_id = dag.add_node(
            LineageNode(
                node_id=train_node_id(spec.spec_id),
                kind="train",
                label=f"训练 {spec.spec_id}",
                inputs=[norm_nodes[key]],
                params=spec.to_dict(),
                stats={
                    "vocab_actual": len(tokenizer.vocab),
                    "vocab_target": spec.vocab_size,
                    "seconds": round(train_seconds, 4),
                },
            )
        )
        dag.add_node(
            LineageNode(
                node_id=bench_node_id(spec.spec_id),
                kind="bench",
                label=f"评测 {spec.spec_id}",
                inputs=[train_id],
                params={"metrics": cfg["metrics"]},
                stats={
                    "tokens_per_char": round(metrics.tokens_per_char, 6),
                    "unknown_rate": round(metrics.unknown_rate, 6),
                    "hard_unknown_rate": round(metrics.hard_unknown_rate, 6),
                    "coverage": round(metrics.vocab_coverage, 6),
                },
            )
        )
        log(
            f"[{index}/{len(specs)}] {spec.spec_id:<24} "
            f"vocab={len(tokenizer.vocab):<5} "
            f"tok/char={metrics.tokens_per_char:.4f} "
            f"unk={metrics.unknown_rate:.4f} "
            f"({train_seconds:.2f}s)"
        )

    result.runs = runs

    # ---- 对比与前沿 --------------------------------------------------
    compare.assemble(result, cfg)

    # ---- Diff --------------------------------------------------------
    result.diffs = diff.compute_diffs(
        result,
        cfg,
        tokenizers,
        lines=bundle.diff_lines,
        form=bundle.form,
    )

    # ---- 收拢：sel → art --------------------------------------------
    frontier_ids = result.frontiers.get(
        compare.frontier_key(cfg["pareto"]["objectives"]), []
    )
    select_inputs = sorted({bench_node_id(sid) for sid in frontier_ids})
    if result.selected and bench_node_id(result.selected) not in select_inputs:
        select_inputs.append(bench_node_id(result.selected))
    select_inputs = [n for n in select_inputs if n in dag.nodes]
    if not select_inputs:
        select_inputs = sorted(dag.by_kind("bench"))

    selected_run = result.find(result.selected)
    select_id = dag.add_node(
        LineageNode(
            node_id=select_node_id(SELECT_TAG),
            kind="select",
            label=f"选型（{cfg['select']['rule']}）",
            inputs=select_inputs,
            params=dict(cfg["select"]),
            stats={
                "selected": result.selected,
                "candidates": len(select_inputs),
                "reason": result.counters.get("selected_reason", ""),
            },
        )
    )
    dag.add_node(
        LineageNode(
            node_id=artifact_node_id(SELECT_TAG),
            kind="artifact",
            label="Tokenizer Artifact",
            inputs=[select_id],
            params={"version_tag": cfg["version_tag"]},
            stats={
                "spec_id": result.selected,
                "vocab_size": len(selected_run.vocab) if selected_run else 0,
                "vocab_digest": selected_run.vocab_digest[:16] if selected_run else "",
            },
        )
    )
    result.dag = dag

    # ---- 版本 --------------------------------------------------------
    digest = (
        sha256_text(f"{CODE_VERSION}\n{selected_run.vocab_digest}")
        if selected_run
        else sha256_text(CODE_VERSION)
    )
    result.version = Version(
        version_id=f"{cfg['version_tag']}-{digest[:8]}",
        created_at=utcnow(),
        digest=digest,
        tag=str(cfg["version_tag"]),
        counts={
            "n_specs": len(runs),
            "n_algorithms": len(result.algorithms()),
            "n_vocab_sizes": len(result.vocab_sizes()),
            "selected_vocab_size": len(selected_run.vocab) if selected_run else 0,
            "total_train_seconds": round(sum(r.train_seconds for r in runs), 3),
        },
    )
    result.counters["n_tokens_train"] = sum(
        len(bundle.words_and_counts(*k)[0]) for k in norm_nodes
    )
    return result, tokenizers


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
def write_outputs(
    result: TokenizerResult,
    tokenizers: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Path]:
    """把产物写到 ``out_dir``，返回 ``文件名 → 路径``。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def dump_json(name: str, payload: Any) -> None:
        path = out / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written[name] = path

    # 每个 spec 一个**可载入**的分词器产物
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for spec_id, tokenizer in sorted(tokenizers.items()):
        path = artifacts_dir / f"{spec_id}.json"
        path.write_text(
            json.dumps(tokenizer.export_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    written["artifacts/"] = artifacts_dir

    dump_json("result.json", _result_payload(result))
    dump_json("diffs.json", result.diffs)
    dump_json("version.json", result.version.to_dict() if result.version else {})
    dump_json(
        "lineage.json",
        result.dag.to_dict() if result.dag is not None else {},
    )
    written["lineage.dot"] = _write_text(
        out, "lineage.dot", result.dag.to_dot() if result.dag is not None else ""
    )
    written["lineage_selected.dot"] = _write_text(
        out,
        "lineage_selected.dot",
        result.dag.to_dot(only_selected_ancestry=True) if result.dag is not None else "",
    )
    written["matrix.csv"] = _write_csv(out, "matrix.csv", result.matrix)
    written["frontier.csv"] = _write_csv(
        out, "frontier.csv", _frontier_csv_rows(result)
    )
    written["runs.jsonl"] = _write_text(
        out,
        "runs.jsonl",
        "\n".join(
            json.dumps(
                # ``to_dict()`` 刻意不含 ``sample``（它在 schema 里是运行时数据），
                # 但 ``explain`` 要展示切分样本，所以这里显式补上。
                # 少了它，``explain`` 会静默地只显示度量、不显示切分 —— 不报错。
                {**run.to_dict(), "sample": [[w, p] for w, p in run.sample]},
                ensure_ascii=False,
            )
            for run in result.runs
        )
        + "\n",
    )
    return written


def _write_text(out: Path, name: str, text: str) -> Path:
    path = out / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_csv(out: Path, name: str, rows: list[dict[str, Any]]) -> Path:
    path = out / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    # 合并所有行出现过的键，保证列不丢 —— 直接拿第一行的 keys 会在
    # 「第一行恰好没有某个可选字段」时静默丢列
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _frontier_csv_rows(result: TokenizerResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    points = {r.spec_id: point_of(r) for r in result.runs}
    for key, ids in sorted(result.frontiers.items()):
        for rank, sid in enumerate(
            sorted(ids, key=lambda i: (points[i]["vocab_size"], i))
        ):
            rows.append(
                {
                    "frontier": key,
                    "rank": rank,
                    "spec_id": sid,
                    "vocab_size": int(points[sid]["vocab_size"]),
                    "vocab_size_actual": int(points[sid]["vocab_size_actual"]),
                    **{
                        name: round(points[sid][name], 6)
                        for name in key.split("+")
                    },
                    "selected": sid == result.selected,
                }
            )
    return rows


def _result_payload(result: TokenizerResult) -> dict[str, Any]:
    """``result.json`` 的内容。

    ``runs`` 里**不含** ``sample`` —— 它是每个 run 前 40 个词的切分，
    12 个 spec 加起来会把文件撑到几百 KB，而它的用途（``explain`` 看切分）
    在 ``runs.jsonl`` 里已经有了。这里只留度量与设计变量。
    """
    return {
        "pipeline": "tokenizerfactory",
        "code_version": CODE_VERSION,
        "config": result.cfg,
        "corpus": result.corpus.to_dict(),
        "runs": [
            {k: v for k, v in run.to_dict().items() if k != "sample"}
            for run in result.runs
        ],
        "matrix": result.matrix,
        "frontiers": result.frontiers,
        "diffs": result.diffs,
        "selected": result.selected,
        "version": result.version.to_dict() if result.version else {},
        "counters": result.counters,
        "lineage": result.dag.summary() if result.dag is not None else {},
        "vocab_digests": {r.spec_id: r.vocab_digest[:16] for r in result.runs},
        "vocab_digest_full": {r.spec_id: r.vocab_digest for r in result.runs},
    }
