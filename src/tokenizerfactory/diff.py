"""Tokenizer Diff —— 本项目最有信息量的一个产物。

对比矩阵告诉你「谁少几个 token」，Pareto 前沿告诉你「哪些配置不被支配」，
但两者都没有回答**为什么**。Diff 回答的就是为什么：

    中华人民共和国
      bpe       v512    ▁中华 | 人民 | 共和国          3
      unigram   v512    ▁中华 | 人民 | 共和国          3
      wordpiece v512    ▁中 | ##华 | ##人 | ##民 | ...  7
      bpe       v2048   ▁中华人民共和国                 1

同一个词、同一份语料，切法可以差 7 倍。差异直接决定三件事：

1. **序列长度与推理成本**。注意力开销随序列长度增长，7 个 token 与 1 个 token
   在长文本上是数量级的差别。
2. **建模粒度**。整词一个 token 意味着模型只在「词」这一层做预测，
   学不到「中华」与「人民」之间的组合关系；切成单字则相反。
3. **跨模型迁移对齐难度**。这一点最容易被忽略：把两个分词器的词表混用、
   或者换掉分词器继续训练时，**边界不一致**会造成同一个字串在两套 id 空间里
   落在完全不同的位置上。下面算的「边界一致度」就是这件事的量化。

边界一致度怎么算
----------------
对每一行文本，把每个 spec 的 token 结束位置（累积字符下标，**不含末位**）
收集成一个集合，然后算两个 spec 的 Jaccard 系数，再对所有行取平均。

用 ``plain_piece`` 算长度而不是 ``len(token)``：WordPiece 的续接 token 带
``##``，直接取长度会让它的边界整体右移 2，算出来的「不一致」全是假的。
未登录 token 按 1 个字符算 —— 它代表的那段信息已经丢了，按 ``[UNK]``
字面量的 5 个字符算同样是假的。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .compare import frontier_key
from .corpus import pre_tokenize
from .normalize import normalize, normalize_words
from .schema import UNK_TOKEN, TokenizerResult, TokenizerRun
from .tokenizers.base import Tokenizer


def tokenize_line(tokenizer: Tokenizer, line: str, *, form: str = "NFC") -> list[str]:
    """把一行文本切成 token（跨词的完整序列）。

    走的是与度量**完全相同**的路径（归一化 → 预分词 → 可选小写化 → 编码），
    否则 Diff 展示的切分与实际评测的切分可能不是一回事 —— 那种不一致
    不会有任何报错，只会让「用 Diff 解释差异」这句话变成空话。
    """
    words = normalize_words(
        pre_tokenize(normalize(line, form=form), tokenizer.spec.pre_tokenizer),
        tokenizer.spec.lowercase,
    )
    tokens: list[str] = []
    for word in words:
        tokens.extend(tokenizer.pieces_of(word))
    return tokens


def piece_length(tokenizer: Tokenizer, token: str) -> int:
    """token 在原文里占几个字符。"""
    if token == UNK_TOKEN:
        return 1
    return len(tokenizer.plain_piece(token))


def boundary_set(tokenizer: Tokenizer, tokens: Sequence[str]) -> set[int]:
    """token 结束位置的累积字符下标（不含末位，末位恒等于总长，没有信息）。"""
    out: set[int] = set()
    pos = 0
    for token in tokens[:-1]:
        pos += piece_length(tokenizer, token)
        out.add(pos)
    return out


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    """两个集合的 Jaccard 系数。两个都空时返回 1.0（空集视为完全一致）。"""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def diff_specs(result: TokenizerResult, cfg: dict[str, Any], max_specs: int = 8) -> list[str]:
    """挑出参与 Diff 的 spec。

    不是全部 spec 都放进来：Diff 的价值在于**并排看清差异**，二十七个 spec
    并排只会变成一张看不清的表。选法按**设计轴**来，于是表里能逐轴读出差异：

    1. 最终入选的那个（它是要交付的东西，必须在）；
    2. 主前沿上的全部（它们是「同样好」的候选，差异最值得看）；
    3. 每个**算法**各补一个，且与入选者**同预分词规则** —— 隔离「算法」这个轴；
    4. 每条**预分词规则**各补一个，且与入选者**同算法** —— 隔离「规则」这个轴。

    第 3 条的「同规则」不能省。第一版只写了「每个算法各补一个压缩最好的」，
    实测下来它给 unigram 与 wordpiece 补的都是 ``whitespace`` 变体 ——
    因为 ``whitespace`` 的 ``tokens_per_char`` 恰好略好一点
    （unigram 0.5827 vs 0.5847）。于是那张表**一次动了两个变量**，
    读者无法判断某一行变长是因为换了算法还是换了规则。
    同一条纪律在 ``report.worst_alternative_at_same_budget`` 里也写着。

    第 4 条同理：``char`` 规则（每个码点一个词）是生育率的**上界**，
    没有它，「粒度」这个轴在表里就只有一个端点，
    「``中华人民共和国`` 是 2 个 token 还是 8 个」这句话就没有参照。

    最后按 ``(vocab_size, algorithm)`` 排序 —— 让表从上到下读起来像
    「预算从小到大」，而不是像一串随机 id。
    """
    picked: list[str] = []
    if result.selected:
        picked.append(result.selected)
    for sid in result.frontiers.get(frontier_key(cfg["pareto"]["objectives"]), []):
        if sid not in picked:
            picked.append(sid)

    budget = int(cfg["select"]["vocab_budget"])
    selected = result.find(result.selected)
    if selected is None:
        return []

    def best_of(pool: list[TokenizerRun]) -> str | None:
        if not pool:
            return None
        return min(pool, key=lambda r: (r.metrics.tokens_per_char, r.spec_id)).spec_id

    def add(sid: str | None) -> None:
        if sid and sid not in picked:
            picked.append(sid)

    within_budget = [r for r in result.runs if r.spec.vocab_size <= budget]
    # 3) 算法轴：同规则
    for algorithm in result.algorithms():
        add(
            best_of(
                [
                    r
                    for r in within_budget
                    if r.algorithm == algorithm
                    and r.spec.pre_tokenizer == selected.spec.pre_tokenizer
                ]
            )
            or best_of([r for r in within_budget if r.algorithm == algorithm])
        )
    # 4) 规则轴：同算法
    for rule in sorted({r.spec.pre_tokenizer for r in result.runs}):
        add(
            best_of(
                [
                    r
                    for r in within_budget
                    if r.spec.pre_tokenizer == rule
                    and r.algorithm == selected.spec.algorithm
                ]
            )
            or best_of([r for r in within_budget if r.spec.pre_tokenizer == rule])
        )

    def order(r: TokenizerRun) -> tuple[int, str, str]:
        return (r.spec.vocab_size, r.algorithm, r.spec_id)

    runs = [r for r in (result.find(sid) for sid in picked) if r is not None]
    runs.sort(key=order)
    kept = runs[:max_specs] if max_specs else runs
    if result.selected and not any(r.spec_id == result.selected for r in kept):
        # 截断不能把入选者挤掉 —— 它是要交付的东西
        chosen = next(r for r in runs if r.spec_id == result.selected)
        kept = sorted(kept[:-1] + [chosen], key=order) if kept else [chosen]
    return [r.spec_id for r in kept]


def compute_diffs(
    result: TokenizerResult,
    cfg: dict[str, Any],
    tokenizers: dict[str, Tokenizer],
    *,
    lines: Sequence[str],
    form: str = "NFC",
) -> dict[str, Any]:
    """计算 Diff 的完整结果。

    ``tokenizers`` 是 ``spec_id → 训练好的分词器``。必须在流水线里现算 ——
    分词器对象不落盘，事后重建就要重训，而 Diff 恰恰需要真实编码。
    """
    spec_ids = diff_specs(result, cfg)
    runs: dict[str, TokenizerRun] = {
        sid: run for sid in spec_ids if (run := result.find(sid)) is not None
    }
    spec_ids = [sid for sid in spec_ids if sid in runs]

    missing = [sid for sid in spec_ids if sid not in tokenizers]
    if missing:
        raise ValueError(
            f"compute_diffs 缺少这些 spec 的分词器：{missing[:5]}"
            f"（共 {len(missing)} 个）。Diff 必须现算真实编码 —— 分词器对象不落盘，"
            "事后重建就要重训，所以调用方必须把流水线里训练好的对象一起传进来。"
            "这里刻意报错而不是静默跳过：静默跳过会让报告里的 Diff 少几个 spec，"
            "而读者不会知道少了什么。"
        )

    table: list[dict[str, Any]] = []
    boundaries: dict[str, list[set[int]]] = {sid: [] for sid in spec_ids}
    token_counts: dict[str, list[int]] = {sid: [] for sid in spec_ids}

    for line in lines:
        per_spec: dict[str, dict[str, Any]] = {}
        for sid in spec_ids:
            tokenizer = tokenizers[sid]
            tokens = tokenize_line(tokenizer, line, form=form)
            per_spec[sid] = {
                "tokens": tokens,
                "n_tokens": len(tokens),
                "rendered": " | ".join(tokens),
            }
            boundaries[sid].append(boundary_set(tokenizer, tokens))
            token_counts[sid].append(len(tokens))
        table.append({"line": line, "specs": per_spec})

    agreement: dict[str, float] = {}
    for i, a in enumerate(spec_ids):
        for b in spec_ids[i + 1 :]:
            per_line = [
                jaccard(ba, bb)
                for ba, bb in zip(boundaries[a], boundaries[b])
            ]
            agreement[f"{a}~{b}"] = sum(per_line) / len(per_line) if per_line else 1.0

    mean_agreement = {
        sid: (
            sum(
                value
                for key, value in agreement.items()
                if sid in key.split("~")
            )
            / max(1, sum(1 for key in agreement if sid in key.split("~")))
        )
        for sid in spec_ids
    }

    return {
        "lines": list(lines),
        "specs": spec_ids,
        "selected": result.selected,
        "table": table,
        "agreement": {k: round(v, 6) for k, v in sorted(agreement.items())},
        "mean_agreement": {k: round(v, 6) for k, v in sorted(mean_agreement.items())},
        "mean_tokens_per_line": {
            sid: round(sum(counts) / len(counts), 4) if counts else 0.0
            for sid, counts in token_counts.items()
        },
        "vocab_size": {sid: len(runs[sid].vocab) for sid in spec_ids},
        "algorithm": {sid: runs[sid].algorithm for sid in spec_ids},
    }
