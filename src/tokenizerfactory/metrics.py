"""五维度量 —— README「评测维度」那一节的可执行版本。

| 维度 | 字段 | 方向 | 为什么单独量它 |
| --- | --- | --- | --- |
| Compression | ``tokens_per_char`` | 越小越好 | 直接决定「同样一段文本吃掉多少上下文」 |
| Fertility | ``tokens_per_word`` | 越小越好 | 跨语言比较的关键：同一句话，中文/英文各要几个 token |
| Coverage | ``vocab_coverage`` | 越大越好 | 词表能**完整**表示多少比例的 benchmark 词 |
| Speed | ``encode/decode_tokens_per_sec`` | 越大越好 | 词表大了编码会变慢，这是真实成本 |
| Unknown | ``unknown_rate`` / ``hard_unknown_rate`` | 越小越好 | 中文、emoji、代码符号最容易在这里暴露 |

四个必须说清楚的度量口径
------------------------

1. **分母是「归一化后文本的字符数」，不是「预分词结果的字符数」。**
   预分词会给第一个词加一个词首标记 ``▁``，于是 ``sum(len(w))`` 比文本长度
   多 1。若用后者当分母，等于把词首标记当成一个真实字符，
   ``tokens_per_char`` 会被系统性地压低一点点 —— 小到不会引起注意，
   但足以让「谁压得更狠」的结论在接近的两个 spec 之间翻面。

2. **Coverage 是「词级」而不是「字符级」。**
   字符级覆盖率恒等于 1（未知字符也会被字符级回退吃掉），毫无信息量。
   词级覆盖率问的是：benchmark 里有多少比例的**词**能被词表完整表示
   （一个 UNK 都不产生）。这才对应「词表够不够用」。

   **但有一条必须说明的口径限制**：词级覆盖率**依赖「词」是怎么定义的**。
   同一个中文句子，``unicode`` 规则下「词」是一个汉字串，``char`` 规则下
   每个汉字都是一个词。前者只要串里有一个字符不认识就整词不算覆盖，
   后者的判定粒度细得多 —— 于是 ``char`` 的覆盖率会**天然偏高**
   （实测 0.977 vs 0.916），这不是因为它的词表更好，而是因为它的分母更细。
   所以**跨预分词规则比较覆盖率是拿苹果比橘子**，只在同一规则内部比才有意义。
   报告里会写明这一点，而不是把这个数字当成一条结论。

3. **速度用多轮打乱顺序后的中位数，不用单轮均值。**
   CPU 降频会让靠后的轮次系统性偏低，固定顺序跑就会把「跑得晚」
   误读成「更慢」。打乱顺序 + 取中位数可以同时抵消降频和偶发抖动。
   这也是本项目引入 ``rng`` 的第二个理由。

   **但中位数只是把噪声压小，不能消掉它 —— 这一点有可执行的后果**：
   速度是**实测值**，所以**以速度为轴的 Pareto 前沿不是逐位可复现的**。
   两个编码速度本来就很接近的 spec，会在两次运行之间交换前沿成员。
   实测：同一份配置跑两次，``tokens_per_char+encode_tokens_per_sec``
   这条前沿的成员从 ``wordpiece-v128`` 变成了 ``wordpiece-v128+whitespace``，
   而其它三条前沿与全部词表摘要逐位相同。
   所以报告里这条前沿会被标注为「含实测轴」，它只能当参考，不能当硬结论 ——
   要下结论请用不含实测轴的前沿（``vocab_size+tokens_per_char``）。
   ``TIMING_METRICS`` 是这个约定的可执行形式。

4. **未知 token 用 ``encode_word`` 返回的计数，不从 tokens 里数 ``[UNK]``。**
   语料里可能真的出现过 ``[UNK]`` 这个字符串并被学进词表，
   数 tokens 会把「恰好长这样的已知 token」也算成未知。
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Sequence

from .rng import make_rng
from .schema import TokenizerMetrics, TokenizerRun
from .tokenizers.base import Tokenizer

#: 指标方向。Pareto 与排序全靠它，**不要在别处再写一遍方向**
METRIC_DIRECTIONS: dict[str, str] = {
    "vocab_size": "min",
    "vocab_size_actual": "min",
    "tokens_per_char": "min",
    "tokens_per_word": "min",
    "vocab_coverage": "max",
    "unknown_rate": "min",
    "hard_unknown_rate": "min",
    "encode_tokens_per_sec": "max",
    "decode_tokens_per_sec": "max",
    "train_seconds": "min",
}

#: 实测类指标（由计时得到）。它们**不是逐位可复现的** —— 见模块文档第 3 条。
#: 以它们为轴算出来的 Pareto 前沿会在两次运行之间小幅变动，
#: 报告里会被标注为「含实测轴」。
TIMING_METRICS: frozenset[str] = frozenset(
    {"encode_tokens_per_sec", "decode_tokens_per_sec", "train_seconds"}
)


def has_timing_axis(objectives: Sequence[str]) -> bool:
    """这组目标轴里是否含实测指标。含的话，它的前沿不可逐位复现。"""
    return any(o in TIMING_METRICS for o in objectives)


#: 中文标签（报告表头与 CLI 共用，避免两处各写一遍）
METRIC_LABELS: dict[str, str] = {    "vocab_size": "词表目标",
    "vocab_size_actual": "实际词表",
    "tokens_per_char": "压缩(tok/字符)",
    "tokens_per_word": "生育率(tok/词)",
    "vocab_coverage": "覆盖率",
    "unknown_rate": "未知率",
    "hard_unknown_rate": "硬语料未知率",
    "encode_tokens_per_sec": "编码吞吐",
    "decode_tokens_per_sec": "解码吞吐",
    "train_seconds": "训练秒",
}

#: 直接从 ``TokenizerMetrics`` 取的字段（其余由 spec 派生）
_METRIC_FIELDS: tuple[str, ...] = (
    "tokens_per_char",
    "tokens_per_word",
    "vocab_coverage",
    "unknown_rate",
    "hard_unknown_rate",
    "encode_tokens_per_sec",
    "decode_tokens_per_sec",
    "n_tokens",
    "n_chars",
    "n_words",
)


def point_of(run: TokenizerRun) -> dict[str, float]:
    """把一个 run 摊平成「指标名 → 数值」，供 Pareto / 排序使用。

    ``vocab_size``（目标）与 ``vocab_size_actual``（实际）都在里面：
    前者是设计变量，后者是结果。BPE 在 pair 用尽时会提前停下，
    于是 actual < target —— 那本身就是一个有信息量的结果，
    报告里两个数都要出现。
    """
    point = {name: float(getattr(run.metrics, name)) for name in _METRIC_FIELDS}
    point["vocab_size"] = float(run.spec.vocab_size)
    point["vocab_size_actual"] = float(len(run.vocab))
    point["train_seconds"] = float(run.train_seconds)
    return point


# ----------------------------------------------------------------------
# 速度基准
# ----------------------------------------------------------------------
def measure_speed(
    tokenizer: Tokenizer,
    words: Sequence[str],
    *,
    repeats: int,
    seed: int,
    tag: str,
) -> tuple[float, float]:
    """返回 ``(编码吞吐, 解码吞吐)``，单位 token/秒，取多轮中位数。"""
    if not words:
        return 0.0, 0.0
    probe = tokenizer.encode(words)
    ids = probe.ids
    n_tokens = len(probe.tokens)
    if n_tokens == 0:
        return 0.0, 0.0

    jobs = ["encode"] * max(1, repeats) + ["decode"] * max(1, repeats)
    make_rng(seed, "speed", tag).shuffle(jobs)

    encode_rates: list[float] = []
    decode_rates: list[float] = []
    for job in jobs:
        start = time.perf_counter()
        if job == "encode":
            produced = tokenizer.encode(words)
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                encode_rates.append(len(produced.tokens) / elapsed)
        else:
            tokenizer.decode(ids)
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                decode_rates.append(n_tokens / elapsed)

    return (
        statistics.median(encode_rates) if encode_rates else 0.0,
        statistics.median(decode_rates) if decode_rates else 0.0,
    )


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def measure(
    tokenizer: Tokenizer,
    bench_words: Sequence[str],
    hard_words: Sequence[str],
    *,
    bench_chars: int,
    hard_chars: int,
    cfg: dict[str, Any],
    tag: str,
) -> TokenizerMetrics:
    """在 benchmark 与 hard 两份语料上度量一个训练好的分词器。

    ``bench_words`` / ``hard_words`` 必须是**已经预分词、已经按 spec 决定是否
    小写化**的词序列；``bench_chars`` / ``hard_chars`` 是**归一化后原文**的
    字符数（见模块文档第 1 条，不能用词的字符数之和）。
    """
    m = cfg["metrics"]
    seed = int(cfg["seed"])

    bench = tokenizer.encode(bench_words)
    n_tokens = len(bench.tokens)
    n_words = len(bench_words)
    n_unknown = bench.n_unknown

    # 词级覆盖：能被完整编码（一个 UNK 都不产生）的词占比。
    # 这里多走了一遍 encode_word —— 词序列只有几千个词，代价可以忽略，
    # 换来的是「覆盖率的定义」与「压缩率的计算」各自独立、互不牵连。
    covered = sum(1 for w in bench_words if tokenizer.encode_word(w)[1] == 0)

    hard = tokenizer.encode(hard_words)
    hard_tokens = len(hard.tokens)

    sample_size = min(int(m["speed_sample_words"]), len(bench_words))
    speed_words = (
        make_rng(seed, "speed-sample", tag).sample(list(bench_words), sample_size)
        if sample_size
        else []
    )
    enc_rate, dec_rate = measure_speed(
        tokenizer, speed_words, repeats=int(m["speed_repeats"]), seed=seed, tag=tag
    )

    return TokenizerMetrics(
        tokens_per_char=n_tokens / bench_chars if bench_chars else 0.0,
        tokens_per_word=n_tokens / n_words if n_words else 0.0,
        vocab_coverage=covered / n_words if n_words else 0.0,
        unknown_rate=n_unknown / n_tokens if n_tokens else 0.0,
        hard_unknown_rate=hard.n_unknown / hard_tokens if hard_tokens else 0.0,
        encode_tokens_per_sec=enc_rate,
        decode_tokens_per_sec=dec_rate,
        n_tokens=n_tokens,
        n_chars=bench_chars,
        n_words=n_words,
    )


def metric_line(run: TokenizerRun) -> str:
    """一行摘要，用于日志与 CLI 的紧凑输出。"""
    mt = run.metrics
    return (
        f"{run.spec_id:<24} vocab={len(run.vocab):<5} "
        f"tok/char={mt.tokens_per_char:.4f} tok/word={mt.tokens_per_word:.3f} "
        f"cov={mt.vocab_coverage:.3f} unk={mt.unknown_rate:.4f} "
        f"hard_unk={mt.hard_unknown_rate:.4f} "
        f"enc={mt.encode_tokens_per_sec:,.0f}/s train={run.train_seconds:.2f}s"
    )
