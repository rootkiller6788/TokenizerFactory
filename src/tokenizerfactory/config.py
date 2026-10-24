"""配置加载与默认值合并。

配置为纯 JSON（避免引入 PyYAML），约定放在 ``<root>/configs/*.json``，
因此 ``_root`` 取配置文件的上两级目录，配置里的相对路径都相对 ``_root`` 解析。

配置分成三块，刻意的：

- ``corpus``（指向 ``data/corpus/*.txt``）：**问题是什么** ——
  语料在哪、benchmark 与 hard 测试集在哪、Diff 用哪些词。
  换语料只改这一块，不动任何代码。
- ``normalize`` + ``specs``：**设计空间是什么** —— 归一化规则、
  要对比哪些算法、哪些词表规模、哪些预分词规则。改这里就是改「搜索空间」。
- ``metrics`` / ``pareto`` / ``select``：**怎么比、怎么选** ——
  速度测几轮、Pareto 用哪两个目标、最终按什么规则选出 Artifact。

把三块分开的好处很实际：回答「换个语料，结论还成立吗」时只需要替换 ``corpus``；
回答「换个词表预算，选出来的分词器会变吗」时只需要改 ``specs.vocab_sizes``。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .schema import TokenizerSpec

DEFAULTS: dict[str, Any] = {
    "pipeline_id": "tokenizerfactory",
    "version_tag": "v1",
    "seed": 20260929,
    # 语料 —— 「问题本身」
    "corpus": {
        "train": "data/corpus/train.txt",
        "benchmark": "data/corpus/benchmark.txt",
        "hard": "data/corpus/hard.txt",
        "diff_words": "data/corpus/diff_words.txt",
    },
    # 归一化：Unicode 归一化形式（管线级）。小写化是**每个 spec 自己的选择**，
    # 见 specs.lowercase_variants —— 因为它会改变词表内容，属于设计空间而不是预处理。
    "normalize": {"form": "NFC"},
    # 设计空间 —— 对比哪些配置
    "specs": {
        "algorithms": ["bpe", "wordpiece", "unigram"],
        # 词表规模的下限不是随便定的：WordPiece 的基础词表是 2×|字符集|+1，
        # 预算必须装得下它（见 validate 里的 minimum_vocab_size）。
        # 本语料 |字符集| ≈ 500 ⇒ WordPiece 基础词表 1001，所以最小档取 2048。
        "vocab_sizes": [2048, 4096, 8192],
        # 三条预分词规则都跑。这不是「多试一个参数」，而是本项目的一个主结论：
        # 规则的选择对结果的影响与算法、词表规模同级，有时还更大。
        "pre_tokenizers": ["unicode", "whitespace", "char"],
        # 是否额外再生成一批 lowercase=True 的 spec（用于演示
        # 「小写化省多少 token、代价是什么」）。默认关闭：它会让 spec 数翻倍
        # （27 → 54），而本项目已经有三个维度的交叉了。
        # 想单独做这个实验，把它打开即可，不需要改任何代码。
        "lowercase_variants": False,
        "unigram": {
            # 种子词表：语料中长度 <= seed_max_len 且出现次数 >= seed_min_count 的子串
            "seed_max_len": 6,
            "seed_min_count": 2,
            # 剪枝-重估的最大轮数。**目标是硬约束**：每轮剪掉 prune_ratio 比例，
            # 一旦达到目标就停下，所以实际轮数通常小于这个上限。
            # 它的作用是给 EM 留出足够的重估次数 —— 剪完不重估，
            # 留下来的 piece 的概率还是在「包含已删 piece」的格子上估的。
            "prune_rounds": 8,
            # 每轮剪掉的比例（相对当前词表）
            "prune_ratio": 0.4,
            # 每轮 EM 的最大迭代数（内部还有基于对数似然的提前停止）
            "em_iters": 8,
        },
    },
    # 度量
    "metrics": {
        # 速度基准重复几轮（取中位数）。单轮会被 CPU 降频与调度噪声带偏
        "speed_repeats": 5,
        # 速度基准用多少个词（太少则计时精度不够，太多则整体变慢）
        "speed_sample_words": 400,
        # 每个 run 存多少词的切分样本，供 explain / diff 使用
        "sample_words": 40,
    },
    "pareto": {
        # 主前沿的两个目标（方向见 metrics.METRIC_DIRECTIONS）
        "objectives": ["vocab_size", "tokens_per_char"],
        # 额外再算几条前沿，用于回答「换个取舍轴结论会不会变」：
        # - vocab_size_actual：当算法撞上饱和（实际词表 < 目标）时，
        #   目标预算不再等于真实成本，换这个轴看前沿会不一样；
        # - encode_tokens_per_sec：把速度拉进来，压缩最好 ≠ 综合最好。
        "extra_objectives": [
            ["vocab_size_actual", "tokens_per_char"],
            ["tokens_per_char", "encode_tokens_per_sec"],
        ],
    },
    "select": {
        # max_compression_within_budget | knee
        "rule": "max_compression_within_budget",
        # 词表预算：真实部署里 embedding 表大小是硬约束（参数表规模 / 显存），
        # 所以「在预算内压缩最好的那个」是最常用的选法。
        # 注意预算本身会改变赢家，见报告里的预算敏感性表。
        "vocab_budget": 2048,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """递归合并：override 覆盖 base，dict 逐层深入，其余整体替换。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(root: str | Path, rel: str | Path) -> Path:
    """把配置里的相对路径解析为绝对路径（相对 ``root``）。

    签名收的是 ``root`` 而不是整个 ``cfg`` —— 子配置里没有 ``_root``，
    调用方请显式传 ``cfg["_root"]``。
    """
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 JSON 配置并与 DEFAULTS 合并，附带 ``_config_path`` / ``_root``。"""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    cfg = deep_merge(DEFAULTS, raw)
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(p.parent.parent)
    validate(cfg)
    return cfg


def load_text_file(cfg: dict[str, Any], key: str) -> str:
    """读取 ``corpus`` 里某个键指向的语料文件。"""
    rel = cfg["corpus"][key]
    p = resolve_path(cfg["_root"], rel)
    if not p.exists():
        raise FileNotFoundError(
            f"语料文件不存在: {p}（配置项 corpus.{key} = {rel!r}）"
        )
    return p.read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 校验 —— 尽量挡在运行之前
# ----------------------------------------------------------------------
def _validate_corpus(cfg: dict[str, Any]) -> None:
    for key in ("train", "benchmark", "hard"):
        if not cfg["corpus"].get(key):
            raise ValueError(f"corpus.{key} 不能为空")
    # diff_words 允许缺省（缺省时用 benchmark 的前几个词）


def _validate_specs(cfg: dict[str, Any]) -> None:
    # 延迟导入：避免 config ← corpus ← config 的循环导入
    from .corpus import PRE_TOKENIZERS
    from .tokenizers import ALGORITHMS

    s = cfg["specs"]
    algos = [str(a) for a in s["algorithms"]]
    if not algos:
        raise ValueError("specs.algorithms 不能为空")
    if len(set(algos)) != len(algos):
        raise ValueError(f"specs.algorithms 有重复项：{algos}")
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"未知算法 {unknown}；可用：{', '.join(sorted(ALGORITHMS))}"
        )

    sizes = [int(v) for v in s["vocab_sizes"]]
    if not sizes:
        raise ValueError("specs.vocab_sizes 不能为空")
    if len(set(sizes)) != len(sizes):
        raise ValueError(f"specs.vocab_sizes 有重复项：{sizes}")
    if any(v < 1 for v in sizes):
        raise ValueError(f"specs.vocab_sizes 必须为正：{sizes}")

    rules = [str(r) for r in s["pre_tokenizers"]]
    if not rules:
        raise ValueError("specs.pre_tokenizers 不能为空")
    unknown = [r for r in rules if r not in PRE_TOKENIZERS]
    if unknown:
        raise ValueError(
            f"未知预分词规则 {unknown}；可用：{', '.join(sorted(PRE_TOKENIZERS))}"
        )

    uni = s["unigram"]
    if int(uni["seed_max_len"]) < 1:
        raise ValueError("specs.unigram.seed_max_len 至少为 1")
    if int(uni["seed_min_count"]) < 1:
        raise ValueError("specs.unigram.seed_min_count 至少为 1")
    if int(uni["prune_rounds"]) < 0:
        raise ValueError("specs.unigram.prune_rounds 不能为负")
    if not 0.0 < float(uni["prune_ratio"]) < 1.0:
        raise ValueError("specs.unigram.prune_ratio 必须在 (0, 1)")
    if int(uni.get("em_iters", 1)) < 1:
        raise ValueError("specs.unigram.em_iters 至少为 1")

    # 同一批 spec_id 必须唯一 —— DAG 节点 id、矩阵行名、前沿标签全用它。
    # 直接复用 TokenizerSpec.spec_id，避免这里手拼的后缀规则与它悄悄分家。
    lowers = [False, True] if s["lowercase_variants"] else [False]
    ids = [
        TokenizerSpec(
            algorithm=a,
            vocab_size=v,
            pre_tokenizer=r,
            lowercase=low,
        ).spec_id
        for a in algos
        for v in sizes
        for r in rules
        for low in lowers
    ]
    if len(set(ids)) != len(ids):
        raise ValueError(f"specs 展开后出现重复的 spec_id：{sorted(ids)}")


def _validate_metrics(cfg: dict[str, Any]) -> None:
    m = cfg["metrics"]
    if int(m["speed_repeats"]) < 1:
        raise ValueError("metrics.speed_repeats 至少为 1")
    if int(m["speed_sample_words"]) < 1:
        raise ValueError("metrics.speed_sample_words 至少为 1")
    if int(m["sample_words"]) < 0:
        raise ValueError("metrics.sample_words 不能为负")


def _validate_pareto(cfg: dict[str, Any]) -> None:
    from .metrics import METRIC_DIRECTIONS

    pairs = [cfg["pareto"]["objectives"]] + [
        list(p) for p in cfg["pareto"]["extra_objectives"]
    ]
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"目标对必须是两个指标，得到 {pair}")
        for name in pair:
            if name not in METRIC_DIRECTIONS:
                raise ValueError(
                    f"未知指标 {name!r}；可用：{', '.join(sorted(METRIC_DIRECTIONS))}"
                )


def _validate_select(cfg: dict[str, Any]) -> None:
    sel = cfg["select"]
    if sel["rule"] not in ("max_compression_within_budget", "knee"):
        raise ValueError(
            f"未知选型规则 {sel['rule']!r}；可用：max_compression_within_budget / knee"
        )
    if int(sel["vocab_budget"]) < 1:
        raise ValueError("select.vocab_budget 至少为 1")


def validate(cfg: dict[str, Any]) -> None:
    """把配置错误尽量挡在运行之前。"""
    _validate_corpus(cfg)
    _validate_metrics(cfg)
    _validate_pareto(cfg)
    _validate_select(cfg)
    # 先校验设计空间本身（算法名、规则名是否合法），下面的联合校验要用到它们
    _validate_specs(cfg)

    # 设计空间与语料需要联合校验：词表预算必须装得下**基础词表**，
    # 否则「训练一个 2048 词表的 WordPiece」这种配置会在跑了几十秒之后才失败，
    # 而那时前面几十个 spec 已经白跑完了。
    #
    # 基础词表的大小**因算法而异**，这是本项目一个真实的设计约束：
    # WordPiece 的非词首子词一律带 ## 前缀，所以它需要每个字符的两种形式，
    # 基础词表是别人的两倍。min 档必须对所有算法都够用。
    from .corpus import pre_tokenize
    from .normalize import normalize
    from .tokenizers import TOKENIZERS, minimum_vocab_size

    train = load_text_file(cfg, "train")
    normalized = normalize(train, form=cfg["normalize"]["form"])
    rules = [str(r) for r in cfg["specs"]["pre_tokenizers"]]
    alphabet: set[str] = set()
    for rule in rules:
        for w in pre_tokenize(normalized, rule):
            alphabet |= set(w)
    if not alphabet:
        raise ValueError("训练语料在归一化 + 预分词后为空，无法训练任何分词器")

    algorithms = [str(a) for a in cfg["specs"]["algorithms"]]
    floor = minimum_vocab_size(algorithms, alphabet)
    smallest = min(int(v) for v in cfg["specs"]["vocab_sizes"])
    if smallest <= floor:
        # 逐算法报出各自的占用，读者一眼能看出差在哪
        detail = "、".join(
            f"{a} {len(TOKENIZERS[a].base_vocab(alphabet))}" for a in algorithms
        )
        raise ValueError(
            f"最小词表规模 {smallest} 不大于基础词表 {floor}（字符集 {len(alphabet)} 个；"
            f"各算法占用：{detail}）。词表连语料里出现过的字符都装不下（或刚好只装得下），"
            "训练出来的不是 BPE/WordPiece/Unigram，而是「按字符切分」。"
            f"请把 specs.vocab_sizes 的最小值提到 > {floor}，"
            "或者缩小训练语料的字符集。"
        )

    # benchmark 语料不能为空 —— 度量全建立在它上面
    bench = load_text_file(cfg, "benchmark")
    if not pre_tokenize(normalize(bench, form=cfg["normalize"]["form"]), rules[0]):
        raise ValueError("benchmark 语料在预分词后为空，无法度量")
