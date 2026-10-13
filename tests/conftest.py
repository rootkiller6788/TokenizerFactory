"""测试夹具。

**走真实的 ``load_config`` 路径**，不构造内存里的假配置：
配置校验（尤其是「词表预算必须装得下基础词表」那条）是产品的一部分，
绕过它测出来的东西不能保证线上成立。

语料刻意做得很小，但**结构完整**：中英混排、有标点、有数字、
有重复词（否则 BPE 学不到东西）、也有只出现一次的词（否则 Unigram 的
剪枝没有东西可剪）。小语料的一个好处是：所有断言都能手算出来。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

#: 训练算法的测试语料。比 ``TINY_TRAIN`` 稍大一点：要让 BPE 学出几十个合并、
#: 让 Unigram 的剪枝有东西可剪、让 WordPiece 的分数有区分度。
#: 放在 ``conftest`` 而不是某个测试文件里，是因为「分词算法」与「度量/对比」
#: 两层测试都要用它 —— 各写一份迟早会漂移。
COUNT_TRAIN = """the cat sat on the mat. the cat sat on a log.
the dog sat on the mat. a dog and a cat sat on a log.
数据 与 模型 的 训练 需要 语料。语料 的 质量 决定 模型 的 效果。
tokenization token tokenizer tokenize tokens tokenizing.
if (a != b) { return a -> b; } 版本 1.2.3 与 3.14。
"""

TINY_TRAIN = """the cat sat on the mat. the cat sat on a log.
the dog sat on the mat. a dog and a cat sat on a log.
数据 与 模型 的 训练 需要 语料。语料 的 质量 决定 模型 的 效果。
a cat and a dog and a log and a mat.
tokenization token tokenizer tokenize tokens.
if (a != b) { return a -> b; } 版本 1.2.3 与 3.14。
"""

TINY_BENCH = """the cat sat on a mat. a dog sat on the log.
模型 的 训练 需要 高质量 的 语料。token tokenizer tokenization.
if (a == b) { return a -> b; } 版本 2.0.1 与 2.71。
"""

TINY_HARD = """罕见汉字：龘靐齉爩 与 與 與
Emoji：😀 🎉 🚀
符号：→ ← ≠ ≈ ∑ ∫ √ ∞
拉丁扩展：ÆØÅ ß Þ ð
希腊字母：α β γ δ
路径：C:\\Users\\test\\data.json
"""

TINY_DIFF = """the cat
tokenization
数据 与 模型
if (a != b)
"""

TINY_CONFIG = {
    "pipeline_id": "tokenizerfactory",
    "version_tag": "tiny",
    "seed": 12345,
    "corpus": {
        "train": "data/corpus/train.txt",
        "benchmark": "data/corpus/benchmark.txt",
        "hard": "data/corpus/hard.txt",
        "diff_words": "data/corpus/diff_words.txt",
    },
    "normalize": {"form": "NFC"},
    "specs": {
        "algorithms": ["bpe", "wordpiece", "unigram"],
        # 下限不是随便定的：这份小语料的字符集是 55 个字符，而 WordPiece 的
        # 基础词表是 2×55+1 = 111 —— 预算必须大于它，否则配置加载期就会报错。
        # 把这个约束写进夹具而不是绕开它，是因为它本身就是产品的一部分。
        "vocab_sizes": [128, 256],
        "pre_tokenizers": ["unicode", "whitespace"],
        "lowercase_variants": False,
        "unigram": {
            "seed_max_len": 4,
            "seed_min_count": 1,
            "prune_rounds": 4,
            "prune_ratio": 0.4,
            "em_iters": 4,
        },
    },
    "metrics": {"speed_repeats": 2, "speed_sample_words": 20, "sample_words": 5},
    "pareto": {
        "objectives": ["vocab_size", "tokens_per_char"],
        "extra_objectives": [["tokens_per_char", "encode_tokens_per_sec"]],
    },
    "select": {"rule": "max_compression_within_budget", "vocab_budget": 128},
}


# ----------------------------------------------------------------------
# 分词算法测试用的词频夹具
# ----------------------------------------------------------------------
def synthetic_counts() -> dict[str, int]:
    """够大的合成词频，让「合并预算能被用满」这件事真的发生。

    小语料只能学出几十个合并，于是 ``budget`` 那条路径（用满预算）根本走不到。
    这里用 6 个字母的全部 4 字组合：字符集小、pair 种类多、
    合并可以一直进行下去，于是预算能被用满。
    """
    words = ["".join(t) for t in itertools.product("abcdef", repeat=4)]
    return {f"\u2581{w}": (i % 11) + 1 for i, w in enumerate(words)}


@pytest.fixture(scope="session")
def train_text() -> str:
    return COUNT_TRAIN


@pytest.fixture(scope="session")
def counts() -> dict[str, int]:
    """真实小语料的词频。**只读** —— 用例不要就地改它。"""
    from tokenizerfactory.corpus import pre_tokenize, word_counts
    from tokenizerfactory.normalize import normalize, normalize_words

    words = normalize_words(pre_tokenize(normalize(COUNT_TRAIN), "unicode"), False)
    return word_counts(words)


@pytest.fixture(scope="session")
def alphabet(counts: dict[str, int]) -> set[str]:
    from tokenizerfactory.corpus import alphabet_of

    return alphabet_of(counts)


@pytest.fixture(scope="session")
def big_counts() -> dict[str, int]:
    return synthetic_counts()


def write_workspace(root: Path, config: dict | None = None) -> Path:
    """把配置与语料写进 ``root``，返回配置文件路径。"""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "data" / "corpus").mkdir(parents=True, exist_ok=True)
    (root / "data" / "corpus" / "train.txt").write_text(TINY_TRAIN, encoding="utf-8")
    (root / "data" / "corpus" / "benchmark.txt").write_text(TINY_BENCH, encoding="utf-8")
    (root / "data" / "corpus" / "hard.txt").write_text(TINY_HARD, encoding="utf-8")
    (root / "data" / "corpus" / "diff_words.txt").write_text(TINY_DIFF, encoding="utf-8")
    path = root / "configs" / "tiny.json"
    path.write_text(
        json.dumps(config or TINY_CONFIG, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("tokenizerfactory")


@pytest.fixture(scope="session")
def tiny_config_path(tiny_root: Path) -> Path:
    return write_workspace(tiny_root)


@pytest.fixture(scope="session")
def tiny_cfg(tiny_config_path: Path) -> dict:
    from tokenizerfactory.config import load_config

    return load_config(tiny_config_path)


@pytest.fixture(scope="session")
def tiny_result(tiny_cfg: dict):
    from tokenizerfactory.pipeline import run_pipeline

    result, tokenizers = run_pipeline(tiny_cfg, verbose=False)
    return result, tokenizers


@pytest.fixture(scope="session")
def tiny_out(tiny_root: Path, tiny_result):
    """产物目录，内容与 ``run`` 命令落盘的一致（含 ``report.md``）。

    ``report.md`` 由 ``cli._cmd_run`` 写，不在 ``write_outputs`` 里 ——
    夹具补上它，否则「落盘的东西」与「命令落盘的东西」会差一个文件，
    而测试恰好测不出这个差别。
    """
    from tokenizerfactory import report
    from tokenizerfactory.pipeline import write_outputs

    result, tokenizers = tiny_result
    out = tiny_root / "out"
    write_outputs(result, tokenizers, out)
    (out / "report.md").write_text(report.render_markdown(result), encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def tiny_diff_lines() -> list[str]:
    """Diff 用的行，与 ``data/corpus/diff_words.txt`` 同源（去掉注释与空行）。"""
    return [ln.strip() for ln in TINY_DIFF.splitlines() if ln.strip()]
