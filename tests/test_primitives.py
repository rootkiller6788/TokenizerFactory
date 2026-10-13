"""基础层：schema / rng / 归一化 / 语料与预分词。

这一层最值得守的两条不变量：

1. **预分词是单射且可逆的**（只多出一个前导空格）。它一旦被破坏，
   ``decode(encode(t))`` 就还原不了原文，而分词器连基本功能都不完整。
2. **三条预分词规则的词首标记必须一致**。不一致的话，规则之间的差异里
   会混进「标记策略不同」这一项，整个对比的结论就脏了。
"""

from __future__ import annotations

import pytest

from tokenizerfactory.config import deep_merge, load_config, resolve_path, validate
from tokenizerfactory.corpus import (
    PRE_TOKENIZERS,
    SPACE_MARKER,
    alphabet_of,
    char_class,
    corpus_stats,
    detokenize_words,
    pre_tokenize,
    word_counts,
)
from tokenizerfactory.normalize import FORMS, normalize, normalize_words
from tokenizerfactory.rng import derive_seed, make_rng
from tokenizerfactory.schema import (
    CorpusStats,
    Tokenization,
    TokenizerMetrics,
    TokenizerSpec,
    sha256_text,
    stable_id,
)

SAMPLES = [
    "the quick brown fox",
    "中华人民共和国成立于一九四九年",
    "if (a != b) { return a -> b; }",
    "版本 1.2.3 与 3.14 以及 2026-09-29",
    "混合 mixed 文本 text 与 数字 123",
    "emoji 😀 与 箭头 → 以及 ÆØÅ",
    "  前后   有    多余   空白  ",
    "",
]


# ----------------------------------------------------------------------
# rng / schema
# ----------------------------------------------------------------------
def test_derive_seed_is_stable_and_sensitive() -> None:
    assert derive_seed("a", "b") == derive_seed("a", "b")
    assert derive_seed("a", "b") != derive_seed("b", "a")
    # 分隔符不能省：("ab","") 与 ("a","b") 必须不同
    assert derive_seed("ab", "") != derive_seed("a", "b")


def test_make_rng_is_reproducible_and_independent() -> None:
    a = make_rng("x", 1)
    b = make_rng("x", 1)
    c = make_rng("x", 2)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    assert make_rng("x", 1).random() != c.random()


def test_stable_id_does_not_use_builtin_hash() -> None:
    # 内置 hash() 按 PYTHONHASHSEED 加盐，跨进程不稳定 —— 这里断言的是
    # 「同样的输入给出同样的输出」这条可复现性前提
    assert stable_id("a", 1) == stable_id("a", 1)
    assert len(stable_id("a", 1)) == 12
    assert stable_id("a", "b") != stable_id("a", 1)


def test_sha256_text_matches_known_vector() -> None:
    assert sha256_text("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_spec_id_is_stable_and_encodes_the_axes() -> None:
    spec = TokenizerSpec(algorithm="bpe", vocab_size=1024)
    assert spec.spec_id == "bpe-v1024"
    assert TokenizerSpec("bpe", 1024, lowercase=True).spec_id == "bpe-v1024+lower"
    assert (
        TokenizerSpec("bpe", 1024, pre_tokenizer="char").spec_id == "bpe-v1024+char"
    )
    assert (
        TokenizerSpec("bpe", 1024, pre_tokenizer="char", lowercase=True).spec_id
        == "bpe-v1024+lower+char"
    )
    # digest 必须对每个轴敏感：extra 不同 → 摘要不同
    a = TokenizerSpec("unigram", 512, extra={"prune_ratio": 0.4})
    b = TokenizerSpec("unigram", 512, extra={"prune_ratio": 0.5})
    assert a.digest != b.digest


def test_metrics_get_rejects_unknown_metric() -> None:
    metrics = TokenizerMetrics(
        tokens_per_char=1.0,
        tokens_per_word=1.0,
        vocab_coverage=1.0,
        unknown_rate=0.0,
        hard_unknown_rate=0.0,
        encode_tokens_per_sec=1.0,
        decode_tokens_per_sec=1.0,
        n_tokens=1,
        n_chars=1,
        n_words=1,
    )
    assert metrics.get("tokens_per_char") == 1.0
    with pytest.raises(KeyError):
        metrics.get("nope")


def test_corpus_stats_dominant_script() -> None:
    stats = CorpusStats(
        name="x",
        n_chars=10,
        n_words=2,
        n_unique_words=2,
        script_shares={"latin": 0.7, "cjk": 0.3},
    )
    assert stats.dominant_script() == "latin"
    assert CorpusStats(name="x", n_chars=0, n_words=0, n_unique_words=0).dominant_script() == "other"


def test_tokenization_len_matches_tokens() -> None:
    tok = Tokenization(tokens=["a", "b"], ids=[0, 1], n_unknown=0)
    assert len(tok) == 2


# ----------------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------------
@pytest.mark.parametrize("form", FORMS)
def test_normalize_accepts_all_documented_forms(form: str) -> None:
    assert isinstance(normalize("abc", form=form), str)


def test_normalize_rejects_unknown_form() -> None:
    with pytest.raises(ValueError):
        normalize("abc", form="NFX")


def test_normalize_nfc_composes_combining_marks() -> None:
    decomposed = "e\u0301"  # e + 组合重音
    assert normalize(decomposed, "NFC") == "\u00e9"
    assert normalize(decomposed, "none") == decomposed


def test_normalize_collapses_whitespace_and_strips() -> None:
    assert normalize("  a \n\n  b \t c  ") == "a b c"


def test_normalize_words_lowercases_only_when_asked() -> None:
    words = ["The", "ABC"]
    assert normalize_words(words, False) == ["The", "ABC"]
    assert normalize_words(words, True) == ["the", "abc"]
    # 不能原地改（调用方常常要复用同一个列表）
    assert words == ["The", "ABC"]


# ----------------------------------------------------------------------
# 预分词
# ----------------------------------------------------------------------
@pytest.mark.parametrize("rule", PRE_TOKENIZERS)
@pytest.mark.parametrize("text", SAMPLES)
def test_pre_tokenization_is_reversible_up_to_leading_marker(rule: str, text: str) -> None:
    """核心不变量：``detokenize(pre_tokenize(t)) == " " + t``（t 已归一化）。

    多出的那个前导空格就是词首标记本身。三条规则都必须满足 ——
    否则「规则之间的差异」里就混进了「标记策略不同」。
    """
    normalized = normalize(text)
    words = pre_tokenize(normalized, rule)
    if not normalized:
        assert words == [] or set(words) <= {SPACE_MARKER}
        return
    assert detokenize_words(words) == " " + normalized


@pytest.mark.parametrize("rule", ["unicode", "whitespace"])
def test_word_boundaries_are_marked_consistently(rule: str) -> None:
    """``▁`` 出现在**每一个词边界**上（含句首）。

    ``unicode`` 与 ``whitespace`` 把标记**贴在**词首，``char`` 把空白变成
    独立的 ``▁`` —— 三种形式不同，但「标记出现在哪里」这件事必须一致，
    否则规则之间的差异里会混进「标记策略不同」这一项。

    这里断言的是统一的那个口径：**``▁`` 的个数 = 1 + 空白段数**。
    """
    text = normalize("alpha beta  gamma 中文 测试")
    words = pre_tokenize(text, rule)
    markers = sum(w.count(SPACE_MARKER) for w in words)
    expected = 1 + sum(1 for i, ch in enumerate(text) if ch == " " and (i == 0 or text[i - 1] != " "))
    assert markers == expected, (rule, words, markers, expected)


def test_char_rule_puts_the_marker_on_whitespace_positions() -> None:
    """``char`` 规则下标记不贴在词首，而是**自己成为一个词**。

    它仍然是「一个词边界一个标记」，只是形态不同 —— 见上一条的统一口径。
    """
    words = pre_tokenize(normalize("ab cd"), "char")
    assert words == [SPACE_MARKER, "a", "b", SPACE_MARKER, "c", "d"]
    assert sum(w.count(SPACE_MARKER) for w in words) == 2  # 句首 1 个 + 空格 1 个


@pytest.mark.parametrize("rule", PRE_TOKENIZERS)
@pytest.mark.parametrize("text", ["one two three", "one  two   three", "中文 测试 文本"])
def test_marker_count_matches_word_boundaries(rule: str, text: str) -> None:
    """统一的标记口径：**标记数 == 1（句首）+ 空白段数**。

    三条规则的标记**形态**不同（贴词首 vs 独立成词），但**数量**必须一致。
    """
    normalized = normalize(text)
    words = pre_tokenize(normalized, rule)
    markers = sum(w.count(SPACE_MARKER) for w in words)
    runs = sum(
        1
        for i, ch in enumerate(normalized)
        if ch == " " and (i == 0 or normalized[i - 1] != " ")
    )
    assert markers == 1 + runs, (rule, text, words)


def test_pre_tokenize_rejects_unknown_rule() -> None:
    with pytest.raises(ValueError):
        pre_tokenize("abc", "nope")


def test_unicode_rule_groups_cjk_runs() -> None:
    """CJK 必须**成串**切，不能按字切。

    按字切的话「中华」这类词永远学不出来（BPE 只在单个词内部合并），
    本项目 README 里那个旗舰演示就做不出来。这条断言是防止它被改回去。
    """
    words = pre_tokenize(normalize("中华人民共和国"), "unicode")
    assert words == [SPACE_MARKER + "中华人民共和国"]


def test_unicode_rule_splits_scripts_and_punctuation() -> None:
    words = pre_tokenize(normalize("我爱Python 3.14，真的"), "unicode")
    assert words == [
        SPACE_MARKER + "我爱",
        "Python",
        SPACE_MARKER + "3",
        ".",
        "14",
        "，",
        "真的",
    ]


def test_whitespace_rule_splits_less_than_unicode_on_chinese() -> None:
    """反面教材：``whitespace`` 在中文上明显切得更少。

    但这里必须说清楚**它的伤害有多大** —— 这件事在本项目里走过弯路。
    ``unicode`` 规则也把连续的汉字当成一整串（不是按字切），所以
    ``whitespace`` 与它的差别**只在标点和脚本切换处**，并不像
    「整句成一个词」听起来那么戏剧化。实测在真实语料上：
    压缩率只差 0.5%，而生育率差 87%。

    这条测试用带标点的句子，把两者的差别显示出来。
    """
    text = "机器学习模型的训练，需要语料。"
    ws = pre_tokenize(normalize(text), "whitespace")
    uni = pre_tokenize(normalize(text), "unicode")
    assert ws == [SPACE_MARKER + text], "按空白切时整句是一个词"
    assert len(uni) > len(ws), "unicode 会在标点处切开"
    assert "".join(uni) == "".join(ws), "两条规则必须还原同一段文本"


def test_unicode_and_whitespace_agree_on_pure_chinese_without_punctuation() -> None:
    """纯中文、无标点、无脚本切换时，``unicode`` 与 ``whitespace`` **完全一致**。

    这是本项目一个反直觉的实测结果，必须显式记下来：
    它解释了为什么在真实语料上两条规则的压缩率只差 0.5% ——
    因为在中文里 ``unicode`` 也把连续汉字当成一整串。
    不记下来的话，下一个人会以为 ``whitespace`` 的「反面教材」效果
    应该很戏剧化，然后花时间去查为什么不是。
    """
    text = "机器学习模型的训练需要语料"
    assert pre_tokenize(normalize(text), "unicode") == pre_tokenize(
        normalize(text), "whitespace"
    )


def test_char_rule_emits_one_word_per_codepoint() -> None:
    words = pre_tokenize(normalize("ab c"), "char")
    assert words == [SPACE_MARKER, "a", "b", SPACE_MARKER, "c"]


@pytest.mark.parametrize("rule", PRE_TOKENIZERS)
def test_all_rules_share_the_same_alphabet(rule: str) -> None:
    """三条规则的字符集必须一致（都含词首标记）。

    不一致会让「词表预算装不装得下基础词表」这个校验在不同规则下
    给出不同答案，而校验是在**规则并集**上做的 —— 那会静默放过某些规则。
    """
    text = normalize("the cat sat 中文 测试 3.14 (a != b) 😀")
    assert alphabet_of(pre_tokenize(text, rule)) == alphabet_of(
        pre_tokenize(text, "unicode")
    )


def test_char_class_covers_the_documented_buckets() -> None:
    assert char_class(SPACE_MARKER) == "marker"
    assert char_class("中") == "cjk"
    assert char_class("a") == "latin"
    assert char_class("7") == "digit"
    assert char_class(" ") == "space"
    assert char_class("😀") == "emoji"
    assert char_class("，") == "symbol"


def test_word_counts_and_alphabet_of() -> None:
    counts = word_counts(["a", "b", "a"])
    assert counts == {"a": 2, "b": 1}
    assert alphabet_of(["ab", "bc"]) == {"a", "b", "c"}


def test_corpus_stats_reports_word_level_numbers() -> None:
    stats = corpus_stats("t", "the cat the cat", "unicode")
    assert stats.n_words == 4
    assert stats.n_unique_words == 2
    assert stats.n_chars == len("the cat the cat")
    assert abs(sum(stats.script_shares.values()) - 1.0) < 1e-9


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
def test_deep_merge_is_recursive_and_non_mutating() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}, "deep_merge 不能改动入参"


def test_resolve_path_keeps_absolute_paths() -> None:
    assert resolve_path("/root", "a/b").as_posix().endswith("root/a/b")
    assert resolve_path("/root", "/abs/x").as_posix() == "/abs/x"


def test_load_config_attaches_root_and_path(tiny_config_path) -> None:
    cfg = load_config(tiny_config_path)
    assert cfg["_config_path"].endswith("tiny.json")
    assert cfg["_root"].endswith("tokenizerfactory") or cfg["_root"] == str(
        tiny_config_path.parent.parent
    )


def test_load_config_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.json")


def _tiny_cfg_copy(tiny_cfg: dict) -> dict:
    """浅拷贝一份配置用于「改坏了再校验」。

    **必须保留 ``_root`` / ``_config_path``**：``validate`` 会去读训练语料
    （它要算基础字符集，从而算出词表规模的下限）。把这两个键剥掉的话，
    校验会在读文件那一步先炸，测出来的就不是想测的那条规则。
    """
    return deep_merge(tiny_cfg, None)


def test_validate_rejects_unknown_algorithm(tiny_cfg: dict) -> None:
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["specs"]["algorithms"] = ["bpe", "nope"]
    with pytest.raises(ValueError, match="未知算法"):
        validate(cfg)


def test_validate_rejects_unknown_pre_tokenizer(tiny_cfg: dict) -> None:
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["specs"]["pre_tokenizers"] = ["nope"]
    with pytest.raises(ValueError, match="未知预分词规则"):
        validate(cfg)


def test_validate_rejects_duplicate_vocab_sizes(tiny_cfg: dict) -> None:
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["specs"]["vocab_sizes"] = [64, 64]
    with pytest.raises(ValueError, match="重复"):
        validate(cfg)


def test_validate_rejects_unknown_metric(tiny_cfg: dict) -> None:
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["pareto"]["objectives"] = ["vocab_size", "nope"]
    with pytest.raises(ValueError, match="未知指标"):
        validate(cfg)


def test_validate_rejects_vocab_budget_too_small_for_base_vocab(tiny_cfg: dict) -> None:
    """词表预算装不下基础词表时必须**在配置加载期**报错。

    这条校验是 WordPiece 的隐形约束的可执行形式：它的基础词表是
    2×|字符集|+1。不挡在这里，配置会在跑了几十秒之后才失败，
    而那时前面的 spec 已经白跑完了。
    """
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["specs"]["vocab_sizes"] = [8, 16]
    with pytest.raises(ValueError, match="基础词表"):
        validate(cfg)


def test_validate_rejects_bad_prune_ratio(tiny_cfg: dict) -> None:
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["specs"]["unigram"]["prune_ratio"] = 1.5
    with pytest.raises(ValueError, match="prune_ratio"):
        validate(cfg)


def test_validate_rejects_unknown_select_rule(tiny_cfg: dict) -> None:
    cfg = _tiny_cfg_copy(tiny_cfg)
    cfg["select"]["rule"] = "nope"
    with pytest.raises(ValueError, match="未知选型规则"):
        validate(cfg)
