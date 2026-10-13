"""语料层：加载、预分词、画像。

**预分词（pre-tokenization）是分词器质量差异的主要来源之一**，而它最容易被忽略。
本项目把它做成一个可对比的维度，因为三件事都取决于它：

1. **中文没有空格**。若按空白切词，一整句（到标点为止）会变成一个「词」，
   词表里于是存进整句而不是词组，词表预算被大量只出现一次的长串吃掉。

   **但它的伤害有多大，实测比直觉小得多** —— 这一点本项目走过弯路，
   所以记在这里。因为 ``unicode`` 规则也把连续的汉字当成**一整串**
   （不是按字切），``whitespace`` 与它的差别**只在标点和脚本切换处**。
   纯中文、无标点、无脚本切换时，两条规则的输出**完全一样**。
   在真实语料上实测的结果是：压缩率只差 0.5%（0.5401 vs 0.5376），
   而**生育率差 87%**（5.482 vs 2.927 tok/词）。

   也就是说：按空白预分词确实污染了词表，但有字符级回退的分词器
   在压缩率上几乎不受影响（编码总能退回字符）；真正被暴露出来的是
   按「词」算的生育率。**只看压缩率会得出「预分词规则无关紧要」的错误结论。**
   这是本项目量五个维度而不是一个维度的直接理由。

2. **代码与 emoji 的符号边界**。`->`、`::`、`!=` 该不该整体成词？
   符号切得太碎会让序列变长，切得太粗会让稀有组合进不了词表。
3. **词首标记**。本项目采用 SentencePiece 的 ``▁``（U+2581）约定：
   词首的空白被折进词里，于是 ``▁the`` 是**一个** token 而不是两个。
   这样 ``decode(encode(text))`` 对**归一化后**的文本是精确成立的 ——
   分词器必须能还原原文，否则它连基本功能都不完整。

三种预分词规则
--------------
| 规则 | 行为 | 用途 |
| --- | --- | --- |
| ``unicode`` | CJK **成串** / 拉丁词 / 数字串 / emoji / 其余每符号一个 | 默认，真实可用 |
| ``whitespace`` | 只按空白切 | **反面教材**：中文退化成整句成词 |
| ``char`` | 每个码点一个词 | 生育率上界，用于说明「预分词有多重要」 |

**CJK 必须成串切，不能按字切。** 这一点走过弯路：最初 ``unicode`` 规则把每个汉字
切成独立的词，结果是「中华」「人民」这类词**永远不可能被学出来** —— BPE 只在
单个词内部做合并，词已经被切成单字了，合并就无从谈起。于是 `[中华人民共和国]`
在词表开到 4096 时仍然是 7 个 token，本项目 README 里那个旗舰演示直接做不出来。

真实的工业分词器（cl100k、LLaMA 等）都是把连续字母（含汉字）作为**一整串**
交给 BPE/Unigram 去合并的。改成成串之后，同一个词在 v512 下切出
`[中华][人民][共和国]`、在 v256 下切出 `[中][华][人民][共和][国]`，
差异才真的看得见。

「按字切」并没有被丢掉，它就是 ``char`` 规则 —— 于是「按字预分词 vs 成串预分词」
这个对比依然在，只是被放到了它该在的位置（生育率上界），而不是冒充默认规则。

词首标记的统一约定
------------------
三种规则**一律给第一个词加词首标记** ``▁``。这不是审美问题：若第一个词不加、
后面的词加，那么「句首的 the」和「句中的 the」就是两个不同的词，
词表要为同一个词付两份预算 —— 规则之间也就不可比了（``unicode`` 加、
``whitespace`` 不加，两者在英文上的差异会混入标记策略的差异）。

代价是预分词**可逆但多一个前导空格**：

    detokenize_words(pre_tokenize(t)) == " " + t      # t 已归一化、已 strip

这正是 SentencePiece「把输入视作以空格开头」的约定，也是它能让
``▁the`` 在句首句中都成立的原因。有测试守着这条等式。
"""

from __future__ import annotations

import re
from typing import Iterable

from .schema import CorpusStats

#: 词首标记（SentencePiece 约定）。它本身也是一个普通字符，会进入基础字符集
SPACE_MARKER = "\u2581"

#: 支持的预分词规则
PRE_TOKENIZERS: tuple[str, ...] = ("unicode", "whitespace", "char")

# 字符类别（用于语料画像与「未知」维度的解释）
_CJK_RANGES = (
    "\u3400-\u4dbf"  # CJK 扩展 A
    "\u4e00-\u9fff"  # CJK 基本区
    "\uf900-\ufaff"  # CJK 兼容表意
)
_EMOJI_RANGES = (
    "\u2600-\u27bf"  # 杂项符号与装饰符号
    "\u2b00-\u2bff"  # 杂项符号与箭头
    "\u2190-\u21ff"  # 箭头
    "\ufe0f"  # 变体选择符
    "\U0001f000-\U0001faff"  # 麻将/扑克/表情/交通等
)
_LATIN_RANGES = "A-Za-z\u00c0-\u024f\u1e00-\u1eff"

_UNICODE_PATTERN = re.compile(
    f"([{_CJK_RANGES}]+)"                    # 1: 汉字成串，交给算法去合并
    f"|([{_LATIN_RANGES}]+(?:['\u2019][{_LATIN_RANGES}]+)*)"  # 2: 拉丁词（含撇号）
    f"|(\\d+)"                              # 3: 数字串
    f"|([{_EMOJI_RANGES}])"                 # 4: emoji，每个一个词
    f"|(\\s+)"                              # 5: 空白
    f"|(.)"                                 # 6: 其余每个符号一个词
)

_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cjk", re.compile(f"[{_CJK_RANGES}]")),
    ("emoji", re.compile(f"[{_EMOJI_RANGES}]")),
    ("latin", re.compile(f"[{_LATIN_RANGES}]")),
    ("digit", re.compile(r"[0-9]")),
    ("space", re.compile(r"\s")),
    ("symbol", re.compile(r"[^\w\s]", re.UNICODE)),
)


# ----------------------------------------------------------------------
# 预分词
# ----------------------------------------------------------------------
def pre_tokenize(text: str, rule: str = "unicode") -> list[str]:
    """把文本切成「词」。

    返回的每个词都带词首标记 ``▁``（第一个词也带）；空白本身不出现在结果里
    （它被折进了下一个词的词首）。因此下面这条等式成立，且有测试守着：

        detokenize_words(pre_tokenize(t)) == " " + t

    ``t`` 需已归一化并 strip（``normalize`` 会做）。多出的那个前导空格就是
    词首标记本身 —— 换成任何一条规则都成立，这是三条规则可比的前提。
    """
    if rule == "unicode":
        return _pre_tokenize_unicode(text)
    if rule == "whitespace":
        return _pre_tokenize_whitespace(text)
    if rule == "char":
        return _pre_tokenize_char(text)
    raise ValueError(
        f"未知预分词规则 {rule!r}；可用：{', '.join(PRE_TOKENIZERS)}"
    )


def _pre_tokenize_unicode(text: str) -> list[str]:
    words: list[str] = []
    pending_space = False
    for m in _UNICODE_PATTERN.finditer(text):
        cjk, latin, digit, emoji, ws, other = m.groups()
        if ws is not None:
            pending_space = True
            continue
        piece = cjk or latin or digit or emoji or other
        if piece is None:
            continue
        if pending_space or not words:
            # 首词也给标记：这样「文档以空白开头」与「不以空白开头」
            # 不会产生两套词表，语料一致性更好
            piece = SPACE_MARKER + piece
        pending_space = False
        words.append(piece)
    return words


def _pre_tokenize_whitespace(text: str) -> list[str]:
    """只按空白切。**中文会退化成整句成词** —— 这正是要展示的失败。

    注意 ``▁`` 是**无条件**加在每段前面的（包括第一段），与 ``unicode`` 规则
    保持一致；否则两条规则的差异里会混进「标记策略不同」这一项，结论就脏了。

    还要注意：它与 ``unicode`` 的差别**只在标点和脚本切换处**，
    因为 ``unicode`` 也把连续汉字当成一整串。伤害有多大见模块文档第 1 条 ——
    比「整句成词」听起来小得多。
    """
    return [SPACE_MARKER + chunk for chunk in text.split()]


def _pre_tokenize_char(text: str) -> list[str]:
    """每个码点一个词；空白变成独立的 ``▁``，开头再补一个 ``▁``。

    补开头那一个是为了让三条规则给出同一个可逆等式：
    ``"ab cd"`` → ``["▁","a","b","▁","c","d"]`` → ``"▁ab▁cd"`` → ``" ab cd"``。
    空白字符自己映射成 ``▁`` 之后恰好还原成一个空格，所以内部不会多出空格。
    """
    return [SPACE_MARKER] + [SPACE_MARKER if c.isspace() else c for c in text]


def detokenize_words(words: Iterable[str]) -> str:
    """预分词结果的逆操作。

    结果比输入**多一个前导空格**（词首标记的产物），见模块文档。
    用它来验证「预分词无损」时请带上这个 ``" " +``。
    """
    return "".join(words).replace(SPACE_MARKER, " ")


# ----------------------------------------------------------------------
# 字符分类与画像
# ----------------------------------------------------------------------
def char_class(ch: str) -> str:
    """单个字符属于哪一类（用于语料画像）。"""
    if ch == SPACE_MARKER:
        return "marker"
    for name, pattern in _CLASS_PATTERNS:
        if pattern.match(ch):
            return name
    return "other"


def corpus_stats(name: str, text: str, rule: str = "unicode") -> CorpusStats:
    """语料画像。

    ``n_words`` / ``n_unique_words`` 依赖预分词规则 —— 同一份文本在
    ``unicode`` 与 ``whitespace`` 下的「词数」相差一个数量级，
    这不是 bug，而是本项目想展示的东西，所以画像里把规则记进 ``name``。
    """
    words = pre_tokenize(text, rule)
    counts: dict[str, int] = {}
    for ch in text:
        cls = char_class(ch)
        counts[cls] = counts.get(cls, 0) + 1
    n_chars = len(text)
    shares = {k: v / n_chars for k, v in counts.items()} if n_chars else {}
    lengths = [len(w) for w in words] or [0]
    return CorpusStats(
        name=f"{name}（预分词：{rule}）",
        n_chars=n_chars,
        n_words=len(words),
        n_unique_words=len(set(words)),
        script_shares=shares,
        max_word_len=max(lengths),
        mean_word_len=sum(lengths) / len(lengths),
    )


def word_counts(words: Iterable[str]) -> dict[str, int]:
    """词频统计。三个算法都以它为输入，因此**必须先算一次再共用** ——
    各算各的会让「同一语料」这个前提在实现层面悄悄失效。"""
    out: dict[str, int] = {}
    for w in words:
        out[w] = out.get(w, 0) + 1
    return out


def alphabet_of(words: Iterable[str]) -> set[str]:
    """基础字符集：所有出现过的字符。词表规模必须大于它。"""
    out: set[str] = set()
    for w in words:
        # 直接 update，不先给每个词建一个临时 set 再求并 —— 语料上千个词时
        # 那是上千次多余的分配。结果完全相同。
        out.update(w)
    return out
