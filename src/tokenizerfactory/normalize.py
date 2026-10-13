"""归一化：流水线的第一段。

归一化看起来无聊，但它是**分词质量差异的一个主要来源**，所以这里的选择必须
显式、可配置、可解释：

| 选择 | 影响 |
| --- | --- |
| Unicode 形式 | ``NFC`` 把「é」（e + 组合重音）合成单码点，``NFKC`` 还会把全角 ``Ａ`` 折成 ``A``。取错形式会让**同一个字符在词表里出现两份**，直接浪费词表预算 |
| 空白折叠 | 连续空白折成单个空格。不折叠的话「缩进」会吃掉大量词表空间；折叠后 decode 只能还原**归一化后**的文本，这一点必须如实写出来 |
| 首尾去空白 | 否则第一个词会带一个孤立的词首标记 |

**不做的事**（刻意）：不去标点、不转小写、不做 Unicode 大小写折叠。
小写化会改变词表内容，属于**设计空间**而不是预处理 —— 因此它是
``TokenizerSpec.lowercase`` 的职责，由搜索阶段去比较「值不值得」。
把预处理和设计空间混在一起，就没法回答「小写化省了多少 token」了。
"""

from __future__ import annotations

import re
import unicodedata

#: 支持的归一化形式。``none`` 表示不做 Unicode 归一化（用于对照实验）
FORMS: tuple[str, ...] = ("NFC", "NFD", "NFKC", "NFKD", "none")

_WS_RUN = re.compile(r"\s+")


def normalize(
    text: str,
    form: str = "NFC",
    collapse_whitespace: bool = True,
    strip: bool = True,
) -> str:
    """归一化文本。**这是唯一允许改动原文的地方**，之后的每一段都只看归一化结果。"""
    if form not in FORMS:
        raise ValueError(f"未知归一化形式 {form!r}；可用：{', '.join(FORMS)}")
    out = text if form == "none" else unicodedata.normalize(form, text)
    if collapse_whitespace:
        out = _WS_RUN.sub(" ", out)
    if strip:
        out = out.strip()
    return out


def normalize_words(words: list[str], lowercase: bool) -> list[str]:
    """对**已经预分词**的词做可选小写化。

    放在预分词之后而不是之前：``lowercase`` 只应影响词表内容，
    不该影响「词是怎么切开的」—— 先小写再预分词会让「İ」这类字符
    的切分结果随语言环境变化，引入无法解释的差异。
    """
    if not lowercase:
        return list(words)
    return [w.lower() for w in words]


def describe(form: str, collapse_whitespace: bool = True, strip: bool = True) -> str:
    """一句话描述当前归一化配置，用于报告头。"""
    parts = [f"Unicode {form}" if form != "none" else "不做 Unicode 归一化"]
    if collapse_whitespace:
        parts.append("空白折叠")
    if strip:
        parts.append("首尾去空白")
    return " + ".join(parts)
