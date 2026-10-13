"""确定性随机数 —— 可复现性的地基。

本项目的验收标准之一是「可复现的量化结果」，所以任何不确定性都必须被钉死。
两个必须绕开的坑（与同工作区其它项目共用同一条纪律）：

1. **内置 ``hash()`` 对 str 加随机盐**（``PYTHONHASHSEED``），跨进程不稳定。
   用它派生种子会让「重跑结果一致」直接失效，因此这里一律用 sha256。
2. **全局 ``random`` 的状态会被任何一次调用推进**，导致结果依赖调用顺序。
   这里按 ``(seed, 用途, 对象 id)`` 派生**独立实例**。

本项目真正需要随机性的地方很少 —— 三个分词算法都是确定性的（BPE / WordPiece 是
贪心计数，Unigram 的 EM 与剪枝由频率决定）。剩下的用处只有两处，但都不能省：

- **Unigram 剪枝的并列打破**：剪枝时可能出现多个符号的「损失」完全相同，
  按集合遍历顺序决定会让结果随 Python 版本变化；用 rng 显式打破才是可复现的。
- **速度基准的重复顺序**：多轮测量如果按固定顺序跑，CPU 降频会把后面的轮次系统性地压低；
  打乱顺序取中位数可以抵消这个偏差。
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def derive_seed(*parts: Any) -> int:
    """由若干片段派生一个 64bit 整数种子（跨平台、跨进程稳定）。"""
    joined = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "big")


def make_rng(*parts: Any) -> random.Random:
    """派生一个独立的 ``Random`` 实例。同样的 ``parts`` 必得同样的序列。"""
    return random.Random(derive_seed(*parts))
