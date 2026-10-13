# TokenizerFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L1 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | TokenizerFactory |
| 类型 | Factory |
| 真正有意思的核心 | BPE / Unigram 训练、比较、压缩 |

---

## 二、核心命题

**不要只是"训练 BPE"。** 真正的问题应该是：

> 同一语料，什么 Tokenization Strategy 最合适？

## 三、整体架构

```
                  Corpus
                    │
                    ▼
              Corpus Profiler
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       BPE       Unigram     WordPiece
        │           │           │
        └───────────┼───────────┘
                    ▼
                Tokenizers
                    │
                    ▼
               Benchmark
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 Compression    Fertility    Coverage
        │        Speed        Unknown
        └───────────┼───────────┘
                    ▼
                 Compare
                    │
                    ▼
               Pareto Search
                    │
                    ▼
            Tokenizer Artifact
```

## 四、主流水线

```
Corpus
 ↓
Normalize
 ↓
Train BPE / Unigram / ...
 ↓
Encode Benchmark Corpus
 ↓
Measure
 ↓
Compare
 ↓
Search Vocab/Algorithm
 ↓
Select
 ↓
Export
```

## 五、评测维度

| 维度 | 含义 |
| --- | --- |
| Compression | 同语料下的 token 数（越低越省上下文） |
| Fertility | 每词平均 token 数，跨语言比较的关键指标 |
| Coverage | 词表覆盖率 / 未知 token 比例 |
| Speed | 编解码吞吐 |
| Unknown | 未登录字符处理能力（中文、emoji、代码符号） |

## 六、特别值得做的：Tokenizer Diff

```
"中华人民共和国"

Tokenizer A:
[中华人民共和国]

Tokenizer B:
[中华][人民][共和国]

Tokenizer C:
[中][华][人民][共和][国]
```

**让 tokenizer 的设计空间可视化。** 同一个词在不同词表下的切分差异，直接决定了：
- 序列长度与推理成本
- 模型对中文 / 代码 / 多语言的实际建模粒度
- 跨模型迁移时的对齐难度

## 七、阶段性落地计划

- **M0｜语料与训练骨架**：定义 Corpus / Tokenizer Spec 的 schema，跑通 BPE 训练与编码一次（非自动）。
- **M1｜多算法与度量**：实现 BPE / Unigram / WordPiece 训练，接入 Compression / Fertility / Coverage / Speed / Unknown 五维度量。
- **M2｜搜索与择优**：把词表大小、算法、预分词规则参数化，接 Pareto Search 输出候选前沿。
- **M3｜Tokenizer Diff 与发布**：实现切分可视化对比，产出 Tokenizer Artifact 与对比报告。

## 八、验收标准

能端到端完成「TokenizerFactory」，并在其「核心」（BPE / Unigram 训练、比较、压缩）上给出可复现的量化结果：

- 在同一语料上给出多算法 × 多词表规模的对比矩阵
- 输出 Pareto Frontier，并能用 Tokenizer Diff 解释差异来源

---

## 附录 A、工程骨架

```
L1-tokenizerfactory/
├── README.md
├── main.py                    零安装入口：把 src/ 塞进 sys.path 后转交 cli:main
├── pyproject.toml             打包 + pytest 配置；dependencies = []，hf extra 是「接真 HF tokenizers」的扩展点
├── requirements.txt           注释说明「核心链路零依赖」
├── .gitignore
├── configs/
│   └── default.json           语料路径 / 归一化形式 / 设计空间（算法×词表×规则）/ 度量 / Pareto 目标对 / 选型规则
├── data/corpus/
│   ├── train.txt              训练语料（中英混排 + 代码 + 数字 + 中文合成词）
│   ├── benchmark.txt          同分布、句子不重复
│   ├── hard.txt               刻意在分布之外：罕见汉字 / emoji / 各种文种 / 路径 / URL / 全角
│   └── diff_words.txt         Diff 用文本（16 行，含旗舰演示词）
├── src/tokenizerfactory/
│   ├── __init__.py            导出全部核心对象
│   ├── schema.py              CorpusStats / TokenizerSpec / Tokenization / TokenizerMetrics /
│   │                          TokenizerRun / Version / TokenizerResult
│   ├── rng.py                 确定性随机（sha256 派生种子，绝不用内置 hash）
│   ├── lineage.py             LineageNode + LineageDAG（含 from_dict 重建 + 确定性导出）
│   ├── normalize.py           NFC/NFD/NFKC/NFKD/none + 可选小写化
│   ├── corpus.py              三条预分词规则 + 词频 + 字符类统计（**CJK 必须成串切**，见附录 C）
│   ├── config.py              配置加载与**加载期**校验（含「词表预算必须装得下基础词表」）
│   ├── metrics.py             五维度量 + 速度基准（多轮打乱顺序取中位数）+ TIMING_METRICS
│   ├── compare.py             支配 / 非支配 / 超体积（min 形式一套坐标）/ 拐点 / 选型 / 组装
│   ├── diff.py                Tokenizer Diff：边界集合 / Jaccard / 按设计轴选列
│   ├── pipeline.py            扇形展开再收拢的编排 + 落盘
│   ├── report.py              报告表格行 + Markdown 报告 + explain
│   ├── cli.py                 run / compare / frontier / diff / explain / list / dag / encode
│   └── tokenizers/
│       ├── base.py            Tokenizer 抽象基类（字符级 UNK 回退 / 基础词表 / 词表摘要）
│       ├── merge.py           增量合并训练器 + **朴素参照实现**（两者必须给出相同合并序列）
│       ├── bpe.py             按词频加权 pair 计数 + 贪心合并
│       ├── wordpiece.py       count(pair)/(count(l)*count(r)) + ## 续接符号
│       └── unigram.py         种子词表 + 前向后向 EM + 剪枝 + Viterbi 编码
├── tests/                     pytest（5 个文件 + conftest，257 个用例）
└── out/                       运行产物（每次 run 重建）
```

## 附录 B、快速开始

```bash
# 核心链路零第三方依赖，直接用标准库跑
python main.py run --config configs/default.json --out out

# 对比矩阵（与报告第三节同一份行构造函数，不会两处分家）
python main.py compare --out out

# Pareto 前沿（--pair 可只看某一条）
python main.py frontier --out out
python main.py frontier --out out --pair vocab_size+tokens_per_char

# Tokenizer Diff + 边界一致度
python main.py diff --out out

# 某个 spec 的完整解释（为什么它在/不在前沿上）
python main.py explain wordpiece-v2048 --out out

# 用**导出的 artifact** 编码任意文本 —— 「产物真的能用」的可执行检查
python main.py encode bpe-v2048 中华人民共和国 的 分词 --out out

# 血缘图（ASCII / Graphviz DOT；--frontier-only 只画入选祖先链）
python main.py dag --out out
python main.py dag --out out --dot --frontier-only > lineage_selected.dot

# 测试
pytest
```

设计约定（与同工作区其它项目共用一套）：

- **`--out` 只属于 `run`**。其余子命令是**只读观察者**，跑完产物目录一个字节都不变（有测试守着）。
- **只读子命令不做训练**。两处例外都是*载入*而非重算：`encode` 载入 artifact，
  `dag` 从 `lineage.json` 重建血缘图（否则 `--dot` 只能打印一句提示）。
- **未知 `spec_id` 返回 1 并给相近建议**，不抛栈。
- 刻意**没有** `rounds` 子命令：本项目不是一轮一轮迭代的，用「第几轮」描述会误导。

## 附录 C、已验证状态

本骨架已实际跑通，不是纸面设计：

- `pytest` → **257 passed**（`test_primitives` 76 / `test_tokenizers` 59 /
  `test_metrics_compare` 38 / `test_diff` 26 / `test_end_to_end` 58），
  覆盖 schema / rng / 归一化 / 三条预分词规则 / 三个算法（含**朴素参照实现比对**）/
  配置校验 / 度量口径 / 支配与超体积 / Diff / 编排 / 落盘 / 报告 / CLI 契约 /
  随项目发布的语料本身的覆盖性
- `python main.py run` → **27 个 spec**（3 算法 × 3 词表规模 × 3 预分词规则），
  整条流水线 **19.8s**（其中训练 19.2s）；DAG **60 节点 / 60 边 / 1 个收拢点**；
  版本 `v1-282de845`

### 语料画像（所有结论的前提）

`data/corpus/train.txt`：**6471 字符 / 1309 词 / 633 唯一词**，字符集 **551** 个，
最长词 25 字符、平均 4.94 字符。字符构成 **cjk 42.5% / latin 39.7% / space 9.6% /
symbol 6.8% / digit 1.2%** —— 中英几乎对半，所以「中文要不要成串切」「英文词能不能合并」
两个问题在这份语料上同等重要。

基础词表占用（预算里先被扣掉的那一块，**因算法而异**）：

| 算法 | 基础词表 | 说明 |
| --- | --- | --- |
| bpe / unigram | **552** | `[UNK]` + 551 个字符 |
| wordpiece | **1103** | `[UNK]` + 551 个字符 + 551 个 `##` 续接形式 |

### 验收标准一 ✅：多算法 × 多词表规模的对比矩阵

27 行 × 13 列，五维度量全在（`matrix.csv` 与报告第三节同一份数据）。
同一份语料、同一份词频（**三个算法共用同一份词频**，这是对比成立的前提）。

关键读数（`unicode` 规则）：

| 词表预算 | bpe | unigram | wordpiece |
| --- | --- | --- | --- |
| 2048 | **0.5376** | 0.5847 | 0.9208 |
| 4096 | **0.5282**（实际 2985，已饱和） | 0.5822（实际 2729，已饱和） | 0.6426 |
| 8192 | **0.5282**（实际 2985） | 0.5822（实际 2729） | 0.5559（实际 4554） |

三个真实结论：

1. **WordPiece 在小预算下落后 71%**（0.9208 vs 0.5376），原因是它的基础词表占 1103，
   2048 的预算里只剩 945 个额度用来学合并 —— 而 BPE / Unigram 有 1496 个。
   这不是它的挑选准则更差，是**起点不同**。到 8192 它才追上（0.5559）。
2. **BPE 与 Unigram 在 4096 就饱和**（实际词表停在 2985 / 2729），再往上加预算完全无效。
   `vocab_size` 这个设计变量在饱和点之后**不再是变量**。
3. **`char` 规则下词表规模这个变量彻底失效**：它的词表恒为 552（551 个字符 + UNK），
   三个预算档给出逐位相同的度量。这也是它在 `vocab_size_actual+tokens_per_char`
   那条前沿上占了 5 个「超体积增量 = 0.0」的重复点、让总超体积只有 1,911 的原因
   （9 个点里有 6 个增量为 0）。

### 验收标准二 ✅：Pareto Frontier + 用 Tokenizer Diff 解释差异来源

三条前沿（`out/frontier.csv`）：

| 目标对 | 前沿点数 | 超体积 | 参考点 |
| --- | --- | --- | --- |
| `vocab_size+tokens_per_char` | 2 | 3,403.7 | (8601.6, 1.0505) |
| `vocab_size_actual+tokens_per_char` | 9 | 1,911.2 | (5580.8, 1.0505) |
| `tokens_per_char+encode_tokens_per_sec` | 5 | 211,827 | (1.0505, −288665.7) |

主前沿只有 **2 个点**：`bpe-v2048`（0.5376 @ 2048）与 `bpe-v4096`（0.5282 @ 2985）。
**「前沿上只有 2 个点」本身就是一个结论**：在这份语料上，词表规模与压缩率之间
几乎不存在真实的取舍 —— 多花一倍预算只能换 1.7% 的压缩。

**Diff 就是「为什么」**（`python main.py diff --out out`）。旗舰演示：

```
中华人民共和国
  bpe-v4096              ▁ | 中华人民共和国                        2
  bpe-v2048              ▁ | 中华 | 人民 | 共和国                  4
  unigram-v2048          ▁ | 中华 | 人民 | 共和国                  4
  wordpiece-v2048        ▁ | ##中 | ##华人民共 | ##和国            4
  bpe-v2048+whitespace   ▁ | 中 | 华 | 人民 | 共和国                5
  bpe-v2048+char         ▁ | 中 | 华 | 人 | 民 | 共 | 和 | 国      8
```

同一个词、同一份语料，**2 个 token 到 8 个 token，差 4 倍**。差距的来源一眼可见：

- `wordpiece` 的 `##华人民共` 是一个语义上不存在的子串 —— 它的挑选准则
  `count(pair)/(count(l)·count(r))` 偏向高频共现，在小语料上学不出「中华」这种词；
- `bpe-v2048+whitespace` 的 `▁ | 中 | 华` 说明「整句成一个词」之后，
  词表里没有足够的合并机会，只能退回字符级；
- `bpe-v2048+char` 是粒度极端值，给出上界。

英文侧的对比同样锋利：`tokenization` 在 `bpe-v2048` 是 **1** 个 token，
在 `wordpiece-v2048` 是 **13** 个（`▁|##t|##o|##k|##e|##n|##i|##z|##a|##t|##i|##o|##n`）；
`internationalization` 是 **6 vs 21**。

边界一致度（跨模型迁移的对齐难度，Diff 文本上的 Jaccard 均值）：
`bpe-v2048` **0.6413** 最高，`wordpiece-v2048` **0.5166** 最低。
平均每行 token 数：`bpe-v2048` **6.25** / `bpe-v4096` **6.00** /
`bpe-v2048+char` **13.13** / `wordpiece-v2048` **12.06**。

### 「搜索优于基线」在这份语料上是 +0.00% —— 而且这是个结论

基线定义为**不比较就直接上的那个配置**：`bpe` + 最小预算 + `unicode` 预分词 + 不小写。
实测它就是 `bpe-v2048`，也就是最终入选者 —— 所以「比基线好多少」是一个恒为 0 的差。

报告里没有把它当成主结论，因为它读起来像「搜索没用」，而真相是
**搜索确认了默认选择**：在这份语料上「不比较就直接上」恰好是对的，
只是这个结论只有比较过之后才敢说出口。真正可决策的数字是**「选错算法的代价」**：

| 词表预算 | 该档算法差距（固定规则） | 该档规则差距（固定算法） |
| --- | --- | --- |
| 2048 | **+71.3%** | +86.1% |
| 4096 | +21.6% | +89.4% |
| 8192 | **+10.2%** | +89.4% |

算法差距随预算从 71.3% 降到 10.2%（预算越大，WordPiece 被 `##` 吃掉的那一半额度越摊越薄），
而规则差距恒定在 ~89%（`char` 是粒度极端值，与预算无关）。
**只看一个预算档就下结论，是本项目最容易犯的错误。**

### 跑起来才发现的问题

按发现顺序，每一条都是实测撞出来的，不是想出来的：

1. **CJK 必须成串切，不能按字切。** 第一版 `_UNICODE_PATTERN` 是每个汉字一个词，
   于是「中华」「人民」「共和国」**永远学不出来**（BPE 只在词内合并），
   README 第六节那个旗舰演示根本做不出来。改成 `([{CJK_RANGES}]+)` 之后才对。
   「按字切」没有删掉 —— 它保留成 `char` 规则，因为它正好是生育率的上界。
2. **三条预分词规则的词首标记必须形态统一。** 第一版 `whitespace` / `char` 的首词不加 `▁`，
   于是规则之间的对比里混进了「标记策略」这个额外变量。现在三条规则统一，
   不变量是「标记数 == 1 + 空白段数」，有测试参数化守着。
3. **`whitespace` 的伤害比直觉小得多。** 直觉是「中文没有空格，所以按空白切会把整句变成一个词，
   压缩率会爆炸」。实测：压缩率只差 **0.5%**（0.5401 vs 0.5376），
   因为编码总能退回字符级。真正被暴露出来的是**生育率**：**5.482 vs 2.927，差 87%**。
   这条教训直接写进了报告 —— 只看压缩率会得出「预分词规则无关紧要」的错误结论。
   这也正是要量五个维度的原因。
4. **覆盖率跨预分词规则不可比。** `char` 的覆盖率 **0.977**，`unicode` 是 **0.916** ——
   看起来 `char` 的词表更好，其实只是因为它把「词」定义成单个码点，判定粒度更细。
   报告第三节把这条口径限制写成了显式警告，第六节还加了「这张表一半的赢家是 `char`」的交叉引用。
5. **速度是实测值 ⇒ 含实测轴的前沿不可逐位复现。** 同一份配置跑两次，
   `tokens_per_char+encode_tokens_per_sec` 这条前沿的成员从 `wordpiece-v128`
   变成 `wordpiece-v128+whitespace`，超体积从 211,827 变成 304,431；
   而另外两条前沿与**全部词表摘要逐位相同**。中位数只能压小噪声，消不掉它。
   现在 `metrics.TIMING_METRICS` 是这个约定的可执行形式，报告里该前沿带「含实测轴」标注，
   测试也分成两段断言（不含实测轴的部分要求严格相等）。
6. **前沿表出现过两个同名「实际词表」列。** `objectives[0]` 是 `vocab_size_actual` 时，
   表头里固定加的那一列与 `METRIC_LABELS` 给出的标签撞名，一个显示 `552`、
   另一个显示 `552.0000`。小语料夹具的 `extra_objectives` 里没有这个目标对，
   所以只有**在真实配置上跑 `frontier` 命令**才看得出来。现在有「任意两个指标组合都不得
   出现重复列名」的穷举测试。
7. **预算敏感性表第一版把两条决策轴合并了。** 原来只有一列「该档最大差距」
   （该档最差 spec 比最好 spec），实测恒定在 +86%~+89% —— 而那个数字由 `char`
   单独决定，跟「选哪个算法」无关。于是同一份报告里出现了「预算越大差距越大」
   与「预算越小选错的代价越大」两句互相矛盾的话。现在拆成「算法差距」与「规则差距」两列，
   且正文里的数字**从表里算**而不是写死。
8. **Diff 的第一版一次动了两个变量。** 「每个算法各补一个压缩最好的」实测给 unigram
   与 wordpiece 补的都是 `whitespace` 变体（它的 `tokens_per_char` 恰好略好一点，
   0.5827 vs 0.5847），于是那张表里算法与规则同时在变。现在按设计轴选列：
   算法轴固定规则、规则轴固定算法 —— 顺带把 `char` 那一列带进来，
   「粒度」这个轴在表里才两端齐全。
9. **`lineage.json` 曾经不可复现。** 节点的 `outputs` 与边的顺序取决于
   「节点是何时被加进来的」，而 `from_dict` 重建时是按拓扑序加进来的 ——
   同一个图的两条构建路径导出两份不同的 JSON。现在导出前排序，
   并有 `test_dag_rebuild_from_json_is_equivalent` 守着。
10. **`dag --dot` 曾经是个空转的开关。** CLI 从磁盘载入的结果对象里没有图，
    于是 `--dot` 静默退化成「打印摘要 + 一句提示」。现在从 `lineage.json` 重建图，
    只读子命令因此真的只读且功能完整。
11. **Diff 词必须刻意为语料做覆盖。** 旗舰演示词 `中华人民共和国` 一开始在 Diff 里
    全是 `[UNK]` —— 训练语料里根本没有「人」「共」「华」「国」「民」这些字。
    这类错误不报错，只让演示变成一堆 `[UNK]`，所以它现在是一条测试
    （`test_shipped_corpus_covers_every_diff_word_character`：逐行取出 Diff 词，
    断言每个字符都在训练语料的字符集里），而不是一次人工检查。

### 可执行的不变量

这些不是文档里的口号，都有测试守着：

| 不变量 | 为什么它值得守 |
| --- | --- |
| **增量合并 == 朴素合并**（BPE / WordPiece，逐项相同） | 簿记错误不报错，只让词表悄悄变，而报告照常生成 |
| **`_encode_run` 绝不返回 `[UNK]`** | 违反它只表现为未知率悄悄偏高 |
| **实际词表 ≤ 目标词表** | 否则「同等预算下谁压得更狠」的前提被破坏 |
| **超体积增量之和 == 总超体积** | 阶梯求和写错方向会多算/少算面积而不报错 |
| **参考点取全部候选点，不取前沿点** | 否则不同前沿子集的超体积不可比 |
| **`plain_piece` 的边界跨算法可比** | 用 `len(token)` 会让 WordPiece 的边界整体右移 2，一致度全是假的 |
| **每个 spec 的血缘链是图里的一条真实路径** | 扁平列表与图分家时 `explain` 会悄悄缺一块 |
| **`sel` 节点的输入恰好是主前沿候选** | 这是「一眼看出谁进了前沿」的唯一依据 |
| **只读子命令不改产物目录** | 观察者会改状态是最容易把人绕晕的一类设计 |
| **导出的 artifact 载入后逐位复现编码** | 否则 `encode` 只是「读了个 JSON 然后假装能用」 |

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L1 深化设计稿；_
_附录 A/B/C 为落地后补写的工程记录。_

