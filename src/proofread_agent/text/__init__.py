"""中文文本处理：分句、分段、分词、Aho-Corasick 多模式匹配。

全部为纯标准库实现，避免在生产内网环境中引入难以审计的第三方依赖。
性能上，Aho-Corasick 对 1 万条术语 × 10 万字稿件的匹配在毫秒级完成。
"""

from __future__ import annotations

import re
from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------- 分句
_SENT_END = "。！？；!?;\n"
_CLOSERS = "”』」）)》】'\"')"


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """按中文句末标点切句，返回 (start, end, sentence)。"""
    out: List[Tuple[int, int, str]] = []
    buf_start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _SENT_END:
            j = i + 1
            # 吸收紧跟的引号/括号与连续句末符号
            while j < n and (text[j] in _CLOSERS or text[j] in _SENT_END):
                j += 1
            seg = text[buf_start:j]
            if seg.strip():
                out.append((buf_start, j, seg))
            buf_start = j
            i = j
        else:
            i += 1
    if buf_start < n and text[buf_start:].strip():
        out.append((buf_start, n, text[buf_start:]))
    return out


# ---------------------------------------------------------------- 分段
_HEADING = re.compile(r"^\s{0,3}(#{1,6}\s+.+|第[一二三四五六七八九十百]+[章部分节]|[一二三四五六七八九十]+、)")
_REF_HEAD = re.compile(r"^\s*(#{1,6}\s*)?(参考文献|references|引用文献|注释)\s*$", re.I)


def split_paragraphs(text: str) -> List[Dict]:
    """按空行/换行切段，标记标题与参考文献区。

    返回 [{index, start, end, text, is_heading, in_reference_section}]
    """
    paras: List[Dict] = []
    offset = 0
    in_ref = False
    for idx, raw in enumerate(text.split("\n")):
        line_start = offset
        offset += len(raw) + 1
        if not raw.strip():
            continue
        if _REF_HEAD.match(raw):
            in_ref = True
        is_heading = bool(_HEADING.match(raw)) and len(raw.strip()) < 60
        paras.append({
            "index": len(paras),
            "line": idx + 1,
            "start": line_start,
            "end": line_start + len(raw),
            "text": raw,
            "is_heading": is_heading,
            "in_reference_section": in_ref,
        })
    return paras


def line_of(text: str, pos: int) -> int:
    """字符偏移 → 行号（1 基）。"""
    return text.count("\n", 0, min(pos, len(text))) + 1


# ---------------------------------------------------------------- 分词
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+(?:\.\d+)?%?")


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """去标点后的字符 n-gram，用于词汇兜底相似度。"""
    cleaned = "".join(_TOKEN_RE.findall(text))
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i:i + n] for i in range(len(cleaned) - n + 1)]


def tokenize(text: str) -> List[str]:
    """极简分词：中文按二元组、英文/数字按整词。

    目的是在没有 jieba 的内网环境里，为 TF-IDF / BM25 提供稳定的词元，
    无需外部词典，且对同义改写有较好召回。
    """
    toks: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        if re.fullmatch(r"[\u4e00-\u9fff]+", tok):
            if len(tok) == 1:
                toks.append(tok)
            else:
                toks.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            toks.append(tok.lower())
    return toks


# ---------------------------------------------------------------- Aho-Corasick
class AhoCorasick:
    """AC 自动机：一次扫描匹配全部模式，支持同位置多模式与重叠模式。

    相比 ``re.compile("|".join(patterns))``：
    * 不受正则元字符影响（术语中含 . * + 等无需转义）；
    * 关键词数量上万时仍保持线性扫描；
    * 可拿到所有命中位置，而正则 finditer 在重叠场景会漏报。
    """

    __slots__ = ("goto", "fail", "output", "pattern_info")

    def __init__(self) -> None:
        self.goto: List[Dict[str, int]] = [{}]
        self.fail: List[int] = [0]
        self.output: List[List[str]] = [[]]
        self.pattern_info: Dict[str, Dict] = {}

    def add(self, pattern: str, info: Dict | None = None) -> None:
        if not pattern:
            return
        self.pattern_info[pattern] = info or {}
        node = 0
        for ch in pattern:
            nxt = self.goto[node].get(ch)
            if nxt is None:
                self.goto.append({})
                self.fail.append(0)
                self.output.append([])
                nxt = len(self.goto) - 1
                self.goto[node][ch] = nxt
            node = nxt
        self.output[node].append(pattern)

    def build(self) -> "AhoCorasick":
        q: deque[int] = deque()
        for ch, nxt in self.goto[0].items():
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            cur = q.popleft()
            for ch, nxt in self.goto[cur].items():
                f = self.fail[cur]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[nxt] = self.goto[f].get(ch, 0) if f or ch in self.goto[0] else 0
                if self.fail[nxt] == nxt:
                    self.fail[nxt] = 0
                self.output[nxt] = self.output[nxt] + self.output[self.fail[nxt]]
                q.append(nxt)
        return self

    def iter_matches(self, text: str) -> Iterable[Tuple[int, int, str]]:
        """产出 (start, end, pattern)。"""
        node = 0
        for i, ch in enumerate(text):
            while node and ch not in self.goto[node]:
                node = self.fail[node]
            node = self.goto[node].get(ch, 0)
            for pat in self.output[node]:
                yield i - len(pat) + 1, i + 1, pat

    def find_all(self, text: str) -> List[Tuple[int, int, str]]:
        return list(self.iter_matches(text))

    @classmethod
    def from_patterns(cls, patterns: Sequence[str]) -> "AhoCorasick":
        ac = cls()
        for p in patterns:
            ac.add(p)
        return ac.build()


# ---------------------------------------------------------------- 编辑距离
def levenshtein(a: str, b: str, max_dist: int = 3) -> int:
    """带上界剪枝的编辑距离（Ukkonen 行优化）。"""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            if v < best:
                best = v
        if best > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """归一化相似度 0-1。"""
    if not a or not b:
        return 0.0
    d = levenshtein(a, b, max_dist=max(len(a), len(b)))
    return 1.0 - d / max(len(a), len(b))


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
