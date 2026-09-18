"""文档对象：所有引擎共享的只读视图。

一次性完成分句、分段、参考文献区识别与偏移映射，避免每个引擎重复切分。
同时提供"引号区间""否定语境"等误报压制所需的辅助索引。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .text import line_of, split_paragraphs, split_sentences

_QUOTE_PAIRS = [("“", "”"), ("‘", "’"), ("《", "》")]
#: 直引号在中文稿件中大量存在（尤其是从聊天工具/网页复制的文本），
#: 必须同样被识别为"引述区间"，否则引文豁免机制会整体失效。
_SYMMETRIC_QUOTES = ['"', "'"]
_REF_ITEM = re.compile(r"^\s*\[(\d+)\]\s*")
_NEGATION = re.compile(r"(不|非|无|未|没有|禁止|防止|避免|杜绝|反对|遏制)[^，。；]{0,6}$")


@dataclass
class Document:
    text: str
    doc_id: str = ""
    title: str = ""
    doc_type: str = "调研专报"
    source_path: str = ""
    paragraphs: List[Dict[str, Any]] = field(default_factory=list)
    sentences: List[Tuple[int, int, str]] = field(default_factory=list)
    quote_spans: List[Tuple[int, int]] = field(default_factory=list)
    ref_start: int = -1
    ref_items: Dict[int, str] = field(default_factory=dict)
    in_text_citations: List[int] = field(default_factory=list)

    # ---- 构造 ------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path, doc_type: str = "调研专报") -> "Document":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        return cls.from_text(text, doc_type=doc_type, source_path=str(p))

    @classmethod
    def from_text(cls, text: str, doc_type: str = "调研专报",
                  source_path: str = "", title: str = "") -> "Document":
        doc = cls(
            text=text,
            doc_id=hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
            title=title or _guess_title(text),
            doc_type=doc_type,
            source_path=source_path,
        )
        doc.paragraphs = split_paragraphs(text)
        doc.sentences = split_sentences(text)
        doc.quote_spans = _find_quotes(text)
        doc._locate_references()
        return doc

    # ---- 定位辅助 --------------------------------------------------
    def paragraph_of(self, pos: int) -> Dict[str, Any]:
        for p in self.paragraphs:
            if p["start"] <= pos < p["end"]:
                return p
        return {"index": -1, "line": line_of(self.text, pos), "text": "", "is_heading": False,
                "in_reference_section": False}

    def sentence_of(self, pos: int) -> str:
        for s, e, sent in self.sentences:
            if s <= pos < e:
                return sent
        return ""

    def line_of(self, pos: int) -> int:
        return line_of(self.text, pos)

    def in_reference_section(self, pos: int) -> bool:
        return self.ref_start >= 0 and pos >= self.ref_start

    def in_quote(self, pos: int) -> bool:
        return any(s <= pos < e for s, e in self.quote_spans)

    def in_citation(self, pos: int, min_quoted_len: int = 20) -> bool:
        """是否位于**引述性**引号内（引号内容足够长，属原文引用）。

        只在长引述中豁免检查。术语级短语加引号（如"一带一路战略"）恰恰是
        错误高发位置，不能因为加了引号就放过——这是误报压制与漏报之间的
        关键取舍，宁可豁免偏保守。
        """
        return any(s <= pos < e and (e - s) >= min_quoted_len
                   for s, e in self.quote_spans)

    def is_negated(self, pos: int) -> bool:
        """判断 pos 之前 8 字内是否存在否定词，用于压制"不得使用 X"类表述的误报。"""
        left = self.text[max(0, pos - 8):pos]
        return bool(_NEGATION.search(left))

    def context(self, pos: int, before: int = 20, after: int = 20) -> str:
        return self.text[max(0, pos - before): min(len(self.text), pos + after)]

    @property
    def char_count(self) -> int:
        return len(re.sub(r"\s", "", self.text))

    @property
    def body_text(self) -> str:
        """正文（排除参考文献区）。"""
        return self.text[:self.ref_start] if self.ref_start >= 0 else self.text

    @property
    def reference_text(self) -> str:
        return self.text[self.ref_start:] if self.ref_start >= 0 else ""

    # ---- 参考文献解析 ----------------------------------------------
    def _locate_references(self) -> None:
        for p in self.paragraphs:
            if p["in_reference_section"] and not _is_ref_head(p["text"]):
                m = _REF_ITEM.match(p["text"])
                if m:
                    self.ref_items[int(m.group(1))] = p["text"]
                    if self.ref_start < 0:
                        self.ref_start = p["start"]
        if self.ref_start < 0:
            # 没有标准"参考文献"标题时，退化为识别连续的 [n] 条目
            numbered = [p for p in self.paragraphs if _REF_ITEM.match(p["text"])]
            if len(numbered) >= 2:
                self.ref_start = numbered[0]["start"]
                for p in numbered:
                    self.ref_items[int(_REF_ITEM.match(p["text"]).group(1))] = p["text"]
        body = self.body_text
        self.in_text_citations = sorted({int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", body)})

    def to_meta(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "source_path": self.source_path,
            "char_count": self.char_count,
            "paragraph_count": len(self.paragraphs),
            "sentence_count": len(self.sentences),
            "reference_items": len(self.ref_items),
        }


def _is_ref_head(text: str) -> bool:
    return bool(re.match(r"^\s*(#{1,6}\s*)?(参考文献|references|引用文献|注释)\s*$", text, re.I))


def _guess_title(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:60]
    return "未命名文档"


def _find_quotes(text: str) -> List[Tuple[int, int]]:
    """识别成对引号/书名号覆盖区间。引号内的原文引述可豁免部分检查。

    * 中文弯引号与书名号按左右符号配对；
    * 直引号（" 与 '）左右同形，按"奇偶交替"配对——第 1、3、5… 个为左，
      第 2、4、6… 个为右；
    * 未闭合的引号延伸到行尾，避免整段被判为"引述区间"。
    """
    spans: List[Tuple[int, int]] = []
    for left, right in _QUOTE_PAIRS:
        stack: List[int] = []
        for i, ch in enumerate(text):
            if ch == left:
                stack.append(i)
            elif ch == right and stack:
                spans.append((stack.pop(), i + 1))
        for s in stack:                       # 未闭合：延伸到行尾
            eol = text.find("\n", s)
            spans.append((s, eol if eol != -1 else len(text)))

    for q in _SYMMETRIC_QUOTES:
        positions = [i for i, ch in enumerate(text) if ch == q]
        for k in range(0, len(positions) - 1, 2):
            start, end = positions[k], positions[k + 1]
            if "\n" in text[start:end]:       # 跨行的直引号多半不是配对引号
                continue
            spans.append((start, end + 1))
    return spans
