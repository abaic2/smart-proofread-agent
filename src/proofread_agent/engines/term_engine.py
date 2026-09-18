"""政治术语精准审校引擎。

三级匹配流水线
--------------
L1 精确匹配（Aho-Corasick）
    对术语库中所有"标准表述 + 错误变体"构建 AC 自动机，一次扫描全量命中。
    优点：零误报、可解释（命中即错，能直接给出规范表述）。

L2 近似匹配（编辑距离 + 倒排索引）
    捕获未登记的字形错误：别字、漏字、增字、词序微调。
    用字符二元组倒排索引把候选位置从 O(n·|Term|) 降到接近 O(n·候选数)，
    再对候选做带上界剪枝的 Levenshtein 距离验证。

L3 语义匹配（本地嵌入向量 + 余弦）
    捕获"字面完全不同但政治含义偏移"的表达，例如"以人民为核心"
    应为"以人民为中心的发展思想"。这一级依赖本地嵌入模型，
    不可用时自动退化为词形近似并显式标注 embedding_mode。

附加校验
--------
* **组合提法顺序校验**：四个意识 / 四个自信 / 新发展理念 / 社会主义核心价值观
  的顺序固定，错序与缺项均报错。
* **提法演进校验**：类型为 deprecated 的条目（如"系列重要讲话精神"）
  按当时时点给出更新建议。
* **禁用/慎用模式**：正则匹配 + 语境豁免。

误报压制（生产的生命线）
------------------------
* 引号内原文引述豁免（引注历史文献属正当引用）；
* 否定语境豁免（"不得使用 X"中的 X 不报错）；
* 参考文献区豁免（文献题名不可改）；
* 标准表述保护（本身正确的表述绝不报错）；
* 重叠消解（同一处只保留最严重的一条）。
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from ..document import Document
from ..llm import EmbeddingClient
from ..report import Category, Dimension, Issue, Severity, Span, make_issue
from ..terminology import TermEntry, TerminologyStore
from ..text import levenshtein
from .base import BaseEngine, EngineResult

CUE_HINT = re.compile(r"(四个意识|四个自信|五位一体|四个全面|新发展理念|社会主义核心价值观)")


class TermEngine(BaseEngine):
    name = "term_engine"
    dimension = Dimension.POLITICAL
    description = "中央政策术语精准校验（正则 + 语义三级匹配）"

    def __init__(self, settings: Any, store: TerminologyStore,
                 embedding: Optional[EmbeddingClient] = None) -> None:
        super().__init__(settings, store)
        self.cfg = settings.engine.get("term_matching", {}) or {}
        self.sup = settings.engine.get("suppression", {}) or {}
        self.term_cfg = settings.terminology.get("update_policy", {}) or {}
        self.embedding = embedding or EmbeddingClient(
            (settings.llm.get("embedding") or {}), settings.llm
        )
        self.max_edit = int(self.cfg.get("max_edit_distance", 2))
        self.fuzzy_threshold = float(self.cfg.get("fuzzy_threshold", 0.86))
        self.semantic_threshold = float(self.cfg.get("semantic_threshold", 0.82))
        self._bigram_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._surface_to_entry: Dict[str, TermEntry] = {}
        self._last_standard_spans: List[Tuple[int, int]] = []
        self._last_standard_texts: Set[str] = set()
        self._standard_terms: Optional[List[str]] = None
        self._build_fuzzy_index()

    # 术语库可能在运行期热更新，索引需随之重建
    def rebuild_index(self) -> None:
        self._bigram_index.clear()
        self._surface_to_entry.clear()
        self._standard_terms = None
        self._build_fuzzy_index()

    # ==================================================================
    # 索引构建
    # ==================================================================
    def _all_surface_forms(self) -> List[Tuple[str, TermEntry]]:
        out: List[Tuple[str, TermEntry]] = []
        for e in self.store.entries:
            if e.term:
                out.append((e.term, e))
            for v in e.variants:
                out.append((v, e))
        return out

    def _build_fuzzy_index(self) -> None:
        """字符二元组倒排索引：bigram -> [(表面形式, 该 bigram 在其中的偏移)]。"""
        for surface, entry in self._all_surface_forms():
            self._surface_to_entry.setdefault(surface, entry)
            if len(surface) < 4:
                continue
            for i in range(len(surface) - 1):
                self._bigram_index[surface[i:i + 2]].append((surface, i))

    # ==================================================================
    # 主流程
    # ==================================================================
    def run(self, doc: Document) -> EngineResult:
        res = self.result(
            terminology_version=self.store.version,
            embedding_mode=getattr(self.embedding, "mode", "unknown"),
        )
        notes: List[str] = []

        l1 = self._level1_exact(doc)
        res.stats["L1_exact_hits"] = len(l1)

        l2 = self._level2_fuzzy(doc, {i.span.start for i in l1})
        res.stats["L2_fuzzy_hits"] = len(l2)

        l3 = self._level3_semantic(doc, [i.span for i in l1 + l2])
        res.stats["L3_semantic_hits"] = len(l3)
        res.stats["embedding_mode"] = getattr(self.embedding, "mode", "unknown")

        ordering = self._check_ordering(doc)
        res.stats["ordering_hits"] = len(ordering)

        banned = self._check_banned(doc)
        res.stats["banned_hits"] = len(banned)

        all_issues = l1 + l2 + l3 + ordering + banned
        kept, suppressed = self._suppress(doc, all_issues)
        res.stats["suppressed"] = suppressed
        res.issues = self.resolve_overlaps(kept)
        res.stats["final"] = len(res.issues)
        res.notes = notes
        return res

    # ==================================================================
    # L1 精确匹配
    # ==================================================================
    def _level1_exact(self, doc: Document) -> List[Issue]:
        if not self.cfg.get("exact", True):
            return []
        issues: List[Issue] = []
        text = doc.text
        standard_spans: List[Tuple[int, int]] = []

        for start, end, pattern in self.store.ac.iter_matches(text):
            info = self.store.ac.pattern_info.get(pattern) or {}
            entry: Optional[TermEntry] = info.get("entry")
            if entry is None:
                continue
            if info.get("role") == "standard":
                standard_spans.append((start, end))
                continue

            suggestion = entry.replaces or entry.term
            # 若错误变体本身就包含标准表述，避免产生"把正确的改错"的反向建议
            if entry.term and entry.term == pattern:
                continue

            kind = "提法演进" if entry.type == "deprecated" else "表述错误"
            issues.append(make_issue(
                category=Category.POLITICAL_TERM,
                severity=entry.severity,
                rule_id=entry.id,
                message=f"政治术语{kind}：「{pattern}」应为「{suggestion}」",
                span=Span(start, end),
                original=pattern,
                suggestion=suggestion,
                explain=entry.explain,
                source=entry.source,
                confidence=entry.confidence,
                agent="terminology",
                meta={"level": "L1", "term_id": entry.id, "tags": entry.tags},
            ))

        self._last_standard_spans = standard_spans
        self._last_standard_texts = {text[s:e] for s, e in standard_spans}
        return issues

    # ==================================================================
    # L2 近似匹配
    # ==================================================================
    def _level2_fuzzy(self, doc: Document, occupied: Set[int]) -> List[Issue]:
        if not self.cfg.get("fuzzy", True):
            return []
        text = doc.text
        issues: List[Issue] = []
        seen: Set[Tuple[int, int]] = set()
        if not self._bigram_index:
            return []

        # ① 倒排索引投票：只对"与某条术语共享二元组"的位置生成候选
        candidate_hits: Dict[int, Set[str]] = defaultdict(set)
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            bucket = self._bigram_index.get(bigram)
            if not bucket:
                continue
            for surface, off in bucket:
                start = i - off
                if 0 <= start and start + len(surface) <= len(text):
                    candidate_hits[start].add(surface)

        # ② 对候选做编辑距离验证
        for start, surfaces in candidate_hits.items():
            if start in occupied:
                continue
            for surface in surfaces:
                entry = self._surface_to_entry.get(surface)
                if entry is None:
                    continue
                window = text[start:start + len(surface)]
                if window == surface:
                    continue                       # 完全一致，交给 L1
                if window in self._last_standard_texts:
                    continue                       # 窗口是另一条标准表述
                # 关键误报防护：窗口内若包含任一完整标准表述，说明是"正确术语 + 后续文字"
                # 被当成近似错误，必须跳过（例如「习近平新时代中国特色社会主义思想为指导」
                # 与变体「……思想体系」仅尾部两字不同，属于典型的边界误报）。
                if self._contains_standard(window):
                    continue
                dist = levenshtein(window, surface, max_dist=self.max_edit)
                if dist == 0 or dist > self.max_edit:
                    continue
                ratio = 1.0 - dist / max(len(window), len(surface))
                if ratio < self.fuzzy_threshold:
                    continue
                # 过滤"标准表述被截断"造成的伪命中
                if surface.startswith(window) and len(window) < len(surface) - self.max_edit:
                    continue
                key = (start, start + len(window))
                if key in seen:
                    continue
                seen.add(key)
                filled = self._fill_sentence(doc, start, start + len(window), surface)
                issues.append(make_issue(
                    category=Category.POLITICAL_TERM,
                    severity=entry.severity if entry.severity != Severity.INFO else Severity.MINOR,
                    rule_id=f"{entry.id}-FUZZY",
                    message=f"疑似术语书写错误：「{window}」→「{surface}」",
                    span=Span(start, start + len(window)),
                    original=window,
                    suggestion=surface,
                    explain=f"与规范表述编辑距离 {dist}，相似度 {ratio:.2f}。{entry.explain}",
                    source=entry.source or "术语库近似匹配",
                    confidence=round(max(0.6, entry.confidence * ratio), 2),
                    agent="terminology",
                    meta={"level": "L2", "edit_distance": dist, "ratio": round(ratio, 3),
                          "term_id": entry.id, "sentence": filled[:120]},
                ))
        return issues

    def _entry_of(self, surface: str) -> Optional[TermEntry]:
        return self._surface_to_entry.get(surface)

    def _contains_standard(self, window: str) -> bool:
        """窗口中是否包含某条术语的标准表述（长度 ≥ 6，避免短词误伤）。"""
        if self._standard_terms is None:
            self._standard_terms = sorted(
                {e.term for e in self.store.entries if len(e.term) >= 6}, key=len, reverse=True)
        return any(t in window for t in self._standard_terms)

    @staticmethod
    def _fill_sentence(doc: Document, start: int, end: int, replacement: str) -> str:
        """把建议替换回原句，供报告展示"修改后效果"。"""
        sent = doc.sentence_of(start)
        if not sent:
            return ""
        # 找到句子在原稿中的起点，做局部替换
        offset = doc.text.find(sent, max(0, start - len(sent)), start + 1)
        if offset < 0:
            return ""
        rel_s, rel_e = start - offset, end - offset
        if 0 <= rel_s < rel_e <= len(sent):
            return sent[:rel_s] + replacement + sent[rel_e:]
        return ""

    # ==================================================================
    # L3 语义匹配
    # ==================================================================
    def _level3_semantic(self, doc: Document, occupied: List[Span]) -> List[Issue]:
        if not self.cfg.get("semantic", True):
            return []
        entries = self.store.semantic_entries()
        if not entries:
            return []

        seed_cache: Dict[str, List[float]] = {}
        for e in entries:
            for seed in e.semantic_seeds:
                seed_cache[f"{e.id}|{seed}"] = self.embedding.encode(seed)

        issues: List[Issue] = []
        min_chars = int(self.sup.get("min_context_chars", 12))

        for s_start, s_end, sent in doc.sentences:
            if len(sent) < min_chars or doc.in_reference_section(s_start):
                continue
            if any(not (s_end <= sp.start or sp.end <= s_start) for sp in occupied):
                continue
            sent_vec = self.embedding.encode(sent)
            for e in entries:
                best = 0.0
                best_seed = ""
                for seed in e.semantic_seeds:
                    cos = EmbeddingClient.cosine(sent_vec, seed_cache[f"{e.id}|{seed}"])
                    if cos > best:
                        best, best_seed = cos, seed
                if best < self.semantic_threshold:
                    continue
                span = self._best_span(s_start, sent, best_seed)
                issues.append(make_issue(
                    category=Category.POLITICAL_TERM,
                    severity=e.severity,
                    rule_id=f"{e.id}-SEM",
                    message=f"疑似不规范语义表达，建议规范为「{e.term}」",
                    span=span,
                    original=doc.text[span.start:span.end],
                    suggestion=e.term,
                    explain=f"与规则种子「{best_seed}」语义相似度 {best:.2f}。{e.explain}",
                    source=e.source or "语义匹配",
                    confidence=round(best, 3),
                    agent="terminology",
                    meta={"level": "L3", "similarity": round(best, 3),
                          "seed": best_seed, "embedding_mode": getattr(self.embedding, "mode", "")},
                ))
        return issues

    @staticmethod
    def _best_span(sent_start: int, sent: str, seed: str) -> Span:
        """在句内定位与种子最相似的片段，作为高亮区间。"""
        matcher = SequenceMatcher(None, sent, seed)
        blocks = matcher.get_matching_blocks()
        best = max(blocks, key=lambda b: b.size) if blocks else None
        if best and best.size >= 3:
            return Span(sent_start + best.a, sent_start + best.a + best.size)
        return Span(sent_start, sent_start + min(len(sent), max(6, len(seed))))

    # ==================================================================
    # 组合提法顺序校验
    # ==================================================================
    def _check_ordering(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for entry in self.store.ordering_entries():
            for m in re.finditer(entry.pattern, doc.text):
                seg = m.group(0)
                base = m.start()
                found: List[Tuple[int, str]] = []
                for item in entry.order:
                    idx = seg.find(item)
                    if idx >= 0:
                        found.append((idx, item))
                missing = [i for i in entry.order if i not in seg]
                present = [it for _, it in sorted(found)]

                if missing and len(present) >= 2:
                    issues.append(make_issue(
                        category=Category.POLITICAL_TERM,
                        severity=Severity.MAJOR,
                        rule_id=f"{entry.id}-ORDER-MISS",
                        message=f"「{entry.term}」表述不完整，缺少：{'、'.join(missing)}",
                        span=Span(base, base + len(seg)),
                        original=seg,
                        suggestion=f"{entry.term}：" + "、".join(entry.order),
                        explain=entry.explain,
                        source=entry.source,
                        confidence=0.9,
                        agent="terminology",
                        meta={"level": "ORDER", "missing": missing},
                    ))
                    continue

                if len(present) < 2:
                    continue
                expected = [i for i in entry.order if i in present]
                if present != expected:
                    first_bad = next(
                        (k for k, (a, b) in enumerate(zip(present, expected)) if a != b), 0
                    )
                    issues.append(make_issue(
                        category=Category.POLITICAL_TERM,
                        severity=entry.severity,
                        rule_id=f"{entry.id}-ORDER",
                        message=(f"「{entry.term}」内部顺序错误："
                                 f"第 {first_bad + 1} 项「{present[first_bad]}」"
                                 f"应为「{expected[first_bad]}」"),
                        span=Span(base, base + len(seg)),
                        original=seg,
                        suggestion=self._rebuild_order(seg, present, expected),
                        explain=entry.explain,
                        source=entry.source,
                        confidence=entry.confidence,
                        agent="terminology",
                        meta={"level": "ORDER", "present": present, "expected": expected},
                    ))
        return issues

    @staticmethod
    def _rebuild_order(seg: str, present: List[str], expected: List[str]) -> str:
        """给出重排后的建议文本。

        做法：定位所有已出现项在其所在连续区间内，用规范顺序整体重建该区间，
        避免逐项 str.replace 造成的连锁替换（例如"核心意识"既是待替换项
        又是其他项的组成部分时会被重复替换）。
        """
        hits = [(seg.find(it), it) for it in set(present)]
        hits = [(p, it) for p, it in hits if p >= 0]
        if not hits:
            return seg
        start = min(p for p, _ in hits)
        end = max(p + len(it) for p, it in hits)
        gap = seg[start:end]
        sep = "、" if "、" in gap else ("，" if "，" in gap else "、")
        return seg[:start] + sep.join(expected) + seg[end:]

    # ==================================================================
    # 禁用 / 慎用模式
    # ==================================================================
    def _check_banned(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for rule in self.store.banned:
            try:
                for m in re.finditer(rule.pattern, doc.text):
                    issues.append(make_issue(
                        category=Category.POLITICAL_TERM
                        if rule.severity == Severity.FATAL else Category.LANGUAGE_NORM,
                        severity=rule.severity,
                        rule_id=rule.id,
                        message=rule.message,
                        span=Span(m.start(), m.end()),
                        original=m.group(0),
                        suggestion=rule.suggestion,
                        explain=f"规则类别：{rule.category}",
                        source=rule.reference,
                        confidence=0.92,
                        agent="terminology",
                        meta={"level": "BANNED", "category": rule.category},
                    ))
            except re.error:
                continue
        return issues

    # ==================================================================
    # 误报压制
    # ==================================================================
    def _suppress(self, doc: Document, issues: List[Issue]) -> Tuple[List[Issue], int]:
        kept: List[Issue] = []
        suppressed = 0
        for issue in issues:
            if self.sup.get("quoted_text_exempt", True) \
                    and issue.meta.get("level") not in ("ORDER",) \
                    and doc.in_citation(issue.span.start):
                suppressed += 1
                continue
            if self.sup.get("negated_context_exempt", True) and doc.is_negated(issue.span.start):
                suppressed += 1
                continue
            if self.sup.get("reference_section_exempt", True) \
                    and doc.in_reference_section(issue.span.start):
                suppressed += 1
                continue
            # 标准表述保护：命中片段本身就是某条标准表述时降级为提示
            if issue.original and issue.original in getattr(self, "_last_standard_texts", set()) \
                    and issue.suggestion and issue.original != issue.suggestion:
                issue.severity = Severity.INFO
                issue.message = f"[需人工确认] 该处命中标准表述，请核对上下文：{issue.message}"
            kept.append(issue)
        return kept, suppressed
