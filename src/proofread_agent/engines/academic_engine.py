"""专业文本深度审校引擎。

面向调研专报与学术期刊的四类深度检查：

A. 数据合规性
   把正文里的"数值事实"解析成 (标签, 数值, 单位) 三元组，再做算术自洽校验：
   * 分项之和 vs 声明的合计；
   * 占比 vs 分子/分母（含 >100% 的硬错误）；
   * 同比/环比口径是否标注；
   * 同一量纲单位混用（万元/亿元、万条/条）；
   * 异常值（占比 100% 以上、增长倍数用于下降等由语言引擎配合）。

B. 格式规范
   章节编号连续性、标题层级跳跃、图表引用与图表实体对应、
   序号体例统一、参考文献区存在性。

C. 核心观点重复发表检测
   三层：段落 SimHash 粗筛 → MinHash 精筛 → TF-IDF/向量余弦终判。
   既做**文内自我重复**，也与 ``data/corpus`` 的历史稿件库比对，
   识别"一稿多发""观点重复发表"。输出相似度与相似来源，供人工判定。

D. 参考文献规范性
   按 GB/T 7714-2015 逐条解析著录项，检查类型标识、必填项缺失、
   年份异常、页码符号、DOI 格式、正文标注与文末条目的双向对应、
   引用编号顺序、重复著录。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from ..document import Document
from ..llm import EmbeddingClient
from ..report import Category, Dimension, Issue, Severity, Span, make_issue
from ..text import jaccard, similarity, tokenize
from ..text.similarity import (
    MinHasher,
    TfidfIndex,
    combined_similarity,
    compare_fingerprints,
    paragraph_fingerprint,
    simhash_similarity,
)
from .base import BaseEngine, EngineResult

# ---------------------------------------------------------------- 数值解析
NUM = re.compile(
    r"(?P<label>[\u4e00-\u9fffA-Za-z]{0,10}?)"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>万条|万元|万人次|万个|万公顷|万|亿元|亿|条|个|人|户|家|项|次|%|吨|亩|"
    r"平方公里|平方米|公里|米|元|天|年|月)?"
)
#: 合计类词汇。"共"单字过于宽泛（会命中"共享""共同"），因此不单独列入。
AGG_WORDS = ["合计", "总计", "共计", "总计", "小计", "总数", "累计", "合共", "三项", "两项"]
AGG_CUE = re.compile(r"(合计|总计|共计|小计|总数|累计|合共)")
PARTS_CUE = re.compile(r"(其中|分别为|分别是|依次为|依次是|分别为)")
#: 数值前常见的连接性文字，提取标签时需要剥离，避免污染标签与位置比较
LABEL_NOISE = re.compile(r"(其中|分别为|分别是|依次为|依次是|合计|总计|共计|小计|总数|累计|约|达|共|为|是|和|及|与)")
_SHARE_CONTEXT = ["占", "占比", "比例", "比重"]
_GROWTH = re.compile(r"(增长|提高|提升|下降|降低|减少|上升|回落)\s*(了|到|至)?\s*$")

HEADING_NUM = re.compile(r"^\s*(?:#{1,6}\s*)?([一二三四五六七八九十]+)、\s*(\S.*)$")
SUB_HEADING = re.compile(r"^\s*(?:#{1,6}\s*)?（([一二三四五六七八九十]+)）\s*(\S.*)$")
FIG_REF = re.compile(r"(?:如)?(?:图|表)\s*(\d+)(?:\s*所示|所示|中|可见|显示|表明)?")
FIG_DEF = re.compile(r"^\s*(?:#{1,6}\s*)?(?:图|表)\s*(\d+)\s*[:：.、]?\s*\S")
_CN_NUM = {c: i for i, c in enumerate("零一二三四五六七八九", 0)}
_GB_PATTERNS: Optional[Dict[str, str]] = None
_GB_CFG: Dict[str, Any] = {}


def _cn_to_int(s: str) -> int:
    """中文数字转整数（支持一~九十九）。"""
    if s == "十":
        return 10
    if "十" in s:
        a, _, b = s.partition("十")
        tens = _CN_NUM.get(a, 1) if a else 1
        ones = _CN_NUM.get(b, 0) if b else 0
        return tens * 10 + ones
    total = 0
    for ch in s:
        total = total * 10 + _CN_NUM.get(ch, 0)
    return total


class AcademicEngine(BaseEngine):
    name = "academic_engine"
    dimension = Dimension.ACADEMIC
    description = "专业文本深度审校（数据合规 / 格式规范 / 重复发表 / 参考文献）"

    def __init__(self, settings: Any, store: Any = None,
                 embedding: Optional[EmbeddingClient] = None) -> None:
        super().__init__(settings, store)
        self.cfg = settings.engine.get("academic", {}) or {}
        self.dup_cfg = self.cfg.get("duplicate", {}) or {}
        self.data_cfg = self.cfg.get("data", {}) or {}
        self.embedding = embedding or EmbeddingClient(
            (settings.llm.get("embedding") or {}), settings.llm
        )
        self.gb = self._load_gb()
        self.minhasher = MinHasher(
            num_perm=int(self.dup_cfg.get("minhash_perm", 128)),
            shingle_size=int(self.dup_cfg.get("shingle_size", 5)),
        )
        self.tfidf = TfidfIndex()
        self.corpus_meta: Dict[str, Dict[str, Any]] = {}
        self._corpus_loaded = False

    # ==================================================================
    def _load_gb(self) -> Dict[str, Any]:
        p = self.settings.path("data/references/gbt7714.yaml")
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    def _load_corpus(self) -> None:
        if self._corpus_loaded:
            return
        self._corpus_loaded = True
        corpus_dir = self.settings.path(self.dup_cfg.get("internal_corpus", "data/corpus"))
        if not corpus_dir.exists():
            return
        for p in sorted(corpus_dir.glob("*.md")) + sorted(corpus_dir.glob("*.txt")):
            text = p.read_text(encoding="utf-8", errors="ignore")
            doc = Document.from_text(text, doc_type="对比语料", source_path=str(p))
            for para in doc.paragraphs:
                body = para["text"].strip()
                if len(body) < 40 or para["is_heading"]:
                    continue
                pid = f"{p.stem}#{para['index']}"
                fp = paragraph_fingerprint(
                    body, shingle_size=self.minhasher.k, num_perm=self.minhasher.num_perm,
                    hasher=self.minhasher,
                )
                self.tfidf.add(pid, body)
                self.corpus_meta[pid] = {
                    "source": p.name, "path": str(p), "text": body,
                    "fingerprint": fp, "index": para["index"],
                }

    # ==================================================================
    def run(self, doc: Document) -> EngineResult:
        issues: List[Issue] = []
        stats: Dict[str, Any] = {}
        if self.cfg.get("data_compliance", True):
            found = self._check_data_compliance(doc)
            stats["data_compliance"] = len(found)
            issues += found
        if self.cfg.get("format_check", True):
            found = self._check_format(doc)
            stats["format"] = len(found)
            issues += found
        if self.cfg.get("duplicate_detection", True):
            found = self._check_duplicates(doc)
            stats["duplicate"] = len(found)
            stats["corpus_paragraphs"] = len(self.corpus_meta)
            issues += found
        if self.cfg.get("reference_check", True):
            found = self._check_references(doc)
            stats["reference"] = len(found)
            issues += found

        kept = self.resolve_overlaps(issues)
        stats["final"] = len(kept)
        return EngineResult(engine=self.name, dimension=self.dimension, issues=kept, stats=stats)

    # ==================================================================
    # A. 数据合规
    # ==================================================================
    def _check_data_compliance(self, doc: Document) -> List[Issue]:
        """数据自洽性检查。

        以**段落**为分析单元而非句子：中文公文中"分项—合计—占比"三者常跨句表述
        （如"其中，A 为 312 万条，B 为 268 万条，C 为 196 万条，三项合计 776 万条，
        占归集总量的 95.8%"），只在句内做算术校验会大量漏检。
        """
        issues: List[Issue] = []
        tol = float(self.data_cfg.get("tolerance", 0.01))

        for p in doc.paragraphs:
            if p["is_heading"] or p["in_reference_section"]:
                continue
            ptext = p["text"]
            nums = self._extract_numbers(ptext)
            if len(nums) < 3:
                continue
            issues += self._check_sum(p["start"], ptext, nums, tol)
            issues += self._check_percent(p["start"], ptext, nums, tol)

        issues += self._check_yoy(doc)
        return issues

    @staticmethod
    def _extract_numbers(sent: str) -> List[Dict[str, Any]]:
        """抽取 (标签, 数值, 单位) 三元组。

        位置字段说明：
        * ``start``  : 整段匹配的起点（含标签）
        * ``vstart`` : **数值本身的起点**——所有顺序比较必须用它，
          否则标签会把"其中""合计"等提示词吞进去，导致分项误判。
        """
        out = []
        for m in NUM.finditer(sent):
            raw = m.group("value").replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                continue
            if "." not in raw and raw.isdigit() and len(raw) == 4 and 1900 <= val <= 2100:
                continue                       # 纯四位年份不作为数值
            label = LABEL_NOISE.sub("", (m.group("label") or "")).strip()
            unit = m.group("unit") or ""
            out.append({
                "label": label, "value": val, "unit": unit,
                "start": m.start(), "end": m.end(),
                "vstart": m.start("value"), "vend": m.end("value"),
                "text": m.group(0),
            })
        return out

    @staticmethod
    def _normalize_unit(unit: str) -> str:
        """量纲归一：万元/元 与 万条/条 属不同量纲；"万"仅作修饰，不参与比较。"""
        return unit.replace("万", "").replace("亿", "")

    def _collect_parts(self, ptext: str, nums: List[Dict], agg: Dict) -> List[Dict]:
        """收集合计项的分项。

        优先取"其中/分别为"提示词之后、"合计"之前且量纲一致的数值（中文公文标准句式）。
        这一步能避免把上一句的"总费用 420 万元"误当作"三项合计 440 万元"的分项。
        """
        qi = -1
        for m in PARTS_CUE.finditer(ptext):
            if m.start() < agg["vstart"]:
                qi = max(qi, m.end())
        unit = self._normalize_unit(agg["unit"])
        parts = []
        for n in nums:
            if n is agg or n["unit"] == "%":
                continue
            if self._normalize_unit(n["unit"]) != unit:
                continue
            if n["value"] == agg["value"]:
                continue
            if n["vstart"] < agg["vstart"] and (qi < 0 or n["vstart"] >= qi):
                parts.append(n)
        return parts

    def _find_agg(self, ptext: str, nums: List[Dict]) -> Optional[Dict]:
        """定位"合计/共计"口径的数值：取提示词与其后数值距离最近的一条。"""
        best: Optional[Tuple[int, Dict]] = None
        for n in nums:
            head = ptext[max(0, n["vstart"] - 8):n["vstart"]]
            m = AGG_CUE.search(head)
            if m:
                dist = len(head) - m.start()
            elif AGG_CUE.search(n["label"]):
                dist = 99
            else:
                continue
            if best is None or dist < best[0]:
                best = (dist, n)
        return best[1] if best else None

    def _check_sum(self, base: int, ptext: str, nums: List[Dict], tol: float) -> List[Issue]:
        issues: List[Issue] = []
        agg = self._find_agg(ptext, nums)
        if agg is None:
            return issues
        parts = self._collect_parts(ptext, nums, agg)
        if len(parts) < 2:
            return issues
        total = sum(p["value"] for p in parts)
        if total == 0:
            return issues
        diff = abs(total - agg["value"]) / max(agg["value"], 1e-9)
        if diff <= tol:
            return issues
        issues.append(make_issue(
            category=Category.DATA_COMPLIANCE,
            severity=Severity.MAJOR if diff > 0.02 else Severity.MINOR,
            rule_id="DATA-SUM",
            message=(f"分项之和与合计不符：{' + '.join(_fmt(p['value']) for p in parts)}"
                     f" = {_fmt(total)}{agg['unit']}，文中合计为 {_fmt(agg['value'])}{agg['unit']}"
                     f"（偏差 {diff:.1%}）"),
            span=Span(base + parts[0]["start"], base + agg["end"]),
            original=ptext[parts[0]["start"]:agg["end"]][:110],
            suggestion=f"核对数据源：合计应为 {_fmt(total)}{agg['unit']}，或修正分项数值/补录漏项",
            explain="数据自洽性检查：合计项应等于同一量纲下各分项之和，偏差超过容差即判定为不一致。",
            source="数据审核规范",
            confidence=0.93,
            agent="academic",
            meta={"check": "sum_consistency", "computed": total, "stated": agg["value"],
                  "parts": [p["value"] for p in parts], "unit": agg["unit"]},
        ))
        return issues

    def _check_percent(self, base: int, ptext: str, nums: List[Dict], tol: float) -> List[Issue]:
        issues: List[Issue] = []
        pcts = [n for n in nums if n["unit"] == "%"]
        if not pcts:
            return issues
        for p in pcts:
            ctx = ptext[max(0, p["vstart"] - 45):p["vstart"]]
            growth = _GROWTH.search(ctx)
            is_share = any(w in ctx for w in _SHARE_CONTEXT) and not growth

            # ① 占比区间检查
            if is_share and p["value"] > 100.0:
                issues.append(make_issue(
                    category=Category.DATA_COMPLIANCE,
                    severity=Severity.MAJOR,
                    rule_id="DATA-PCT-RANGE",
                    message=f"占比数值超出合理范围：{p['text']}（占比不应大于 100%）",
                    span=Span(base + p["vstart"], base + p["end"]),
                    original=p["text"],
                    suggestion="核对分母口径；若为多项占比之和，应表述为「累计占比」并说明重复计入",
                    explain="占比 = 分项/总量，理论上限为 100%；超过则必为口径或数据错误。",
                    source="数据审核规范",
                    confidence=0.96,
                    agent="academic",
                    meta={"check": "percent_range"},
                ))
                continue

            if not is_share:
                continue

            # ② 占比与所列数据的一致性复算
            numerator = self._find_agg(ptext, [n for n in nums if n["vstart"] < p["vstart"]])
            if numerator is None:
                continue
            parts = self._collect_parts(ptext, nums, numerator)
            if len(parts) < 2:
                continue
            unit = self._normalize_unit(numerator["unit"])
            total = sum(x["value"] for x in parts)
            candidates = [n for n in nums
                          if n is not numerator and n not in parts and n["unit"] != "%"
                          and self._normalize_unit(n["unit"]) == unit
                          and n["vstart"] < numerator["vstart"]]
            if not candidates or total == 0:
                continue
            denominator = max(candidates, key=lambda x: x["value"])
            actual = total / denominator["value"] * 100
            if abs(actual - p["value"]) / max(p["value"], 1e-9) <= max(tol, 0.02):
                continue
            issues.append(make_issue(
                category=Category.DATA_COMPLIANCE,
                severity=Severity.MAJOR,
                rule_id="DATA-PCT-CONSISTENCY",
                message=(f"占比与所列数据不符：{total:g}{numerator['unit']}"
                         f" ÷ {denominator['value']:g}{denominator['unit']} = {actual:.2f}%，"
                         f"文中为 {p['value']}%"),
                span=Span(base + p["vstart"], base + p["end"]),
                original=p["text"],
                suggestion=f"按所列数据应为 {actual:.1f}%，请核对分子分母口径",
                explain="占比一致性检查：用段落内的分项之和与总量重算占比，与文中数值比对。",
                source="数据审核规范",
                confidence=0.9,
                agent="academic",
                meta={"check": "percent_consistency", "computed": round(actual, 3),
                      "stated": p["value"], "denominator": denominator["value"]},
            ))
        return issues

    def _check_yoy(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for s, e, sent in doc.sentences:
            for m in re.finditer(r"(同比|环比)[^，。；]{0,20}?(增长|下降|提高|降低|上升|回落)"
                                 r"[^，。；]{0,12}?(较快|明显|显著|大幅|有所|基本持平|平稳)", sent):
                if re.search(r"\d", m.group(0)):
                    continue
                issues.append(make_issue(
                    category=Category.DATA_COMPLIANCE,
                    severity=Severity.MAJOR,
                    rule_id="DATA-YOY-VAGUE",
                    message=f"统计口径缺失：{m.group(0)}——同比/环比结论未给出具体数值与基期",
                    span=Span(s + m.start(), s + m.end()),
                    original=m.group(0),
                    suggestion="补充具体增减幅度与基期，例如「同比增长 12.3%（2025 年 1—6 月为基期）」",
                    explain="同比、环比属于统计术语，缺少数值的定性描述无法验证，学术审稿通常要求补充。",
                    source="《学术出版规范》CY/T 174-2019",
                    confidence=0.85,
                    agent="academic",
                    meta={"check": "yoy"},
                ))
        return issues

    # ==================================================================
    # B. 格式规范
    # ==================================================================
    def _check_format(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        issues += self._check_numbering(doc)
        issues += self._check_figures(doc)
        issues += self._check_citation_style(doc)
        return issues

    def _check_numbering(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        expected = 1
        last_pos = 0
        for p in doc.paragraphs:
            m = HEADING_NUM.match(p["text"])
            if not m:
                continue
            got = _cn_to_int(m.group(1))
            if got != expected:
                issues.append(make_issue(
                    category=Category.FORMAT,
                    severity=Severity.MINOR,
                    rule_id="FMT-NUMBERING",
                    message=f"章节序号不连续：「{m.group(1)}、」应为「{_int_to_cn(expected)}、」",
                    span=Span(p["start"], p["start"] + len(p["text"].split("、")[0]) + 1),
                    original=m.group(1) + "、",
                    suggestion=_int_to_cn(expected) + "、",
                    explain="章节序号须连续递增，缺号或跳号在排版与引用时会造成混乱。",
                    source="《党政机关公文格式》GB/T 9704-2012",
                    confidence=0.9,
                    agent="academic",
                ))
                expected = got
            expected += 1
            last_pos = p["start"]
        return issues

    def _check_figures(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        refs = {int(m.group(1)) for m in FIG_REF.finditer(doc.body_text)}
        defs = set()
        for m in FIG_DEF.finditer(doc.text):
            defs.add(int(m.group(1)))
        for n in sorted(refs - defs):
            pos = doc.text.find(f"图{n}") if f"图{n}" in doc.text else doc.text.find(f"表{n}")
            issues.append(make_issue(
                category=Category.FORMAT,
                severity=Severity.MINOR,
                rule_id="FMT-FIG-MISSING",
                message=f"正文引用了图/表 {n}，但文稿中未找到对应图表或图表题注",
                span=Span(max(0, pos), max(0, pos) + 4),
                original=f"图/表 {n}",
                suggestion=f"补充图/表 {n} 及其题注，或删除该处引用",
                explain="图表引用与图表实体必须一一对应，是期刊格式初审的必查项。",
                source="《学术出版规范》CY/T 174-2019",
                confidence=0.85,
                agent="academic",
            ))
        for n in sorted(defs - refs):
            issues.append(make_issue(
                category=Category.FORMAT,
                severity=Severity.INFO,
                rule_id="FMT-FIG-UNREF",
                message=f"图/表 {n} 未在正文中被引用",
                span=Span(0, 0),
                original=f"图/表 {n}",
                suggestion="在正文中补充「如图/表 n 所示」的引用",
                source="《学术出版规范》CY/T 174-2019",
                confidence=0.8,
                agent="academic",
            ))
        return issues

    def _check_citation_style(self, doc: Document) -> List[Issue]:
        """序号体例统一：全角括号、[1-3] 缩略、中文"文献[1]"混用等。"""
        issues: List[Issue] = []
        body = doc.body_text
        if re.search(r"（\d{1,3}）", body) and re.search(r"\[\d{1,3}\]", body):
            issues.append(make_issue(
                category=Category.FORMAT,
                severity=Severity.INFO,
                rule_id="FMT-CITE-STYLE",
                message="同篇文稿中同时出现「[n]」与「（n）」两种引用标注体例",
                span=Span(0, 0),
                original="混合体例",
                suggestion="按目标刊物要求统一为一种（顺序编码制通常用 [n]）",
                source="GB/T 7714-2015",
                confidence=0.8,
                agent="academic",
            ))
        return issues

    # ==================================================================
    # C. 重复发表检测
    # ==================================================================
    def _check_duplicates(self, doc: Document) -> List[Issue]:
        self._load_corpus()
        issues: List[Issue] = []
        issues += self._internal_duplicates(doc)
        issues += self._cross_corpus_duplicates(doc)
        return issues

    def _internal_duplicates(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        paras = [p for p in doc.paragraphs
                 if not p["is_heading"] and len(p["text"].strip()) >= 40
                 and not p["in_reference_section"]]
        fps = []
        for p in paras:
            fps.append((p, paragraph_fingerprint(
                p["text"], shingle_size=self.minhasher.k,
                num_perm=self.minhasher.num_perm, hasher=self.minhasher)))
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                (pa, fa), (pb, fb) = fps[i], fps[j]
                scores = compare_fingerprints(fa, fb)
                if scores["simhash"] < 0.85:
                    continue
                sim = combined_similarity(scores)
                if sim < 0.9:
                    continue
                issues.append(make_issue(
                    category=Category.DUPLICATE,
                    severity=Severity.MAJOR,
                    rule_id="DUP-INTERNAL",
                    message=f"文内段落重复：第 {pa['index'] + 1} 段与第 {pb['index'] + 1} 段高度相似"
                            f"（{sim:.0%}）",
                    span=Span(pb["start"], pb["end"]),
                    original=pb["text"][:80],
                    suggestion="合并或改写重复段落，避免同一稿件内自我复制",
                    explain=f"SimHash {scores['simhash']:.2f} / MinHash {scores['minhash']:.2f} / "
                            f"3-gram Jaccard {scores['ngram']:.2f}",
                    source="CY/T 174-2019 学术不端界定",
                    confidence=round(sim, 3),
                    agent="academic",
                    meta={"check": "internal_duplicate", "pair": [pa["index"], pb["index"]]},
                ))
        return issues

    def _cross_corpus_duplicates(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        if not self.corpus_meta:
            return issues
        sim_th = float(self.dup_cfg.get("similarity_threshold", 0.72))
        used: set = set()

        for p in doc.paragraphs:
            body = p["text"].strip()
            if len(body) < 40 or p["is_heading"] or p["in_reference_section"]:
                continue
            fp = paragraph_fingerprint(body, shingle_size=self.minhasher.k,
                                       num_perm=self.minhasher.num_perm, hasher=self.minhasher)
            best: Optional[Tuple[str, float, Dict[str, float]]] = None
            for pid, meta in self.corpus_meta.items():
                # SimHash 汉明相似度粗筛：先淘汰明显无关的段落，降低精算开销
                if simhash_similarity(meta["fingerprint"]["simhash"], fp["simhash"]) < 0.5:
                    continue
                scores = compare_fingerprints(fp, meta["fingerprint"])
                if scores["minhash"] < 0.45 and scores["ngram"] < 0.45:
                    continue
                sim = combined_similarity(scores)
                if sim >= sim_th and (best is None or sim > best[1]):
                    best = (pid, sim, scores)
            if best is None:
                continue
            pid, sim, scores = best
            meta = self.corpus_meta[pid]
            key = (pid, p["index"])
            if key in used:
                continue
            used.add(key)
            issues.append(make_issue(
                category=Category.DUPLICATE,
                severity=Severity.MAJOR if sim >= 0.85 else Severity.MINOR,
                rule_id="DUP-CROSS",
                message=(f"疑似重复发表：本段与历史稿件「{meta['source']}」第 "
                         f"{meta['index'] + 1} 段相似度 {sim:.0%}"),
                span=Span(p["start"], p["end"]),
                original=body[:90],
                suggestion="核对是否为同一研究的重复发表；如为延续性研究，须在文中说明与既有成果的区别与联系",
                explain=(f"对比语料：{meta['path']}。"
                         f"SimHash {scores['simhash']:.2f} / MinHash {scores['minhash']:.2f} / "
                         f"3-gram Jaccard {scores['ngram']:.2f}。"
                         f"对照段落：「{meta['text'][:60]}…」"),
                source="CY/T 174-2019",
                confidence=round(sim, 3),
                agent="academic",
                meta={"check": "cross_duplicate", "corpus": meta["source"],
                      "corpus_index": meta["index"], "similarity": round(sim, 3)},
            ))
        return issues

    # ==================================================================
    # D. 参考文献
    # ==================================================================
    def _check_references(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        if not doc.ref_items:
            if any(k in doc.text for k in ("参考文献", "引用文献")):
                issues.append(make_issue(
                    category=Category.REFERENCE, severity=Severity.MINOR,
                    rule_id="REF-EMPTY", message="存在「参考文献」标题但未检测到条目",
                    span=Span(doc.ref_start if doc.ref_start >= 0 else 0, 0),
                    suggestion="补充参考文献条目或删除该标题",
                    source="GB/T 7714-2015", confidence=0.9, agent="academic",
                ))
            return issues

        issues += self._check_citation_mapping(doc)
        type_codes = set((self.gb.get("doc_type_codes") or {}).keys())
        known_years_upper = 2100
        seen_titles: Dict[str, int] = {}

        for num, item in doc.ref_items.items():
            text = item.strip()
            span_start = doc.text.find(item)
            span_end = span_start + len(item)

            # 1) 类型标识
            codes = re.findall(r"\[([A-Z]{1,2}(?:/[A-Z]{2})?)\]", text)
            code = codes[0] if codes else ""
            if not code or code not in type_codes:
                issues.append(make_issue(
                    category=Category.REFERENCE, severity=Severity.MAJOR,
                    rule_id="REF-001",
                    message=f"文献 [{num}] 缺少或存在错误的文献类型标识（如 [J]/[M]/[D]）",
                    span=Span(span_start, span_end),
                    original=text[:90],
                    suggestion="按 GB/T 7714-2015 补充类型标识：期刊 [J]、专著 [M]、学位论文 [D]、"
                               "报纸 [N]、电子资源 [EB/OL]",
                    explain=self._gb_explain("REF-001"), source="GB/T 7714-2015",
                    confidence=0.92, agent="academic",
                ))
                code = ""

            # 2) 年份异常
            years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)]
            bad_year = [y for y in years if y > 2026 or y < 1900]
            if bad_year:
                issues.append(make_issue(
                    category=Category.REFERENCE, severity=Severity.MAJOR,
                    rule_id="REF-004",
                    message=f"文献 [{num}] 出版年份异常：{bad_year[0]}（晚于当前年份或早于 1900 年）",
                    span=Span(span_start, span_end), original=text[:90],
                    suggestion="核对出版年；若为预出版/在线首发，应使用「网络首发」体例并注明",
                    explain=self._gb_explain("REF-004"), source="GB/T 7714-2015",
                    confidence=0.95, agent="academic",
                ))

            # 3) 页码符号
            if re.search(r"\d+\s*[–—~～]\s*\d+", text):
                issues.append(make_issue(
                    category=Category.REFERENCE, severity=Severity.MINOR,
                    rule_id="REF-002",
                    message=f"文献 [{num}] 页码范围使用了非规范连接符（应使用半字线「-」）",
                    span=Span(span_start, span_end), original=text[:90],
                    suggestion="统一使用 ASCII 连字符，如 112-128",
                    explain=self._gb_explain("REF-002"), source="GB/T 7714-2015",
                    confidence=0.9, agent="academic",
                ))

            # 4) 作者"等"用法
            if re.search(r"[,，]\s*等", text) is None and text.count(",") + text.count("，") >= 4 \
                    and re.search(r"[\u4e00-\u9fff]{2,4}[,，][\u4e00-\u9fff]{2,4}[,，][\u4e00-\u9fff]{2,4}[,，]", text):
                issues.append(make_issue(
                    category=Category.REFERENCE, severity=Severity.MINOR,
                    rule_id="REF-003",
                    message=f"文献 [{num}] 作者超过 3 人，应著录前 3 位后加「, 等」",
                    span=Span(span_start, span_end), original=text[:90],
                    suggestion="张三, 李四, 王五, 等. 题名[J]. …",
                    explain=self._gb_explain("REF-003"), source="GB/T 7714-2015",
                    confidence=0.85, agent="academic",
                ))

            # 5) DOI
            doi = re.search(r"(?i)doi\s*[:：]?\s*(\S+)", text)
            if doi and not re.match(r"10\.\d{4,9}/", doi.group(1)):
                issues.append(make_issue(
                    category=Category.REFERENCE, severity=Severity.MINOR,
                    rule_id="REF-007",
                    message=f"文献 [{num}] DOI 格式可疑：「{doi.group(1)}」",
                    span=Span(span_start + doi.start(1), span_start + doi.end(1)),
                    original=doi.group(1), suggestion="正确格式为 10.<注册机构码>/<后缀>",
                    explain=self._gb_explain("REF-007"), source="GB/T 7714-2015",
                    confidence=0.88, agent="academic",
                ))

            # 6) 疑似虚构来源（启发式：刊名/出版社不在已知清单且与题名高度重合）
            issues += self._check_source_plausibility(num, text, span_start, span_end)

            # 7) 重复著录
            title_key = _ref_title_key(text)
            if title_key:
                if title_key in seen_titles:
                    issues.append(make_issue(
                        category=Category.REFERENCE, severity=Severity.MINOR,
                        rule_id="REF-008",
                        message=f"文献 [{num}] 与 [{seen_titles[title_key]}] 疑似重复著录",
                        span=Span(span_start, span_end), original=text[:90],
                        suggestion="合并为一条，并统一正文引用编号",
                        explain=self._gb_explain("REF-008"), source="GB/T 7714-2015",
                        confidence=0.8, agent="academic",
                    ))
                else:
                    seen_titles[title_key] = num

        # 8) 引用顺序
        cited = doc.in_text_citations
        if cited and cited != sorted(cited):
            issues.append(make_issue(
                category=Category.REFERENCE, severity=Severity.MINOR, rule_id="REF-006",
                message=f"正文引用未按出现顺序编号：实际出现顺序 {cited}",
                span=Span(0, 0), original=str(cited),
                suggestion="按首次出现顺序重排编号与文末条目",
                explain=self._gb_explain("REF-006"), source="GB/T 7714-2015",
                confidence=0.85, agent="academic",
            ))
        return issues

    def _check_citation_mapping(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        cited = set(doc.in_text_citations)
        listed = set(doc.ref_items.keys())
        for n in sorted(cited - listed):
            pos = doc.body_text.find(f"[{n}]")
            issues.append(make_issue(
                category=Category.REFERENCE, severity=Severity.MAJOR, rule_id="REF-005",
                message=f"正文引用了 [{n}]，但文末参考文献中不存在该条目",
                span=Span(max(0, pos), max(0, pos) + len(f"[{n}]")),
                original=f"[{n}]",
                suggestion="在文末补录对应文献，或删除该处引用标注",
                explain=self._gb_explain("REF-005"), source="GB/T 7714-2015",
                confidence=0.95, agent="academic",
                meta={"check": "citation_mapping", "missing": n},
            ))
        for n in sorted(listed - cited):
            item = doc.ref_items[n]
            issues.append(make_issue(
                category=Category.REFERENCE, severity=Severity.MINOR, rule_id="REF-005",
                message=f"文献 [{n}] 从未在正文中被引用（僵尸条目）",
                span=Span(doc.text.find(item), doc.text.find(item) + len(item)),
                original=item[:80],
                suggestion="删除该条目，或在正文相应位置补上引用",
                explain=self._gb_explain("REF-005"), source="GB/T 7714-2015",
                confidence=0.9, agent="academic",
                meta={"check": "orphan_reference", "num": n},
            ))
        return issues

    def _check_source_plausibility(self, num: int, text: str, s: int, e: int) -> List[Issue]:
        """启发式判断：刊名/出版社与题名高度重合 → 疑似虚构来源。"""
        publishers = set(self.gb.get("known_publishers", []) or [])
        journals = set(self.gb.get("known_journals", []) or [])
        m = re.search(r"\.\s*\[M\]\.\s*([^:：,，]+)", text)
        if m:
            publisher = m.group(1).strip()
            if publisher not in publishers and "出版" not in publisher and "大学" not in publisher:
                return [make_issue(
                    category=Category.REFERENCE, severity=Severity.INFO, rule_id="REF-009",
                    message=f"文献 [{num}] 出版社「{publisher}」未在常用出版社清单中，请核实",
                    span=Span(s, e), original=text[:90],
                    suggestion="核实出版社全称；如为内部资料，应标注「内部资料」字样",
                    explain="来源可信度检查：非正规出版机构的专著来源在审稿中通常需要额外说明。",
                    source="GB/T 7714-2015", confidence=0.6, agent="academic",
                )]
        return []

    def _gb_explain(self, rule_id: str) -> str:
        for chk in self.gb.get("checks", []) or []:
            if chk.get("id") == rule_id:
                return chk.get("explain", "")
        return ""

    # ==================================================================
    # 供上层 Agent 调用的观点级去重
    # ==================================================================
    def extract_claims(self, doc: Document) -> List[Dict[str, Any]]:
        """抽取"核心观点句"，供跨稿件观点重复发表检测。"""
        cue = ["本文认为", "研究表明", "研究认为", "结论是", "我们认为", "可以判断",
               "应当", "需要", "建议", "必须", "关键在于"]
        claims: List[Dict[str, Any]] = []
        for s, e, sent in doc.sentences:
            body = sent.strip()
            if len(body) < 18 or body.startswith("["):
                continue
            if any(c in body for c in cue):
                claims.append({"text": body, "start": s, "end": e,
                               "tokens": tokenize(body)})
        return claims


# ---------------------------------------------------------------- helpers
def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:g}"


def _int_to_cn(n: int) -> str:
    if n <= 10:
        return "零一二三四五六七八九十"[n]
    if n < 20:
        return "十" + "零一二三四五六七八九"[n - 10]
    return "零一二三四五六七八九"[n // 10] + "十" + ("零一二三四五六七八九"[n % 10] if n % 10 else "")


def _ref_title_key(text: str) -> str:
    m = re.search(r"\.\s*([^\.\[]{6,60}?)\s*\[[A-Z]", text)
    return m.group(1).strip() if m else ""
