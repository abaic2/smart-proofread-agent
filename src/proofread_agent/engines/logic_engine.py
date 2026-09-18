"""逻辑严密性评估引擎。

把"逻辑"拆成可机判的六类缺陷，每一类都有明确的检测信号与举证要求：

1. **以偏概全 / 推论越界**：局部样本（某县、12 个乡镇）推出全域结论
   （全国、普遍、全面建成）。信号 = 因果连接词 + 全域词 + 上下文局部范围词。
2. **因果跳步**：出现"因此/由此可见"等结论标记，但前文既无数据也无引证。
3. **绝对化断言**：众所周知、显而易见、必然、完全等词自带举证义务，
   在无数据句中出现即报。
4. **强度不一致**：同一句内同时使用强断言与模糊限定（"必然……可能"），
   或数量词（全部/大多数/个别）与给出的占比互相矛盾。
5. **指代不明**：段落首句以"该做法/这一情况/其"等指代开篇，而上文无相应先行语。
6. **论证要素缺失**：段内有论点标记（本文认为/结论是）但整段无证据标记。

引擎输出统一附带 ``reason`` 字段，说明"为什么这是逻辑问题"，
使编辑可以判断是否接受，而不是面对一个黑盒告警。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from ..document import Document
from ..report import Category, Dimension, Issue, Severity, Span, make_issue
from .base import BaseEngine, EngineResult

# 全域性结论词
_UNIVERSAL = ["全国", "全世界", "普遍", "全部", "所有地区", "各地普遍", "全面建成",
              "整体上已", "无一例外", "放之四海"]
# 局部性范围词
_LOCAL = ["本县", "某县", "该县", "我县", "本区", "样本", "抽样", "个别乡镇", "部分乡镇",
          "调研的", "实地调研", "走访", "试点", "本省", "个案"]
# 结论连接词
_CONCLUDE = ["因此", "所以", "由此可见", "这说明", "这表明", "可见", "足以说明", "由此可见"]
# 数据/证据信号
_EVIDENCE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|万|亿|人|户|家|个|项|条|次|元)|据[^，。]{0,10}统计|调查(显示|发现)|数据显示|样本|案例|如表|见图|见表|图\s*\d+")
_BUT = ["但是", "然而", "不过", "但"]


class LogicEngine(BaseEngine):
    name = "logic_engine"
    dimension = Dimension.LOGIC
    description = "逻辑严密性评估（推论越界/因果跳步/绝对化/强度不一致/指代不明）"

    def __init__(self, settings: Any, store: Any = None) -> None:
        super().__init__(settings, store)
        self.cfg = settings.engine.get("logic", {}) or {}
        res = self._load()
        self.absolute = res.get("absolute_claims", []) or []
        self.hedges = res.get("hedges", []) or []
        self.causal = res.get("causal_connectives", []) or []
        self.contrast = res.get("contrast_connectives", []) or []
        self.quantity_map: Dict[str, Dict] = res.get("quantity_claim_map", {}) or {}
        self.vague = res.get("vague_reference", []) or []
        self.markers = res.get("argument_markers", {}) or {}

    def _load(self) -> Dict[str, Any]:
        path = self.settings.path(
            self.settings.style.get("presets_file", "data/terminology/semantic_clusters.yaml")
        )
        if not path.exists():
            return {}
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("logic", {}) or {}

    # ==================================================================
    def run(self, doc: Document) -> EngineResult:
        issues: List[Issue] = []
        if self.cfg.get("check_claim_evidence", True):
            issues += self._overgeneralization(doc)
            issues += self._claim_without_evidence(doc)
        if self.cfg.get("check_causal_leap", True):
            issues += self._causal_leap(doc)
        issues += self._absolute_claims(doc)
        issues += self._intensity_conflict(doc)
        if self.cfg.get("check_vague_reference", True):
            issues += self._vague_reference(doc)
        if self.cfg.get("check_quantifier", True):
            issues += self._quantifier_conflict(doc)

        kept = self.resolve_overlaps(issues)
        return EngineResult(
            engine=self.name, dimension=self.dimension, issues=kept,
            stats={"raw": len(issues), "final": len(kept),
                   "absolute_terms": len(self.absolute), "checked_sentences": len(doc.sentences)},
        )

    # ---- 1. 以偏概全 -------------------------------------------------
    def _overgeneralization(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        sents = doc.sentences
        for idx, (s, e, sent) in enumerate(sents):
            if not any(c in sent for c in _CONCLUDE + ["说明", "表明", "标志"]):
                continue
            uni = [u for u in _UNIVERSAL if u in sent]
            if not uni:
                continue
            window = "".join(x[2] for x in sents[max(0, idx - 3):idx + 1])
            local = [l for l in _LOCAL if l in window]
            if not local:
                continue
            issues.append(make_issue(
                category=Category.LOGIC,
                severity=Severity.MAJOR,
                rule_id="LOGIC-OVERGEN",
                message=f"推论越界：以局部样本（{'、'.join(local[:2])}）推及全域结论（{'、'.join(uni[:2])}）",
                span=Span(s, e),
                original=sent.strip(),
                suggestion=("限定结论范围，例如改为："
                            "「调研范围内的基层数字治理已进入精细化管理阶段」；"
                            "若确需全域结论，须补充全国性样本与来源"),
                explain=("逻辑缺陷：样本代表性不足（以偏概全）。"
                         f"上文出现局部范围限定词 {local[:3]}，本句使用全域结论词 {uni[:3]}，"
                         "两者覆盖范围不匹配。"),
                source="论证逻辑规范",
                confidence=0.88,
                agent="logic",
                meta={"defect": "overgeneralization", "local_scope": local[:3], "universal": uni[:3]},
            ))
        return issues

    # ---- 2. 结论无据 -------------------------------------------------
    def _claim_without_evidence(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        claims = self.markers.get("claim", []) or []
        for p in doc.paragraphs:
            if p["is_heading"] or len(p["text"]) < 30:
                continue
            has_claim = any(c in p["text"] for c in claims)
            if not has_claim:
                continue
            if _EVIDENCE.search(p["text"]) or any(
                w in p["text"] for w in ("根据", "依据", "参照", "数据显示", "文献")
            ):
                continue
            issues.append(make_issue(
                category=Category.LOGIC,
                severity=Severity.MINOR,
                rule_id="LOGIC-NO-EVIDENCE",
                message="存在明确论点但全段缺少数据、案例或文献支撑",
                span=Span(p["start"], p["end"]),
                original=p["text"][:70] + ("…" if len(p["text"]) > 70 else ""),
                suggestion="补充数据来源（机构+报告名+时间）或案例，或将判断降级为「初步观察」",
                explain="论证链不完整：claim 已给出，缺少 evidence 与 warrant 环节。",
                source="论证逻辑规范",
                confidence=0.75,
                agent="logic",
                meta={"defect": "missing_evidence", "paragraph": p["index"]},
            ))
        return issues

    # ---- 3. 因果跳步 -------------------------------------------------
    def _causal_leap(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        sents = doc.sentences
        for idx, (s, e, sent) in enumerate(sents):
            if not any(c in sent for c in self.causal):
                continue
            if _EVIDENCE.search(sent):
                continue
            prev = "".join(x[2] for x in sents[max(0, idx - 2):idx])
            if _EVIDENCE.search(prev) or len(prev.strip()) < 10:
                continue
            issues.append(make_issue(
                category=Category.LOGIC,
                severity=Severity.MINOR,
                rule_id="LOGIC-CAUSAL-LEAP",
                message="因果推理缺少中间环节：使用结论性连接词，但前文无数据或事实支撑",
                span=Span(s, e),
                original=sent.strip(),
                suggestion="补充支撑该因果判断的事实/数据，或改为「这可能与……有关」等留有余地的表述",
                explain=f"检测到因果连接词 {[c for c in self.causal if c in sent][:2]}，"
                        "但上下文未发现可验证的论据。",
                source="论证逻辑规范",
                confidence=0.72,
                agent="logic",
                meta={"defect": "causal_leap"},
            ))
        return issues

    # ---- 4. 绝对化断言 -----------------------------------------------
    def _absolute_claims(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for s, e, sent in doc.sentences:
            hits = [w for w in self.absolute if w in sent]
            if not hits:
                continue
            has_evidence = bool(_EVIDENCE.search(sent))
            if has_evidence and len(hits) <= 1:
                continue
            severity = Severity.MAJOR if len(hits) >= 2 or not has_evidence else Severity.MINOR
            span_start = min(s + sent.find(w) for w in hits)
            span_end = max(s + sent.find(w) + len(w) for w in hits)
            issues.append(make_issue(
                category=Category.LOGIC,
                severity=severity,
                rule_id="LOGIC-ABSOLUTE",
                message=f"绝对化断言「{'、'.join(hits[:3])}」缺少充分举证，易被质疑严谨性",
                span=Span(span_start, span_end),
                original=sent.strip()[:80],
                suggestion="改为可度量、可验证的表述（如「在调研样本中」「显著」），或补充统计依据",
                explain="绝对化表述在审稿与舆情复核中是高风险点：一旦存在反例即构成事实性瑕疵。",
                source="学术写作规范 / 公文用语规范",
                confidence=0.8,
                agent="logic",
                meta={"defect": "absolute_claim", "terms": hits[:5], "has_evidence": has_evidence},
            ))
        return issues

    # ---- 5. 强度不一致 -----------------------------------------------
    def _intensity_conflict(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for s, e, sent in doc.sentences:
            strong = [w for w in self.absolute if w in sent]
            weak = [w for w in self.hedges if w in sent]
            if strong and weak:
                issues.append(make_issue(
                    category=Category.LOGIC,
                    severity=Severity.MINOR,
                    rule_id="LOGIC-INTENSITY",
                    message=f"同一句内断言强度冲突：强断言「{strong[0]}」与限定语「{weak[0]}」并用",
                    span=Span(s, e),
                    original=sent.strip()[:80],
                    suggestion="统一论证强度：有依据则用强断言并标注来源，否则改用限定语",
                    explain="论证强度不一致会让读者无法判断作者的真实确信程度。",
                    source="论证逻辑规范",
                    confidence=0.7,
                    agent="logic",
                    meta={"defect": "intensity_conflict"},
                ))
        return issues

    # ---- 6. 指代不明 -------------------------------------------------
    def _vague_reference(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        sents = doc.sentences
        for idx, (s, e, sent) in enumerate(sents):
            stripped = sent.strip().lstrip("#* ")
            for word in self.vague:
                if not stripped.startswith(word):
                    continue
                # 先行语检测：指代词的"中心语"应在上文出现过（如前文出现"……做法"，
                # 则"该做法"有先行语；若上文只有"……现象"，则"该做法"属指代不明）。
                head = _refer_head(word)
                prev = "".join(x[2] for x in sents[max(0, idx - 3):idx])
                if head and re.search(r"[\u4e00-\u9fff]{1,6}" + re.escape(head), prev):
                    continue
                if not head and len(prev.strip()) > 12:
                    continue
                issues.append(make_issue(
                    category=Category.LOGIC,
                    severity=Severity.MINOR,
                    rule_id="LOGIC-VAGUE-REF",
                    message=f"指代不明：以「{word}」开篇，但上文未出现与之对应的先行语"
                            f"（缺少「…{head or '所指对象'}」类表述）",
                    span=Span(s + sent.find(word), s + sent.find(word) + len(word)),
                    original=sent.strip()[:80],
                    suggestion=f"将「{word}」替换为具体所指对象（如「数据重复录入问题」）",
                    explain="指代对象的缺位会使论证链断裂，是审稿人高频指出的问题。",
                    source="学术写作规范",
                    confidence=0.72,
                    agent="logic",
                    meta={"defect": "vague_reference", "word": word, "head": head},
                ))
                break
        return issues

    # ---- 7. 数量词与数据矛盾 ------------------------------------------
    def _quantifier_conflict(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for s, e, sent in doc.sentences:
            for word, constraint in self.quantity_map.items():
                if word not in sent:
                    continue
                pcts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", sent)]
                if not pcts:
                    continue
                val = max(pcts) / 100.0
                lo = constraint.get("min_ratio")
                hi = constraint.get("max_ratio")
                bad = (lo is not None and val < lo) or (hi is not None and val > hi)
                if not bad:
                    continue
                issues.append(make_issue(
                    category=Category.LOGIC,
                    severity=Severity.MAJOR,
                    rule_id="LOGIC-QUANTIFIER",
                    message=f"数量表述与数据不符：「{word}」对应占比约 {val:.0%}，"
                            f"{'低于' if lo is not None and val < lo else '高于'}该词的可接受区间",
                    span=Span(s, e),
                    original=sent.strip()[:90],
                    suggestion=constraint.get("advice", "调整数量表述或核对数据"),
                    explain="数量词蕴含份额约束，与文中百分比冲突时必有一处错误。",
                    source="论证逻辑规范",
                    confidence=0.82,
                    agent="logic",
                    meta={"defect": "quantifier_conflict", "value": round(val, 4), "word": word},
                ))
        return issues


_HEAD_NOUNS = ("做法", "情况", "现象", "问题", "举措", "机制", "模式", "经验", "矛盾")


def _refer_head(word: str) -> str:
    """从指代词中提取中心语，用于先行语匹配（如「该做法」→「做法」）。"""
    for h in _HEAD_NOUNS:
        if word.endswith(h) or h in word:
            return h
    return ""


def _core_nouns(text: str) -> Set[str]:
    """抽取中文 2-4 字名词性片段（极简），用于判断先行语是否存在。"""
    skip = {"我们", "他们", "可以", "已经", "因此", "但是", "不过", "同时", "进行",
            "通过", "并且", "以及", "由于", "如果", "虽然", "这样", "他们"}
    out: Set[str] = set()
    for m in re.finditer(r"[\u4e00-\u9fff]{2,4}(?:问题|情况|现象|做法|举措|机制|平台|数据|系统|工作)", text):
        out.add(m.group(0))
    for m in re.finditer(r"[\u4e00-\u9fff]{2,4}", text):
        w = m.group(0)
        if w not in skip:
            out.add(w)
    return out
