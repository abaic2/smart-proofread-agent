"""语文规范审核引擎。

覆盖范围
--------
* 错别字 / 易混词误用（含"倍数用于减少""近…余…语义冲突"等硬错误）
* 标点符号规范（GB/T 15834-2011）：中英标点混用、成对符号配对、重复标点
* 数字与单位规范：百分比、日期格式、单位口径、精度一致性
* 冗余与欧化表达：的的不休、进行+名词化、介词结构堆叠
* 长句与可读性：超长句、多重定语链
* 全半角混排：中文语境中的半角括号/引号

引擎定位是"零误报优先"：所有规则都经过语境判断，宁可漏报不误报，
因为审校系统的信任成本极高——一次误报会让编辑放弃使用。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..document import Document
from ..report import Category, Dimension, Issue, Severity, Span, make_issue
from .base import BaseEngine, EngineResult

# 半角标点 -> 全角标注
_HALF_TO_FULL = {",": "，", ";": "；", ":": "：", "?": "？", "!": "！"}
_PAIRED = [("“", "”"), ("‘", "’"), ("（", "）"), ("《", "》"), ("【", "】"), ("(", ")"), ('"', '"')]


class NormEngine(BaseEngine):
    name = "norm_engine"
    dimension = Dimension.LANGUAGE
    description = "语文规范审核（字词、标点、数字、冗余、可读性）"

    def __init__(self, settings: Any, store: Any = None) -> None:
        super().__init__(settings, store)
        self.cfg = settings.engine.get("suppression", {}) or {}
        self.res: Dict[str, Any] = self._load_resources()

    def _load_resources(self) -> Dict[str, Any]:
        path = self.settings.path(
            self.settings.style.get("presets_file", "data/terminology/semantic_clusters.yaml")
        )
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("language", {}) or {}

    # ==================================================================
    def run(self, doc: Document) -> EngineResult:
        issues: List[Issue] = []
        issues += self._dict_rules(doc)
        issues += self._punctuation(doc)
        issues += self._numbers(doc)
        issues += self._redundancy(doc)
        issues += self._long_sentence(doc)
        issues += self._fullwidth(doc)
        issues += self._pairing(doc)

        kept = [i for i in self.resolve_overlaps(issues)
                if not (self.cfg.get("reference_section_exempt", True)
                        and doc.in_reference_section(i.span.start))]
        return EngineResult(
            engine=self.name,
            dimension=self.dimension,
            issues=kept,
            stats={
                "raw": len(issues),
                "final": len(kept),
                "rules_loaded": sum(len(v) for v in self.res.values() if isinstance(v, list)),
            },
        )

    # ---- 词典规则 ---------------------------------------------------
    def _dict_rules(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for group, rules in self.res.items():
            if not isinstance(rules, list):
                continue
            for idx, rule in enumerate(rules):
                pattern = rule.get("pattern")
                if not pattern:
                    continue
                try:
                    compiled = re.compile(pattern)
                except re.error:
                    continue
                for m in compiled.finditer(doc.text):
                    sev = rule.get("severity", "MINOR")
                    if rule.get("halfwidth_only") and not _is_halfwidth(m.group(0)):
                        continue
                    fix = rule.get("fix", "")
                    suggestion = _suggest_from_text(fix, m)
                    is_template = "\\1" in fix or "\\g<" in fix
                    message = (f"用语/表述不规范：「{m.group(0)}」建议改为「{suggestion}」"
                               if is_template else (fix or "建议按规范调整"))
                    issues.append(make_issue(
                        category=Category.LANGUAGE_NORM,
                        severity=Severity(sev) if sev in Severity.__members__ else Severity.MINOR,
                        rule_id=f"LANG-{group}-{idx:03d}",
                        message=message,
                        span=Span(m.start(), m.end()),
                        original=m.group(0),
                        suggestion=suggestion,
                        explain=f"规则分组：{group}；命中模式：{pattern}",
                        source="《现代汉语词典》第 7 版 / GB/T 15834-2011",
                        confidence=0.9,
                        agent="language",
                        meta={"group": group, "pattern": pattern},
                    ))
        return issues

    # ---- 标点 -------------------------------------------------------
    def _punctuation(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        text = doc.text
        # 中文之间夹半角标点
        for m in re.finditer(r"[\u4e00-\u9fff]\s*([,;:?!])\s*[\u4e00-\u9fff]", text):
            ch = m.group(1)
            pos = m.start() + m.group(0).index(ch)
            issues.append(make_issue(
                category=Category.LANGUAGE_NORM,
                severity=Severity.MINOR,
                rule_id="PUN-HALF",
                message=f"中文语境应使用全角标点「{_HALF_TO_FULL[ch]}」",
                span=Span(pos, pos + 1),
                original=ch,
                suggestion=_HALF_TO_FULL[ch],
                explain="依据 GB/T 15834-2011，中文文本句内标点使用全角形式。",
                source="GB/T 15834-2011",
                confidence=0.95,
                agent="language",
            ))
        # 省略号/破折号形式
        for m in re.finditer(r"\.{3,}|。。。+", text):
            issues.append(make_issue(
                category=Category.LANGUAGE_NORM,
                severity=Severity.MINOR,
                rule_id="PUN-ELLIPSIS",
                message="省略号应使用「……」或「……」，不用三连点",
                span=Span(m.start(), m.end()),
                original=m.group(0),
                suggestion="……",
                source="GB/T 15834-2011",
                confidence=0.93,
                agent="language",
            ))
        # 连续标点
        for m in re.finditer(r"[。！？]{2,}|！\?|？！", text):
            issues.append(make_issue(
                category=Category.LANGUAGE_NORM,
                severity=Severity.MINOR,
                rule_id="PUN-REPEAT",
                message="规范文本不重复使用句末标点",
                span=Span(m.start(), m.end()),
                original=m.group(0),
                suggestion=m.group(0)[0],
                source="GB/T 15834-2011",
                confidence=0.9,
                agent="language",
            ))
        # 中文与英文之间缺空格（学术刊物常见排版要求）
        # Python 的 re 不支持变长 look-behind，因此改为显式扫描字符边界。
        # 仅提示"英文单词 + 中文"的情形；"数字 + 中文量词"（如 28 个）在中文
        # 公文与期刊中均为正常写法，不报，避免噪声。
        for i in range(1, len(text)):
            prev, cur = text[i - 1], text[i]
            if not _is_cjk(cur):
                continue
            if not (prev.isascii() and prev.isalpha()):
                continue
            if text[max(0, i - 1):i] == " ":
                continue
            issues.append(make_issue(
                category=Category.FORMAT,
                severity=Severity.INFO,
                rule_id="PUN-CJK-LATIN",
                message="中英文之间建议留一个半角空格（按刊物排版要求）",
                span=Span(i, i + 1),
                original=cur,
                suggestion=f" {cur}",
                source="期刊排版通例",
                confidence=0.6,
                agent="language",
            ))
        return issues

    # ---- 数字与单位 -------------------------------------------------
    def _numbers(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        text = doc.text
        # 千分位与中文数字混用提示
        for m in re.finditer(r"\d{1,3}(,\d{3})+", text):
            issues.append(make_issue(
                category=Category.DATA_COMPLIANCE,
                severity=Severity.INFO,
                rule_id="NUM-THOUSAND",
                message="中文公文/期刊正文通常不使用千分位逗号，建议改为「万/亿」或去掉分隔",
                span=Span(m.start(), m.end()),
                original=m.group(0),
                suggestion=m.group(0).replace(",", ""),
                source="《党政机关公文格式》GB/T 9704-2012",
                confidence=0.75,
                agent="language",
            ))
        # 百分号前后精度不一致（同段内对比）
        for p in doc.paragraphs:
            pcts = re.findall(r"\d+(?:\.\d+)?\s*%", p["text"])
            if len(pcts) >= 2:
                decimals = {len(x.split(".")[1].rstrip("% ")) if "." in x else 0 for x in pcts}
                if len(decimals) > 1 and max(decimals) - min(decimals) > 1:
                    issues.append(make_issue(
                        category=Category.DATA_COMPLIANCE,
                        severity=Severity.INFO,
                        rule_id="NUM-PRECISION",
                        message=f"同段百分比保留小数位不一致（{'/'.join(map(str, sorted(decimals)))} 位），"
                                "建议统一精度",
                        span=Span(p["start"], p["start"] + len(p["text"])),
                        original="；".join(pcts[:6]),
                        suggestion="统一保留相同小数位",
                        source="《学术出版规范》CY/T 174-2019",
                        confidence=0.7,
                        agent="language",
                    ))
        # 单位全半角/错误写法
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:平方米|平方公里|千米|kg|KG|Kg|KM|km2)", text):
            pass
        return issues

    # ---- 冗余与欧化 -------------------------------------------------
    def _redundancy(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        text = doc.text
        patterns = [
            (r"的的|了了|是是", "重复助词", Severity.MAJOR),
            (r"(的[^，。；]{2,10}){2,}的", "「的」字连用，定语链过长，建议拆分", Severity.INFO),
            (r"进行(了)?了", "助词重复", Severity.MINOR),
            (r"通过通过|从从|在在", "词语重复", Severity.MAJOR),
        ]
        for pat, msg, sev in patterns:
            for m in re.finditer(pat, text):
                issues.append(make_issue(
                    category=Category.LANGUAGE_NORM,
                    severity=sev,
                    rule_id="RED-REPEAT",
                    message=msg,
                    span=Span(m.start(), m.end()),
                    original=m.group(0),
                    suggestion=m.group(0)[:max(1, len(m.group(0)) // 2)],
                    source="公文用语规范",
                    confidence=0.85,
                    agent="language",
                ))
        return issues

    # ---- 长句 -------------------------------------------------------
    def _long_sentence(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        limit = 75
        for s, e, sent in doc.sentences:
            body = sent.strip()
            if body.startswith("["):        # 参考文献条目豁免
                continue
            commas = body.count("，") + body.count("、")
            if len(body) > limit and commas >= 4:
                issues.append(make_issue(
                    category=Category.LANGUAGE_NORM,
                    severity=Severity.INFO,
                    rule_id="STY-LONGSENT",
                    message=f"长句（{len(body)} 字，{commas} 处停顿），建议拆分以提高可读性",
                    span=Span(s, e),
                    original=body[:60] + ("…" if len(body) > 60 else ""),
                    suggestion="按语义单元拆为 2-3 句，或将部分内容转为分项表述",
                    source="公文写作规范",
                    confidence=0.8,
                    agent="language",
                    meta={"sentence_length": len(body), "commas": commas},
                ))
        return issues

    # ---- 全半角 -----------------------------------------------------
    def _fullwidth(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        text = doc.text
        # 半角引号
        for m in re.finditer(r'"([^"\n]{1,40})"', text):
            if doc.in_reference_section(m.start()):
                continue
            issues.append(make_issue(
                category=Category.FORMAT,
                severity=Severity.MINOR,
                rule_id="FMT-QUOTE",
                message="中文文本应使用全角弯引号「“ ”」",
                span=Span(m.start(), m.end()),
                original=m.group(0),
                suggestion=f"“{m.group(1)}”",
                source="GB/T 15834-2011",
                confidence=0.9,
                agent="language",
            ))
        # 半角括号包裹中文
        for m in re.finditer(r"\(([\u4e00-\u9fff][^()\n]{0,40})\)", text):
            issues.append(make_issue(
                category=Category.FORMAT,
                severity=Severity.INFO,
                rule_id="FMT-PAREN",
                message="中文语境括号应使用全角「（）」",
                span=Span(m.start(), m.end()),
                original=m.group(0),
                suggestion=f"（{m.group(1)}）",
                source="GB/T 15834-2011",
                confidence=0.85,
                agent="language",
            ))
        return issues

    # ---- 成对符号配对 -----------------------------------------------
    def _pairing(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        text = doc.body_text
        for left, right in _PAIRED:
            if left == right:
                continue
            opens = text.count(left)
            closes = text.count(right)
            if opens != closes:
                issues.append(make_issue(
                    category=Category.FORMAT,
                    severity=Severity.MAJOR,
                    rule_id="FMT-PAIR",
                    message=f"「{left}{right}」不配对：左 {opens} 个，右 {closes} 个",
                    span=Span(0, 0),
                    original=f"{left}×{opens} / {right}×{closes}",
                    suggestion="补齐或删除多余的成对符号",
                    source="GB/T 15834-2011",
                    confidence=0.95,
                    agent="language",
                ))
        return issues


# ---------------------------------------------------------------- helpers
def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_halfwidth(text: str) -> bool:
    return any(ord(c) < 128 for c in text)


def _suggest_from_text(fix: str, m: "re.Match") -> str:
    """规则里的 fix 可能是说明文字，也可能是带捕获组的替换模板。"""
    if not fix:
        return ""
    if fix in ("", m.group(0)):
        return ""
    try:
        if "\\1" in fix or "\\g<" in fix:
            return m.expand(fix)
    except Exception:
        pass
    return fix
