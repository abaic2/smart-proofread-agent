"""风格迁移与一致性引擎。

不是"把文章改得像某种风格"，而是**量化差距 + 给出可执行的迁移动作**——
这是审校场景的正确姿势：编辑需要知道差在哪里、改哪一句，而不是拿到一篇
被模型重写过的稿子（那会引入新的政治与事实风险）。

能力
----
1. **风格画像**：平均句长、口语词密度、绝对化词密度、数据标注率、
   被动/欧化结构占比、段旨句命中率。
2. **差距诊断**：与目标文体（公文专报 / 学术期刊 / 新闻通稿）的目标区间比对，
   输出偏差项与偏差幅度。
3. **迁移清单**：do/avoid 词表命中 → 逐条给出替换建议；
   长句 → 给出拆分方案；缺段旨句 → 给出补写模板。
4. **风格评分**：0-100，用于报告中的"风格一致性"维度。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..document import Document
from ..report import Category, Dimension, Issue, Severity, Span, make_issue
from ..text import split_sentences
from .base import BaseEngine, EngineResult

# 口语/非正式表达
_COLLOQUIAL = ["我觉得", "说白了", "搞", "挺好的", "大概可能", "网上说", "据说", "呀", "嘛",
               "咱们", "挺", "特别特别", "超级", "搞定", "弄", "搞活", "抓瞎"]
_ABSOLUTE = ["毫无疑问", "众所周知", "显然", "必然", "极大地", "非常好的", "完全", "绝对",
             "史上最", "前所未有", "一劳永逸", "彻底解决"]
_EUROPEAN = [r"对[^，。]{2,12}进行[了]?[^，。]{2,10}", r"关于[^，。]{2,15}的[^，。]{2,10}的",
             r"在[^，。]{2,12}上[，,]?[^，。]{2,12}方面", r"通过[^，。]{2,15}的方式"]
_DATA_SOURCE = re.compile(r"数据来源|来源[:：]|据[^，。]{0,12}(统计|调查|报告)|"
                          r"(国家|省|市|县)[^，。]{0,8}(统计局|局|委|院|中心)|"
                          r"《[^》]{2,40}》")

_SEGMENT_CUES = ["现将", "经调研", "建议", "存在的主要问题", "下一步", "综上", "为此", "据此"]


class StyleEngine(BaseEngine):
    name = "style_engine"
    dimension = Dimension.STYLE
    description = "风格迁移与文体一致性（量化画像 + 迁移动作清单）"

    def __init__(self, settings: Any, store: Any = None, preset: Optional[str] = None) -> None:
        super().__init__(settings, store)
        self.cfg = settings.style
        self.res = self._load()
        self.preset_name = preset or self.cfg.get("default_preset", "公文专报")
        self.presets: Dict[str, Dict[str, Any]] = self.res.get("presets", {}) or {}
        self.metrics_cfg: List[Dict[str, Any]] = self.res.get("metrics", []) or []
        self.profile: Dict[str, Any] = {}

    def _load(self) -> Dict[str, Any]:
        path = self.settings.path(self.cfg.get("presets_file", "data/terminology/semantic_clusters.yaml"))
        if not path.exists():
            return {}
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("style", {}) or {}

    # ==================================================================
    def run(self, doc: Document) -> EngineResult:
        preset = self.presets.get(self.preset_name, {}) or {}
        issues: List[Issue] = []
        self.profile = self._profile(doc)
        issues += self._check_avoid(doc, preset)
        issues += self._check_metrics(doc)
        issues += self._check_structure(doc, preset)

        kept = self.resolve_overlaps(issues)
        score = self.score()
        return EngineResult(
            engine=self.name, dimension=self.dimension, issues=kept,
            stats={"preset": self.preset_name, "profile": self.profile,
                   "style_score": score, "raw": len(issues), "final": len(kept)},
        )

    # ---- 风格画像 ---------------------------------------------------
    def _profile(self, doc: Document) -> Dict[str, Any]:
        body = doc.body_text
        sents = [s for s in split_sentences(body) if len(s[2].strip()) >= 4]
        n_sent = max(1, len(sents))
        lengths = [len(s[2]) for s in sents]
        chars = max(1, len(re.sub(r"\s", "", body)))

        def density(words: List[str]) -> float:
            hit = sum(body.count(w) for w in words)
            return hit / chars

        data_marks = len(_DATA_SOURCE.findall(body))
        data_sentences = sum(1 for _, _, s in sents if re.search(r"\d", s))
        european = sum(len(re.findall(p, body)) for p in _EUROPEAN)

        return {
            "平均句长": round(sum(lengths) / n_sent, 1),
            "最长句": max(lengths) if lengths else 0,
            "口语词密度": round(density(_COLLOQUIAL), 5),
            "绝对化词密度": round(density(_ABSOLUTE), 5),
            "数据标注率": round(data_marks / max(1, data_sentences), 3),
            "欧化结构数": european,
            "段旨句命中率": round(self._segment_lead_ratio(doc), 3),
            "句子数": n_sent,
            "字数": chars,
        }

    @staticmethod
    def _segment_lead_ratio(doc: Document) -> float:
        """段旨句：段首 30 字内出现判断/结论性标记的段落占比。"""
        marks = ["应", "须", "需", "表明", "说明", "是", "存在", "建议", "取得", "总体", "基本"]
        paras = [p for p in doc.paragraphs if not p["is_heading"] and len(p["text"]) > 40
                 and not p["in_reference_section"]]
        if not paras:
            return 1.0
        hit = sum(1 for p in paras if any(m in p["text"][:30] for m in marks))
        return hit / len(paras)

    # ---- avoid 词表 --------------------------------------------------
    def _check_avoid(self, doc: Document, preset: Dict[str, Any]) -> List[Issue]:
        issues: List[Issue] = []
        avoid = preset.get("avoid", []) or []
        for word in avoid:
            for m in re.finditer(re.escape(word), doc.body_text):
                issues.append(make_issue(
                    category=Category.STYLE,
                    severity=Severity.MINOR if word in _ABSOLUTE else Severity.INFO,
                    rule_id="STYLE-AVOID",
                    message=f"目标文体「{self.preset_name}」不宜使用「{word}」",
                    span=Span(m.start(), m.end()),
                    original=word,
                    suggestion=_style_fix(word),
                    explain=f"文体 {self.preset_name} 的禁用表达表命中。{preset.get('description', '')}",
                    source=f"文体预设：{self.preset_name}",
                    confidence=0.8,
                    agent="style",
                    meta={"preset": self.preset_name, "word": word},
                ))
        return issues

    # ---- 指标差距 ---------------------------------------------------
    def _check_metrics(self, doc: Document) -> List[Issue]:
        issues: List[Issue] = []
        for item in self.metrics_cfg:
            name = item.get("name")
            target = (item.get("target") or {}).get(self.preset_name)
            if not name or not target or name not in self.profile:
                continue
            lo, hi = target[0], target[1]
            val = self.profile[name]
            if lo <= val <= hi:
                continue
            direction = "偏高" if val > hi else "偏低"
            bound = hi if val > hi else lo
            issues.append(make_issue(
                category=Category.STYLE,
                severity=Severity.MINOR if abs(val - bound) / max(bound or 1, 1e-6) > 0.3
                else Severity.INFO,
                rule_id="STYLE-METRIC",
                message=f"风格指标「{name}」{direction}：当前 {val}，目标区间 {lo}–{hi}",
                span=Span(0, 0),
                original="",
                suggestion=_metric_advice(name, val, bound, direction),
                explain=f"文体 {self.preset_name} 的量化风格区间设定。",
                source=f"文体预设：{self.preset_name}",
                confidence=0.75,
                agent="style",
                meta={"metric": name, "value": val, "target": [lo, hi]},
            ))
        return issues

    # ---- 结构 -------------------------------------------------------
    def _check_structure(self, doc: Document, preset: Dict[str, Any]) -> List[Issue]:
        issues: List[Issue] = []
        if self.preset_name != "公文专报":
            return issues
        body = doc.body_text
        required = {"问题段": "存在问题", "建议段": "建议"}
        if "问题" not in body and "不足" not in body:
            issues.append(make_issue(
                category=Category.STYLE, severity=Severity.MAJOR, rule_id="STYLE-STRUCT-PROBLEM",
                message="专报缺少「存在问题」部分，不符合「现状—问题—建议」三段式要求",
                span=Span(0, 0), original="（缺失）",
                suggestion="补充「存在的主要问题」部分，至少列出 2 项并给出表现与成因",
                explain="调研专报的文体惯例要求问题意识显性化，仅罗列成绩会被上级退回补充。",
                source="公文写作规范", confidence=0.85, agent="style",
            ))
        if "建议" not in body and "拟" not in body:
            issues.append(make_issue(
                category=Category.STYLE, severity=Severity.MAJOR, rule_id="STYLE-STRUCT-ADVICE",
                message="专报缺少「工作建议」部分",
                span=Span(0, 0), original="（缺失）",
                suggestion="补充「下一步工作建议」，建议与问题一一对应，可量化、可考核",
                explain="专报的价值在于为决策提供可执行选项，建议部分是必备项。",
                source="公文写作规范", confidence=0.85, agent="style",
            ))
        return issues

    # ---- 评分 -------------------------------------------------------
    def score(self) -> float:
        if not self.profile:
            return 100.0
        score = 100.0
        for item in self.metrics_cfg:
            target = (item.get("target") or {}).get(self.preset_name)
            name = item.get("name")
            if not target or name not in self.profile:
                continue
            lo, hi = target
            val = self.profile[name]
            if lo <= val <= hi:
                continue
            span = max(hi - lo, 1e-6)
            excess = (lo - val) if val < lo else (val - hi)
            score -= min(12.0, 12.0 * excess / span)
        return round(max(0.0, score), 1)

    def transfer_plan(self) -> List[Dict[str, str]]:
        """生成风格迁移动作清单（供报告"修改建议"章节使用）。"""
        plan: List[Dict[str, str]] = []
        preset = self.presets.get(self.preset_name, {}) or {}
        for rule in preset.get("rules", []) or []:
            plan.append({"动作": rule, "说明": "文体硬性要求"})
        if self.profile:
            if self.profile.get("平均句长", 0) > 55:
                plan.append({"动作": "拆分长句", "说明": f"当前平均句长 "
                            f"{self.profile['平均句长']} 字，建议压到 45 字以内"})
            if self.profile.get("数据标注率", 1) < 0.8:
                plan.append({"动作": "补充数据来源", "说明": f"数据句标注率仅 "
                            f"{self.profile['数据标注率']:.0%}，需标注机构+报告名+时间"})
            if self.profile.get("绝对化词密度", 0) > 0.002:
                plan.append({"动作": "弱化绝对化表述", "说明": "将「完全/必然/众所周知」改为可度量的表述"})
        return plan

    # ---- 定向改写（供 LLM Agent 使用）--------------------------------
    def target_preset(self) -> Dict[str, Any]:
        return self.presets.get(self.preset_name, {}) or {}


def _style_fix(word: str) -> str:
    mapping = {
        "众所周知": "（直接陈述事实，删除）",
        "毫无疑问": "（删除，或用数据支撑）",
        "显然": "（改为「从调研数据看」）",
        "必然": "（改为「可能导致」或标注因果依据）",
        "我觉得": "经调研分析",
        "说白了": "换言之",
        "大概可能": "约",
        "网上说": "据公开报道",
        "据说": "据了解",
        "搞定": "完成",
        "咱们": "我们",
    }
    return mapping.get(word, f"替换「{word}」为规范书面表述")


def _metric_advice(name: str, val: float, bound: float, direction: str) -> str:
    if name == "平均句长":
        return "过长句拆分为「判断 + 依据」两句；过短则合并同类信息，减少碎片化短句"
    if name == "口语词密度":
        return "将口语词替换为规范书面语"
    if name == "绝对化词密度":
        return "删除或弱化绝对化修饰，改为有据可查的表述"
    if name == "数据标注率":
        return "每个数据结论都补上「来源 + 时间 + 口径」三要素"
    return f"将该指标从 {val} 调整至 {bound} 附近"
