"""术语库动态更新。

术语库的生命力在于"动态"。本模块提供三条更新管道：

1. **官方文件管道**：输入一份新的政策文件/新闻稿正文（txt/md/docx 文本），
   自动抽取"坚持/贯彻/落实/推进 + 术语"结构的候选表述，生成待审条目；
2. **接口管道**：从内网术语服务（JSON API）或权威媒体页面增量拉取；
3. **台账管道**：读取本单位术语台账 ``data/terminology/local_ledger.yaml``，
   优先级最高，可覆盖中央库条目。

所有自动抽取的结果**一律进入待审区**（``data/terminology/pending/``），
经人工确认或影子校验通过后才写入 active 版本，避免"机器改错术语"。
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .store import BannedRule, TermEntry, TerminologyStore, TerminologyVersion, diff_versions


# 政策文件中的术语承载句式：(正则, 提示语标签)
_CUE_PATTERNS = [
    (r"坚持([\u4e00-\u9fff“”‘’]{4,26}?)(?=[，。；、]|$)", "坚持…"),
    (r"深入(?:学习)?(?:贯彻|落实)([\u4e00-\u9fff]{4,26}?)(?=[，。；]|$)", "贯彻落实…"),
    (r"(?:牢固)?树立([\u4e00-\u9fff]{4,24}?)(?=[，。；]|$)", "树立…"),
    (r"(?:坚决)?(?:做到|践行)([\u4e00-\u9fff]{4,24}?)(?=[，。；]|$)", "践行…"),
    (r"(铸牢|筑牢|树立|培养)([\u4e00-\u9fff]{4,20}?意识)(?=[，。；]|$)", "铸牢…意识"),
    (r"([\u4e00-\u9fff]{2,10}?(?:总基调|总布局|战略布局|总体布局|发展格局|共同体|重要任务))", "固定提法"),
]

#: 抽取结果前端常见的动词性噪声，需剥离后才能得到稳定的术语本体
_LEAD_NOISE = [
    "坚持不懈", "坚持不懈用", "坚持", "继续", "持续", "不断", "牢牢", "着力", "切实",
    "深入", "全面", "坚决", "始终", "不懈", "进一步", "大力", "注重", "加强",
    "铸牢", "筑牢", "树立", "培养", "深入贯彻", "贯彻落实", "践行", "做到", "用",
]

# 明显不是术语的噪声词
_STOP = {
    "问题导向", "系统观念", "底线思维", "一岗双责", "久久为功", "狠抓落实",
    "问题", "工作", "事情", "原则", "要求", "标准", "措施", "机制", "情况",
}


@dataclass
class CandidateTerm:
    """候选术语条目。"""

    term: str
    cue: str
    source_id: str
    source_ref: str = ""
    occurrences: int = 1
    severity: str = "MAJOR"
    suggested_type: str = "literal"
    variants: List[str] = field(default_factory=list)
    confidence: float = 0.6

    def to_entry_dict(self) -> Dict[str, Any]:
        return {
            "id": f"AUTO-{abs(hash(self.term)) % 10**7:07d}",
            "term": self.term,
            "type": self.suggested_type,
            "severity": self.severity,
            "tags": ["自动抽取", "待审"],
            "variants": self.variants,
            "explain": f"由「{self.cue}」句式自动抽取，来源：{self.source_id}"
                       + (f"（{self.source_ref}）" if self.source_ref else ""),
            "source": self.source_ref or self.source_id,
            "confidence": round(self.confidence, 2),
        }


class TerminologyUpdater:
    """术语库更新器。"""

    def __init__(self, store: TerminologyStore, settings: Any) -> None:
        self.store = store
        self.settings = settings
        self.cfg = settings.terminology
        self.pending_dir = settings.path("data/terminology/pending")

    # ---- 抽取 ------------------------------------------------------
    def extract_candidates(self, text: str, source_id: str,
                           source_ref: str = "") -> List[CandidateTerm]:
        """从政策文件正文抽取候选术语。

        三步：句式抽取 → 剥离动词性噪声 → 包含关系去重
        （"坚持稳中求进工作总基调"与"稳中求进工作总基调"只保留后者）。
        """
        found: Dict[str, CandidateTerm] = {}
        for pat, cue_label in _CUE_PATTERNS:
            for m in re.finditer(pat, text):
                raw = next((g for g in reversed(m.groups()) if g), "")
                term = _clean_term(raw)
                if len(term) < 4 or term in _STOP or term.startswith("的"):
                    continue
                if term in found:
                    found[term].occurrences += 1
                    continue
                found[term] = CandidateTerm(
                    term=term, cue=cue_label, source_id=source_id, source_ref=source_ref,
                    confidence=min(0.9, 0.55 + 0.04 * len(term)),
                )

        terms = _drop_contained(list(found))
        result = [found[t] for t in terms]
        for c in result:
            c.confidence = min(0.95, c.confidence + 0.05 * (c.occurrences - 1))
        return sorted(result, key=lambda x: -x.confidence)

    # ---- 来源抓取 ---------------------------------------------------
    def fetch_source(self, source: Dict[str, Any]) -> List[CandidateTerm]:
        stype = source.get("type")
        sid = source.get("id", "unknown")
        if not source.get("enabled", True):
            return []
        try:
            if stype == "file":
                p = self.settings.path(source["path"]) if not Path(source["path"]).is_absolute() \
                    else Path(source["path"])
                if not p.exists():
                    return []
                return self.extract_candidates(p.read_text(encoding="utf-8"), sid, str(p))
            if stype == "url_text":
                req = urllib.request.Request(source["url"], headers={"User-Agent": "proofread-agent/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                return self.extract_candidates(_strip_html(raw), sid, source["url"])
            if stype == "json_api":
                req = urllib.request.Request(source["url"], headers={"User-Agent": "proofread-agent/1.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data", data if isinstance(data, list) else [])
                return [
                    CandidateTerm(
                        term=it.get("term", ""), cue=it.get("cue", "接口下发"),
                        source_id=sid, source_ref=it.get("source", ""),
                        severity=it.get("severity", "MAJOR"),
                        variants=it.get("variants", []) or [],
                        confidence=float(it.get("confidence", 0.9)),
                    )
                    for it in items if it.get("term")
                ]
        except Exception as exc:  # 更新失败不应中断审校
            return [CandidateTerm(term="", cue=f"来源 {sid} 拉取失败：{exc}", source_id=sid, confidence=0.0)]
        return []

    # ---- 主流程 ----------------------------------------------------
    def run_update(self, auto_promote: bool = False) -> Dict[str, Any]:
        """执行一轮更新：抓取 → 去重 → 落待审区 → 影子校验 → （可选）转正。"""
        sources = self.cfg.get("update_sources", []) or []
        collected: List[CandidateTerm] = []
        for src in sources:
            collected.extend(self.fetch_source(src))

        existing = {e.term for e in self.store.entries}
        fresh = [c for c in collected if c.term and c.term not in existing and c.confidence >= 0.6]

        pending_file = self._write_pending(fresh)

        # 构建候选版本并做影子校验
        beta = TerminologyVersion(
            version=f"{self.store.version}+auto{time.strftime('%m%d%H%M')}",
            loaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            entries=list(self.store.entries) + [TermEntry.from_dict(c.to_entry_dict()) for c in fresh],
            banned=list(self.store.banned),
        )
        diff = self.store.stage_shadow(beta)

        promoted = False
        policy = self.cfg.get("update_policy", {}) or {}
        if auto_promote or not policy.get("require_human_approval", True):
            self.store.promote_shadow(approved=True)
            promoted = True

        return {
            "sources_scanned": len(sources),
            "candidates_raw": len(collected),
            "candidates_new": len(fresh),
            "pending_file": str(pending_file) if pending_file else None,
            "shadow_diff": diff,
            "promoted": promoted,
            "mode": policy.get("mode", "shadow"),
        }

    def _write_pending(self, candidates: List[CandidateTerm]) -> Optional[Path]:
        if not candidates:
            return None
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        path = self.pending_dir / f"pending_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
        payload = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(candidates),
                "warning": "自动抽取结果，需人工确认后方可转正；请核对 severity 与 variants。",
            },
            "entries": [c.to_entry_dict() for c in candidates],
        }
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def approve_pending(self, pending_file: str | Path, ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """人工审批：把待审条目合并进本地台账（最高优先级）。"""
        p = Path(pending_file)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        entries = data.get("entries", []) or []
        if ids:
            entries = [e for e in entries if e.get("id") in ids]
        ledger = self.settings.root / "data" / "terminology" / "local_ledger.yaml"
        existing: List[Dict[str, Any]] = []
        if ledger.exists():
            existing = (yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}).get("entries", []) or []
        merged_ids = {e.get("id") for e in existing}
        added = [e for e in entries if e.get("id") not in merged_ids]
        for e in added:
            e["tags"] = sorted(set(list(e.get("tags", [])) + ["已审核"]))
        payload = {
            "meta": {
                "name": "本单位增量术语台账",
                "version": time.strftime("%Y.%m.%d"),
                "priority": "highest",
                "note": "本台账条目优先级高于中央术语库，用于覆盖本地口径。",
            },
            "entries": existing + added,
        }
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self.store.load()  # 热重载
        return {"approved": len(added), "total_ledger": len(existing) + len(added), "ledger": str(ledger)}


def _clean_term(raw: str) -> str:
    """剥离动词性噪声，得到术语本体。"""
    term = (raw or "").strip().strip("，。；、：: ")
    changed = True
    while changed and len(term) > 4:
        changed = False
        for w in _LEAD_NOISE:
            if term.startswith(w) and len(term) - len(w) >= 4:
                term = term[len(w):]
                changed = True
                break
    while term and term[0] in "懈持续实抓":
        term = term[1:]
    return term.strip("，。；、：: ")


def _drop_contained(terms: List[str]) -> List[str]:
    """包含关系去重：优先保留更完整的表述。

    经过 _clean_term 剥离动词噪声后，剩下的包含关系通常意味着较短者是被截断的
    片段（如"中华民族共同体" vs "中华民族共同体意识"），因此保留较长者。
    """
    ordered = sorted(set(terms), key=len, reverse=True)
    kept: List[str] = []
    for t in ordered:
        if any(k != t and t in k for k in kept):
            continue
        kept.append(t)
    return kept


def _strip_html(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;?", " ", html)
    return re.sub(r"\s{2,}", " ", html)
