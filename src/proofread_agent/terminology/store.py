"""术语库存储层：加载、版本化、热更新、影子校验。

核心能力
--------
* **多版本共存**：每次更新生成新版本快照，报告记录 ``terminology_version``，
  保证历史审校结论可复现（编辑申诉时可回溯当时依据的术语版本）。
* **影子模式**：新版本先用于"影子校验"，与线上版本比对差异并产出 diff 报告，
  经人工审批（或满足自动合并策略）后才切换为 active，防止错误术语污染全量稿件。
* **索引化**：术语变体统一构建 Aho-Corasick 索引，加载后匹配复杂度与术语量无关。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from ..report import Category, Severity, Span, make_issue  # noqa: F401  (类型契约)
from ..text import AhoCorasick


@dataclass
class TermEntry:
    """一条术语标准条目。"""

    id: str
    term: str
    type: str                       # literal | ordering | semantic | deprecated
    severity: Severity
    tags: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    pattern: str = ""
    semantic_seeds: List[str] = field(default_factory=list)
    explain: str = ""
    source: str = ""
    confidence: float = 1.0
    replaces: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TermEntry":
        sev = d.get("severity", "MINOR")
        return cls(
            id=d.get("id", ""),
            term=d.get("term", ""),
            type=d.get("type", "literal"),
            severity=Severity(sev if isinstance(sev, str) else str(sev)),
            tags=list(d.get("tags", []) or []),
            variants=list(d.get("variants", []) or []),
            order=list(d.get("order", []) or []),
            pattern=d.get("pattern") or "",
            semantic_seeds=list(d.get("semantic_seeds", []) or []),
            explain=(d.get("explain") or "").strip(),
            source=d.get("source", ""),
            confidence=float(d.get("confidence", 1.0)),
            replaces=d.get("replaces", "") or "",
        )


@dataclass
class BannedRule:
    """一条禁用/慎用词模式规则。"""

    id: str
    pattern: str
    severity: Severity
    category: str
    message: str
    suggestion: str = ""
    allow_context: List[str] = field(default_factory=list)
    reference: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BannedRule":
        return cls(
            id=d.get("id", ""),
            pattern=d.get("pattern", ""),
            severity=Severity(d.get("severity", "MINOR")),
            category=d.get("category", "其他"),
            message=d.get("message", ""),
            suggestion=d.get("suggestion", ""),
            allow_context=list(d.get("allow_context", []) or []),
            reference=d.get("reference", ""),
        )


@dataclass
class TerminologyVersion:
    """术语库版本快照。"""

    version: str
    loaded_at: str
    entries: List[TermEntry]
    banned: List[BannedRule]
    sources: List[str] = field(default_factory=list)
    checksum: str = ""


class TerminologyStore:
    """术语库仓储。"""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.cfg = settings.terminology
        self.versions: List[TerminologyVersion] = []
        self.active: Optional[TerminologyVersion] = None
        self.shadow: Optional[TerminologyVersion] = None
        self._ac: Optional[AhoCorasick] = None
        self._variant_map: Dict[str, TermEntry] = {}
        self.history_path = settings.path(".cache/terminology_history.jsonl")

    # ---- 加载 ------------------------------------------------------
    def load(self) -> TerminologyVersion:
        meta, entries, sources = self._read_core()
        banned = self._read_banned()
        local = self._read_local_ledger()
        entries.extend(local)
        version = TerminologyVersion(
            version=meta.get("version", "0.0.0"),
            loaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            entries=entries,
            banned=banned,
            sources=list(meta.get("sources", []) or []) + sources,
            checksum=_checksum(entries, banned),
        )
        self.active = version
        self.versions.append(version)
        self._build_index(version)
        self._persist_history(version)
        return version

    def _read_core(self) -> Tuple[Dict[str, Any], List[TermEntry], List[str]]:
        p = self.settings.path(self.cfg["core_terms"])
        if not p.exists():
            return {}, [], [f"术语库文件缺失：{p}"]
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        meta = data.get("meta", {}) or {}
        entries = [TermEntry.from_dict(d) for d in data.get("entries", []) or []]
        return meta, entries, []

    def _read_banned(self) -> List[BannedRule]:
        p = self.settings.path(self.cfg["banned_patterns"])
        if not p.exists():
            return []
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return [BannedRule.from_dict(d) for d in data.get("rules", []) or []]

    def _read_local_ledger(self) -> List[TermEntry]:
        """读取本单位增量术语台账（最高优先级）。"""
        p = self.settings.root / "data" / "terminology" / "local_ledger.yaml"
        if not p.exists():
            return []
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return []
        return [TermEntry.from_dict(d) for d in data.get("entries", []) or []]

    # ---- 索引 ------------------------------------------------------
    def _build_index(self, version: TerminologyVersion) -> None:
        ac = AhoCorasick()
        vmap: Dict[str, TermEntry] = {}
        for e in version.entries:
            # 标准表述本身也注册，但标记为 standard，用于"误报压制"
            ac.add(e.term, {"entry": e, "role": "standard"})
            for v in e.variants:
                ac.add(v, {"entry": e, "role": "variant"})
                vmap[v] = e
        ac.build()
        self._ac = ac
        self._variant_map = vmap

    # ---- 查询接口 --------------------------------------------------
    @property
    def ac(self) -> AhoCorasick:
        if self._ac is None:
            self.load()
        assert self._ac is not None
        return self._ac

    @property
    def entries(self) -> List[TermEntry]:
        return self.active.entries if self.active else []

    @property
    def banned(self) -> List[BannedRule]:
        return self.active.banned if self.active else []

    @property
    def version(self) -> str:
        return self.active.version if self.active else "unknown"

    def semantic_entries(self) -> List[TermEntry]:
        return [e for e in self.entries if e.type == "semantic" and e.semantic_seeds]

    def ordering_entries(self) -> List[TermEntry]:
        return [e for e in self.entries if e.type == "ordering" and e.order and e.pattern]

    def by_id(self, entry_id: str) -> Optional[TermEntry]:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def all_standard_terms(self) -> List[str]:
        return [e.term for e in self.entries if e.term]

    # ---- 影子校验与激活 --------------------------------------------
    def stage_shadow(self, candidate: TerminologyVersion) -> Dict[str, Any]:
        """将候选版本挂为影子版本，并产出与当前 active 的差异。"""
        diff = diff_versions(self.active, candidate)
        self.shadow = candidate
        return diff

    def promote_shadow(self, approved: bool = True) -> Optional[TerminologyVersion]:
        """影子版本转正。未获批准则不生效，保证术语库变更可控。"""
        if not approved or self.shadow is None:
            return None
        self.active = self.shadow
        self.versions.append(self.shadow)
        self._build_index(self.shadow)
        self._persist_history(self.shadow, note="promoted_from_shadow")
        self.shadow = None
        return self.active

    # ---- 持久化 ----------------------------------------------------
    def _persist_history(self, version: TerminologyVersion, note: str = "loaded") -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "version": version.version,
                "checksum": version.checksum,
                "loaded_at": version.loaded_at,
                "entries": len(version.entries),
                "banned": len(version.banned),
                "note": note,
            }
            with self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 历史记录失败不影响主流程

    def export_snapshot(self, out_dir: str | Path) -> Path:
        """导出版本快照，用于归档与跨环境同步。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        assert self.active is not None
        payload = {
            "version": self.active.version,
            "checksum": self.active.checksum,
            "entries": [e.__dict__ for e in self.active.entries],
            "banned": [b.__dict__ for b in self.active.banned],
        }
        path = out / f"terminology_{self.active.version}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _checksum(entries: Iterable[TermEntry], banned: Iterable[BannedRule]) -> str:
    import hashlib
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e.id}|{e.term}|{'/'.join(e.variants)}".encode("utf-8"))
    for b in banned:
        h.update(f"{b.id}|{b.pattern}".encode("utf-8"))
    return h.hexdigest()[:16]


def diff_versions(old: Optional[TerminologyVersion], new: TerminologyVersion) -> Dict[str, Any]:
    """比较两个术语库版本，输出可读的变更清单。"""
    old_ids = {e.id: e for e in (old.entries if old else [])}
    new_ids = {e.id: e for e in new.entries}
    added = [e.id for e in new.entries if e.id not in old_ids]
    removed = [i for i in old_ids if i not in new_ids]
    modified: List[Dict[str, Any]] = []
    for eid, e in new_ids.items():
        o = old_ids.get(eid)
        if o and (o.term != e.term or set(o.variants) != set(e.variants) or o.severity != e.severity):
            modified.append({
                "id": eid,
                "term_changed": o.term != e.term,
                "variants_added": sorted(set(e.variants) - set(o.variants)),
                "variants_removed": sorted(set(o.variants) - set(e.variants)),
                "severity": f"{o.severity.value}->{e.severity.value}" if o.severity != e.severity else "",
            })
    return {
        "old_version": old.version if old else None,
        "new_version": new.version,
        "added": added,
        "removed": removed,
        "modified": modified,
        "summary": f"新增 {len(added)} 条，下架 {len(removed)} 条，修改 {len(modified)} 条",
    }
