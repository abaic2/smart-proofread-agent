"""配置加载：读取 config/settings.yaml，支持环境变量覆盖与路径解析。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

try:  # PyYAML 为唯一外部数据依赖，缺失时给出明确指引
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 PyYAML：pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """环境变量覆盖，便于容器化部署。

    PFRD_LLM__BASE_URL=http://vllm:8000/v1  ->  cfg["llm"]["openai_compatible"]["base_url"]
    """
    prefix = "PFRD_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        node: Any = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
        leaf = path[-1]
        # 类型保持：若原配置该键为数值/布尔，则按原类型转换，避免 "0.1" 污染
        if isinstance(node.get(leaf), bool):
            node[leaf] = value.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(node.get(leaf), (int, float)):
            try:
                node[leaf] = type(node[leaf])(value)
            except ValueError:
                node[leaf] = value
        else:
            node[leaf] = value
    return cfg


@dataclass
class Settings:
    """全局配置对象。"""

    raw: Dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    # ---- 访问器 ----------------------------------------------------
    @property
    def llm(self) -> Dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def terminology(self) -> Dict[str, Any]:
        return self.raw.get("terminology", {})

    @property
    def engine(self) -> Dict[str, Any]:
        return self.raw.get("engine", {})

    @property
    def style(self) -> Dict[str, Any]:
        return self.raw.get("style", {})

    @property
    def report(self) -> Dict[str, Any]:
        return self.raw.get("report", {})

    @property
    def graph(self) -> Dict[str, Any]:
        return self.raw.get("graph", {})

    def path(self, relative: str) -> Path:
        """把配置中的相对路径解析为绝对路径。"""
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_settings(config_path: str | os.PathLike | None = None) -> Settings:
    """加载配置，默认使用 config/settings.yaml。"""
    if config_path is None:
        config_path = os.environ.get(
            "PFRD_CONFIG", PROJECT_ROOT / "config" / "settings.yaml"
        )
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = _apply_env_overrides(cfg)
    return Settings(raw=cfg, root=path.parent.parent if path.parent.name == "config" else PROJECT_ROOT)
