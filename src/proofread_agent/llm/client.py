"""本地部署大语言模型适配层。

支持三种本地推理后端，全部通过 HTTP 访问内网/本机服务，数据不出域：

* ``ollama``             —— Ollama（``/api/chat``）
* ``openai_compatible``  —— vLLM / LM Studio / Xinference / TGI（``/v1/chat/completions``）
* ``mock``               —— 离线确定性兜底，用于无 GPU 环境演示与 CI

设计原则
--------
1. **零第三方依赖**：使用 ``urllib.request`` 直接发起 HTTP，避免内网审计负担。
2. **故障不阻塞**：模型不可达时自动降级为规则引擎 + 模板化生成，
   并在报告上标记 ``degraded=True``，工作流始终可端到端跑通。
3. **可观测**：每次调用记录耗时、token 估算与后端名，写入审计 trace。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..text import char_ngrams


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str
    latency_ms: float = 0.0
    degraded: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 本地模型客户端
# --------------------------------------------------------------------------
class LocalLLM:
    """本地大语言模型统一客户端。"""

    def __init__(self, settings: Dict[str, Any], timeout: int = 120) -> None:
        self.cfg = settings or {}
        self.backend_name = self.cfg.get("backend", "openai_compatible")
        self.timeout = timeout
        self._available: Optional[bool] = None
        self.calls: List[Dict[str, Any]] = []

        oc = self.cfg.get("openai_compatible", {}) or {}
        ol = self.cfg.get("ollama", {}) or {}
        self.base_url = (oc.get("base_url") if self.backend_name == "openai_compatible"
                         else ol.get("base_url")) or "http://127.0.0.1:8000/v1"
        self.model = (oc.get("model") if self.backend_name == "openai_compatible"
                      else ol.get("model")) or "local-model"
        self.api_key = oc.get("api_key", "EMPTY")
        self.temperature = oc.get("temperature", 0.1)
        self.max_tokens = oc.get("max_tokens", 2048)

    # ---- 可用性探测 -------------------------------------------------
    def probe(self, force: bool = False) -> bool:
        """探测本地推理服务是否在线。结果缓存，避免每个节点都探测。"""
        if self._available is not None and not force:
            return self._available
        try:
            if self.backend_name == "ollama":
                url = self.base_url.rstrip("/") + "/api/tags"
            else:
                url = self.base_url.rstrip("/") + "/models"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "proofread-agent/1.0"}
        if self.backend_name == "openai_compatible" and self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # ---- 推理 -------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]], temperature: float | None = None,
             max_tokens: int | None = None, expect_json: bool = False) -> LLMResponse:
        """同步对话补全。失败抛 LLMError，由上层决定是否降级。"""
        if self.backend_name == "ollama":
            url = self.base_url.rstrip("/") + "/api/chat"
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature if temperature is None else temperature,
                    "num_predict": max_tokens or self.max_tokens,
                },
            }
            if expect_json:
                payload["format"] = "json"
            extract = lambda d: (d.get("message") or {}).get("content", "")  # noqa: E731
        else:
            url = self.base_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": max_tokens or self.max_tokens,
                "stream": False,
            }
            if expect_json:
                payload["response_format"] = {"type": "json_object"}
            extract = lambda d: d["choices"][0]["message"]["content"]  # noqa: E731

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        started = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - 依赖外部服务
            raise LLMError(f"HTTP {exc.code}: {exc.read()[:200]!r}") from exc
        except Exception as exc:  # pragma: no cover
            raise LLMError(f"本地模型调用失败：{exc}") from exc

        latency = (time.time() - started) * 1000
        text = extract(data) or ""
        self.calls.append({
            "backend": self.backend_name, "model": self.model,
            "latency_ms": round(latency, 1), "chars": len(text),
        })
        return LLMResponse(text=text, backend=self.backend_name, model=self.model,
                           latency_ms=latency, raw=data)

    def json_chat(self, messages: List[Dict[str, str]], **kw: Any) -> Dict[str, Any]:
        """要求模型输出 JSON，并做宽松解析（容忍 ```json 包裹与前后噪声）。"""
        resp = self.chat(messages, expect_json=True, **kw)
        return parse_json_loose(resp.text)

    @property
    def stats(self) -> Dict[str, Any]:
        if not self.calls:
            return {"calls": 0}
        return {
            "calls": len(self.calls),
            "backend": self.backend_name,
            "model": self.model,
            "avg_latency_ms": round(sum(c["latency_ms"] for c in self.calls) / len(self.calls), 1),
        }


def parse_json_loose(text: str) -> Dict[str, Any]:
    """从模型输出里抠出 JSON 对象。"""
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------
# 嵌入向量客户端
# --------------------------------------------------------------------------
class EmbeddingClient:
    """语义向量客户端。

    优先使用本地嵌入模型（bge-m3 / bge-large-zh）；不可用时退化为
    特征哈希向量（feature hashing）。后者严格来说是**词形近似**而非真语义，
    会在报告中通过 ``embedding_mode`` 字段明确标注，避免误导使用者。
    """

    def __init__(self, cfg: Dict[str, Any], llm_cfg: Dict[str, Any], dim: int = 1024) -> None:
        self.cfg = cfg or {}
        self.llm_cfg = llm_cfg or {}
        self.dim = int(self.cfg.get("dim", dim))
        self.model = self.cfg.get("model", "bge-m3")
        self.mode = "uninitialized"

    def probe(self) -> str:
        """返回实际可用的模式：vector | lexical。"""
        backend = self.cfg.get("backend", "auto")
        if backend == "lexical":
            self.mode = "lexical"
            return self.mode
        try:
            vec = self._remote_embed("连通性测试")
            self.mode = "vector" if vec else "lexical"
        except Exception:
            self.mode = "lexical"
        return self.mode

    def _remote_embed(self, text: str) -> List[float]:
        backend = self.llm_cfg.get("backend", "openai_compatible")
        if backend == "ollama":
            oc = self.llm_cfg.get("ollama", {}) or {}
            url = oc.get("base_url", "http://127.0.0.1:11434").rstrip("/") + "/api/embeddings"
            payload = {"model": self.model, "prompt": text}
        else:
            oc = self.llm_cfg.get("openai_compatible", {}) or {}
            url = oc.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/") + "/embeddings"
            payload = {"model": self.model, "input": text}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if oc.get("api_key"):
            headers["Authorization"] = f"Bearer {oc['api_key']}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "embedding" in data:
            return data["embedding"]
        if "data" in data and data["data"]:
            return data["data"][0].get("embedding", [])
        return []

    # ---- 对外统一接口 ----------------------------------------------
    def encode(self, text: str) -> List[float]:
        if self.mode == "uninitialized":
            self.probe()
        if self.mode == "vector":
            try:
                vec = self._remote_embed(text)
                if vec:
                    return _l2(vec)
            except Exception:
                self.mode = "lexical"
        return self._hashed(text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]

    def _hashed(self, text: str) -> List[float]:
        """特征哈希：字符 2/3-gram 哈希到固定维度并带符号累加。"""
        vec = [0.0] * self.dim
        grams = char_ngrams(text, 2) + char_ngrams(text, 3)
        if not grams:
            return vec
        for g in grams:
            h = hash_stable(g)
            idx = h % self.dim
            sign = 1.0 if (h >> 20) & 1 else -1.0
            vec[idx] += sign
        return _l2(vec)

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        return sum(a[i] * b[i] for i in range(n))


def hash_stable(text: str) -> int:
    """跨进程稳定的字符串哈希（不使用 Python 内置 hash，避免随机盐）。"""
    import hashlib
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _l2(vec: List[float]) -> List[float]:
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


# --------------------------------------------------------------------------
# 离线兜底生成器
# --------------------------------------------------------------------------
class MockLLM:
    """离线兜底：不调用任何模型，用模板生成结构化结论。

    作用是保证在无 GPU / 无模型服务的环境下，LangGraph 工作流仍能
    端到端跑通并被测试。产物会带 ``degraded=True``。
    """

    backend_name = "mock"
    model = "rule-engine-fallback"

    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls: List[Dict[str, Any]] = []

    def probe(self, force: bool = False) -> bool:
        return False

    def chat(self, messages: List[Dict[str, str]], **_: Any) -> LLMResponse:
        self.calls.append({"backend": "mock", "latency_ms": 0.0})
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return LLMResponse(
            text=f"[离线规则引擎生成] 已基于规则命中结果完成该节点处理，输入长度 {len(user)} 字。",
            backend="mock", model=self.model, degraded=True,
        )

    def json_chat(self, messages: List[Dict[str, str]], **kw: Any) -> Dict[str, Any]:
        self.chat(messages, **kw)
        return {}

    @property
    def stats(self) -> Dict[str, Any]:
        return {"calls": len(self.calls), "backend": "mock", "model": self.model}


def build_llm(settings: Any) -> Any:
    """工厂：根据配置与可用性返回 LocalLLM 或 MockLLM。"""
    llm_cfg = settings.llm
    fallback = llm_cfg.get("fallback_to_rule_engine", True)
    client = LocalLLM(llm_cfg, timeout=int((llm_cfg.get("openai_compatible") or {}).get("timeout", 120)))
    if client.probe():
        return client
    if fallback:
        return MockLLM()
    raise LLMError(
        "本地大模型服务不可达，且未开启 fallback_to_rule_engine。"
        f"请检查 {client.backend_name} 服务：{client.base_url}"
    )
