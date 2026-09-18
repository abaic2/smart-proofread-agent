"""文档级相似度算法：SimHash、MinHash/LSH、TF-IDF 余弦。

用于"核心观点重复发表"检测，分三层：
1. 段落级 SimHash 粗筛（汉明距离）；
2. MinHash + 签名分桶精筛（集合相似度）；
3. TF-IDF / 向量余弦终判（语义相似度）。
三层串联后，10 万段落级别的语料比对可在秒级完成。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from . import char_ngrams, tokenize


# ---------------------------------------------------------------- SimHash
def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")


def simhash(tokens: Sequence[str], bits: int = 64) -> int:
    """经典 SimHash：按词频加权，逐位投票。"""
    if not tokens:
        return 0
    counts = Counter(tokens)
    vector = [0] * bits
    for tok, w in counts.items():
        h = _hash64(tok)
        for i in range(bits):
            vector[i] += w if (h >> i) & 1 else -w
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def simhash_similarity(a: int, b: int, bits: int = 64) -> float:
    return 1.0 - hamming(a, b) / bits


# ---------------------------------------------------------------- MinHash
_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32)


def _base_hashes(tokens: Sequence[str], k: int, num_perm: int) -> List[int]:
    """对 k-shingle 集合计算 num_perm 个独立哈希（系数固定，保证可比性）。"""
    shingles = {" ".join(tokens[i:i + k]) for i in range(max(0, len(tokens) - k + 1))}
    if not shingles:
        shingles = {" ".join(tokens)} if tokens else set()
    hashes: List[int] = []
    for s in shingles:
        hashes.append(_hash64(s) % _MAX_HASH)
    return hashes


def _coeffs(num_perm: int, seed: int = 20260918) -> List[Tuple[int, int]]:
    """确定性生成 (a, b) 系数，避免每次运行签名不一致。"""
    out: List[Tuple[int, int]] = []
    state = seed
    for _ in range(num_perm):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        a = (state >> 16) | 1
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        b = state >> 16
        out.append((a, b))
    return out


class MinHasher:
    """可复用的 MinHash 签名器。"""

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 20260918) -> None:
        self.num_perm = num_perm
        self.k = shingle_size
        self.coeffs = _coeffs(num_perm, seed)

    def signature(self, tokens: Sequence[str]) -> Tuple[int, ...]:
        hashes = _base_hashes(tokens, self.k, self.num_perm)
        if not hashes:
            return tuple([0] * self.num_perm)
        sig = []
        for a, b in self.coeffs:
            sig.append(min(((a * h + b) % _MERSENNE) for h in hashes) % _MAX_HASH)
        return tuple(sig)

    @staticmethod
    def estimate_jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
        if not sig_a or not sig_b:
            return 0.0
        same = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return same / len(sig_a)


# ---------------------------------------------------------------- TF-IDF
class TfidfIndex:
    """轻量 TF-IDF 向量空间模型，支持增量语料与余弦检索。"""

    def __init__(self) -> None:
        self.df: Counter = Counter()
        self.docs: List[Tuple[str, Counter]] = []
        self.norm_cache: List[float] = []
        self._dirty = True

    def add(self, doc_id: str, text: str) -> None:
        tf = Counter(tokenize(text))
        self.docs.append((doc_id, tf))
        for tok in tf:
            self.df[tok] += 1
        self._dirty = True

    def _weights(self, tf: Counter) -> Dict[str, float]:
        n = max(1, len(self.docs))
        out: Dict[str, float] = {}
        for tok, c in tf.items():
            idf = math.log((1 + n) / (1 + self.df.get(tok, 0))) + 1.0
            out[tok] = (1 + math.log(c)) * idf
        return out

    def _ensure_norms(self) -> None:
        if not self._dirty:
            return
        self._vec_cache = [self._weights(tf) for _, tf in self.docs]
        self.norm_cache = [
            math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in self._vec_cache
        ]
        self._dirty = False

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """返回 [(doc_id, cosine), ...]，按相似度降序。"""
        if not self.docs:
            return []
        self._ensure_norms()
        qtf = Counter(tokenize(text))
        qvec = self._weights(qtf)
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored: List[Tuple[str, float]] = []
        for idx, (doc_id, _) in enumerate(self.docs):
            vec = self._vec_cache[idx]
            dot = 0.0
            smaller, larger = (qvec, vec) if len(qvec) < len(vec) else (vec, qvec)
            for tok, w in smaller.items():
                w2 = larger.get(tok)
                if w2:
                    dot += w * w2
            if dot:
                scored.append((doc_id, dot / (qnorm * self.norm_cache[idx])))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


# ---------------------------------------------------------------- 组合器
def paragraph_fingerprint(text: str, shingle_size: int = 5, num_perm: int = 128,
                          hasher: MinHasher | None = None) -> Dict:
    """生成段落指纹，供跨稿件比对复用。"""
    toks = tokenize(text)
    hasher = hasher or MinHasher(num_perm=num_perm, shingle_size=shingle_size)
    return {
        "tokens": toks,
        "simhash": simhash(toks),
        "minhash": hasher.signature(toks),
        "ngrams": char_ngrams(text, 3),
    }


def compare_fingerprints(a: Dict, b: Dict) -> Dict[str, float]:
    """三路相似度打分，便于按置信度分层输出。"""
    from . import jaccard

    return {
        "simhash": simhash_similarity(a["simhash"], b["simhash"]),
        "minhash": MinHasher.estimate_jaccard(a["minhash"], b["minhash"]),
        "ngram": jaccard(a["ngrams"], b["ngrams"]),
    }


def combined_similarity(scores: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    weights = weights or {"simhash": 0.25, "minhash": 0.35, "ngram": 0.40}
    total_w = sum(weights.values())
    return sum(scores.get(k, 0.0) * w for k, w in weights.items()) / total_w
