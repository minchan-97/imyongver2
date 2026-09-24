"""
passage_cluster.py — 기출 지문(제시문) 전용 관리 + 유사 지문 클러스터링.

구조:
  기출 = 지문(passage) 1개 + 딸린 문제(question) N개  = 한 세트.
  지문은 통으로 보관(외우기용), 문제는 개념 매핑용.
  지문 유형은 미리 정하지 않는다. 시스템이 유사 지문끼리 '묶어주기'만 하고,
  사용자가 그 묶음을 보고 유형을 확정(이름 붙이기/나누기/옮기기)한다.

클러스터링:
  지문은 서사적 텍스트라 개념 SOM과 성격이 다름 → 전용 임베딩 + k-means.
  sklearn 있으면 사용, 없으면 numpy k-means 폴백(의존성 최소화).
"""
from __future__ import annotations
CORE_VERSION = "13.3"
import os, pickle, hashlib
import numpy as np

try:
    from cloud import push as _cloud_push
except Exception:  # cloud 계층이 없어도 로컬 동작은 유지
    def _cloud_push(path, **kw): return False



# ── 지문 세트 저장 ────────────────────────────────────────────
class PassageBank:
    """지문 + 딸린 문제 세트 은행."""
    def __init__(self, subject):
        self.subject = subject
        self.sets = []   # {id, passage, questions[], year, level, source, ptype}

    def add(self, passage, questions, year=None, level=None,
            source="", ptype=None):
        pid = hashlib.md5(passage.encode("utf-8")).hexdigest()[:10]
        if any(s["id"] == pid for s in self.sets):
            return False
        self.sets.append({
            "id": pid, "passage": passage,
            "questions": questions if isinstance(questions, list) else [questions],
            "year": year, "level": level, "source": source,
            "ptype": ptype,   # 지문 유형(확정 전엔 None)
        })
        return True

    def set_type(self, pid, ptype):
        for s in self.sets:
            if s["id"] == pid:
                s["ptype"] = ptype
                return True
        return False

    def types(self):
        return sorted(set(s["ptype"] for s in self.sets if s["ptype"]))

    def by_type(self, ptype):
        return [s for s in self.sets if s["ptype"] == ptype]

    def untyped(self):
        return [s for s in self.sets if not s["ptype"]]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"subject": self.subject, "sets": self.sets}, f)
        _cloud_push(path)

    @staticmethod
    def load(path, subject):
        b = PassageBank(subject)
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f)
            b.sets = d.get("sets", [])
        return b


# ── numpy k-means (sklearn 없을 때 폴백) ─────────────────────
def _kmeans_numpy(X, k, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    centers = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new_labels = d.argmin(1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                centers[j] = pts.mean(0)
    return labels


def cluster_passages(passages, emb, k=None, tokenize_fn=None):
    """
    지문 리스트 → 유사 지문끼리 군집 라벨.
    emb: 고정 임베딩(embed_tokens 사용). 지문을 문장벡터로.
    k: 군집 수(None이면 대략 sqrt(n/2)).
    반환: labels(list[int]), 유효 인덱스(벡터화 성공한 것만)
    """
    if tokenize_fn is None:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from korean_tokenizer import tokenize as tokenize_fn

    vecs, valid_idx = [], []
    for i, p in enumerate(passages):
        v = emb.embed_tokens(tokenize_fn(p))
        if v is not None:
            vecs.append(v); valid_idx.append(i)
    if len(vecs) < 2:
        return [0] * len(valid_idx), valid_idx
    X = np.array(vecs)

    if k is None:
        k = max(2, int(np.sqrt(len(X) / 2)))

    try:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=min(k, len(X)), n_init=10,
                        random_state=42).fit_predict(X)
    except Exception:
        labels = _kmeans_numpy(X, k)
    return list(labels), valid_idx


def group_by_cluster(passages, labels, valid_idx):
    """군집 라벨 → {군집번호: [지문인덱스,...]} 로 묶기."""
    from collections import defaultdict
    groups = defaultdict(list)
    for lab, idx in zip(labels, valid_idx):
        groups[int(lab)].append(idx)
    return dict(groups)

