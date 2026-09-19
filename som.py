"""
som.py — 고정 임베딩 위에서 도는 SOM.

이전 발산의 두 번째 원인 대비:
  - 전과목을 한 지도에 부으면 벡터가 사방으로 흩어져 수렴 못 함.
  → 이 SOM은 '한 과목(국어)'의 레코드만 학습한다. 좁고 일관된 도메인.
  - 입력 벡터는 embedding.py의 고정 임베딩에서 나온다(좌표계 불변).

각 노드는:
  - w        : 가중치 벡터(임베딩 공간)
  - rec_ids  : 이 노드가 BMU였던 레코드 id 목록 (출처 추적용 역인덱스)
  - years    : 그 레코드들의 연도(있으면) → L1 시계열 분석
CPU only, numpy only, 결정론적.
"""
from __future__ import annotations
import numpy as np
import pickle
from collections import defaultdict

try:
    from cloud import push as _cloud_push
except Exception:  # cloud 계층이 없어도 로컬 동작은 유지
    def _cloud_push(path, **kw): return False



class SOM:
    def __init__(self, grid=(12, 12), dim=64, seed=42):
        self.gh, self.gw = grid
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (self.gh * self.gw, dim))
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-9)
        # 격자 좌표(이웃 계산용)
        self.coords = np.array([[i // self.gw, i % self.gw]
                                for i in range(self.gh * self.gw)], dtype=float)
        self.node_rec_ids = defaultdict(list)
        self.node_years = defaultdict(list)
        self.node_codes = defaultdict(list)
        self.qe = np.zeros(self.gh * self.gw)  # quantization error per node

    def _bmu(self, x):
        # 코사인 유사도(정규화돼 있으므로 내적) 최대 = 거리 최소
        sims = self.W @ x
        return int(np.argmax(sims))

    def train(self, X, iters=2000, lr0=0.3, sigma0=None, seed=42):
        """X: (N, dim) 정규화된 입력 벡터들."""
        rng = np.random.default_rng(seed)
        N = len(X)
        if N == 0:
            raise ValueError("학습할 벡터가 없음 (임베딩 후 전부 OOV였을 수 있음)")
        if sigma0 is None:
            sigma0 = max(self.gh, self.gw) / 2.0
        for t in range(iters):
            frac = t / iters
            lr = lr0 * (1 - frac)
            sigma = sigma0 * (1 - frac) + 0.5
            x = X[rng.integers(N)]
            b = self._bmu(x)
            d2 = ((self.coords - self.coords[b]) ** 2).sum(1)
            h = np.exp(-d2 / (2 * sigma * sigma))
            self.W += (lr * h)[:, None] * (x - self.W)
            self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-9)

    def assign(self, X, records):
        """학습 후: 각 레코드를 BMU 노드에 배정하고 역인덱스/QE 구축."""
        self.node_rec_ids.clear(); self.node_years.clear()
        self.node_codes.clear()
        errs = np.zeros(self.gh * self.gw)
        counts = np.zeros(self.gh * self.gw)
        for x, rec in zip(X, records):
            b = self._bmu(x)
            self.node_rec_ids[b].append(rec.rec_id)
            if rec.year is not None:
                self.node_years[b].append(rec.year)
            if rec.code:
                self.node_codes[b].append(rec.code)
            errs[b] += 1.0 - float(self.W[b] @ x)  # 1 - cos = 거리
            counts[b] += 1
        self.qe = np.where(counts > 0, errs / np.maximum(counts, 1), 0.0)

    def bmu_of(self, x):
        return self._bmu(x)

    def high_qe_nodes(self, topn=10):
        """QE 높은 노드 = '잘 못 담은 영역'. L3 검색 후보의 씨앗."""
        occupied = [i for i in range(self.gh * self.gw) if len(self.node_rec_ids[i]) > 0]
        occupied.sort(key=lambda i: -self.qe[i])
        return occupied[:topn]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "grid": (self.gh, self.gw), "dim": self.dim, "W": self.W,
                "node_rec_ids": dict(self.node_rec_ids),
                "node_years": dict(self.node_years),
                "node_codes": dict(self.node_codes),
                "qe": self.qe,
            }, f)
        _cloud_push(path)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        s = SOM(grid=d["grid"], dim=d["dim"])
        s.W = d["W"]
        s.node_rec_ids = defaultdict(list, d["node_rec_ids"])
        s.node_years = defaultdict(list, d["node_years"])
        s.node_codes = defaultdict(list, d["node_codes"])
        s.qe = d["qe"]
        return s
