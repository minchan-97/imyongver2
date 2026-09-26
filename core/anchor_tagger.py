"""
anchor_tagger.py — 은닉층에 '출처 앵커'를 심은 태거.

왜 앵커인가
  은닉층은 보통 알 수 없는 숫자 뭉치다. 여기서는 은닉 뉴런 일부를 **출처 축**으로
  고정한다. "이 판단은 교육과정에서 왔나, 지도서에서 왔나, 웹에서 왔나"가
  층 안에 박힌다. 판단을 자료에 묶는 원칙이 화면 표시가 아니라 구조가 된다.

앵커 축 (출처 묶음 — 개별 파일이 아니라 종류로 묶어 확장에 안전)
  교육과정 / 지도서 / 기출 / 내답변 / 웹수집 / 개인필기
  + 자유 뉴런 몇 개 (출처로 설명 안 되는 패턴을 담는 자리)

학습
  손실 = 라벨 맞히기(교차 엔트로피) + λ · 앵커 제약
  앵커 제약: 지도서에서 온 기록이면 '지도서' 뉴런이 켜지고 나머지 앵커는 눌린다.
  (자유 뉴런은 제약하지 않는다)

검사 (cogito_trace의 앵커 프로브를 은닉층 안으로 가져온 것)
  · 기여도   — 판단이 어느 앵커에서 나왔나 (활성 × 출력가중치)
  · 절제     — 앵커 뉴런을 끄면 판단이 얼마나 움직이나
               끄고도 그대로면 그 출처를 실제로 쓰지 않은 것
  · 경고     — 웹수집·내답변에만 기대어 내린 판단

정직한 범위
  앵커는 '이 판단이 어느 출처 성격의 자료와 닮았나'를 본다.
  특정 파일에서 인용했다는 증명이 아니다. 그건 근거 검색이 따로 한다.
"""
from __future__ import annotations
import os, json, time, pickle
import numpy as np

import paths
from korean_tokenizer import tokenize
from embedding import FrozenEmbedding

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

# 출처 묶음 → 앵커 축
ANCHORS = ["교육과정", "지도서", "기출", "내답변", "웹수집", "개인필기"]
DOC2ANCHOR = {
    "교육과정_성취기준": "교육과정", "교육과정_총론": "교육과정",
    "지도서_각론": "지도서", "지도서_총론": "지도서",
    "내_답변": "내답변", "웹수집": "웹수집",
    "개인_필기": "개인필기", "2차_자료": "개인필기",
}
FREE_UNITS = 6          # 앵커에 묶이지 않는 자유 뉴런
TRUST = {"교육과정": 1.0, "지도서": 0.9, "기출": 0.95,
         "개인필기": 0.6, "내답변": 0.6, "웹수집": 0.5}


def tagger_path(subject):
    return paths._p(f"anchor_tagger_{subject}.pkl")


def anchor_of(rec):
    """기록 → 앵커 축 이름."""
    if getattr(rec, "layer", "") == "L1_pattern":
        return "기출"
    return DOC2ANCHOR.get(getattr(rec, "doc_type", None) or "", None)


# ── 수식 ─────────────────────────────────────────────────────
LEAK = 0.01          # 뉴런이 죽어 앵커 축이 사라지는 것 방지 (leaky ReLU)


def _relu(z):
    return np.where(z > 0, z, LEAK * z)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _forward(X, P):
    H = _relu(X @ P["W1"] + P["b1"])
    return H, _softmax(H @ P["W2"] + P["b2"])


def train(X, Y, A, n_hidden, n_anchor, epochs=400, lr=0.3, l2=1e-4,
          anchor_w=0.5, seed=0):
    """
    X: (N, dim) 입력   Y: (N, C) 정답 라벨   A: (N, n_anchor) 앵커 정답(없으면 전부 0)
    앞의 n_anchor개 은닉 뉴런이 앵커 축, 나머지는 자유 뉴런.
    anchor_w=0 이면 평범한 2층 신경망(비교용).
    """
    rng = np.random.default_rng(seed)
    d, C = X.shape[1], Y.shape[1]
    P = {"W1": rng.normal(0, 0.1, (d, n_hidden)), "b1": np.zeros(n_hidden),
         "W2": rng.normal(0, 0.1, (n_hidden, C)), "b2": np.zeros(C)}
    P["b1"][:n_anchor] = 0.1        # 앵커 뉴런은 살짝 켠 상태에서 시작
    # 앵커 뉴런을 그 출처의 '평균 벡터' 방향으로 초기화한다.
    # 무작위로 시작하면, 입력이 비슷한 출처끼리는 한 축이 못 일어서고 죽는다
    # (교육과정과 지도서처럼 어휘가 겹치는 경우).
    for k in range(n_anchor):
        m = A[:, k] == 1
        if m.sum() >= 3:
            proto = X[m].mean(axis=0)
            n = np.linalg.norm(proto)
            if n > 1e-9:
                P["W1"][:, k] = proto / n * 2.0
    N = len(X)
    has_anchor = A.sum(axis=1) > 0          # 앵커를 아는 기록만 제약
    hist = []
    for ep in range(epochs):
        H, Pr = _forward(X, P)
        # 1) 라벨 손실
        loss = -np.log(np.clip((Pr * Y).sum(1), 1e-12, None)).mean()
        dZ2 = (Pr - Y) / N
        dH = dZ2 @ P["W2"].T
        # 2) 앵커 제약: 앵커 뉴런끼리 '경쟁'시킨다.
        #    제곱오차로 각 축을 따로 맞추면, 입력이 비슷한 출처끼리는 한 축이
        #    통째로 죽어버린다(활성 0). softmax 경쟁이면 한 축이 이겨야 하므로
        #    모든 축이 살아남는다.
        if anchor_w > 0 and has_anchor.any():
            na = max(int(has_anchor.sum()), 1)
            Pa = _softmax(H[:, :n_anchor])
            loss += anchor_w * (-np.log(np.clip((Pa * A).sum(1), 1e-12, None))
                                * has_anchor).sum() / na
            dH = dH.copy()
            dH[:, :n_anchor] += anchor_w * ((Pa - A) * has_anchor[:, None]) / na
        dZ1 = dH * np.where(H > 0, 1.0, LEAK)
        gW2 = H.T @ dZ2 + l2 * P["W2"]
        gW1 = X.T @ dZ1 + l2 * P["W1"]
        P["W2"] -= lr * gW2
        P["b2"] -= lr * dZ2.sum(0)
        P["W1"] -= lr * gW1
        P["b1"] -= lr * dZ1.sum(0)
        if ep % 50 == 0:
            hist.append(round(float(loss), 4))
    P["hist"] = hist
    return P


# ── 학습 진입점 ──────────────────────────────────────────────
def _vectors(records, emb):
    X, keep = [], []
    for r in records:
        v = emb.embed_tokens(tokenize(r.text))
        if v is not None:
            X.append(v)
            keep.append(r)
    return (np.array(X) if X else np.zeros((0, emb.dim))), keep


def fit(subject, records, field="area", n_free=FREE_UNITS, holdout=0.2,
        min_per_class=5, anchor_w=0.5, epochs=400, seed=0, compare=True):
    """
    라벨(기본: 영역)을 맞히는 태거를 앵커 은닉층으로 학습.
    compare=True면 앵커 없는 같은 크기 신경망도 함께 학습해 정확도를 나란히 낸다.
    """
    import labeler as lb
    if not os.path.exists(paths.emb_path(subject)):
        raise ValueError(f"{subject}: 임베딩이 없어요. 먼저 학습이 필요합니다.")
    emb = FrozenEmbedding.load(paths.emb_path(subject))
    store = lb.load_labels(subject)
    labeled = [r for r in records
               if r.rec_id in store and "error" not in store[r.rec_id]
               and (store[r.rec_id].get(field) or "").strip()]
    X, keep = _vectors(labeled, emb)
    if len(keep) < 40:
        raise ValueError(f"라벨 있는 자료가 적어요 ({len(keep)}건, 은닉층은 40건 이상 권장)")

    vals = [store[r.rec_id][field].strip() for r in keep]
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    classes = sorted([c for c, n in counts.items() if n >= min_per_class])
    if len(classes) < 2:
        raise ValueError("라벨 종류가 부족해요")
    ci = {c: i for i, c in enumerate(classes)}
    idx = [i for i, v in enumerate(vals) if v in ci]
    X = X[idx]
    keep = [keep[i] for i in idx]
    vals = [vals[i] for i in idx]

    Y = np.zeros((len(keep), len(classes)))
    for i, v in enumerate(vals):
        Y[i, ci[v]] = 1
    A = np.zeros((len(keep), len(ANCHORS)))
    for i, r in enumerate(keep):
        a = anchor_of(r)
        if a in ANCHORS:
            A[i, ANCHORS.index(a)] = 1.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keep))
    n_te = max(5, int(len(keep) * holdout))
    te, tr = perm[:n_te], perm[n_te:]
    n_hidden = len(ANCHORS) + n_free

    def _run(aw):
        P = train(X[tr], Y[tr], A[tr], n_hidden, len(ANCHORS),
                  epochs=epochs, anchor_w=aw, seed=seed)
        _, Pr = _forward(X[te], P)
        acc = float((Pr.argmax(1) == Y[te].argmax(1)).mean())
        return P, acc

    P, acc = _run(anchor_w)
    base_acc = None
    if compare:
        _, base_acc = _run(0.0)              # 앵커 없는 같은 크기 신경망
    major = max(np.bincount(Y[te].argmax(1), minlength=len(classes))) / len(te)

    # 앵커 축이 실제로 서 있는지: 앵커별 평균 활성
    H_all, _ = _forward(X, P)
    axis = {}
    for k, a in enumerate(ANCHORS):
        m = A[:, k] == 1
        if m.any():
            own = float(H_all[m, k].mean())
            other = float(H_all[~m, k].mean()) if (~m).any() else 0.0
            axis[a] = {"자기활성": round(own, 3), "남의활성": round(other, 3),
                       "표본": int(m.sum())}

    model = {"params": P, "classes": classes, "field": field, "dim": emb.dim,
             "anchors": ANCHORS, "n_free": n_free, "anchor_w": anchor_w,
             "acc": acc, "base_acc": base_acc, "baseline": float(major),
             "n_train": len(tr), "n_test": len(te), "axis": axis,
             "at": time.time()}
    p = tagger_path(subject)
    with open(p, "wb") as f:
        pickle.dump(model, f)
    _cloud_push(p)
    return {k: v for k, v in model.items() if k != "params"}


def load(subject):
    p = tagger_path(subject)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ── 판단 + 검사 ──────────────────────────────────────────────
def explain(subject, text, model=None, emb=None):
    """
    한 문장을 판단하고, 그 판단이 어느 출처 앵커에서 나왔는지까지 낸다.
      label       예측 라벨
      anchors     앵커별 기여도 (활성 × 출력가중치, 합이 1)
      ablation    앵커를 끄면 판단이 얼마나 움직이나
      warning     근거가 약한 출처에만 기댄 경우
    """
    model = model or load(subject)
    if not model:
        return None
    emb = emb or FrozenEmbedding.load(paths.emb_path(subject))
    v = emb.embed_tokens(tokenize(text))
    if v is None:
        return None
    P = model["params"]
    X = v[None, :]
    H, Pr = _forward(X, P)
    k = int(Pr.argmax())
    label = model["classes"][k]

    n_a = len(model["anchors"])
    contrib = H[0, :n_a] * P["W2"][:n_a, k]
    pos = np.clip(contrib, 0, None)
    total = pos.sum() + 1e-12
    anchors = {a: round(float(pos[i] / total), 3)
               for i, a in enumerate(model["anchors"])}

    abl = {}
    for i, a in enumerate(model["anchors"]):
        H2 = H.copy()
        H2[0, i] = 0.0
        p2 = _softmax(H2 @ P["W2"] + P["b2"])
        abl[a] = round(float(abs(p2[0, k] - Pr[0, k])), 3)

    weak = sum(anchors.get(a, 0) for a in ("웹수집", "내답변", "개인필기"))
    warning = None
    if weak > 0.6:
        warning = "이 판단은 공식 자료(교육과정·지도서·기출)보다 개인·웹 자료에 기대고 있어요"
    elif max(anchors.values()) < 0.25:
        warning = "특정 출처 성격에 뚜렷이 기대지 않은 판단이에요"
    trust = round(sum(anchors.get(a, 0) * TRUST.get(a, 0.5)
                      for a in model["anchors"]), 3)
    return {"label": label, "prob": round(float(Pr[0, k]), 3),
            "anchors": anchors, "ablation": abl,
            "trust": trust, "warning": warning}
