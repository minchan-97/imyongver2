"""
labeler.py — 2층: LLM이 붙인 라벨을 쌓아 '로컬 태거'로 증류한다.

흐름
  1. LLM이 기출/자료 쪽마다 라벨을 붙인다 (영역·발문유형·난도).
     rec_id로 캐시하므로 같은 쪽은 두 번 부르지 않는다. → data/labels_{과목}.pkl
  2. 라벨이 쌓이면 로컬 분류기를 학습한다.
     입력 = 기존 FrozenEmbedding 벡터(추가 의존성 없음), 모델 = numpy 다항 로지스틱 회귀.
     → data/tagger_{과목}.pkl
  3. 그 뒤로는 LLM 없이도 새 쪽에 영역·유형을 즉시 붙일 수 있다.
     라벨이 늘수록 다시 학습해 좋아진다(홀드아웃 정확도로 확인).

로컬 태거는 '제안'이다. 자료에 쓰는 값은 사람이 확정한다.
"""
from __future__ import annotations
CORE_VERSION = "13.2"
import os, json, pickle, time
import numpy as np

import paths
from korean_tokenizer import tokenize
from embedding import FrozenEmbedding

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

QTYPES = ["개념설명", "사례적용", "지도방안", "자료해석", "비교분석", "서술평가", "기타"]
LEVELS = ["상", "중", "하"]

SYSTEM = """너는 임용 기출/자료 한 쪽을 읽고 라벨만 붙이는 분류기다. JSON 하나만 출력한다.
{"area":"영역(듣기·말하기/읽기/쓰기/문법/문학/매체 등, 모르면 \\"\\")",
 "qtype":"개념설명|사례적용|지도방안|자료해석|비교분석|서술평가|기타",
 "level":"상|중|하",
 "why":"한 줄 근거"}
본문에 없는 내용을 추측해 넣지 말 것."""


def labels_path(subject): return paths._p(f"labels_{subject}.pkl")
def tagger_path(subject): return paths._p(f"tagger_{subject}.pkl")


def load_labels(subject):
    p = labels_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def save_labels(subject, d):
    p = labels_path(subject)
    with open(p, "wb") as f:
        pickle.dump(d, f)
    _cloud_push(p)


# ── 1) LLM 라벨링 ────────────────────────────────────────────
def label_records(subject, records, api_key, model="gpt-4o-mini", limit=None,
                  workers=6, progress=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    store = load_labels(subject)
    todo = [r for r in records if r.rec_id not in store][:limit or len(records)]
    if not todo:
        return store, 0
    client = OpenAI(api_key=api_key)

    def work(r):
        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": r.text[:3000]}])
        d = json.loads(resp.choices[0].message.content)
        return {"area": str(d.get("area") or "").strip(),
                "qtype": d.get("qtype") if d.get("qtype") in QTYPES else "기타",
                "level": d.get("level") if d.get("level") in LEVELS else "중",
                "why": str(d.get("why") or "")[:200],
                "source": r.source, "year": r.year, "at": time.time(), "by": model}

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                store[r.rec_id] = fut.result()
            except Exception as e:
                store[r.rec_id] = {"error": str(e), "at": time.time()}
            done += 1
            if progress:
                progress(done, len(todo))
    save_labels(subject, store)
    return store, done


# ── 2) 로컬 태거 (다항 로지스틱 회귀, numpy만) ────────────────
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit(X, y, n_class, epochs=300, lr=0.5, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (X.shape[1], n_class))
    b = np.zeros(n_class)
    Y = np.zeros((len(y), n_class))
    Y[np.arange(len(y)), y] = 1
    for _ in range(epochs):
        P = _softmax(X @ W + b)
        G = (P - Y) / len(y)
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def _vectors(records, emb):
    X, keep = [], []
    for r in records:
        v = emb.embed_tokens(tokenize(r.text))
        if v is not None:
            X.append(v); keep.append(r)
    return (np.array(X) if X else np.zeros((0, emb.dim))), keep


def train_tagger(subject, records, fields=("area", "qtype", "level"), holdout=0.2,
                 min_per_class=3, seed=0):
    """라벨 있는 기록으로 분류기 학습. 반환: 필드별 홀드아웃 정확도."""
    emb = FrozenEmbedding.load(paths.emb_path(subject))
    store = load_labels(subject)
    labeled = [r for r in records if r.rec_id in store and "error" not in store[r.rec_id]]
    X, keep = _vectors(labeled, emb)
    if len(keep) < 10:
        raise ValueError(f"라벨 있는 자료가 너무 적어요 ({len(keep)}건, 최소 10건)")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(keep))
    n_te = max(1, int(len(keep) * holdout))
    te, tr = idx[:n_te], idx[n_te:]

    model = {"dim": emb.dim, "fields": {}, "n_train": len(tr), "n_test": len(te),
             "at": time.time()}
    report = {}
    for f in fields:
        vals = [store[keep[i].rec_id].get(f) or "" for i in range(len(keep))]
        classes = [c for c, n in _count(vals).items() if c and n >= min_per_class]
        if len(classes) < 2:
            report[f] = {"skip": "라벨 종류가 부족해요"}
            continue
        ci = {c: i for i, c in enumerate(classes)}
        tr_i = [i for i in tr if vals[i] in ci]
        te_i = [i for i in te if vals[i] in ci]
        if len(tr_i) < 5 or not te_i:
            report[f] = {"skip": "학습/검증 표본 부족"}
            continue
        W, b = _fit(X[tr_i], np.array([ci[vals[i]] for i in tr_i]), len(classes))
        pred = _softmax(X[te_i] @ W + b).argmax(1)
        truth = np.array([ci[vals[i]] for i in te_i])
        acc = float((pred == truth).mean())
        major = max(_count([vals[i] for i in te_i]).values()) / len(te_i)
        model["fields"][f] = {"W": W, "b": b, "classes": classes}
        report[f] = {"accuracy": acc, "baseline": major, "classes": len(classes),
                     "train": len(tr_i), "test": len(te_i)}
    p = tagger_path(subject)
    with open(p, "wb") as f_:
        pickle.dump(model, f_)
    _cloud_push(p)
    model["report"] = report
    return report


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def load_tagger(subject):
    p = tagger_path(subject)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def tag(subject, text, tagger=None, emb=None):
    """LLM 없이 로컬 태거로 라벨 제안. 반환: {field: (값, 확신도)}"""
    tagger = tagger or load_tagger(subject)
    if not tagger:
        return None
    emb = emb or FrozenEmbedding.load(paths.emb_path(subject))
    v = emb.embed_tokens(tokenize(text))
    if v is None:
        return None
    out = {}
    for f, m in tagger["fields"].items():
        p = _softmax((v[None, :] @ m["W"]) + m["b"])[0]
        i = int(p.argmax())
        out[f] = (m["classes"][i], float(p[i]))
    return out
