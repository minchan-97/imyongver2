"""
selfcheck.py — 자기검증. 앱 버튼으로도, 나중에 워커/크론으로도 같은 코드가 돈다.

    python core/selfcheck.py --subject 국어            # 점검만
    python core/selfcheck.py --subject 국어 --fix      # 기준 미달이면 재학습까지

세 갈래:
  1) 데이터 위생  — 출처 부실, 중복 쪽, 손글씨 판독 불안(?·[판독불가]) → 재스캔 후보
  2) 모델 품질    — SOM 양자화 오차 추이, 노드 붕괴, 임베딩 커버리지 → 나빠지면 재학습
  3) 회귀 테스트  — 내가 푼 문항의 근거(성취기준 코드·SOM 노드 자료)가 지금도 실재하는지

결과는 data/health_{과목}.pkl에 최근 30회까지 쌓이고 서버에도 백업된다.
추이(직전 회차 대비)가 있어야 '나빠졌다'를 말할 수 있어서 이력을 남긴다.
"""
from __future__ import annotations
import os, re, sys, time, pickle, hashlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import paths
from schema import load_records_pkl
from embedding import FrozenEmbedding, train_embedding
from som import SOM
from korean_tokenizer import tokenize
from study_state import StudyState

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

# 기준값 — 넘으면 경고, --fix면 재학습
TH = {
    "qe_worsen": 0.15,        # 직전 대비 양자화 오차 15% 악화
    "dead_nodes": 0.45,       # 빈 노드 비율
    "coverage": 0.85,         # 새 자료 벡터화 성공률
    "ungrounded": 0.20,       # 근거 없는 문항 비율
    "uncertain_page": 3,      # 한 쪽에 (?)/[판독불가] 이 이상이면 재스캔 후보
}
UNCERTAIN = re.compile(r"\(\?\)|\[판독불가\]")


def health_path(subject):
    return paths._p(f"health_{subject}.pkl")


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w가-힣]", "", t or "")).strip().lower()


# ── 1) 데이터 위생 ────────────────────────────────────────────
def check_hygiene(l2, l1, common):
    recs = list(l2) + list(l1) + list(common)
    weak_source, dup, uncertain, short, no_code, no_year = [], [], [], [], [], []
    seen = {}
    for r in recs:
        src = (r.source or "").strip()
        if len(src) < 3 or src.lower() in ("미상", "unknown", "출처", "-"):
            weak_source.append({"rec_id": r.rec_id, "source": src, "text": r.text[:60]})
        key = _norm(r.text)[:300]
        if len(key) >= 15:      # 너무 짧은 문장은 우연히 같을 수 있어 제외
            if key in seen:
                dup.append({"rec_id": r.rec_id, "source": src,
                            "same_as": seen[key], "text": r.text[:60]})
            else:
                seen[key] = src
        n_unc = len(UNCERTAIN.findall(r.text))
        if n_unc >= TH["uncertain_page"]:
            uncertain.append({"rec_id": r.rec_id, "source": src, "marks": n_unc,
                              "text": r.text[:60]})
        if len(r.text.strip()) < 25:
            short.append({"rec_id": r.rec_id, "source": src, "text": r.text[:60]})
        if r.doc_type == "교육과정_성취기준" and not r.code:
            no_code.append({"rec_id": r.rec_id, "source": src, "text": r.text[:60]})
        if r.layer == "L1_pattern" and not r.year:
            no_year.append({"rec_id": r.rec_id, "source": src, "text": r.text[:60]})
    return {"total": len(recs), "weak_source": weak_source, "duplicate": dup,
            "uncertain_scan": uncertain, "too_short": short,
            "achievement_no_code": no_code, "exam_no_year": no_year,
            "rescan_candidates": [x["source"] for x in uncertain][:50]}


# ── 2) 모델 품질 ─────────────────────────────────────────────
def check_model(subject, l2, prev=None):
    out = {"trained": False}
    ep, sp = paths.emb_path(subject), paths.som_path(subject)
    if not (os.path.exists(ep) and os.path.exists(sp)):
        out["note"] = "임베딩/SOM이 아직 없어요 — 학습 필요"
        out["needs_retrain"] = len(l2) >= 3
        return out
    emb, som = FrozenEmbedding.load(ep), SOM.load(sp)
    X, kept = [], []
    for r in l2:
        v = emb.embed_tokens(tokenize(r.text))
        if v is not None:
            X.append(v); kept.append(r)
    coverage = len(kept) / max(len(l2), 1)
    qe = dead = top_share = None
    if X:
        Xa = np.array(X)
        bmus = [som.bmu_of(x) for x in Xa]
        qe = float(np.mean([1.0 - float(som.W[b] @ x) for b, x in zip(bmus, Xa)]))
        n_nodes = som.gh * som.gw
        hits = Counter(bmus)
        dead = 1.0 - len(hits) / n_nodes
        top_share = max(hits.values()) / len(bmus)
    prev_qe = (prev or {}).get("model", {}).get("qe")
    worsen = (qe - prev_qe) / prev_qe if (qe and prev_qe) else 0.0
    reasons = []
    if coverage < TH["coverage"]:
        reasons.append(f"새 자료 벡터화 성공률 {coverage:.0%} (< {TH['coverage']:.0%})")
    if dead is not None and dead > TH["dead_nodes"]:
        reasons.append(f"빈 노드 {dead:.0%} (> {TH['dead_nodes']:.0%})")
    if worsen > TH["qe_worsen"]:
        reasons.append(f"양자화 오차 {worsen:+.0%} 악화")
    out.update({"trained": True, "vocab": len(emb.word2idx), "dim": emb.dim,
                "grid": [som.gh, som.gw], "vectorized": len(kept), "records": len(l2),
                "coverage": coverage, "qe": qe, "qe_prev": prev_qe, "qe_delta": worsen,
                "dead_nodes": dead, "top_node_share": top_share,
                "needs_retrain": bool(reasons), "reasons": reasons})
    return out


def retrain(subject, l2, dim=32, grid=10, iters=4000):
    """임베딩 + SOM 재학습 (앱의 '학습 시작'과 같은 절차)."""
    texts = [r.text for r in l2]
    emb = train_embedding(texts, dim=dim, min_count=1, epochs=30)
    emb.save(paths.emb_path(subject))
    X, kept = [], []
    for r in l2:
        v = emb.embed_tokens(tokenize(r.text))
        if v is not None:
            X.append(v); kept.append(r)
    if not X:
        raise RuntimeError("벡터화 실패 — 자료가 너무 적거나 토큰이 없음")
    som = SOM(grid=(grid, grid), dim=emb.dim)
    som.train(np.array(X), iters=iters)
    som.assign(np.array(X), kept)
    som.save(paths.som_path(subject))
    return {"vocab": len(emb.word2idx), "vectorized": len(kept), "records": len(l2)}


# ── 3) 회귀 테스트 (근거 실재성) ──────────────────────────────
def check_regression(subject, l2, common):
    """
    내가 실제로 푼 문항들이 근거로 삼았던 것:
      - 성취기준 코드 → 그 코드를 가진 자료가 지금도 있는가
      - SOM 노드      → 그 노드에 배정된 자료(rec_id)가 지금도 있는가
    하나도 없으면 '근거 없는 문항'. LLM 없이 결정적으로 돌아간다.
    """
    st_ = StudyState.load(paths.study_path(subject), subject)
    corpus = list(l2) + list(common)
    have_ids = {r.rec_id for r in corpus}
    have_codes = {r.code for r in corpus if r.code}
    som = SOM.load(paths.som_path(subject)) if os.path.exists(paths.som_path(subject)) else None

    codes = getattr(st_, "code_stats", None) or {}
    nodes = getattr(st_, "node_stats", None) or {}
    dead_codes = sorted(c for c in codes if c and c not in have_codes)
    dead_nodes, checked_nodes = [], 0
    if som is not None:
        for n in nodes:
            try:
                n = int(n)
            except Exception:
                continue
            checked_nodes += 1
            ids = som.node_rec_ids.get(n, [])
            if not ids or not (set(ids) & have_ids):
                dead_nodes.append(n)
    items = len(codes) + checked_nodes
    ungrounded = len(dead_codes) + len(dead_nodes)
    rate = ungrounded / items if items else 0.0
    return {"checked_codes": len(codes), "checked_nodes": checked_nodes,
            "ungrounded_codes": dead_codes[:30], "ungrounded_nodes": dead_nodes[:30],
            "ungrounded_rate": rate,
            "alert": rate > TH["ungrounded"] and items >= 5,
            "note": "인용한 근거가 지금 자료에 실재하는지만 봅니다(내용 정확성 평가 아님)."}


# ── 실행 ─────────────────────────────────────────────────────
def load_history(subject):
    p = health_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return []


def run(subject, fix=False, dim=32, grid=10):
    hist = load_history(subject)
    prev = hist[-1] if hist else None
    l2 = load_records_pkl(paths.l2_path(subject))
    l1 = load_records_pkl(paths.l1_path(subject))
    common = load_records_pkl(paths.common_chongron_path())

    rep = {"subject": subject, "epoch": time.time(),
           "when": time.strftime("%Y-%m-%d %H:%M", time.localtime())}
    rep["hygiene"] = check_hygiene(l2, l1, common)
    try:
        import resubject
        rows = resubject.audit([subject])
        rep["subject_mix"] = {"suspect": len(rows), "moves": resubject.summary(rows),
                              "examples": rows[:10]}
    except Exception as e:
        rep["subject_mix"] = {"error": str(e)}
    rep["model"] = check_model(subject, l2, prev)
    rep["regression"] = check_regression(subject, l2, common)
    rep["retrained"] = None
    if fix and rep["model"].get("needs_retrain") and len(l2) >= 3:
        try:
            rep["retrained"] = retrain(subject, l2, dim=dim, grid=grid)
            rep["model_after"] = check_model(subject, l2, prev)
        except Exception as e:
            rep["retrained"] = {"error": str(e)}

    h = rep["hygiene"]
    rep["alerts"] = []
    for label, key in [("출처 부실", "weak_source"), ("중복 쪽", "duplicate"),
                       ("판독 불안(재스캔 후보)", "uncertain_scan"),
                       ("코드 없는 성취기준", "achievement_no_code"),
                       ("연도 없는 기출", "exam_no_year")]:
        if h[key]:
            rep["alerts"].append(f"{label} {len(h[key])}건")
    _sm = rep.get("subject_mix", {})
    if _sm.get("suspect"):
        _mv = ", ".join(f"{m['from']}→{m['to']} {m['n']}건" for m in _sm["moves"][:3])
        rep["alerts"].append(f"과목이 다르게 판정된 자료 {_sm['suspect']}건 ({_mv})")
    rep["alerts"] += rep["model"].get("reasons", [])
    if rep["regression"]["alert"]:
        rep["alerts"].append(f"근거 없는 문항 {rep['regression']['ungrounded_rate']:.0%}")

    hist.append(rep)
    hist = hist[-30:]
    p = health_path(subject)
    with open(p, "wb") as f:
        pickle.dump(hist, f)
    _cloud_push(p)
    return rep


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="국어")
    ap.add_argument("--fix", action="store_true", help="기준 미달이면 재학습까지")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--grid", type=int, default=10)
    a = ap.parse_args()
    r = run(a.subject, a.fix, a.dim, a.grid)
    print(json.dumps({k: v for k, v in r.items() if k != "hygiene"},
                     ensure_ascii=False, indent=2, default=str))
    print("경고:", r["alerts"] or "없음")
