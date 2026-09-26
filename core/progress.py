"""
progress.py — "좋아지고 있나"를 한 화면에서 답한다.

지표는 이미 여기저기 쌓여 있는데(자기검증 이력, 자가 시험 회차, 질문함, 경향 랩)
화면에는 늘 마지막 값 하나만 보였다. 그래서 매번 CSV를 뽑아 눈으로 비교해야 했다.
이 모듈은 그 기록들을 시간순으로 묶어 돌려준다.

  health(subject)   자기검증 이력 → 자료 수, 벡터화율, 양자화 오차, 빈 노드, 경고 수
  exam(subject)     자가 시험 회차 → 근거 검색·빈칸 성공률, 구멍 수, 전략 성적
  answers(subject)  질문함 → 답변 수, 그중 구멍을 메운 수
  trend(subject)    경향 예측 → 적중률, Brier, 기준선
  headline(subject) 위 넷에서 '문장으로 쓸 수 있는 변화'만 뽑음
"""
from __future__ import annotations
import time

import paths


def _t(epoch):
    return time.strftime("%m-%d %H:%M", time.localtime(epoch or 0))


def health(subject, n=30):
    try:
        import selfcheck
        hist = selfcheck.load_history(subject)[-n:]
    except Exception:
        return []
    out = []
    for h in hist:
        m = h.get("model") or {}
        out.append({"시각": _t(h.get("epoch")),
                    "자료": m.get("records"),
                    "벡터화": m.get("coverage"),
                    "양자화오차": m.get("qe"),
                    "빈노드": m.get("dead_nodes"),
                    "경고": len(h.get("alerts") or []),
                    "재학습": bool(h.get("retrained")) and not isinstance(
                        h.get("retrained"), str)})
    return out


def exam(subject, n=30):
    try:
        import self_exam as se
        lab = se.load(subject)
    except Exception:
        return [], []
    runs = []
    for r in (lab.get("runs") or [])[-n:]:
        rt, cz = r.get("retrieval") or {}, r.get("cloze") or {}
        runs.append({"시각": _t(r.get("at")),
                     "근거검색": rt.get("rate"),
                     "검색시행": rt.get("n"),
                     "빈칸복원": cz.get("rate"),
                     "구멍": r.get("gaps"),
                     "메움": len((r.get("retest") or {}).get("fixed") or [])})
    arms = []
    try:
        arms = se.arm_table(lab)
    except Exception:
        pass
    return runs, arms


def answers(subject):
    try:
        import ask_box as ab
        s = ab.summary(subject)
        box = ab.load(subject)
    except Exception:
        return {}, []
    rows = []
    for a in (box.get("answers") or [])[-20:]:
        rows.append({"시각": _t(a.get("at")), "종류": a.get("kind"),
                     "항목": str(a.get("key"))[:24],
                     "도움": {True: "○", False: "×", None: "확인중"}.get(a.get("helped"))})
    return s, rows


def trend(subject):
    try:
        import trend_lab as tl
        lab = tl.load_lab(subject)
        return tl.summary(lab["predictions"]), tl.top_rules(lab, n=5, min_n=1)
    except Exception:
        return {}, []


def _delta(series, key):
    vals = [r[key] for r in series if r.get(key) is not None]
    if len(vals) < 2:
        return None
    return vals[0], vals[-1]


def headline(subject):
    """문장으로 쓸 수 있는 변화만. (없으면 빈 목록)"""
    out = []
    h = health(subject)
    r0 = _delta(h, "자료")
    if r0 and r0[1] != r0[0]:
        out.append(f"자료 {r0[0]}건 → {r0[1]}건")
    q0 = _delta(h, "양자화오차")
    if q0 and abs(q0[1] - q0[0]) > 1e-4:
        out.append(f"양자화 오차 {q0[0]:.3f} → {q0[1]:.3f}"
                   + (" (개선)" if q0[1] < q0[0] else ""))
    d0 = _delta(h, "빈노드")
    if d0 and abs(d0[1] - d0[0]) > 0.01:
        out.append(f"빈 노드 {d0[0]:.0%} → {d0[1]:.0%}")
    runs, arms = exam(subject)
    e0 = _delta(runs, "근거검색")
    if e0:
        out.append(f"근거 검색 성공률 {e0[0]:.0%} → {e0[1]:.0%}")
    best = [a for a in arms if a.get("성공률") is not None]
    if best:
        b = best[0]
        out.append(f"가장 잘 맞는 전략: {b['전략']} {b['성공률']:.0%} ({b['시행']}회)")
    s, _ = answers(subject)
    if s.get("answered"):
        out.append(f"내 답변 {s['answered']}개 · 그중 구멍을 메운 것 {s.get('helped', 0)}개")
    t, _ = trend(subject)
    if t.get("scored"):
        out.append(f"경향 예측 {t['scored']}건 채점 · Brier {t['brier']:.3f} "
                   f"(기준선 {t['brier_baseline']:.3f})")
    return out


def overview():
    """전 과목 한 줄 요약."""
    from schema import SUBJECTS
    rows = []
    for s in sorted(SUBJECTS):
        h = health(s, n=2)
        if not h:
            continue
        last = h[-1]
        runs, arms = exam(s, n=1)
        rows.append({"과목": s, "자료": last["자료"],
                     "빈노드": last["빈노드"], "경고": last["경고"],
                     "근거검색": (runs[-1]["근거검색"] if runs else None),
                     "회차": len(health(s))})
    rows.sort(key=lambda r: -(r["자료"] or 0))
    return rows
