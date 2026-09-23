"""
daily_digest.py — 매일 아침 '오늘 읽을 자료집'을 과목별로 자동 편성.

무엇을 읽힐지는 결정적으로 고른다(LLM이 고르지 않는다):
  · 약점      — 내가 확정한 채점 이력에서 정답률 낮은 성취기준/개념영역
  · 복습 주기 — 마지막으로 읽은 뒤 1·3·7·14·30일이 지난 항목(맞을수록 간격이 늘어남)
  · 경향      — 경향 랩이 다음 시험에 나온다고 본 대상(확률 높은 것 가산)
  · 미개척    — 아직 한 번도 읽지 않은 항목

각 항목에 실제 자료 원문을 출처와 함께 붙인다(없는 얘기를 만들지 않으려고
요약도 그 원문만 근거로 시킨다). OpenAI 키가 있으면 3줄 요약 + 확인 질문 2개까지.

저장: data/digest_{과목}.pkl
  {"days": {날짜: digest}, "schedule": {키: {last_read, streak, due}}}
"""
from __future__ import annotations
CORE_VERSION = "13.2"
import os, re, json, time, pickle, random
from collections import defaultdict, Counter

import paths
from schema import load_records_pkl
from study_state import StudyState

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

INTERVALS = [1, 3, 7, 14, 30, 60]      # 복습 간격(일) — 연속으로 읽을수록 뒤로
DAY = 86400


def digest_path(subject):
    return paths._p(f"digest_{subject}.pkl")


def load_store(subject):
    p = digest_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"days": {}, "schedule": {}}


def save_store(subject, store):
    store["days"] = dict(sorted(store["days"].items())[-30:])   # 최근 30일만
    p = digest_path(subject)
    with open(p, "wb") as f:
        pickle.dump(store, f)
    _cloud_push(p)


def today_str(ts=None):
    return time.strftime("%Y-%m-%d", time.localtime(ts or time.time()))


# ── 항목 고르기 ──────────────────────────────────────────────
KIND_BONUS = {"code": 0.6, "concept": 0.4, "node": 0.3, "area": 0.2,
              "unit": 0.2, "exam": 0.3, "source": 0.0}
MIN_GROUP = 2          # 이만큼은 모여야 한 항목


def _groups(subject, corpus):
    """
    묶을 수 있는 방법을 전부 쓴다. 코드가 없는 자료(스캔본·필기)도
    반드시 어딘가에 묶이도록 마지막에 '출처 묶음'이 받쳐준다.
    """
    g = {}

    def put(key, kind, title, r):
        g.setdefault(key, {"kind": kind, "title": title, "recs": []})["recs"].append(r)

    for r in corpus:
        if r.code:
            put(f"code:{r.code}", "code", r.code, r)
        for c in (r.concepts or []):
            put(f"concept:{c}", "concept", c, r)
        if r.area:
            put(f"area:{r.area}", "area", f"{r.area} 영역", r)
        if r.unit:
            put(f"unit:{r.unit}", "unit", r.unit, r)

    # SOM 개념영역(노드)별 묶음 — 학습된 지도가 있을 때
    try:
        from som import SOM
        import os as _os
        sp = paths.som_path(subject)
        if _os.path.exists(sp):
            som = SOM.load(sp)
            by_id = {r.rec_id: r for r in corpus}
            for node, ids in (som.node_rec_ids or {}).items():
                rs = [by_id[i] for i in ids if i in by_id]
                for r in rs:
                    put(f"node:{node}", "node", f"개념영역 {node}번", r)
    except Exception:
        pass

    # 기출은 시험 단위로도 묶는다 (기출만 있는 과목도 자료집이 나오도록)
    for r in corpus:
        if r.layer == "L1_pattern":
            base = r.source.rsplit(" p.", 1)[0] if " p." in r.source else r.source
            put(f"exam:{base}", "exam", f"기출 · {base}", r)

    # 출처 묶음(항상 가능) — 같은 자료를 4쪽씩 끊어서
    by_src = {}
    for r in corpus:
        base = r.source.rsplit(" p.", 1)[0] if " p." in r.source else r.source
        by_src.setdefault(base, []).append(r)
    for base, rs in by_src.items():
        rs = sorted(rs, key=lambda x: x.source)
        for k in range(0, len(rs), 4):
            part = rs[k:k + 4]
            if len(part) >= MIN_GROUP:
                key = f"source:{base}#{k // 4 + 1}"
                g[key] = {"kind": "source", "recs": part,
                          "title": f"{base} ({k + 1}~{k + len(part)}쪽)"}
    # 개념영역(노드) 제목에 대표 낱말을 붙인다 ('개념영역 12번' → '개념영역 12 · 쓰기/과정')
    from korean_tokenizer import tokenize
    stop = set("국어 학생 교사 지도 내용 활동 수업 학습 방법 경우 위해 대해 이다 하는 것이 "
               "자료 단원 차시 개정 교육 교육과정 평가 성취 기준 NUM URL".split())
    for key, gi in g.items():
        if gi["kind"] != "node":
            continue
        c = Counter()
        for r in gi["recs"][:40]:
            for t in set(tokenize(r.text)):
                if len(t) >= 2 and t not in stop:
                    c[t] += 1
        kw = [w for w, _ in c.most_common(3)]
        if kw:
            gi["title"] = f"{gi['title'].replace('번', '')} · {'/'.join(kw)}"
    return {k: v for k, v in g.items() if len(v["recs"]) >= MIN_GROUP}


def _candidates(subject, l2, common, st_, store, trend_targets):
    """묶음마다 점수를 매긴다(약점·복습주기·출제예상·미개척)."""
    corpus = list(l2) + list(common)
    groups = _groups(subject, corpus)
    now = time.time()
    sched = store["schedule"]
    out = []
    for key, gi in groups.items():
        s = sched.get(key, {})
        last, due = s.get("last_read"), s.get("due", 0)
        never = last is None
        score, why = KIND_BONUS.get(gi["kind"], 0.0), []
        if never:
            score += 1.0
            why.append("아직 안 읽음")
        elif now >= due:
            score += min(2.0, 0.6 + (now - due) / DAY * 0.1)
            why.append(f"복습 주기 도래(마지막 {int((now - last) / DAY)}일 전)")
        else:
            score -= 1.5                                   # 아직 이른 항목
        code = gi["title"] if gi["kind"] == "code" else None
        if code:
            c, t = st_.code_stats.get(code, [0, 0])
            if t:
                score += (1.0 - c / t) * 2.0
                why.append(f"정답률 {c / t:.0%} ({c}/{t})")
            if code in trend_targets:
                score += trend_targets[code]
                why.append(f"다음 시험 출제 예상 {trend_targets[code]:.0%}")
        out.append({"key": key, "kind": gi["kind"], "title": gi["title"],
                    "score": score, "why": why, "recs": gi["recs"], "never": never})
    out.sort(key=lambda d: -d["score"])
    return out


def _select(cands, n_items, st_):
    """
    항목을 고르면서 읽을 원문까지 정한다.
    이미 다른 항목에 들어간 원문은 빼고 고르므로, 큰 묶음이 작은 묶음을
    통째로 잡아먹어 항목 수가 줄어드는 일이 없다.
    """
    used, picked = set(), []
    for c in cands:
        left = [r for r in c["recs"] if r.rec_id not in used]
        if len(left) < 1:
            continue
        reads = _pick_reads(left, st_)
        if not reads:
            continue
        c = dict(c, reads=reads)
        picked.append(c)
        used |= {r["rec_id"] for r in reads}
        if len(picked) >= n_items:
            break
    return picked


def _trend_targets(subject):
    """경향 랩의 '채점 대기 중' 예측 중 코드 대상만 {코드: 확률}."""
    try:
        import trend_lab as tl
        lab = tl.load_lab(subject)
        out = {}
        for p in lab["predictions"]:
            if p["scope"] == "code" and not p["scored"]:
                out[p["target"]] = max(out.get(p["target"], 0), p["prob"])
        return out
    except Exception:
        return {}


def _pick_reads(recs, st_, limit=3, chars=1200):
    """읽을 원문 고르기 — 신뢰도 높은 것, 성취기준 문서 우선, 너무 길면 자름."""
    order = {"교육과정_성취기준": 0, "지도서_각론": 1, "지도서_총론": 2,
             "교육과정_총론": 3, "개인_필기": 4}
    rs = sorted(recs, key=lambda r: (order.get(r.doc_type, 5),
                                     -st_.trust.get(r.rec_id, 1.0), len(r.text)))
    out = []
    for r in rs[:limit]:
        t = r.text.strip()
        label = r.doc_type or ("기출" if r.layer == "L1_pattern" else None)
        out.append({"source": r.source, "doc_type": label, "rec_id": r.rec_id,
                    "text": t[:chars] + ("…" if len(t) > chars else "")})
    return out


# ── LLM 요약(선택) ───────────────────────────────────────────
SYSTEM = """너는 임용 수험생의 자료집을 만드는 조교다. 아래 '원문'만 근거로 쓴다.
원문에 없는 내용은 절대 쓰지 말 것. JSON 하나만 출력한다.
{"summary":["핵심 3줄. 각 줄 40자 내외"],
 "questions":["원문만 보고 답할 수 있는 확인 질문 2개"],
 "keyword":"이 항목을 한 단어로"}"""


def _summarize(item, api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    body = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in item["reads"])
    r = client.chat.completions.create(
        model=model, temperature=0.2, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"항목: {item['title']}\n\n원문:\n{body[:6000]}"}])
    d = json.loads(r.choices[0].message.content)
    return {"summary": [str(x) for x in (d.get("summary") or [])][:3],
            "questions": [str(x) for x in (d.get("questions") or [])][:2],
            "keyword": str(d.get("keyword") or "")[:30]}


# ── 편성 ─────────────────────────────────────────────────────
def build(subject, n_items=5, api_key=None, model="gpt-4o-mini", date=None,
          force=False, progress=None):
    """오늘자 자료집 만들기. 이미 있으면 그대로 돌려줌(force=True면 새로)."""
    store = load_store(subject)
    d = date or today_str()
    if d in store["days"] and not force:
        return store["days"][d], store

    l2 = load_records_pkl(paths.l2_path(subject))
    l1 = load_records_pkl(paths.l1_path(subject))
    common = load_records_pkl(paths.common_chongron_path())
    if not (l2 or l1 or common):
        raise ValueError(f"'{subject}' 자료가 아직 없어요. 자료나 기출을 먼저 넣어주세요.")
    st_ = StudyState.load(paths.study_path(subject), subject)
    cands = _candidates(subject, l2 + l1, common, st_, store, _trend_targets(subject))

    items = []
    for c in _select(cands, n_items, st_):
        items.append({"key": c["key"], "kind": c["kind"], "title": c["title"],
                      "why": c["why"] or ["기본 순환"], "score": round(c["score"], 2),
                      "reads": c["reads"],
                      "summary": [], "questions": [], "read": False})
    if api_key:
        for i, it in enumerate(items):
            if progress:
                progress(i, len(items))
            try:
                it.update(_summarize(it, api_key, model))
            except Exception as e:
                it["summary_error"] = str(e)
        if progress:
            progress(len(items), len(items))

    if not items:
        raise ValueError(f"묶을 자료가 부족해요 — 자료 {len(l2)}건 · 기출 {len(l1)}건 · "
                         f"공통 {len(common)}건. 같은 출처로 2쪽 이상 있으면 편성돼요.")
    digest = {"date": d, "subject": subject, "made_at": time.time(),
              "items": items, "pool": len(cands),
              "chars": sum(len(r["text"]) for it in items for r in it["reads"])}
    store["days"][d] = digest
    save_store(subject, store)
    return digest, store


def mark_read(subject, date, key, ok=True):
    """'읽음' 표시 → 다음 복습일을 뒤로 민다(연속으로 읽을수록 간격 증가)."""
    store = load_store(subject)
    dg = store["days"].get(date)
    if dg:
        for it in dg["items"]:
            if it["key"] == key:
                it["read"] = ok
    s = store["schedule"].setdefault(key, {"last_read": None, "streak": 0, "due": 0})
    if ok:
        s["last_read"] = time.time()
        s["streak"] = min(len(INTERVALS) - 1, s.get("streak", 0) + 1)
        s["due"] = time.time() + INTERVALS[s["streak"]] * DAY
    else:                                   # 읽음 취소
        s["streak"] = max(0, s.get("streak", 0) - 1)
        s["due"] = time.time()
    save_store(subject, store)
    return store


def recent(subject, n=7):
    store = load_store(subject)
    return [store["days"][k] for k in sorted(store["days"])[-n:]]
