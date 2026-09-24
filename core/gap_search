"""
gap_search.py — 자가 시험에서 '근거를 못 찾은 곳'만 웹에서 찾는다.

원칙
  1. 답을 찾으려고 검색하지 않는다. **내게 없는 자료가 무엇인지** 찾으려고 검색한다.
  2. 찾은 것은 바로 자료가 되지 않는다. 수집함에 후보로 쌓이고, 사람이 채택해야 들어간다.
     (출처 강제 원칙을 웹으로 뚫지 않기 위해서)
  3. 채택한 출처는 성적이 추적된다. 그 자료가 들어온 뒤 같은 구멍이 메워졌는지를
     자가 시험이 다시 확인한다. 도움이 된 도메인은 점수가 오르고, 아닌 곳은 내려간다.

저장: data/webfind_{과목}.pkl
  {"candidates": [...], "domains": {도메인: {"seen","accepted","helped"}}, "log": [...]}
"""
from __future__ import annotations
import os, re, time, pickle, hashlib
from urllib.parse import urlparse

import paths

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

# 교육과정 자료로 신뢰할 만한 곳 (도메인 끝부분 기준)
TRUSTED = {
    "moe.go.kr": 1.0, "kice.re.kr": 1.0, "ncic.re.kr": 1.0, "edunet.net": 0.9,
    "kedi.re.kr": 0.9, "keris.or.kr": 0.85, "go.kr": 0.8, "ac.kr": 0.7,
    "or.kr": 0.6, "re.kr": 0.8,
}
BLOCKED = ("blog.", "cafe.", "tistory.com", "pinterest.", "facebook.", "x.com",
           "instagram.", "youtube.")


def find_path(subject):
    return paths._p(f"webfind_{subject}.pkl")


def load(subject):
    p = find_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"candidates": [], "domains": {}, "log": []}


def save(subject, box):
    box["candidates"] = box["candidates"][-300:]
    box["log"] = box["log"][-200:]
    p = find_path(subject)
    with open(p, "wb") as f:
        pickle.dump(box, f)
    _cloud_push(p)


def domain_of(url):
    try:
        h = urlparse(url).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def trust(url, box):
    """도메인 신뢰도 0~1: 기관 가중치 + 지금까지 실제로 도움이 됐는지."""
    d = domain_of(url)
    if not d or any(b in d for b in BLOCKED):
        return 0.0
    base = 0.35
    for suffix, w in TRUSTED.items():
        if d.endswith(suffix):
            base = max(base, w)
    st = box["domains"].get(d)
    if st and st.get("accepted"):
        base = min(1.0, base + 0.3 * (st.get("helped", 0) / st["accepted"]))
    return round(base, 2)


def make_query(gap, subject):
    """구멍 하나 → 검색어. 코드가 있으면 코드 그대로가 제일 정확하다."""
    if gap.get("code"):
        return f'{gap["code"]} 성취기준 해설'
    bits = [subject, gap.get("area") or "", "교육과정", "지도"]
    sample = re.sub(r"\s+", " ", (gap.get("sample") or ""))[:40]
    return " ".join([b for b in bits if b] + [sample])[:120]


def collect(subject, gaps, brave_key, per_query=5, log=print):
    """구멍마다 검색해서 후보를 수집함에 쌓는다. 자료로 넣지는 않는다."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "layer3_trend"))
    from layer3 import brave_search

    box = load(subject)
    seen = {c["url"] for c in box["candidates"]}
    added = 0
    for g in gaps:
        q = make_query(g, subject)
        try:
            results = brave_search(q, brave_key, count=per_query)
        except Exception as e:
            log(f"  검색 실패({q[:30]}): {e}")
            continue
        for r in results:
            url = r.get("source") or ""
            if not url or url in seen:
                continue
            t = trust(url, box)
            if t <= 0:
                continue                      # 블로그·SNS 등은 후보에도 안 넣음
            d = domain_of(url)
            box["domains"].setdefault(d, {"seen": 0, "accepted": 0, "helped": 0})
            box["domains"][d]["seen"] += 1
            box["candidates"].append({
                "id": hashlib.sha1((url + q).encode()).hexdigest()[:10],
                "gap_key": g["key"], "query": q, "url": url, "domain": d,
                "title": r.get("title", "")[:150], "text": (r.get("text") or "")[:1200],
                "trust": t, "status": "new", "at": time.time()})
            seen.add(url)
            added += 1
        g["searched"] = True
        log(f"  · {g['key']} → 후보 {added}건 누적")
    save(subject, box)
    return added


def pending(subject, min_trust=0.0):
    box = load(subject)
    out = [c for c in box["candidates"] if c["status"] == "new"
           and c["trust"] >= min_trust]
    out.sort(key=lambda c: -c["trust"])
    return out


def accept(subject, cand_ids, doc_type="웹수집"):
    """
    채택 = 자료로 편입. 출처는 URL 그대로 남기고, 어느 구멍을 메우려 한 것인지 기록한다.
    (나중에 자가 시험이 그 구멍을 다시 풀어보고 도움이 됐는지 확인한다)
    """
    from schema import Record, load_records_pkl, save_records_pkl
    box = load(subject)
    ids = set(cand_ids)
    path = paths.l2_path(subject)
    recs = load_records_pkl(path)
    have = {r.rec_id for r in recs}
    n = 0
    for c in box["candidates"]:
        if c["id"] not in ids or c["status"] != "new":
            continue
        text = f"{c['title']}\n{c['text']}".strip()
        if len(text) < 40:
            c["status"] = "rejected"
            continue
        try:
            rec = Record(text=text, layer="L2_corpus", subject=subject,
                         source=c["url"], doc_type=doc_type,
                         code=c.get("gap_key") if str(c.get("gap_key", "")).startswith("[")
                         else None)
        except Exception:
            c["status"] = "rejected"
            continue
        if rec.rec_id not in have:
            recs.append(rec)
            have.add(rec.rec_id)
            n += 1
        c["status"] = "accepted"
        c["accepted_at"] = time.time()
        d = box["domains"].setdefault(c["domain"], {"seen": 0, "accepted": 0, "helped": 0})
        d["accepted"] += 1
        box["log"].append({"kind": "accept", "url": c["url"], "gap": c["gap_key"],
                           "at": time.time()})
    if n:
        save_records_pkl(recs, path)
    save(subject, box)
    return n


def reject(subject, cand_ids):
    box = load(subject)
    ids = set(cand_ids)
    n = 0
    for c in box["candidates"]:
        if c["id"] in ids and c["status"] == "new":
            c["status"] = "rejected"
            n += 1
    save(subject, box)
    return n


def mark_helped(subject, gap_key, helped=True):
    """자가 시험이 그 구멍을 다시 풀어 성공하면, 그 구멍에 채택된 출처의 점수를 올린다."""
    box = load(subject)
    for c in box["candidates"]:
        if c["gap_key"] == gap_key and c["status"] == "accepted":
            d = box["domains"].setdefault(c["domain"],
                                          {"seen": 0, "accepted": 0, "helped": 0})
            if helped:
                d["helped"] += 1
            box["log"].append({"kind": "helped" if helped else "no_help",
                               "url": c["url"], "gap": gap_key, "at": time.time()})
    save(subject, box)


def domain_table(subject):
    box = load(subject)
    rows = []
    for d, s in box["domains"].items():
        rows.append({"도메인": d, "노출": s["seen"], "채택": s["accepted"],
                     "도움됨": s["helped"],
                     "도움률": (s["helped"] / s["accepted"]) if s["accepted"] else None})
    rows.sort(key=lambda r: (-(r["도움률"] or -1), -r["채택"]))
    return rows
