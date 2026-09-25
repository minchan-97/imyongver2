"""
inventory.py — 지금 자료가 어디에 얼마나 있는지 한눈에.

"52개 넣었는데 국어에 47건뿐"처럼 자료가 어디로 갔는지 확인할 때 쓴다.
과목을 하나씩 눌러 보지 않아도 전체가 한 화면에 나온다.

  subjects()  과목별 자료·기출 건수, 글자 수, 학습 상태(임베딩·SOM)
  sources()   출처(문서)별 쪽수 — 어느 과목에 들어가 있는지까지
  files()     서버에 보관된 원본 파일 ↔ 실제로 들어간 쪽수 대조
  queue()     스캔 대기열 상태와 실패 목록
"""
from __future__ import annotations
import os
from collections import Counter, defaultdict

import paths
from schema import SUBJECTS, load_records_pkl


def _stats(recs):
    return {"건수": len(recs),
            "글자": sum(len(r.text) for r in recs),
            "영역있음": sum(1 for r in recs if r.area),
            "코드있음": sum(1 for r in recs if r.code)}


def subjects(include_empty=False):
    rows = []
    for s in sorted(SUBJECTS - {"공통"}):
        l2 = load_records_pkl(paths.l2_path(s))
        l1 = load_records_pkl(paths.l1_path(s))
        if not (l2 or l1) and not include_empty:
            continue
        a, b = _stats(l2), _stats(l1)
        rows.append({
            "과목": s, "자료": a["건수"], "기출": b["건수"],
            "글자": a["글자"] + b["글자"],
            "영역": a["영역있음"] + b["영역있음"],
            "코드": a["코드있음"] + b["코드있음"],
            "임베딩": "○" if os.path.exists(paths.emb_path(s)) else "",
            "SOM": "○" if os.path.exists(paths.som_path(s)) else "",
        })
    rows.sort(key=lambda r: -(r["자료"] + r["기출"]))
    common = load_records_pkl(paths.common_chongron_path())
    if common:
        c = _stats(common)
        rows.append({"과목": "공통(총론)", "자료": c["건수"], "기출": 0,
                     "글자": c["글자"], "영역": c["영역있음"],
                     "코드": c["코드있음"], "임베딩": "", "SOM": ""})
    return rows


def _base(src):
    b = (src or "").rsplit(" (", 1)[0]
    return b.rsplit(" p.", 1)[0] if " p." in b else b


def sources():
    """출처(문서) 단위 집계 — 한 문서가 여러 과목에 흩어졌는지도 보인다."""
    agg = defaultdict(lambda: {"쪽": 0, "과목": Counter(), "종류": Counter(),
                               "층": Counter()})
    for s in sorted(SUBJECTS - {"공통"}):
        for path, layer in ((paths.l2_path(s), "자료"), (paths.l1_path(s), "기출")):
            for r in load_records_pkl(path):
                a = agg[_base(r.source)]
                a["쪽"] += 1
                a["과목"][s] += 1
                a["층"][layer] += 1
                if r.doc_type:
                    a["종류"][r.doc_type] += 1
    rows = []
    for name, a in agg.items():
        show = name
        if show.startswith("http"):
            from urllib.parse import urlparse
            show = f"🌐 {urlparse(show).netloc}"
        rows.append({"출처": show, "쪽": a["쪽"],
                     "과목": ", ".join(f"{k} {v}" for k, v in a["과목"].most_common()),
                     "종류": ", ".join(k for k, _ in a["종류"].most_common(2)),
                     "층": ", ".join(k for k, _ in a["층"].most_common())})
    rows.sort(key=lambda r: -r["쪽"])
    return rows


def files():
    """서버에 보관된 원본 ↔ 실제 들어간 쪽수 대조 (안 들어간 파일 찾기)."""
    try:
        import cloud
        ups = cloud.list_uploads()
    except Exception:
        ups = []
    if not ups:
        return []
    src_pages = {r["출처"]: r["쪽"] for r in sources()}
    rows = []
    for u in ups:
        stem = os.path.splitext(u.get("original_name", ""))[0]
        pages = src_pages.get(stem)
        if pages is None:                       # 이름이 조금 달라도 찾아본다
            hit = [k for k in src_pages if stem[:12] and stem[:12] in k]
            pages = src_pages[hit[0]] if hit else 0
        rows.append({"원본 파일": u.get("original_name", ""),
                     "올린 과목": u.get("subject", ""),
                     "크기KB": (u.get("size_bytes") or 0) // 1024,
                     "들어간 쪽": pages,
                     "상태": "✅" if pages else "⚠️ 아직 안 들어감"})
    rows.sort(key=lambda r: (r["들어간 쪽"] != 0, r["원본 파일"]))
    return rows


def queue():
    try:
        import ingest_queue as iq
        q = iq.load()
    except Exception:
        return {"대기": 0, "완료": 0, "실패": 0, "실패목록": [], "최근": []}
    jobs = q.get("jobs", [])
    fail = [j for j in jobs if j["status"] == "error"]
    done = [j for j in jobs if j["status"] == "done"]
    prog = [{"파일": j["source"],
             "진행": f"{j.get('next_page', 0)}/{(j.get('result') or {}).get('total_pages', '?')}쪽"}
            for j in jobs if j["status"] == "queued" and j.get("next_page")]
    return {"진행중": prog,
            "대기": len([j for j in jobs if j["status"] == "queued"]),
            "완료": len(done), "실패": len(fail),
            "실패목록": [{"파일": j["source"],
                        "이유": str((j.get("result") or {}).get("error", ""))[:120]}
                       for j in fail[-10:]],
            "최근": [{"파일": j["source"],
                     "쪽": (j.get("result") or {}).get("added", 0),
                     "과목": (j.get("result") or {}).get("subject", "")}
                    for j in done[-10:]]}


def summary():
    ss = subjects()
    return {"과목수": len(ss), "총자료": sum(r["자료"] for r in ss),
            "총기출": sum(r["기출"] for r in ss),
            "총글자": sum(r["글자"] for r in ss),
            "문서수": len(sources())}
