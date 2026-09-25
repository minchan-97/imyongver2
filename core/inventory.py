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
import os, re
from collections import Counter, defaultdict

import paths
from schema import SUBJECTS, load_records_pkl


def _stats(recs):
    return {"건수": len(recs),
            "글자": sum(len(r.text) for r in recs),
            "영역있음": sum(1 for r in recs if r.area),
            "코드있음": sum(1 for r in recs if r.code)}


def sync_all(log=None):
    """
    전체 현황을 보기 전에 모든 과목 파일을 서버에서 받아온다.
    (앱은 고른 과목만 동기화하므로, 안 들어가 본 과목은 로컬에 파일이 없어
     '자료 0'으로 보인다 — 실제로는 서버에 있다)
    """
    try:
        import cloud
    except Exception:
        return {"받음": 0, "과목": []}
    if not cloud.enabled():
        return {"받음": 0, "과목": []}
    got, subs = 0, []
    names = cloud.list_local_names()
    for s in paths.discover_subjects(names):
        rep = cloud.sync(paths.all_paths(s))
        if rep["down"]:
            got += len(rep["down"])
            subs.append(s)
        if log:
            log(s)
    return {"받음": got, "과목": subs}


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
    """
    서버 보관 원본 ↔ 실제 들어간 쪽수 대조.
    파일명과 출처 이름이 다를 수 있으므로(예: '2021학년도초등학교교육과정A.pdf' →
    '2021학년도 초등학교 교사 임용후보자 선정경쟁시험') 대기열 기록으로 먼저 연결하고,
    없으면 이름 비교로 넘어간다.
    """
    try:
        import cloud
        ups = cloud.list_uploads()
    except Exception:
        ups = []
    if not ups:
        return []

    # 1) 대기열 기록: storage_path → (저장 쪽수, 출처, 진행 상태)
    byjob = {}
    try:
        import ingest_queue as iq
        for j in iq.load().get("jobs", []):
            sp = j.get("storage_path")
            if not sp:
                continue
            r = j.get("result") or {}
            cur = byjob.setdefault(sp, {"쪽": 0, "출처": j.get("source", ""),
                                        "상태": j["status"], "진행": ""})
            cur["쪽"] += int(r.get("added") or 0)
            cur["출처"] = j.get("source", cur["출처"])
            if j["status"] == "queued" and j.get("next_page"):
                cur["진행"] = f"{j['next_page']}/{r.get('total_pages', '?')}쪽 진행중"
            elif j["status"] == "error":
                cur["상태"] = "error"
                cur["오류"] = str(r.get("error", ""))[:100]
    except Exception:
        pass

    src_pages = {r["출처"]: r["쪽"] for r in sources()}

    def _norm(x):
        return re.sub(r"[\s_\-\[\]()]+", "", x or "").lower()

    norm_src = {_norm(k): v for k, v in src_pages.items()}

    rows = []
    for u in ups:
        sp = u.get("storage_path")
        name = u.get("original_name", "")
        stem = os.path.splitext(name)[0]
        j = byjob.get(sp)
        pages = src_pages.get(j["출처"]) if j and j.get("출처") in src_pages else None
        if pages is None and j:
            pages = j["쪽"] or None
        if pages is None:                       # 이름으로 다시 찾아보기
            pages = norm_src.get(_norm(stem))
        if pages is None:
            hit = [k for k in norm_src if _norm(stem)[:14] and _norm(stem)[:14] in k]
            pages = norm_src[hit[0]] if hit else 0
        if j and j.get("진행"):
            status = "⏳ " + j["진행"]
        elif j and j["상태"] == "error":
            status = "❌ " + j.get("오류", "실패")
        elif pages:
            status = "✅"
        else:
            status = "⚠️ 아직 안 들어감"
        rows.append({"원본 파일": name, "올린 과목": u.get("subject", ""),
                     "크기KB": (u.get("size_bytes") or 0) // 1024,
                     "들어간 쪽": pages or 0, "저장된 출처": (j or {}).get("출처", ""),
                     "상태": status})
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
