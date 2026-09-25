"""
seed_rules.py — 파일명으로 정해지는 것은 코드로 박는다.

LLM에게 물어야 할 것과 물을 필요가 없는 것이 있다.
'2021학년도 초등학교 교사 임용후보자 선정경쟁시험.pdf'가 기출이라는 건
읽어보지 않아도 안다. 그런데 지금까지는 이런 것까지 LLM 판정에 맡겨서
수학 기본이론이 국어로, 국어 각론완성이 과학으로 가는 일이 생겼다.

바구니(bucket)
  기출   — 시험지 전체. 과목으로 쪼개지 않고 한 덩어리로 둔다.
           나중에 문항 단위로 나눈 뒤에 과목을 붙인다.
  공통   — 교육과정 총론, 전 과목 평가 등 특정 교과에 속하지 않는 자료
  기타   — 2차 시험용(수업실연·심층면접·교직논술 등)
  <과목> — 국어/수학/… 교과 자료

우선순위: 2차 > 기출 > 과목 교재 > 총론/공통
(같은 '2025 국어 수업실연'이라도 2차 자료이므로 기타로 간다)
"""
from __future__ import annotations
import re

SUBJECT_WORDS = {
    "국어": "국어", "영어": "영어", "수학": "수학", "사회": "사회", "과학": "과학",
    "미술": "미술", "음악": "음악", "체육": "체육", "실과": "실과", "도덕": "도덕",
    "통합교과": "통합교과", "바른생활": "통합교과", "슬기로운생활": "통합교과",
    "즐거운생활": "통합교과", "창의적체험활동": "창의적체험활동", "창체": "창의적체험활동",
}

# 2차 시험 자료 (1차 공부와 섞이면 안 됨)
SECOND = re.compile(r"수업실연|심층면접|교직논술|수업나눔|면접")
# 기출 시험지
EXAM = re.compile(r"임용후보자|선정경쟁시험|\d{4}\s*학년도[_\s]*(초등학교)?[_\s]*(교육과정|교직과정)"
                  r"|\d{4}\s*초등\s*임용|기출")
YEAR = re.compile(r"(20\d{2})\s*학년도|(20\d{2})")
# 교재 성격
MIND = re.compile(r"마인드맵")
GAERON = re.compile(r"각론")
BASIC = re.compile(r"기본\s*이론|기본이론")
CURRIC = re.compile(r"개정.*교육과정|교육과정\s*원문|성취기준|내용\s*체계")
CHONGRON = re.compile(r"총론|전\s*과목|범교과|창의적\s*체험")


def subject_of(name: str):
    """파일명에서 교과 찾기 (가장 먼저 나오는 교과 낱말)."""
    best, pos = None, 10 ** 9
    for w, s in SUBJECT_WORDS.items():
        i = name.find(w)
        if i >= 0 and i < pos:
            best, pos = s, i
    return best


def classify(name: str) -> dict:
    """
    파일명 → 라우팅. 반환:
      {bucket, subject, layer, doc_type, year, level, confidence, reason}
      confidence 1.0 = 파일명만으로 확실 (LLM 판정 불필요)
    """
    n = re.sub(r"\s+", " ", name or "")
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", n)
    subj = subject_of(stem)
    ym = YEAR.search(stem)
    year = int(ym.group(1) or ym.group(2)) if ym else None

    # 1) 2차 시험 자료
    if SECOND.search(stem):
        return _r("기타", "기타", "L2_corpus", "2차_자료", None, None, 1.0,
                  "파일명에 2차 시험 키워드")

    # 2) 기출 시험지 — 과목으로 쪼개지 않고 한 덩어리
    if EXAM.search(stem):
        return _r("기출", "기출", "L1_pattern", None, year, "초등", 1.0,
                  "파일명에 기출/선정경쟁시험")

    # 3) 교과 교재
    if subj:
        if MIND.search(stem):
            dt = "개인_필기"
            why = "마인드맵"
        elif CURRIC.search(stem):
            dt = "교육과정_성취기준"
            why = "교육과정 원문"
        elif GAERON.search(stem):
            dt = "지도서_각론"
            why = "각론"
        elif BASIC.search(stem):
            dt = "지도서_총론"
            why = "기본이론"
        else:
            dt = "지도서_각론"
            why = "교과 자료"
        return _r(subj, subj, "L2_corpus", dt, None, None, 1.0,
                  f"파일명에 '{subj}' + {why}")

    # 4) 총론·전 과목
    if CHONGRON.search(stem):
        return _r("공통", "공통", "L2_corpus", "교육과정_총론", None, None, 0.9,
                  "총론/전 과목")

    return _r(None, None, None, None, year, None, 0.0, "파일명으로 판정 불가")


def _r(bucket, subject, layer, doc_type, year, level, conf, reason):
    return {"bucket": bucket, "subject": subject, "layer": layer,
            "doc_type": doc_type, "year": year, "level": level,
            "confidence": conf, "reason": reason}


# ── 이미 저장된 자료에 적용 ──────────────────────────────────
def _base(src):
    b = (src or "").rsplit(" (", 1)[0]
    return b.rsplit(" p.", 1)[0] if " p." in b else b


def audit(subject_list=None):
    """저장된 기록의 출처(파일명)를 규칙으로 다시 판정 → 옮길 것만."""
    import paths
    from schema import SUBJECTS, load_records_pkl
    subject_list = subject_list or sorted(SUBJECTS)
    rows = {}
    for s in subject_list:
        for path, layer in ((paths.l2_path(s), "L2_corpus"),
                            (paths.l1_path(s), "L1_pattern")):
            for r in load_records_pkl(path):
                base = _base(r.source)
                if base.startswith("http") or r.doc_type in ("내_답변", "웹수집"):
                    continue
                v = classify(base)
                if not v["bucket"] or v["confidence"] < 0.9:
                    continue
                if v["subject"] == s and v["layer"] == layer:
                    continue
                k = (path, layer, base, v["subject"])
                d = rows.setdefault(k, {"path": path, "layer": layer, "source": base,
                                        "current": s, "proposed": v["subject"],
                                        "to_layer": v["layer"], "doc_type": v["doc_type"],
                                        "year": v["year"], "level": v["level"],
                                        "reason": v["reason"], "rec_ids": []})
                d["rec_ids"].append(r.rec_id)
    out = list(rows.values())
    for d in out:
        d["pages"] = len(d["rec_ids"])
    out.sort(key=lambda d: -d["pages"])
    return out


def apply(rows):
    """판정대로 옮긴다 (과목 바구니 + 층(L1/L2) + 자료종류까지 맞춘다)."""
    import paths
    from schema import Record, load_records_pkl, save_records_pkl
    from collections import defaultdict
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["path"]].append(r)
    moved = 0
    for path, rs in by_src.items():
        recs = load_records_pkl(path)
        plan = {}
        for r in rs:
            for rid in r["rec_ids"]:
                plan[rid] = r
        keep = [x for x in recs if x.rec_id not in plan]
        buckets = defaultdict(list)
        for x in recs:
            r = plan.get(x.rec_id)
            if not r:
                continue
            d = x.to_dict()
            d["subject"] = r["proposed"]
            d["layer"] = r["to_layer"]
            if r["to_layer"] == "L1_pattern":
                d["year"] = d.get("year") or r["year"]
                d["level"] = d.get("level") or r["level"] or "초등"
                d["doc_type"] = None
            else:
                d["doc_type"] = r["doc_type"] or d.get("doc_type")
                d["year"] = None
                d["level"] = None
            try:
                nr = Record(**{k: v for k, v in d.items() if k != "rec_id"})
            except Exception:
                keep.append(x)
                continue
            tgt = (paths.l1_path(r["proposed"]) if r["to_layer"] == "L1_pattern"
                   else paths.l2_path(r["proposed"]))
            buckets[tgt].append(nr)
        save_records_pkl(keep, path)
        for tgt, new in buckets.items():
            cur = load_records_pkl(tgt)
            have = {x.rec_id for x in cur}
            for nr in new:
                if nr.rec_id not in have:
                    cur.append(nr)
                    have.add(nr.rec_id)
                    moved += 1
            save_records_pkl(cur, tgt)
    return moved


# ── 기출 한 덩어리 → 문항 단위로 나누기 ──────────────────────
QNO = re.compile(
    r"(?m)^\s*(?:"
    r"(\d{1,2})\s*[.)]\s|"                 # 1. / 1)
    r"【\s*(\d{1,2})\s*】|"                # 【1】
    r"\[\s*(\d{1,2})\s*\]|"                # [1]
    r"문\s*(\d{1,2})\b|"                   # 문 1
    r"(\d{1,2})\s*번\b"                    # 1번
    r")")


def split_questions(text):
    """시험지 본문 → 문항 조각. 번호가 안 잡히면 [] (통째로 둔다)."""
    marks = [(m.start(), next(g for g in m.groups() if g)) for m in QNO.finditer(text)]
    if len(marks) < 2:
        return []
    out = []
    for i, (pos, no) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].strip()
        if len(body) < 25 and out:          # 너무 짧으면 앞 문항에 붙임(번호 오인식 대비)
            out[-1] = (out[-1][0], out[-1][1] + "\n" + body)
        elif len(body) >= 25:
            out.append((int(no), body))
    return out if len(out) >= 2 else []


def split_exam_bucket(dry_run=True, bucket="기출"):
    """
    기출 바구니의 쪽 기록을 문항 단위로 쪼갠다.
    출처는 '시험 이름 p.3 #2'로 남아 어느 쪽 몇 번인지 추적된다.
    (과목 배정은 이 다음 단계에서 문항 본문을 읽고 한다)
    """
    import paths
    from schema import Record, load_records_pkl, save_records_pkl
    path = paths.l1_path(bucket)
    recs = load_records_pkl(path)
    made, touched, new = 0, 0, []
    for r in recs:
        parts = split_questions(r.text)
        if not parts:
            new.append(r)
            continue
        touched += 1
        for no, body in parts:
            d = r.to_dict()
            d["text"] = body
            d["source"] = f"{r.source} #{no}"
            try:
                new.append(Record(**{k: v for k, v in d.items() if k != "rec_id"}))
                made += 1
            except Exception:
                pass
    if not dry_run and touched:
        import maintenance as mt
        mt._backup(path)
        save_records_pkl(new, path)
    return {"쪽": len(recs), "나눈 쪽": touched, "문항": made}
