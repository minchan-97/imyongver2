"""
resubject.py — 이미 저장된 자료의 '과목'이 맞는지 다시 판정하고 옮긴다.

왜 필요한가: 넣을 때 과목 칸의 기본값이 '지금 보고 있는 과목'이라,
자동 분류가 과목을 못 집으면 전부 그 과목(대개 국어)으로 들어간다.
쌓인 뒤에 발견되므로 사후 재판정이 필요하다.

판정 근거(강한 것부터)
  1) 성취기준 코드의 교과 글자  — [4국02-01] → 국어, [2슬03-02] → 통합교과 (거의 확실)
  2) 교과 고유 어휘            — '음운', '분수의 나눗셈', '용해도' 등
  3) (선택) LLM                — 위 둘로 못 가르는 것만

판정 단위는 기본이 '출처(문서)'다. 한 지도서 PDF의 40쪽 중 코드가 붙은 건 3쪽뿐이어도,
그 문서 전체의 근거를 합쳐서 판정하면 나머지 37쪽도 같이 제자리를 찾는다.
(쪽 단위로만 보면 근거 없는 쪽은 영원히 미분류로 남아 자료가 '자라지' 않는다.)

같은 이유로 propagate_tags()는 같은 출처 안에서 영역·자료종류·학년군·단원을
다수결로 채운다. 없는 값을 지어내지 않고, 이미 있는 값은 건드리지 않는다.

이 모듈은 '제안'만 만든다. 실제 이동은 사람이 확인한 것만 apply_moves()로 수행한다.
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict

import paths
from schema import Record, load_records_pkl, save_records_pkl, SUBJECTS

CODE_RE = re.compile(r"\[(\d{1,2})([가-힣]{1,3})(\d{2})[-–](\d{2})\]")

# 성취기준 코드의 교과 글자 → 과목
CODE_SUBJECT = {
    "국": "국어", "수": "수학", "사": "사회", "과": "과학", "영": "영어",
    "도": "도덕", "체": "체육", "음": "음악", "미": "미술", "실": "실과",
    "바": "통합교과", "슬": "통합교과", "즐": "통합교과",
}

# 교과 고유 어휘만. 일반어('문장','학교','나')는 어떤 자료에나 있어 오판을 만든다.
LEX = {
    "국어": "음운 형태소 품사 서술자 운율 비유법 상징 갈래 설명문 논설문 맞춤법 띄어쓰기 "
            "주제문 문단구성 읽기전략 쓰기과정 듣기말하기 문학작품 매체언어 독서토론 어휘지도",
    "수학": "분수 소수점 약분 통분 각도 넓이 부피 비례식 방정식 자연수 곱셈 나눗셈 "
            "수직선 좌표평면 반올림 평면도형 입체도형 대칭 어림하기 규칙성 사칙연산",
    "과학": "용해 용액 증발 응결 지층 화석 태양계 별자리 소화기관 호흡기관 광합성 "
            "전기회로 자석 열전달 산성 염기성 생태계 먹이사슬 용해도 물질의성질 관찰실험",
    "사회": "등고선 축척 민주주의 삼권분립 헌법 시장경제 수요와공급 인구분포 도시화 문화유산 "
            "조선시대 고려시대 삼국시대 독립운동 공공기관 선거제도 조세 기후지역 지형",
    "영어": "phonics alphabet listening speaking vocabulary 파닉스 영단어 의사소통기능 "
            "영어회화 영어문장 영어읽기 chant 알파벳",
    "도덕": "도덕적성찰 배려 정직 공정성 인권 생명존중 통일교육 시민의식 도덕규범 양심 효도 우정",
    "음악": "가창 기악 음악창작 음악감상 박자 리듬꼴 가락 화음 음계 장단 국악 민요 계이름 셈여림",
    "미술": "조형요소 미술표현 미술감상 명도 채도 구도 소묘 판화 조소 시각디자인 미술사 작품감상",
    "체육": "체력운동 운동기능 스포츠 리듬운동 준비운동 심폐지구력 표현활동 경쟁활동 도전활동",
    "실과": "가정생활 발명 소프트웨어교육 로봇 코딩 식품조리 의생활 주생활 진로교육 자원관리",
    "통합교과": "바른생활 슬기로운생활 즐거운생활",
    "총론": "핵심역량 추구하는인간상 교육과정편성 창의적체험활동 범교과학습 학년군 교과군",
}
LEXSET = {s: set(v.split()) for s, v in LEX.items()}


def base_source(src: str) -> str:
    """'국어 지도서 p.12' → '국어 지도서' (같은 문서 묶기)."""
    return src.rsplit(" p.", 1)[0] if " p." in src else src


def _scores(text, code=None):
    """과목별 점수와 근거. judge()와 문서 단위 판정이 함께 쓴다."""
    sc, ev = Counter(), []
    codes = [code] if code else []
    codes += ["[%s%s%s-%s]" % m for m in CODE_RE.findall(text or "")]
    seen = set()
    for c in codes:
        m = CODE_RE.match(c or "")
        if not m or c in seen:
            continue
        seen.add(c)
        subj = CODE_SUBJECT.get(m.group(2)[0])
        if subj:
            sc[subj] += 3
            ev.append(f"코드 {c}")
    t = text or ""
    for subj, words in LEXSET.items():
        hit = [w for w in words if len(w) >= 2 and w in t]
        if hit:
            sc[subj] += min(3, len(hit)) * 0.7
            ev.append(f"{subj} 어휘 {'/'.join(hit[:3])}")
    return sc, ev


def _verdict(sc, ev):
    if not sc:
        return None, 0.0, []
    best, top = sc.most_common(1)[0]
    total = sum(sc.values())
    # 근거의 '양'도 반영: 코드 1개(3점)나 고유어휘 3개(2.1점)면 충분,
    # 어휘 1개(0.7점)만으로는 확신하지 않는다.
    strength = min(1.0, top / 2.1)
    return best, round((top / total) * strength, 2), ev[:4]


def judge(text, code=None):
    """한 기록의 과목 판정. 반환: (과목, 확신도 0~1, 근거)"""
    return _verdict(*_scores(text, code))


def judge_docs(records):
    """출처(문서)별로 근거를 합쳐 판정. 반환: {출처: (과목, 확신도, 근거, 쪽수)}"""
    agg = {}
    for r in records:
        b = base_source(r.source)
        sc, ev = _scores(r.text, r.code)
        cur = agg.setdefault(b, [Counter(), [], 0])
        cur[0].update(sc)
        cur[1] += ev
        cur[2] += 1
    out = {}
    for b, (sc, ev, n) in agg.items():
        subj, conf, e = _verdict(sc, ev)
        out[b] = (subj, conf, e, n)
    return out


def audit(subject_list=None, min_conf=0.6, by_source=True,
          layers=("L2_corpus", "L1_pattern")):
    """
    저장된 기록 중 '지금 과목과 다르게 판정되는' 것을 모은다.
    by_source=True(기본): 문서 단위로 판정하고, 그 문서의 모든 쪽을 함께 옮긴다.
    반환 행: {path, layer, current, proposed, conf, evidence, source, pages, rec_ids, text}
    """
    subject_list = subject_list or sorted(SUBJECTS - {"공통"})
    out = []
    for subj in subject_list:
        for path, layer in ((paths.l2_path(subj), "L2_corpus"),
                            (paths.l1_path(subj), "L1_pattern")):
            if layer not in layers:
                continue
            recs = load_records_pkl(path)
            if not recs:
                continue
            if by_source:
                docs = judge_docs(recs)
                by_base = {}
                for r in recs:
                    by_base.setdefault(base_source(r.source), []).append(r)
                for b, (prop, conf, ev, n) in docs.items():
                    if prop and prop != subj and conf >= min_conf:
                        rs = by_base[b]
                        out.append({"path": path, "layer": layer, "current": subj,
                                    "proposed": prop, "conf": conf, "evidence": ev,
                                    "source": b, "pages": len(rs),
                                    "rec_ids": [r.rec_id for r in rs],
                                    "text": rs[0].text[:100]})
            else:
                for r in recs:
                    prop, conf, ev = judge(r.text, r.code)
                    if prop and prop != subj and conf >= min_conf:
                        out.append({"path": path, "layer": layer, "current": subj,
                                    "proposed": prop, "conf": conf, "evidence": ev,
                                    "source": r.source, "pages": 1,
                                    "rec_ids": [r.rec_id], "text": r.text[:100]})
    out.sort(key=lambda d: (-d["conf"], -d["pages"]))
    return out


def summary(rows):
    """'국어 → 수학 3개 문서 / 41쪽' 식 요약."""
    c = Counter()
    for r in rows:
        c[(r["current"], r["proposed"])] += r["pages"]
    docs = Counter((r["current"], r["proposed"]) for r in rows)
    return [{"from": a, "to": b, "docs": docs[(a, b)], "pages": n}
            for (a, b), n in c.most_common()]


def apply_moves(rows):
    """확인된 것만 실제로 옮긴다 (같은 layer의 대상 과목 pkl로)."""
    by_src = defaultdict(set)
    for r in rows:
        by_src[(r["path"], r["layer"], r["proposed"])].update(r["rec_ids"])

    moved = 0
    for (src_path, layer, target_subj), idset in by_src.items():
        recs = load_records_pkl(src_path)
        keep = [r for r in recs if r.rec_id not in idset]
        move = [r for r in recs if r.rec_id in idset]
        if not move:
            continue
        tgt_path = (paths.l2_path(target_subj) if layer == "L2_corpus"
                    else paths.l1_path(target_subj))
        tgt = load_records_pkl(tgt_path)
        have = {r.rec_id for r in tgt}
        for r in move:
            d = r.to_dict()
            d["subject"] = target_subj
            nr = Record(**{k: v for k, v in d.items() if k != "rec_id"})
            if nr.rec_id not in have:
                tgt.append(nr)
                have.add(nr.rec_id)
        save_records_pkl(keep, src_path)
        save_records_pkl(tgt, tgt_path)
        moved += len(move)
    return moved


# ── 같은 출처끼리 태그 채우기 ─────────────────────────────────
FILLABLE = ("area", "doc_type", "grade_band", "unit")


def tag_gaps(subject):
    """이 과목에서 '같은 출처의 다른 쪽에는 있는데 이 쪽에는 없는' 태그 통계."""
    stat = {}
    for path, _ in ((paths.l2_path(subject), 1), (paths.l1_path(subject), 2)):
        recs = load_records_pkl(path)
        for field in FILLABLE:
            fill, _ = _plan_fill(recs, field)
            if fill:
                stat[field] = stat.get(field, 0) + len(fill)
    return stat


def _plan_fill(recs, field):
    """출처별 다수결 값 → 그 값이 비어 있는 쪽 목록."""
    votes = defaultdict(Counter)
    for r in recs:
        v = getattr(r, field, None)
        if v:
            votes[base_source(r.source)][v] += 1
    fill, chosen = [], {}
    for b, c in votes.items():
        top, n = c.most_common(1)[0]
        if n >= 1 and (len(c) == 1 or n >= sum(c.values()) * 0.6):   # 애매하면 안 채움
            chosen[b] = top
    for r in recs:
        b = base_source(r.source)
        if not getattr(r, field, None) and b in chosen:
            fill.append((r.rec_id, chosen[b]))
    return fill, chosen


def propagate_tags(subject, fields=FILLABLE, dry_run=False):
    """
    같은 출처 안에서 비어 있는 태그를 다수결 값으로 채운다.
    (없는 값을 만들지 않고, 이미 있는 값은 건드리지 않는다. 코드는 쪽마다 달라 제외.)
    """
    filled = Counter()
    for path in (paths.l2_path(subject), paths.l1_path(subject)):
        recs = load_records_pkl(path)
        if not recs:
            continue
        changed = False
        for field in fields:
            plan, _ = _plan_fill(recs, field)
            if not plan:
                continue
            m = dict(plan)
            new = []
            for r in recs:
                if r.rec_id in m:
                    d = r.to_dict()
                    d[field] = m[r.rec_id]
                    r = Record(**{k: v for k, v in d.items() if k != "rec_id"})
                    filled[field] += 1
                    changed = True
                new.append(r)
            recs = new
        if changed and not dry_run:
            save_records_pkl(recs, path)
    return dict(filled)


# ── 규칙으로 못 가르는 문서 → LLM 판정 ───────────────────────
LLM_SYSTEM = """너는 초등 임용 자료를 분류하는 사서다. 한 문서(여러 쪽)의 본문 일부를 보고
아래 JSON 하나만 출력한다. 마인드맵·개념도를 옮긴 글처럼 조각난 낱말 나열일 수도 있다.
{"subject":"국어|영어|수학|사회|과학|미술|음악|체육|실과|도덕|총론|창의적체험활동|통합교과|공통 중 하나",
 "area":"영역(예: 읽기, 수와 연산). 모르면 \"\"",
 "doc_type":"교육과정_총론|교육과정_성취기준|지도서_총론|지도서_각론|개인_필기 중 하나 또는 \"\"",
 "confidence":0.0~1.0,
 "reason":"판단 근거 한 줄(어떤 낱말을 보고 그렇게 봤는지)"}
근거가 약하면 confidence를 낮게 준다. 추측으로 확신하지 말 것."""


def undetermined_docs(subject_list=None, max_conf=0.6, layers=("L2_corpus", "L1_pattern"),
                      sample_chars=4000, skip_tagged=True):
    """규칙으로 과목을 못 가르는(또는 확신 낮은) 문서들. 마인드맵·필기가 주로 여기 걸린다."""
    subject_list = subject_list or sorted(SUBJECTS - {"공통"})
    out = []
    for subj in subject_list:
        for path, layer in ((paths.l2_path(subj), "L2_corpus"),
                            (paths.l1_path(subj), "L1_pattern")):
            if layer not in layers:
                continue
            recs = load_records_pkl(path)
            if not recs:
                continue
            by_base = {}
            for rec in recs:
                by_base.setdefault(base_source(rec.source), []).append(rec)
            docs = judge_docs(recs)
            for b, rs in by_base.items():
                prop, conf, ev, _ = docs.get(b, (None, 0.0, [], 0))
                if all((rec.doc_type in ("내_답변", "웹수집")) for rec in rs):
                    continue          # 내가 답한 것·웹에서 채택한 것은 분류 대상 아님
                try:                  # 파일명 규칙으로 판정되는 문서는 LLM에 묻지 않는다
                    import seed_rules as _sr
                    if _sr.classify(b)["confidence"] >= 0.9:
                        continue
                except Exception:
                    pass
                tagged = sum(1 for rec in rs if rec.area and rec.doc_type)
                if skip_tagged and tagged >= max(1, int(len(rs) * 0.8)):
                    continue          # 이미 사람이 확인해 태그가 채워진 문서는 다시 안 묻는다
                if conf < max_conf:
                    sample, used = [], 0
                    for rec in rs:
                        t = rec.text.strip()
                        if used + len(t) > sample_chars:
                            break
                        sample.append(t)
                        used += len(t)
                    out.append({"path": path, "layer": layer, "current": subj,
                                "source": b, "pages": len(rs),
                                "rec_ids": [rec.rec_id for rec in rs],
                                "rule_guess": prop, "rule_conf": conf,
                                "has_area": sum(1 for rec in rs if rec.area),
                                "sample": "\n".join(sample)})
    out.sort(key=lambda d: -d["pages"])
    return out


def judge_docs_llm(docs, api_key, model="gpt-4o-mini", workers=6, progress=None):
    """undetermined_docs() 결과에 LLM 판정을 붙인다. 반환: 같은 행 + proposed/area/doc_type/conf"""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from schema import DOC_TYPES
    client = OpenAI(api_key=api_key)

    def work(d):
        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": LLM_SYSTEM},
                      {"role": "user", "content": f"문서 이름: {d['source']}\n\n본문:\n{d['sample']}"}])
        j = json.loads(resp.choices[0].message.content)
        subj = j.get("subject") if j.get("subject") in SUBJECTS else None
        dt = j.get("doc_type") if j.get("doc_type") in DOC_TYPES else None
        try:
            conf = max(0.0, min(1.0, float(j.get("confidence", 0.5))))
        except Exception:
            conf = 0.5
        return {"proposed": subj, "area": str(j.get("area") or "").strip(),
                "doc_type": dt, "conf": conf,
                "evidence": ["LLM: " + str(j.get("reason") or "")[:120]]}

    out, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, d): d for d in docs}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                out.append({**d, **fut.result()})
            except Exception as e:
                out.append({**d, "proposed": None, "area": "", "doc_type": None,
                            "conf": 0.0, "evidence": [f"LLM 실패: {e}"]})
            done += 1
            if progress:
                progress(done, len(docs))
    out.sort(key=lambda x: -x["conf"])
    return out


def apply_doc_decisions(rows):
    """
    문서 단위 결정 적용: 과목 이동 + 영역·자료종류 채우기(비어 있는 쪽만).
    rows: judge_docs_llm() 결과 중 사용자가 확인한 것 (proposed/area/doc_type 수정 가능)
    """
    moved = tagged = 0
    for r in rows:
        ids = set(r["rec_ids"])
        recs = load_records_pkl(r["path"])
        stay, hit = [], []
        for rec in recs:
            (hit if rec.rec_id in ids else stay).append(rec)
        if not hit:
            continue
        new_hit = []
        for rec in hit:
            d = rec.to_dict()
            if r.get("area") and not d.get("area"):
                d["area"] = r["area"]
                tagged += 1
            if r.get("doc_type") and not d.get("doc_type"):
                d["doc_type"] = r["doc_type"]
            if r.get("proposed") and r["proposed"] != r["current"]:
                d["subject"] = r["proposed"]
            new_hit.append(Record(**{k: v for k, v in d.items() if k != "rec_id"}))
        if r.get("proposed") and r["proposed"] != r["current"]:
            tgt_path = (paths.l2_path(r["proposed"]) if r["layer"] == "L2_corpus"
                        else paths.l1_path(r["proposed"]))
            tgt = load_records_pkl(tgt_path)
            have = {x.rec_id for x in tgt}
            for rec in new_hit:
                if rec.rec_id not in have:
                    tgt.append(rec)
                    have.add(rec.rec_id)
            save_records_pkl(stay, r["path"])
            save_records_pkl(tgt, tgt_path)
            moved += len(new_hit)
        else:
            save_records_pkl(stay + new_hit, r["path"])
    return moved, tagged


# ── 기출 쪽별 과목 재배정 ────────────────────────────────────
EXAM_SYSTEM = """너는 초등 임용 시험지 한 쪽을 읽고 '어느 교과 문항인지'만 판정한다.
JSON 하나만 출력한다.
{"subject":"국어|영어|수학|사회|과학|미술|음악|체육|실과|도덕|총론|창의적체험활동|통합교과 중 하나",
 "area":"영역 이름 한 낱말(모르면 \"\")",
 "confidence":0.0~1.0,
 "reason":"근거 한 줄"}
교육과정 총론·교직 일반(편성운영, 창의적체험활동 운영 등)은 subject를 '총론'으로.
표지·안내문·배점표만 있는 쪽은 confidence를 0.2 이하로."""


def exam_pages(subject_list=None, layers=("L1_pattern",)):
    """기출(L1) 기록을 쪽 단위로 모은다."""
    subject_list = subject_list or sorted(SUBJECTS - {"공통"})
    out = []
    for subj in subject_list:
        path = paths.l1_path(subj)
        for rec in load_records_pkl(path):
            out.append({"path": path, "layer": "L1_pattern", "current": subj,
                        "rec_id": rec.rec_id, "source": rec.source,
                        "text": rec.text, "code": rec.code, "area": rec.area})
    return out


def judge_exam_pages(pages, api_key=None, model="gpt-4o-mini", workers=6,
                     min_conf=0.6, progress=None):
    """
    쪽마다 과목 판정. 코드가 있으면 규칙으로 끝내고(공짜),
    없을 때만 LLM에 묻는다. 반환: 이동이 필요한 쪽만.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out, need_llm = [], []
    for p in pages:
        subj, conf, ev = judge(p["text"], p["code"])
        if subj and conf >= 0.9:                 # 코드 등 확실한 근거
            if subj != p["current"]:
                out.append({**p, "proposed": subj, "conf": conf, "area": p["area"],
                            "evidence": ev, "by": "규칙"})
        else:
            need_llm.append(p)
    if api_key and need_llm:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def work(p):
            resp = client.chat.completions.create(
                model=model, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": EXAM_SYSTEM},
                          {"role": "user", "content": p["text"][:3000]}])
            j = json.loads(resp.choices[0].message.content)
            s = j.get("subject") if j.get("subject") in SUBJECTS else None
            try:
                c = max(0.0, min(1.0, float(j.get("confidence", 0.5))))
            except Exception:
                c = 0.5
            from labeler import clean_area
            return s, c, clean_area(j.get("area")), str(j.get("reason") or "")[:100]

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, p): p for p in need_llm}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    s, c, area, why = fut.result()
                    if s and c >= min_conf and (s != p["current"] or
                                                (area and not p["area"])):
                        out.append({**p, "proposed": s, "conf": c,
                                    "area": area or p["area"],
                                    "evidence": ["LLM: " + why], "by": "LLM"})
                except Exception as e:
                    pass
                done += 1
                if progress:
                    progress(done, len(need_llm))
    out.sort(key=lambda x: -x["conf"])
    return out


def apply_exam_moves(rows):
    """판정대로 기출 쪽을 과목별 L1로 옮긴다(영역도 비어 있으면 채움)."""
    from collections import defaultdict as _dd
    by_src = _dd(list)
    for r in rows:
        by_src[r["path"]].append(r)
    moved = 0
    for path, rs in by_src.items():
        recs = load_records_pkl(path)
        plan = {r["rec_id"]: r for r in rs}
        keep, move = [], []
        for rec in recs:
            (move if rec.rec_id in plan else keep).append(rec)
        if not move:
            continue
        buckets = {}
        for rec in move:
            r = plan[rec.rec_id]
            d = rec.to_dict()
            d["subject"] = r["proposed"]
            if r.get("area") and not d.get("area"):
                d["area"] = r["area"]
            nr = Record(**{k: v for k, v in d.items() if k != "rec_id"})
            buckets.setdefault(paths.l1_path(r["proposed"]), []).append(nr)
        save_records_pkl(keep, path)
        for tgt, recs2 in buckets.items():
            cur = load_records_pkl(tgt)
            have = {x.rec_id for x in cur}
            for nr in recs2:
                if nr.rec_id not in have:
                    cur.append(nr)
                    have.add(nr.rec_id)
                    moved += 1
            save_records_pkl(cur, tgt)
    return moved

