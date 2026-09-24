"""
maintenance.py — 쌓인 자료를 쓸 수 있는 상태로 정비한다 (전 과목 공통).

단계(권장 순서)
  1) fix_tags   잘못 일괄로 붙은 태그 정리 (예: 기출 42쪽이 전부 '읽기')
  2) dedupe     같은 본문 중복 쪽 제거
  3) split      너무 긴 쪽을 문단 단위로 분할 (8천 자 한 덩어리 → 1천 자 조각)
  4) label      조각마다 영역 라벨 붙이기 (LLM, 비어 있는 것만)
  5) retrain    임베딩·SOM 재학습

왜 분할이 먼저 중요한가: 한 기록이 8천 자면 임베딩 한 벡터에 개념 수십 개가 뭉개진다.
SOM 지도도 흐려지고, 해설의 근거로 인용해도 어디가 근거인지 알 수 없다.

모든 단계는 dry_run으로 미리 볼 수 있고, 실제 변경 전에 로컬 백업을 남긴다.
(서버에도 버전 백업이 쌓이므로 사이드바에서 되돌릴 수 있다.)
"""
from __future__ import annotations
import os, re, time, shutil
from collections import Counter, defaultdict

import paths
from schema import Record, SUBJECTS, load_records_pkl, save_records_pkl

TARGET = 1000          # 조각 목표 길이(자)
HARD_MAX = 1600        # 이보다 길면 문장 단위로 더 쪼갬
KEEP_AS_IS = 1600      # 이보다 짧은 기록은 그대로 둠
SENT = re.compile(r"(?<=[.!?])\s+|(?<=다\.)\s*|(?<=요\.)\s*|(?<=음\.)\s*")


def subject_list(subjects=None):
    if subjects in (None, "all"):
        return paths.discover_subjects()
    if isinstance(subjects, str):
        return [s.strip() for s in subjects.split(",") if s.strip()]
    return list(subjects)


def _files(subject):
    return [(paths.l2_path(subject), "L2_corpus"), (paths.l1_path(subject), "L1_pattern")]


def _backup(path):
    if os.path.exists(path):
        bdir = os.path.join(paths.BASE, "backup")
        os.makedirs(bdir, exist_ok=True)
        shutil.copy2(path, os.path.join(
            bdir, f"{os.path.basename(path)}.{time.strftime('%Y%m%d-%H%M%S')}.bak"))


def _norm(t):
    return re.sub(r"\s+", "", t or "")


# ── 1) 잘못 일괄로 붙은 태그 정리 ─────────────────────────────
def fix_tags(subject, dry_run=True):
    """
    한 파일 전체가 같은 영역 값 하나뿐인데 출처(문서)가 여럿이면,
    넣을 때 일괄로 붙은 값일 가능성이 높다 → 비운다.
    (문서가 하나뿐이면 진짜로 그 영역일 수 있으므로 건드리지 않는다.)
    """
    out = []
    for path, layer in _files(subject):
        recs = load_records_pkl(path)
        if len(recs) < 5:
            continue
        docs = {r.source.rsplit(" p.", 1)[0] for r in recs}
        areas = Counter(r.area for r in recs if r.area)
        if len(docs) >= 2 and len(areas) == 1 and sum(areas.values()) >= len(recs) * 0.9:
            val = next(iter(areas))
            out.append({"path": path, "layer": layer, "field": "area", "value": val,
                        "records": len(recs), "docs": len(docs)})
            if not dry_run:
                _backup(path)
                new = []
                for r in recs:
                    d = r.to_dict()
                    d["area"] = None
                    new.append(Record(**{k: v for k, v in d.items() if k != "rec_id"}))
                save_records_pkl(new, path)
    return out


# ── 1-2) 깨진 글자 걸러내기 ──────────────────────────────────
CID = re.compile(r"\(?cid[:\s]?\d{1,6}\)?", re.I)   # (cid:54) / cid54 둘 다
HANGUL = re.compile(r"[가-힣]")


def is_garbage(text):
    """CID 폰트 PDF를 키 없이 텍스트층으로 읽으면 'cid41083…' 같은 글자가 나온다."""
    t = text or ""
    if len(CID.findall(t)) >= 5:
        return "CID 깨짐"
    if len(t) > 200 and len(HANGUL.findall(t)) / len(t) < 0.15:
        return "한글 거의 없음(글꼴 깨짐 의심)"
    return None


def drop_garbage(subject, dry_run=True):
    """못 쓰는 기록을 뺀다. 원본 파일은 uploads 버킷에 있으니 키를 넣고 다시 스캔하면 살아난다."""
    out = []
    for path, layer in _files(subject):
        recs = load_records_pkl(path)
        keep, bad = [], []
        for r in recs:
            why = is_garbage(r.text)
            (bad if why else keep).append(r)
            if why:
                out.append({"path": path, "source": r.source, "why": why,
                            "text": r.text[:60]})
        if bad and not dry_run:
            _backup(path)
            save_records_pkl(keep, path)
    return out


def clean_areas(subject, dry_run=True):
    """이미 저장된 엉뚱한 영역 값(설명문을 베낀 것 등)을 비운다."""
    from labeler import clean_area
    out = []
    for path, layer in _files(subject):
        recs = load_records_pkl(path)
        bad = [r for r in recs if r.area and not clean_area(r.area)]
        if not bad:
            continue
        out += [{"path": path, "value": r.area[:40], "source": r.source} for r in bad]
        if not dry_run:
            _backup(path)
            new = []
            for r in recs:
                if r.area and not clean_area(r.area):
                    d = r.to_dict()
                    d["area"] = None
                    r = Record(**{k: v for k, v in d.items() if k != "rec_id"})
                new.append(r)
            save_records_pkl(new, path)
    return out


# ── 2) 중복 제거 ─────────────────────────────────────────────
def dedupe(subject, dry_run=True):
    removed = []
    for path, layer in _files(subject):
        recs = load_records_pkl(path)
        seen, keep, drop = {}, [], []
        for r in recs:
            k = _norm(r.text)[:400]
            if len(k) >= 15 and k in seen:
                drop.append({"source": r.source, "same_as": seen[k]})
            else:
                seen[k] = r.source
                keep.append(r)
        if drop:
            removed += [dict(d, path=path) for d in drop]
            if not dry_run:
                _backup(path)
                save_records_pkl(keep, path)
    return removed


# ── 3) 긴 쪽 분할 ────────────────────────────────────────────
def _chunks(text, target=TARGET, hard_max=HARD_MAX):
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[0-9①-⑮가-힣]{1,3}[.)])", text)
             if p.strip()]
    if not paras:
        paras = [text]
    out, cur = [], ""
    for p in paras:
        if len(p) > hard_max:                      # 문단 자체가 길면 문장 단위로
            for s in SENT.split(p):
                s = (s or "").strip()
                if not s:
                    continue
                if len(cur) + len(s) > target and cur:
                    out.append(cur.strip())
                    cur = ""
                cur += s + " "
            continue
        if len(cur) + len(p) > target and cur:
            out.append(cur.strip())
            cur = ""
        cur += p + "\n"
    if cur.strip():
        out.append(cur.strip())
    # 표·마인드맵처럼 문단도 문장부호도 없는 본문은 마지막 수단으로 길이로 자름
    #   (공백 경계에서만 자르고, 낱말을 쪼개지는 않는다)
    final = []
    for c in out:
        if len(c) <= hard_max:
            final.append(c)
            continue
        words, cur2 = c.split(), ""
        for w in words:
            if len(cur2) + len(w) + 1 > target and cur2:
                final.append(cur2.strip())
                cur2 = ""
            cur2 += w + " "
        if cur2.strip():
            final.append(cur2.strip())
    out = final

    # 너무 짧은 꼬리는 앞 조각에 붙임
    merged = []
    for c in out:
        if merged and len(c) < 200:
            merged[-1] += "\n" + c
        else:
            merged.append(c)
    return merged


def split(subject, dry_run=True, keep_as_is=KEEP_AS_IS, target=TARGET):
    """긴 기록을 문단 단위로 나눈다. 출처는 '… p.12 (3/8)'로 남겨 추적 가능."""
    stat = {"scanned": 0, "split": 0, "before": 0, "after": 0, "examples": []}
    for path, layer in _files(subject):
        recs = load_records_pkl(path)
        if not recs:
            continue
        new, changed = [], False
        for r in recs:
            stat["scanned"] += 1
            if len(r.text) <= keep_as_is:
                new.append(r)
                continue
            parts = _chunks(r.text, target=target)
            if len(parts) < 2:
                new.append(r)
                continue
            changed = True
            stat["split"] += 1
            stat["before"] += 1
            stat["after"] += len(parts)
            if len(stat["examples"]) < 5:
                stat["examples"].append({"source": r.source, "len": len(r.text),
                                         "parts": len(parts),
                                         "sizes": [len(p) for p in parts[:6]]})
            for i, p in enumerate(parts, 1):
                d = r.to_dict()
                d["text"] = p
                d["source"] = f"{r.source} ({i}/{len(parts)})"
                new.append(Record(**{k: v for k, v in d.items() if k != "rec_id"}))
        if changed and not dry_run:
            _backup(path)
            save_records_pkl(new, path)
    return stat


# ── 4) 조각에 영역 라벨 ──────────────────────────────────────
def label_areas(subject, api_key, model="gpt-4o-mini", limit=300, dry_run=True,
                progress=None):
    """영역이 비어 있는 기록에만 LLM 라벨을 붙여 채운다."""
    import labeler as lb
    filled = 0
    for path, layer in _files(subject):
        recs = load_records_pkl(path)
        todo = [r for r in recs if not r.area][:limit]
        if not todo:
            continue
        if dry_run:
            filled += len(todo)
            continue
        store, _ = lb.label_records(subject, todo, api_key, model, progress=progress)
        _backup(path)
        new = []
        for r in recs:
            lab = store.get(r.rec_id) or {}
            if not r.area and lab.get("area"):
                d = r.to_dict()
                d["area"] = lab["area"]
                r = Record(**{k: v for k, v in d.items() if k != "rec_id"})
                filled += 1
            new.append(r)
        save_records_pkl(new, path)
    return filled


# ── 전체 실행 ────────────────────────────────────────────────
STEPS = ("fix_tags", "garbage", "dedupe", "exam_subject", "split", "label", "retrain")
APP_STEPS = ("fix_tags", "garbage", "dedupe", "exam_subject", "split", "label")   # 재학습은 오래 걸려 워커에 맡김


def run(subject, steps=STEPS, dry_run=True, api_key=None, model="gpt-4o-mini",
        label_limit=300, progress=None):
    rep = {"subject": subject, "dry_run": dry_run}
    if "fix_tags" in steps:
        rep["fix_tags"] = fix_tags(subject, dry_run)
    if "fix_tags" in steps:
        ca = clean_areas(subject, dry_run)
        rep["clean_areas"] = len(ca)
    if "garbage" in steps:
        g = drop_garbage(subject, dry_run)
        rep["garbage"] = {"n": len(g), "examples": g[:5]}
    if "dedupe" in steps:
        rep["dedupe"] = len(dedupe(subject, dry_run))
    if "exam_subject" in steps:
        try:
            import resubject as rsj
            pages = rsj.exam_pages([subject])
            rows = rsj.judge_exam_pages(pages, api_key, model)
            rep["exam_subject"] = {"pages": len(pages), "moves": len(rows),
                                   "to": dict(Counter(r["proposed"] for r in rows))}
            if rows and not dry_run:
                rep["exam_subject"]["moved"] = rsj.apply_exam_moves(rows)
        except Exception as e:
            rep["exam_subject"] = {"error": str(e)}
    if "split" in steps:
        rep["split"] = split(subject, dry_run)
    if "label" in steps and api_key:
        rep["label"] = label_areas(subject, api_key, model, label_limit, dry_run,
                                   progress)
    if "retrain" in steps and not dry_run:
        try:
            import selfcheck
            corpus = selfcheck.train_corpus(subject)
            rep["retrain"] = (selfcheck.retrain(subject, corpus) if len(corpus) >= 3
                              else "자료 부족")
        except Exception as e:
            rep["retrain"] = {"error": str(e)}
    return rep


def run_all(subjects=None, **kw):
    return {s: run(s, **kw) for s in subject_list(subjects)}
