"""
ingest_queue.py — 스캔까지 워커에 맡기기.

앱에서 '이 자료 다시 읽어줘'를 걸어두면(또는 드라이브에 파일을 넣어두면),
새벽 워커가 원본을 내려받아 비전으로 스캔하고 분류해서 저장한다.

대기열 한 건(job)
  {id, kind: "rescan" | "drive" | "upload",
   source: 저장될 출처 이름, subject, doc_type, year, level,
   storage_path | drive_file, handwriting, replace: 같은 출처의 옛 기록을 지울지,
   status: queued | done | error, result, at}

저장: data/ingest_queue.pkl (서버 백업) — 앱과 워커가 이 파일로 대화한다.
사람이 거는 일(무엇을 다시 읽을지)과 기계가 하는 일(스캔·분류)을 나눠 둔 것.
"""
from __future__ import annotations
import os, time, pickle, hashlib

import paths

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False


def queue_path():
    return paths._p("ingest_queue.pkl")


def load():
    p = queue_path()
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"jobs": [], "log": []}


def save(q):
    q["jobs"] = q["jobs"][-200:]
    q["log"] = q["log"][-200:]
    p = queue_path()
    with open(p, "wb") as f:
        pickle.dump(q, f)
    _cloud_push(p)


def add(kind, source, subject, **kw):
    q = load()
    job = {"id": hashlib.sha1(f"{kind}{source}{time.time()}".encode()).hexdigest()[:10],
           "kind": kind, "source": source, "subject": subject,
           "status": "queued", "at": time.time(), "result": None, **kw}
    q["jobs"].append(job)
    save(q)
    return job


def pending(q=None):
    q = q or load()
    return [j for j in q["jobs"] if j["status"] == "queued"]


def set_status(q, job_id, status, result=None):
    for j in q["jobs"]:
        if j["id"] == job_id:
            j["status"] = status
            j["result"] = result
            j["done_at"] = time.time()
    save(q)


# ── 실제 처리 (워커에서 실행) ────────────────────────────────
def _fetch(job):
    """원본 바이트 가져오기: 서버 보관본 또는 드라이브."""
    if job.get("storage_path"):
        import cloud
        return job.get("filename") or job["source"], cloud.download_upload(job["storage_path"])
    if job.get("drive_file"):
        import drive as gd
        return gd.download(job["drive_file"], job.get("api_key_g"), job.get("sa_json"))
    raise ValueError("원본 위치(storage_path/drive_file)가 없어요")


def process(job, api_key, model=None, concept_names=None, log=print):
    """한 건 처리: 내려받기 → 스캔 → 분류 → 저장."""
    from page_scan import build_pages, scan_pages, ScanCache
    from schema import Record, load_records_pkl, save_records_pkl
    import auto_tag

    name, raw = _fetch(job)
    if not raw:
        raise RuntimeError("원본을 못 받았어요")
    pages = build_pages([(name, raw)])
    res = scan_pages(pages, api_key, concept_names or [],
                     cache=ScanCache(paths.scan_cache_path()),
                     handwriting=bool(job.get("handwriting")), model=model,
                     progress=lambda d, n: log(f"    스캔 {d}/{n}쪽"))
    for r in res:
        r["page_in_file"] = r["page"]

    subject = job.get("subject")
    doc_type = job.get("doc_type")
    year, level = job.get("year"), job.get("level")
    if not subject or not doc_type:                 # 비어 있으면 자동 분류가 채움
        tag = auto_tag.classify(name, res, api_key, model or "gpt-4o-mini")
        subject = subject or tag["subject"] or "국어"
        doc_type = doc_type or (None if tag["category"] == "기출" else tag["category"])
        if tag["category"] == "기출":
            year = year or tag["year"]
            level = level or tag["level"] or "초등"
        job["auto_tag"] = {"category": tag["category"], "subject": tag["subject"],
                           "conf": tag["confidence"], "reason": tag["reason"]}

    is_exam = bool(year)
    path = paths.l1_path(subject) if is_exam else paths.l2_path(subject)
    existing = load_records_pkl(path)
    src = job["source"]
    if job.get("replace"):                           # 같은 출처의 옛 기록 치우기
        before = len(existing)
        existing = [r for r in existing
                    if not (r.source == src or r.source.startswith(src + " p."))]
        log(f"    옛 기록 {before - len(existing)}쪽 제거")

    have = {r.rec_id for r in existing}
    added, skipped = 0, 0
    for r in res:
        if r["skip"] or r.get("error") or not r["text"]:
            skipped += 1
            continue
        try:
            rec = Record(text=r["text"],
                         layer="L1_pattern" if is_exam else "L2_corpus",
                         subject=subject, source=f"{src} p.{r['page_in_file'] + 1}",
                         code=r["codes"][0] if r["codes"] else None,
                         area=r["area"] or None,
                         doc_type=None if is_exam else doc_type,
                         concepts=r.get("concepts") or None,
                         year=int(year) if is_exam else None,
                         level=level if is_exam else None)
        except Exception as e:
            log(f"    거부: {e}")
            continue
        if rec.rec_id not in have:
            existing.append(rec)
            have.add(rec.rec_id)
            added += 1
    save_records_pkl(existing, path)
    return {"pages": len(res), "added": added, "skipped": skipped,
            "subject": subject, "path": os.path.basename(path),
            "auto_tag": job.get("auto_tag")}


def run_queue(api_key, model=None, limit=20, log=print):
    """대기열 전체 처리. 반환: 처리 결과 목록."""
    q = load()
    todo = pending(q)[:limit]
    out = []
    for job in todo:
        log(f"  · {job['source']} ({job['kind']})")
        try:
            r = process(job, api_key, model, log=log)
            set_status(q, job["id"], "done", r)
            out.append({"source": job["source"], **r})
            log(f"    → {r['added']}쪽 저장 ({r['path']})")
        except Exception as e:
            set_status(q, job["id"], "error", {"error": str(e)})
            out.append({"source": job["source"], "error": str(e)})
            log(f"    → 실패: {e}")
        q = load()
    return out


# ── 드라이브 새 파일 자동 수집 ───────────────────────────────
def enqueue_drive_new(folders, api_key_g=None, sa_json=None, log=print):
    """드라이브 폴더에서 아직 안 가져온 파일을 대기열에 넣는다 (워커용)."""
    import drive as gd
    state = gd.DriveState(paths.drive_state_path())
    n = 0
    for u in folders:
        fid = gd.folder_id(u)
        if not fid:
            continue
        for f in gd.dedupe(gd.list_folder(fid, api_key_g, sa_json)):
            if not state.is_new(f):
                continue
            add("drive", source=os.path.splitext(f["name"])[0], subject=None,
                drive_file=f, api_key_g=api_key_g, sa_json=sa_json, replace=False)
            state.mark(f)
            n += 1
            log(f"  + 대기열 추가: {f['path']}")
    if n:
        state.save()
    return n
