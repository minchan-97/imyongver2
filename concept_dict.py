"""
cloud.py — Supabase 동기화 계층.

원칙: 기존 틀(로컬 data/*.pkl로 읽고 쓰기)은 그대로 두고,
      '저장할 때마다 서버로 올리고, 앱 시작할 때 서버에서 내려받는다'.
      → 앱 코드는 거의 안 바뀌고, Streamlit Cloud처럼 디스크가 날아가는
        환경에서도 자료·분석결과가 유지된다.

Supabase에 들어가는 것
  1) Storage 'artifacts' 버킷
       current/{key}.pkl              : 각 pkl의 최신본 (앱이 다시 읽는 원본)
       history/{key}/{시각}.pkl       : 버전 백업 (분석결과 되돌리기용)
  2) Storage 'uploads' 버킷            : 내가 올린 원본 파일(pdf/docx/txt)
  3) DB 테이블
       artifacts : pkl별 해시·크기·갱신시각 (동기화 판단용)
       records   : L1/L2/L3 자료를 행 단위로 미러링 (대시보드에서 조회·검색용)
       uploads   : 원본 파일 목록(원래 파일명·종류·해시)

키가 없으면(SUPABASE_URL/SUPABASE_KEY 미설정) 자동으로 꺼지고
예전처럼 로컬 pkl만으로 동작한다. 서버 오류가 나도 로컬 저장은 막지 않는다.
"""
from __future__ import annotations
import os, time, hashlib

BUCKET_ART = "artifacts"
BUCKET_UP = "uploads"
HISTORY_INTERVAL_SEC = 30 * 60   # 같은 파일의 버전 백업은 최소 30분 간격
HISTORY_KEEP = 30                # 파일당 버전 백업 최대 개수

# Supabase Storage 키는 한글을 못 받음 → 과목명 등을 영문 슬러그로 바꿔 저장
SLUG = {
    "국어": "gukeo", "영어": "english", "수학": "math", "사회": "social",
    "과학": "science", "미술": "art", "음악": "music", "체육": "pe",
    "실과": "practical", "도덕": "ethics", "총론": "chongron",
    "창의적체험활동": "changche", "통합교과": "tonghap", "공통": "common",
    "교육과정총론": "curriculum_general",
}

_CLIENT = None
_ENABLED = None
_LAST = {}          # key -> {"sha256":..., "last_history_epoch":...} (왕복 줄이기용)
ERRORS: list[str] = []


# ── 설정 / 연결 ──────────────────────────────────────────────
def cfg(name, default=None):
    """st.secrets → 환경변수 순으로 읽기 (CLI 실행도 지원)."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def client():
    global _CLIENT, _ENABLED
    if _ENABLED is not None:
        return _CLIENT
    url, key = cfg("SUPABASE_URL"), cfg("SUPABASE_KEY")
    if not url or not key:
        _ENABLED = False
        return None
    try:
        from supabase import create_client
        _CLIENT = create_client(url, key)
        _ENABLED = True
    except Exception as e:
        _err("연결", e)
        _ENABLED = False
    return _CLIENT


def enabled() -> bool:
    return client() is not None


def _err(where, e):
    ERRORS.append(f"[{time.strftime('%H:%M:%S')}] {where}: {e}")
    del ERRORS[:-20]


def slug(s: str) -> str:
    return SLUG.get(s, s)


def storage_key(local_name: str) -> str:
    """'국어_L2.pkl' → 'gukeo_L2' (확장자 제외, ASCII만)."""
    stem = local_name[:-4] if local_name.endswith(".pkl") else local_name
    parts = [slug(p) for p in stem.split("_")]
    out = []
    for p in parts:
        out.append(p if p.isascii() else "x" + p.encode("utf-8").hex())
    return "_".join(out)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metas(keys):
    c = client()
    if not c or not keys:
        return {}
    res = c.table("artifacts").select("*").in_("key", list(keys)).execute()
    return {r["key"]: r for r in (res.data or [])}


# ── pkl 올리기 / 내려받기 ─────────────────────────────────────
def push(path: str, force_history: bool = False) -> bool:
    """로컬 pkl을 서버 최신본으로 올린다. 내용이 같으면 건너뜀."""
    c = client()
    if not c or not os.path.exists(path):
        return False
    name = os.path.basename(path)
    key = storage_key(name)
    try:
        with open(path, "rb") as f:
            data = f.read()
        sha = _sha(data)
        prev = _LAST.get(key)
        if prev is None:
            prev = _metas([key]).get(key, {})
        if prev.get("sha256") == sha and not force_history:
            _LAST[key] = prev
            return True

        now = time.time()
        bucket = c.storage.from_(BUCKET_ART)
        bucket.upload(f"current/{key}.pkl", data,
                      {"content-type": "application/octet-stream", "upsert": "true"})

        last_hist = prev.get("last_history_epoch") or 0
        if force_history or now - last_hist >= HISTORY_INTERVAL_SEC:
            ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
            bucket.upload(f"history/{key}/{ts}.pkl", data,
                          {"content-type": "application/octet-stream", "upsert": "true"})
            last_hist = now
            _prune_history(key)

        row = {"key": key, "local_name": name, "sha256": sha, "size_bytes": len(data),
               "updated_epoch": now, "last_history_epoch": last_hist}
        c.table("artifacts").upsert(row).execute()
        _LAST[key] = row
        return True
    except Exception as e:
        _err(f"업로드 {name}", e)
        return False


def _prune_history(key):
    c = client()
    try:
        items = c.storage.from_(BUCKET_ART).list(f"history/{key}", {"limit": 1000})
        names = sorted(i["name"] for i in items if i.get("name", "").endswith(".pkl"))
        old = names[:-HISTORY_KEEP]
        if old:
            c.storage.from_(BUCKET_ART).remove([f"history/{key}/{n}" for n in old])
    except Exception as e:
        _err(f"버전정리 {key}", e)


def sync(paths_list) -> dict:
    """
    앱 시작 시 호출. 파일마다 '더 최근 쪽'을 기준으로 맞춘다.
      서버가 최신/로컬 없음 → 내려받기
      로컬이 최신(예: 전에 업로드 실패) → 올리기
    """
    report = {"down": [], "up": []}
    c = client()
    if not c:
        return report
    by_key = {storage_key(os.path.basename(p)): p for p in paths_list}
    try:
        metas = _metas(by_key.keys())
    except Exception as e:
        _err("동기화 조회", e)
        return report
    for key, path in by_key.items():
        meta = metas.get(key)
        local = os.path.exists(path)
        try:
            if meta:
                if local:
                    with open(path, "rb") as f:
                        same = _sha(f.read()) == meta["sha256"]
                    if same:
                        _LAST[key] = meta
                        continue
                    if os.path.getmtime(path) > meta["updated_epoch"]:
                        if push(path):
                            report["up"].append(os.path.basename(path))
                        continue
                data = c.storage.from_(BUCKET_ART).download(f"current/{key}.pkl")
                with open(path, "wb") as f:
                    f.write(data)
                _LAST[key] = meta
                report["down"].append(os.path.basename(path))
            elif local:
                if push(path, force_history=True):
                    report["up"].append(os.path.basename(path))
        except Exception as e:
            _err(f"동기화 {os.path.basename(path)}", e)
    return report


# ── 버전 백업 조회 / 복원 ─────────────────────────────────────
def list_history(local_name: str) -> list[str]:
    c = client()
    if not c:
        return []
    key = storage_key(local_name)
    try:
        items = c.storage.from_(BUCKET_ART).list(f"history/{key}", {"limit": 1000})
        return sorted((i["name"] for i in items if i.get("name", "").endswith(".pkl")),
                      reverse=True)
    except Exception as e:
        _err(f"버전목록 {local_name}", e)
        return []


def restore_history(path: str, version_name: str) -> bool:
    """서버의 과거 버전으로 로컬을 되돌리고, 그걸 새 최신본으로 올린다."""
    c = client()
    if not c:
        return False
    key = storage_key(os.path.basename(path))
    try:
        data = c.storage.from_(BUCKET_ART).download(f"history/{key}/{version_name}")
        with open(path, "wb") as f:
            f.write(data)
        return push(path, force_history=True)
    except Exception as e:
        _err(f"버전복원 {version_name}", e)
        return False


# ── 자료(Record)를 DB 테이블로 미러링 ─────────────────────────
def sync_records(path: str, records) -> bool:
    """
    save_records_pkl 직후 호출. pkl 전체 = 이 file_key의 행 전체로 맞춘다.
    행마다 내용 해시(row_sha)를 비교해 '바뀐 행만' 올리고, 빠진 행만 지운다
    → 1건 추가해도 수천 행을 다시 올리지 않음.
    """
    import json
    c = client()
    if not c:
        return False
    file_key = storage_key(os.path.basename(path))
    now = time.time()
    rows = {}
    for r in records:
        d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        d["file_key"] = file_key
        d["row_sha"] = hashlib.sha1(json.dumps(d, ensure_ascii=False, sort_keys=True,
                                               default=str).encode()).hexdigest()
        d["updated_epoch"] = now
        rows[d["rec_id"]] = d            # 같은 id 중복 제거(upsert 충돌 방지)
    try:
        tbl = c.table("records")
        existing, start = {}, 0
        while True:
            res = (tbl.select("rec_id,row_sha").eq("file_key", file_key)
                   .range(start, start + 999).execute())
            got = res.data or []
            existing.update({x["rec_id"]: x.get("row_sha") for x in got})
            if len(got) < 1000:
                break
            start += 1000

        changed = [d for k, d in rows.items() if existing.get(k) != d["row_sha"]]
        for i in range(0, len(changed), 500):
            tbl.upsert(changed[i:i + 500], on_conflict="file_key,rec_id").execute()
        stale = [k for k in existing if k not in rows]
        for i in range(0, len(stale), 200):
            tbl.delete().eq("file_key", file_key).in_("rec_id", stale[i:i + 200]).execute()
        return True
    except Exception as e:
        _err(f"자료 DB {file_key}", e)
        return False


# ── 원본 파일 보관 ────────────────────────────────────────────
_ARCHIVED = set()

def archive_upload(subject: str, kind: str, filename: str, raw: bytes) -> bool:
    """업로드한 원본(pdf/docx/txt)을 한 번만 서버에 보관. 같은 파일은 재업로드 안 함."""
    c = client()
    if not c or not raw:
        return False
    sha = _sha(raw)
    tag = (subject, sha)
    if tag in _ARCHIVED:
        return True
    try:
        hit = (c.table("uploads").select("id").eq("subject", subject)
               .eq("sha256", sha).limit(1).execute())
        if not hit.data:
            ext = os.path.splitext(filename)[1].lower()
            ext = ext if ext.isascii() else ""
            spath = f"{slug(subject)}/{storage_key(kind)}/{sha[:16]}{ext}"
            c.storage.from_(BUCKET_UP).upload(
                spath, raw, {"content-type": "application/octet-stream", "upsert": "true"})
            c.table("uploads").insert({
                "subject": subject, "kind": kind, "original_name": filename,
                "storage_path": spath, "sha256": sha, "size_bytes": len(raw),
                "created_epoch": time.time()}).execute()
        _ARCHIVED.add(tag)
        return True
    except Exception as e:
        _err(f"원본보관 {filename}", e)
        return False
