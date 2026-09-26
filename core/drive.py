"""
drive.py — 구글 드라이브 폴더에서 자료 가져오기.

secrets에 폴더 링크만 넣어두면, 앱이 그 폴더(하위 폴더 포함)를 훑어
'아직 안 가져온 파일'만 내려받아 기존 스캔·자동분류 파이프라인에 태운다.

인증 세 가지 중 하나:
  0) 키 없음(기본) — 공개 폴더 링크만
     폴더를 '링크가 있는 모든 사용자 - 뷰어'로 두면, 구글이 iframe 임베드용으로 제공하는
     공개 목록 페이지(embeddedfolderview)를 읽어 파일 목록을 얻는다.
     공식 문서에 없는 경로라 구글이 바꾸면 깨질 수 있고, 수정 시각도 '날짜'까지만 나온다.
     제대로 쓰려면 아래 A/B를 권한다(API 키는 무료로 발급 가능).
  A) 공개 폴더 + GOOGLE_API_KEY
     폴더 공유를 '링크가 있는 모든 사용자 - 뷰어'로 두고 API 키만 넣으면 됨. 가장 간단.
  B) 비공개 폴더 + GOOGLE_SERVICE_ACCOUNT (서비스 계정 JSON 통째로)
     폴더를 그 서비스 계정 이메일에 '뷰어'로 공유. 공개 안 해도 됨.

구글 문서/슬라이드/스프레드시트는 원본 파일이 없으므로 PDF로 내보내 가져온다.
가져온 파일은 (파일ID, 수정시각)으로 기록해 두어 다음엔 건너뛴다.
"""
from __future__ import annotations
import os, re, io, json, pickle, time

API = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME = "application/vnd.google-apps.folder"
EXPORT = {                      # 구글 네이티브 문서 → 내려받을 형식
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.drawing": ("image/png", ".png"),
}
TAKE_EXT = (".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False


def folder_id(url_or_id: str) -> str | None:
    """드라이브 폴더 링크에서 ID만 뽑기 (ID를 그대로 넣어도 됨)."""
    s = (url_or_id or "").strip()
    if not s:
        return None
    m = re.search(r"/folders/([A-Za-z0-9_-]{10,})", s) or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    return s if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s) else None


# ── 키 없이: 공개 폴더 목록/내려받기 ─────────────────────────
PUB_LIST = "https://drive.google.com/embeddedfolderview?id={}#list"
PUB_DL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
GDOC_EXPORT = {
    "document": ("https://docs.google.com/document/d/{}/export?format=pdf", ".pdf"),
    "presentation": ("https://docs.google.com/presentation/d/{}/export/pdf", ".pdf"),
    "spreadsheet": ("https://docs.google.com/spreadsheets/d/{}/export?format=pdf", ".pdf"),
}
_ENTRY_SPLIT = re.compile(r'<div[^>]*class="flip-entry"')
_ID = re.compile(r'id="entry-([A-Za-z0-9_-]{10,})"')
_TITLE = re.compile(r'flip-entry-title"[^>]*>(.*?)</div>', re.S)
_MOD = re.compile(r'flip-entry-last-modified"[^>]*>\s*<div[^>]*>(.*?)</div>', re.S)
_ICON = re.compile(r'icon_\d+_([a-z]+)_list', re.I)


def _unescape(s):
    import html
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def list_folder_public(fid, recursive=True, _prefix="", _depth=0, _seen=None):
    """API 키 없이 공개 폴더 목록 읽기."""
    import requests
    _seen = set() if _seen is None else _seen
    if fid in _seen:                      # 폴더 순환 방지
        return []
    _seen.add(fid)
    r = requests.get(PUB_LIST.format(fid), timeout=30,
                     headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"공개 폴더를 못 읽었어요 ({r.status_code}). "
                           "폴더 공유가 '링크가 있는 모든 사용자'인지 확인해 주세요.")
    out = []
    for chunk in _ENTRY_SPLIT.split(r.text)[1:]:
        m = _ID.search(chunk)
        if not m:
            continue
        fileid = m.group(1)
        t = _TITLE.search(chunk)
        name = _unescape(t.group(1)) if t else fileid
        mod = _MOD.search(chunk)
        modified = _unescape(mod.group(1)) if mod else ""
        ic = _ICON.search(chunk)
        icon = ic.group(1).lower() if ic else ""
        if icon == "folder":
            if recursive and _depth < 5:
                out += list_folder_public(fileid, True, _prefix + name + "/",
                                          _depth + 1, _seen)
            continue
        if icon in GDOC_EXPORT:
            mime = f"application/vnd.google-apps.{icon}"
        elif name.lower().endswith(TAKE_EXT):
            mime = ""
        else:
            continue                      # 다루지 않는 형식
        out.append({"id": fileid, "name": name, "mime": mime, "modified": modified,
                    "size": 0, "path": _prefix + name, "public": True})
    return out


def download_public(file) -> tuple[str, bytes]:
    """API 키 없이 공개 파일 내려받기."""
    import requests
    kind = (file.get("mime") or "").rsplit(".", 1)[-1]
    if kind in GDOC_EXPORT:
        url, ext = GDOC_EXPORT[kind]
        url = url.format(file["id"])
        name = file["name"] + (ext if not file["name"].lower().endswith(ext) else "")
    else:
        url, name = PUB_DL.format(file["id"]), file["name"]
    r = requests.get(url, timeout=180, allow_redirects=True,
                     headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"내려받기 실패 {file['name']} ({r.status_code})")
    head = r.content[:512].lower()
    if b"<html" in head and not file["name"].lower().endswith((".htm", ".html")):
        raise RuntimeError(f"{file['name']}: 공개 파일이 아니거나 다운로드 한도 초과 "
                           "(공유 설정 확인, 또는 잠시 후 재시도)")
    return name, r.content


def _auth(api_key=None, sa_json=None):
    """반환: (params, headers)"""
    if sa_json:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
        info = json.loads(sa_json) if isinstance(sa_json, str) else dict(sa_json)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        creds.refresh(Request())
        return {}, {"Authorization": f"Bearer {creds.token}"}
    if api_key:
        return {"key": api_key}, {}
    raise ValueError("GOOGLE_API_KEY 또는 GOOGLE_SERVICE_ACCOUNT 중 하나가 필요해요.")


def list_folder(fid, api_key=None, sa_json=None, recursive=True, _prefix="", _depth=0):
    """폴더 안의 파일 목록 (하위 폴더 포함). 키가 없으면 공개 폴더 모드."""
    import requests
    if not (api_key or sa_json):
        return list_folder_public(fid, recursive, _prefix, _depth)

    params, headers = _auth(api_key, sa_json)
    out, page = [], None
    while True:
        q = {**params,
             "q": f"'{fid}' in parents and trashed=false",
             "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,size)",
             "pageSize": 1000, "supportsAllDrives": "true",
             "includeItemsFromAllDrives": "true"}
        if page:
            q["pageToken"] = page
        r = requests.get(API, params=q, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"드라이브 목록 실패 ({r.status_code}): {r.text[:200]}")
        d = r.json()
        for f in d.get("files", []):
            if f["mimeType"] == FOLDER_MIME:
                if recursive and _depth < 5:
                    out += list_folder(f["id"], api_key, sa_json, True,
                                       _prefix + f["name"] + "/", _depth + 1)
                continue
            keep = f["mimeType"] in EXPORT or f["name"].lower().endswith(TAKE_EXT)
            if keep:
                out.append({"id": f["id"], "name": f["name"], "mime": f["mimeType"],
                            "modified": f.get("modifiedTime", ""),
                            "size": int(f.get("size") or 0),
                            "path": _prefix + f["name"]})
        page = d.get("nextPageToken")
        if not page:
            break
    return out


def dedupe(files):
    """
    여러 폴더를 합칠 때 같은 파일이 두 번 들어가지 않게.
    파일 ID뿐 아니라 (이름, 크기)로도 걸러낸다 — 같은 자료를 폴더 두 곳에 둔 경우
    ID가 달라서 두 번 스캔되고 비용이 두 배로 든다.
    """
    seen_id, seen_name, out = set(), set(), []
    for f in files:
        key = (f.get("name"), f.get("size") or 0)
        if f["id"] in seen_id or key in seen_name:
            continue
        seen_id.add(f["id"])
        seen_name.add(key)
        out.append(f)
    return out


def download(file, api_key=None, sa_json=None) -> tuple[str, bytes]:
    """파일 1개 내려받기. 반환: (파일명, bytes). 구글 문서는 PDF로 변환."""
    import requests
    if not (api_key or sa_json):
        return download_public(file)
    params, headers = _auth(api_key, sa_json)
    if file["mime"] in EXPORT:
        mime, ext = EXPORT[file["mime"]]
        url = f"{API}/{file['id']}/export"
        q = {**params, "mimeType": mime}
        name = file["name"] + (ext if not file["name"].lower().endswith(ext) else "")
    else:
        url = f"{API}/{file['id']}"
        q = {**params, "alt": "media", "supportsAllDrives": "true"}
        name = file["name"]
    r = requests.get(url, params=q, headers=headers, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"내려받기 실패 {file['name']} ({r.status_code}): {r.text[:200]}")
    return name, r.content


# ── 가져온 파일 기록 ──────────────────────────────────────────
class DriveState:
    """{file_id: {name, modified, imported_epoch}} — 이미 가져온 파일은 다시 안 가져오게."""

    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.d = pickle.load(f)
            except Exception:
                self.d = {}

    def is_new(self, f) -> bool:
        prev = self.d.get(f["id"])
        return (prev is None) or (prev.get("modified") != f["modified"])

    def mark(self, f):
        self.d[f["id"]] = {"name": f["name"], "modified": f["modified"],
                           "imported_epoch": time.time()}

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.d, f)
        _cloud_push(self.path)
