"""
auto_tag.py — 올린 파일을 읽고 '무슨 자료인지' 자동 분류.

한 파일당 LLM 호출 1번(텍스트만, 이미 스캔한 본문 사용 → 싸고 빠름).
분류 결과는 '제안'이다. 앱에서 파일당 한 줄 표로 보여주고 사람이 확정한다
(최종 판정은 사람 — 자동값이 자료를 오염시키지 않게).

기출은 쪽마다 과목을 따로 판정한다 → 초등 통짜 시험지(전과목 혼합)도
과목별 기출로 자동 분배된다.
"""
from __future__ import annotations
import json, re

CATEGORIES = ["기출", "교육과정_성취기준", "지도서_각론", "지도서_총론",
              "교육과정_총론", "개인_필기"]
SUBJECTS = ["국어", "영어", "수학", "사회", "과학", "미술", "음악", "체육",
            "실과", "도덕", "총론", "창의적체험활동", "통합교과"]
LEVELS = ["초등", "중등", "특수", "공통"]

SYSTEM = """너는 초등 임용시험 준비 자료를 분류하는 사서다.
파일 이름과 각 쪽 본문 앞부분을 보고 아래 JSON 하나만 출력한다.
{
 "category": "기출 | 교육과정_성취기준 | 지도서_각론 | 지도서_총론 | 교육과정_총론 | 개인_필기 중 하나",
 "subject": "국어|영어|수학|사회|과학|미술|음악|체육|실과|도덕|총론|창의적체험활동|통합교과 중 하나",
 "title": "사람이 알아보기 쉬운 자료 이름 (예: '2022 개정 국어과 교육과정', '국어 3-1 교사용 지도서 각론', '2024 초등임용 1교시')",
 "year": 출제연도(기출일 때만, 숫자) 또는 null,
 "level": "초등|중등|특수" 또는 null,
 "grade_band": "학년군(예: 3~4학년군)" 또는 "",
 "area": "영역(예: 읽기)" 또는 "",
 "unit": "단원명" 또는 "",
 "confidence": 0.0~1.0,
 "reason": "판단 근거 한 줄",
 "page_subjects": {"쪽번호": "과목", ...}
}
판단 기준:
- 기출: 문항 번호·배점·'다음을 읽고'·시험 안내문·'○○학년도 ○○교사 임용'. 교육과정 총론·교직 문항은 과목 '총론'.
- 교육과정_총론: 인간상·핵심역량·편성운영 기준 등 전 교과 공통 문서 → subject '총론'.
- 교육과정_성취기준: [4국02-01] 같은 코드와 성취기준 해설이 주를 이룸.
- 지도서_총론: 그 교과의 목표·교수학습 방법·평가 일반론.
- 지도서_각론: 특정 단원·차시의 지도 내용.
- 개인_필기: 손글씨·요약 노트·개인 정리본(문체가 메모투, (?)·[강조] 표시 많음).
- page_subjects는 기출일 때만 채운다. 표지·안내문 쪽은 생략. 기출이 아니면 {}.
확신이 없으면 confidence를 낮게 주고 추측임을 reason에 적는다."""


def _digest(name, pages, per_page=700, total=24000, exam_hint=False):
    out, used = [f"파일 이름: {name}"], 0
    for r in pages:
        t = (r.get("text") or "").strip()
        if not t:
            continue
        chunk = t[: (1200 if exam_hint else per_page)]
        if used + len(chunk) > total:
            break
        out.append(f"[p.{r['page_in_file'] + 1}]\n{chunk}")
        used += len(chunk)
    return "\n\n".join(out)


def _clean(d, n_pages):
    cat = d.get("category") if d.get("category") in CATEGORIES else "지도서_각론"
    subj = d.get("subject") if d.get("subject") in SUBJECTS else None
    if cat == "교육과정_총론":
        subj = "총론"
    year = d.get("year")
    try:
        year = int(year) if year else None
    except Exception:
        year = None
    level = d.get("level") if d.get("level") in LEVELS else None
    ps = {}
    if cat == "기출":
        for k, v in (d.get("page_subjects") or {}).items():
            try:
                k = int(k)
            except Exception:
                continue
            if 1 <= k <= n_pages and v in SUBJECTS:
                ps[k] = v
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except Exception:
        conf = 0.5
    return {"category": cat, "subject": subj, "title": str(d.get("title") or "").strip(),
            "year": year, "level": level,
            "grade_band": str(d.get("grade_band") or "").strip(),
            "area": str(d.get("area") or "").strip(),
            "unit": str(d.get("unit") or "").strip(),
            "confidence": conf, "reason": str(d.get("reason") or "").strip(),
            "page_subjects": ps}


def guess_rules(name, pages, is_image=False):
    """키 없을 때: 파일명·본문 키워드로 대충 추정 (확신도 낮게)."""
    text = name + "\n" + "\n".join((r.get("text") or "")[:800] for r in pages[:5])
    y = re.search(r"(20[0-3]\d)", name)
    subj = next((s for s in SUBJECTS if s in name), None)
    if is_image:
        cat = "개인_필기"
    elif re.search(r"임용|기출|문항|배점|교시", text):
        cat = "기출"
    elif "총론" in text and ("핵심역량" in text or "인간상" in text or "편성" in text):
        cat = "교육과정_총론"
    elif re.search(r"\[\d{1,2}[가-힣]{1,3}\d{2}[-–]\d{2}\]", text):
        cat = "교육과정_성취기준"
    elif "지도서" in text and "총론" in text:
        cat = "지도서_총론"
    else:
        cat = "지도서_각론"
    level = next((l for l in ["초등", "중등", "특수"] if l in text), None)
    return _clean({"category": cat, "subject": subj,
                   "title": re.sub(r"\.[A-Za-z0-9]+$", "", name),
                   "year": int(y.group(1)) if (y and cat == "기출") else None,
                   "level": level, "confidence": 0.3,
                   "reason": "키 없음 — 파일명·키워드 추정"}, len(pages))


def classify(name, pages, api_key=None, model="gpt-4o-mini", is_image=False):
    """
    pages: 이 파일의 스캔 결과 리스트 (각 항목에 page_in_file, text)
    반환: _clean() 형식 dict
    """
    if not api_key:
        return guess_rules(name, pages, is_image)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    exam_hint = bool(re.search(r"임용|기출|교시", name))
    last = None
    for _ in range(2):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": _digest(name, pages,
                                                              exam_hint=exam_hint)}])
            return _clean(json.loads(r.choices[0].message.content), len(pages))
        except Exception as e:
            last = e
    g = guess_rules(name, pages, is_image)
    g["reason"] = f"자동분류 실패({last}) — 키워드 추정"
    return g
