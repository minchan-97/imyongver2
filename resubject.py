"""
resubject.py — 이미 저장된 자료의 '과목'이 맞는지 다시 판정하고 옮긴다.

왜 필요한가: 넣을 때 과목 칸의 기본값이 '지금 보고 있는 과목'이라,
자동 분류가 과목을 못 집으면 전부 그 과목(대개 국어)으로 들어간다.
쌓인 뒤에 발견되므로 사후 재판정이 필요하다.

판정 근거(강한 것부터)
  1) 성취기준 코드의 교과 글자  — [4국02-01] → 국어, [2슬03-02] → 통합교과 (거의 확실)
  2) 교과 고유 어휘            — '음운', '분수의 나눗셈', '용해도' 등
  3) (선택) LLM                — 위 둘로 못 가르는 것만

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


def judge(text, code=None):
    """한 기록의 과목 판정. 반환: (과목, 확신도 0~1, 근거)"""
    scores = Counter()
    ev = []
    codes = [code] if code else []
    codes += ["[%s%s%s-%s]" % m for m in CODE_RE.findall(text or "")]
    for c in codes:
        m = CODE_RE.match(c or "")
        if not m:
            continue
        subj = CODE_SUBJECT.get(m.group(2)[0])
        if subj:
            scores[subj] += 3
            ev.append(f"코드 {c}")
    t = text or ""
    for subj, words in LEXSET.items():
        hit = [w for w in words if len(w) >= 2 and w in t]
        if hit:
            scores[subj] += min(3, len(hit)) * 0.7
            ev.append(f"{subj} 어휘 {'/'.join(hit[:3])}")
    if not scores:
        return None, 0.0, []
    best, top = scores.most_common(1)[0]
    total = sum(scores.values())
    # 근거의 '양'도 반영: 코드 1개(3점)나 고유어휘 3개(2.1점)는 충분,
    # 어휘 1개(0.7점)만으로는 확신하지 않는다.
    strength = min(1.0, top / 2.1)
    return best, round((top / total) * strength, 2), ev[:4]


def audit(subject_list=None, min_conf=0.6, layers=("L2_corpus", "L1_pattern")):
    """
    저장된 기록들을 훑어 '지금 과목과 다르게 판정되는' 것만 모은다.
    반환: [{rec_id, path, current, proposed, conf, evidence, source, text}]
    """
    subject_list = subject_list or sorted(SUBJECTS - {"공통"})
    out = []
    for subj in subject_list:
        for path, layer in ((paths.l2_path(subj), "L2_corpus"),
                            (paths.l1_path(subj), "L1_pattern")):
            if layer not in layers:
                continue
            for r in load_records_pkl(path):
                prop, conf, ev = judge(r.text, r.code)
                if prop and prop != subj and conf >= min_conf:
                    out.append({"rec_id": r.rec_id, "path": path, "layer": layer,
                                "current": subj, "proposed": prop, "conf": conf,
                                "evidence": ev, "source": r.source,
                                "text": r.text[:100]})
    out.sort(key=lambda d: -d["conf"])
    return out


def summary(rows):
    """'국어 → 수학 12건' 식 요약."""
    c = Counter((r["current"], r["proposed"]) for r in rows)
    return [{"from": a, "to": b, "n": n} for (a, b), n in c.most_common()]


def apply_moves(rows):
    """
    확인된 것만 실제로 옮긴다. 같은 layer의 대상 과목 pkl로 이동.
    (rec_id는 출처+본문으로 만들어지므로 과목이 바뀌어도 그대로 → 중복 안 생김)
    """
    by_src = defaultdict(list)
    for r in rows:
        by_src[(r["path"], r["layer"], r["proposed"])].append(r["rec_id"])

    moved, touched = 0, {}
    for (src_path, layer, target_subj), ids in by_src.items():
        recs = load_records_pkl(src_path)
        keep, move = [], []
        idset = set(ids)
        for r in recs:
            (move if r.rec_id in idset else keep).append(r)
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
        touched.setdefault(src_path, None)
        save_records_pkl(keep, src_path)
        save_records_pkl(tgt, tgt_path)
        moved += len(move)
    return moved
