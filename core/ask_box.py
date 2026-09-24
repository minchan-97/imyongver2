"""
ask_box.py — AI가 모르는 것을 사람에게 묻는다.

질문은 '만들어낸 궁금증'이 아니라 **측정된 불확실성**에서 나온다.
  · 자가 시험에서 근거를 못 찾은 성취기준·개념        → 개념 질문
  · 빈칸 복원 실패                                   → 용어 질문
  · 스캔 판독 불안((?)·[판독불가]이 많은 쪽)          → 판독 확인 질문
  · 규칙으로 과목을 못 가른 문서                      → 분류 질문
  · 기출 통계에서 눈에 띄는 변화(영역 증감·신규 코드) → 임용 전반 질문

답을 하면 두 가지가 일어난다.
  1. 답이 자료로 편입된다(doc_type "내_답변", 출처는 '내 답변 · 날짜').
     → 다음부터 해설·자료집이 이 답을 근거로 인용할 수 있다.
  2. 그 질문이 겨냥한 구멍에 표시가 남는다. 다음 자가 시험이 그 구멍을 다시 풀어
     성공하면 '도움이 된 답'으로 기록된다 (사람의 답도 성적이 추적된다).

저장: data/askbox_{과목}.pkl
"""
from __future__ import annotations
import os, re, json, time, pickle, hashlib
from collections import Counter

import paths

try:
    from cloud import push as _cloud_push
except Exception:
    def _cloud_push(path, **kw): return False

KINDS = {
    "concept": "개념",
    "term": "용어",
    "scan": "판독 확인",
    "classify": "분류",
    "meta": "임용 전반",
}


def box_path(subject):
    return paths._p(f"askbox_{subject}.pkl")


def load(subject):
    p = box_path(subject)
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"questions": [], "answers": [], "seen_keys": []}


def save(subject, box):
    box["questions"] = box["questions"][-200:]
    box["answers"] = box["answers"][-300:]
    box["seen_keys"] = box["seen_keys"][-500:]
    p = box_path(subject)
    with open(p, "wb") as f:
        pickle.dump(box, f)
    _cloud_push(p)


def _qid(kind, key):
    return hashlib.sha1(f"{kind}|{key}".encode()).hexdigest()[:10]


def _add(box, kind, key, question, context="", why="", gap_key=None):
    qid = _qid(kind, key)
    if qid in box["seen_keys"]:
        return False
    box["questions"].append({"id": qid, "kind": kind, "key": key,
                             "question": question, "context": context[:600],
                             "why": why, "gap_key": gap_key,
                             "status": "open", "at": time.time()})
    box["seen_keys"].append(qid)
    return True


# ── 질문 만들기 ──────────────────────────────────────────────
def generate(subject, n=6, api_key=None, model="gpt-4o-mini", log=print):
    """측정된 불확실성 → 질문. LLM은 '임용 전반' 질문 한두 개에만 쓴다(선택)."""
    from schema import load_records_pkl
    box = load(subject)
    made = 0

    # 1) 자가 시험이 못 메운 구멍
    try:
        import self_exam as se
        lab = se.load(subject)
        from labeler import clean_area
        for g in se.top_gaps(lab, n=6):
            key = str(g["key"])
            if key.startswith("[") and key.endswith("]"):          # 성취기준 코드
                kind = "concept"
                q = (f"{key} 성취기준은 실제 수업과 기출에서 어떤 형태로 다뤄지나요? "
                     "핵심 개념과 자주 나오는 발문 형태를 알려주세요.")
            elif "빈칸 복원 실패" in (g.get("why") or ""):
                m = re.search(r"\((.+?)\)", g.get("why") or "")
                t = m.group(1) if m else key
                kind = "term"
                q = (f"'{t}'가 무엇인가요? 임용에서 이 용어가 어떤 맥락으로 나오는지도 "
                     "같이 알려주시면 좋겠어요.")
            elif clean_area(key) == key and len(key) <= 8:          # 영역 이름
                kind = "concept"
                q = (f"'{key}' 영역은 제가 가진 자료가 얇아요. 이 영역에서 임용에 자주 나오는 "
                     "개념과 출제 방식은 어떤가요?")
            else:
                kind = "term"
                q = (f"'{key}'에 대해 제가 가진 자료로는 근거를 못 찾았어요. "
                     "어디서 배우는 내용인지, 핵심이 무엇인지 알려주세요.")
            made += _add(box, kind, key, q, context=(g.get("sample") or ""),
                         why=f"자가 시험 {g['misses']}회 실패", gap_key=key)
    except Exception as e:
        log(f"  구멍 질문 생략: {e}")

    # 2) 판독이 불안한 쪽 (사람만 확인 가능)
    UNC = re.compile(r"\(\?\)|\[판독불가\]")
    for r in load_records_pkl(paths.l2_path(subject)):
        hits = UNC.findall(r.text)
        if len(hits) >= 3:
            snippet = r.text[:200]
            made += _add(box, "scan", r.source,
                         f"'{r.source}'을 읽을 때 글자가 흐려 확신이 없었어요. "
                         "아래 부분이 원본에 뭐라고 적혀 있나요?",
                         context=snippet, why=f"판독 불안 {len(hits)}곳")
            if made >= n * 2:
                break

    # 3) 과목·종류를 못 가른 문서
    try:
        import resubject as rsj
        for d in rsj.undetermined_docs([subject])[:3]:
            made += _add(box, "classify", d["source"],
                         f"'{d['source']}'이 어떤 자료인지 헷갈려요. "
                         "어느 과목의 무슨 자료인가요? (교육과정/지도서/기출/개인 정리)",
                         context=d["sample"][:300] if d.get("sample") else "",
                         why="규칙으로 판정 불가")
    except Exception:
        pass

    # 4) 임용 전반 — 기출 통계에서 실제로 관찰된 변화만 근거로 묻는다
    try:
        import trend_lab as tl
        l1 = load_records_pkl(paths.l1_path(subject))
        st_ = tl.stats(l1)
        if st_["years"] and len(st_["years"]) >= 2:
            recent = st_["areas"]["recent"]
            new_codes = st_["codes"]["new"][:3]
            facts = []
            from labeler import clean_area
            recent = {k: v for k, v in (recent or {}).items() if clean_area(k) == k}
            if recent:
                top = ", ".join(f"{k} {v}회" for k, v in list(recent.items())[:3])
                facts.append(f"최근 3년 영역 분포: {top}")
            if new_codes:
                facts.append(f"최근에 새로 등장한 코드: {', '.join(new_codes)}")
            if facts:
                if api_key:
                    q = _meta_question(subject, facts, api_key, model)
                else:
                    q = (f"{facts[0]} 이런 분포가 실제 시험 난이도나 출제 방식과 "
                         "어떻게 연결되나요? 왜 이 영역이 자주 나온다고 보시나요?")
                made += _add(box, "meta", f"trend-{int(time.time()) // 86400}",
                             q, context=" / ".join(facts), why="기출 통계 관찰")
    except Exception as e:
        log(f"  통계 질문 생략: {e}")

    save(subject, box)
    return made


META_SYSTEM = """너는 임용 준비생에게 배우려는 학습 조교다. 아래 '관찰된 통계'만 근거로,
수험생만 답할 수 있는 질문 하나를 만든다. 통계에 없는 사실을 지어내지 말 것.
질문은 임용 전반(난이도, 출제 방식, 채점 기준, 왜 그렇게 출제되는지, 공부 전략 등)에서
고르고, 한 문장으로. JSON: {"question":"..."}"""


def _meta_question(subject, facts, api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model, temperature=0.4, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": META_SYSTEM},
                  {"role": "user", "content": f"과목: {subject}\n관찰:\n- " + "\n- ".join(facts)}])
    return str(json.loads(r.choices[0].message.content).get("question", ""))[:300]


# ── 답하기 ───────────────────────────────────────────────────
def pending(subject, kinds=None):
    box = load(subject)
    qs = [q for q in box["questions"] if q["status"] == "open"
          and (not kinds or q["kind"] in kinds)]
    qs.sort(key=lambda q: -q["at"])
    return qs


def answer(subject, qid, text, keep_as_material=True):
    """
    답변 저장. keep_as_material이면 자료로 편입해 다음부터 근거로 쓰인다.
    (출처는 '내 답변'으로 남아 공식 자료와 구분된다)
    """
    from schema import Record, load_records_pkl, save_records_pkl
    text = (text or "").strip()
    if not text:
        return False
    box = load(subject)
    q = next((x for x in box["questions"] if x["id"] == qid), None)
    if not q:
        return False
    q["status"] = "answered"
    q["answered_at"] = time.time()
    rec_id = None
    if keep_as_material and len(text) >= 10:
        path = paths.l2_path(subject)
        recs = load_records_pkl(path)
        body = f"[질문] {q['question']}\n[내 답변] {text}"
        try:
            rec = Record(text=body, layer="L2_corpus", subject=subject,
                         source=f"내 답변 · {time.strftime('%Y-%m-%d')} · {q['key'][:30]}",
                         doc_type="내_답변",
                         code=q["key"] if str(q["key"]).startswith("[") else None)
            if rec.rec_id not in {r.rec_id for r in recs}:
                recs.append(rec)
                save_records_pkl(recs, path)
                rec_id = rec.rec_id
        except Exception:
            pass
    box["answers"].append({"qid": qid, "kind": q["kind"], "key": q["key"],
                           "question": q["question"], "answer": text,
                           "rec_id": rec_id, "gap_key": q.get("gap_key"),
                           "helped": None, "at": time.time()})
    save(subject, box)
    return True


def skip(subject, qid):
    box = load(subject)
    for q in box["questions"]:
        if q["id"] == qid:
            q["status"] = "skipped"
    save(subject, box)
    return True


def mark_helped(subject, gap_key, helped=True):
    """자가 시험이 그 구멍을 다시 풀어 성공하면, 그 답이 도움이 됐다고 기록."""
    box = load(subject)
    n = 0
    for a in box["answers"]:
        if a.get("gap_key") == gap_key and a.get("helped") is None:
            a["helped"] = bool(helped)
            n += 1
    if n:
        save(subject, box)
    return n


def summary(subject):
    box = load(subject)
    ans = box["answers"]
    helped = [a for a in ans if a.get("helped")]
    return {"open": len([q for q in box["questions"] if q["status"] == "open"]),
            "answered": len(ans), "helped": len(helped),
            "by_kind": dict(Counter(q["kind"] for q in box["questions"]))}
