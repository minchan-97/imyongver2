"""
worker.py — 서버에서 혼자 도는 점검·재학습.

    SUPABASE_URL=... SUPABASE_KEY=... python scripts/worker.py --subject 국어 --fix

1) 서버에서 최신 pkl을 내려받고
2) 자기검증(데이터 위생·모델 품질·회귀 테스트)을 돌리고
3) --fix면 기준 미달일 때 임베딩·SOM을 재학습해서
4) 결과 pkl과 리포트를 다시 서버에 올린다.

GitHub Actions·크론·내 PC 어디서 돌려도 같다. 앱은 사이드바에서 그 리포트를 본다.
"""
import os, re, sys, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import paths, cloud, selfcheck


def run_one(subject, fix, maintain_steps=None, label_limit=300):
    # 0) 자료 정비 (깨진 글자·중복·긴 쪽 분할·영역 라벨) — 변경 전 백업이 남는다
    if maintain_steps:
        try:
            import maintenance as mt
            mrep = mt.run(subject, steps=maintain_steps, dry_run=False,
                          api_key=os.environ.get("OPENAI_API_KEY"),
                          model=os.environ.get("OPENAI_TAG_MODEL", "gpt-4o-mini"),
                          label_limit=label_limit)
            sp = mrep.get("split", {})
            print(f"정비: 깨진글자 {mrep.get('garbage', {}).get('n', 0)} · "
                  f"중복 {mrep.get('dedupe', 0)} · "
                  f"분할 {sp.get('before', 0)}→{sp.get('after', 0)} · "
                  f"영역라벨 {mrep.get('label', 0)} · "
                  f"태그정리 {len(mrep.get('fix_tags', []))}")
        except Exception as e:
            print("정비 건너뜀:", e)

    r = selfcheck.run(subject, fix=fix)
    # 새 기출이 들어왔으면 대기 중인 경향 예측을 자동 채점 (LLM 호출 없음)
    try:
        import trend_lab as tl
        from schema import load_records_pkl
        lab = tl.load_lab(subject)
        n = tl.score(subject, load_records_pkl(paths.l1_path(subject)),
                     lab["predictions"], lab)
        if n:
            tl.save_lab(subject, lab)
        print(f"경향 예측 채점: {n}건 · 누적 {tl.summary(lab['predictions'])}")
    except Exception as e:
        print("경향 채점 건너뜀:", e)
    # 오늘의 자료집 편성 (OPENAI_API_KEY가 있으면 요약·확인 질문까지)
    try:
        import daily_digest as dd
        dg, _ = dd.build(subject, n_items=int(os.environ.get("DIGEST_ITEMS", 5)),
                         api_key=os.environ.get("OPENAI_API_KEY"),
                         model=os.environ.get("OPENAI_DIGEST_MODEL", "gpt-4o-mini"),
                         force=True)
        print(f"자료집: {len(dg['items'])}항목 · {dg['chars']}자 — "
              + ", ".join(i["title"] for i in dg["items"]))
    except Exception as e:
        print("자료집 건너뜀:", e)

    # 자가 시험: 스스로 내고 풀고, 기계적으로 채점 (전략 성적 누적)
    try:
        import self_exam as se
        ex = se.run(subject, api_key=os.environ.get("OPENAI_API_KEY"),
                    model=os.environ.get("OPENAI_TAG_MODEL", "gpt-4o-mini"),
                    n_retrieval=int(os.environ.get("EXAM_RETRIEVAL", 30)),
                    n_cloze=int(os.environ.get("EXAM_CLOZE", 8)))
        if "error" in ex:
            print("자가 시험:", ex["error"])
        else:
            print(f"자가 시험: 검색 {ex['retrieval'].get('ok', 0)}/{ex['retrieval'].get('n', 0)} · "
                  f"빈칸 {ex['cloze'].get('ok', 0)}/{ex['cloze'].get('n', 0)} · "
                  f"전략 {[(a['전략'], round(a['성공률'], 2) if a['성공률'] is not None else '-') for a in ex['arms']]}")
            # 결핍만 웹에서 찾아 수집함에 쌓기 (자료로 넣지는 않음 — 사람이 채택)
            bkey = os.environ.get("BRAVE_API_KEY")
            if bkey:
                import gap_search as gs
                lab = se.load(subject)
                gaps = se.top_gaps(lab, n=int(os.environ.get("GAP_QUERIES", 5)),
                                   only_unsearched=True)
                if gaps:
                    n = gs.collect(subject, gaps, bkey)
                    se.save(subject, lab)
                    print(f"결핍 검색: 구멍 {len(gaps)}곳 → 후보 {n}건 (검토 대기)")
    except Exception as e:
        print("자가 시험 건너뜀:", e)

    # AI의 질문 만들기 (측정된 불확실성 → 아침에 사람이 답함)
    try:
        import ask_box as ab
        made = ab.generate(subject, n=int(os.environ.get("ASK_N", 6)),
                           api_key=os.environ.get("OPENAI_API_KEY"),
                           model=os.environ.get("OPENAI_TAG_MODEL", "gpt-4o-mini"))
        s = ab.summary(subject)
        print(f"AI 질문: 새로 {made}개 · 대기 {s['open']}개 · "
              f"답변 누적 {s['answered']}개(도움 {s['helped']})")
    except Exception as e:
        print("질문 생성 건너뜀:", e)

    print(json.dumps({k: v for k, v in r.items() if k != "hygiene"},
                     ensure_ascii=False, indent=2, default=str))
    print("경고:", r["alerts"] or "없음")
    return r

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default=os.environ.get("SUBJECT", "all"),
                help="과목명, 쉼표로 여러 개, 또는 'all'(자료가 있는 과목 전부)")
ap.add_argument("--fix", action="store_true", help="기준 미달이면 재학습")
ap.add_argument("--maintain", default=os.environ.get("MAINTAIN", ""),
                help="정비 단계(쉼표) 또는 'on'(기본 단계) / 비우면 정비 안 함")
ap.add_argument("--label-limit", type=int,
                default=int(os.environ.get("LABEL_LIMIT", 300)))
a = ap.parse_args()

_m = (a.maintain or "").strip().lower()
if _m in ("", "0", "off", "false"):
    MAINT = None
else:
    import maintenance as _mt
    MAINT = (_mt.APP_STEPS if _m in ("1", "on", "true", "yes")
             else tuple(s.strip() for s in a.maintain.split(",") if s.strip()))
print("정비 단계:", ", ".join(MAINT) if MAINT else "안 함")

if not cloud.enabled():
    print("⚠️  SUPABASE_URL/KEY 없음 — 로컬 파일만 사용합니다.")

if a.subject.strip().lower() in ("all", "*", "전체"):
    subjects = paths.discover_subjects(cloud.list_local_names())
    if not subjects:
        print("자료가 있는 과목을 못 찾았어요. --subject 국어 처럼 직접 지정하세요.")
        sys.exit(0)
else:
    subjects = [s.strip() for s in a.subject.split(",") if s.strip()]
print("대상 과목:", ", ".join(subjects))

# ── 0) 스캔 대기열 처리 (앱에서 '다시 읽기'로 걸어둔 것 + 드라이브 새 파일) ──
OPENAI = os.environ.get("OPENAI_API_KEY")
if OPENAI:
    import ingest_queue as iq
    folders = [x for x in re.split(r"[\n,]+", os.environ.get("DRIVE_FOLDERS", "")) if x.strip()]
    if folders:
        try:
            n = iq.enqueue_drive_new(folders, os.environ.get("GOOGLE_API_KEY"),
                                     os.environ.get("GOOGLE_SERVICE_ACCOUNT"))
            print(f"드라이브 새 파일 {n}개 대기열 추가")
        except Exception as e:
            print("드라이브 수집 실패:", e)
    _rescan = (os.environ.get("RESCAN_ALL") or "").strip()
    if _rescan and _rescan.lower() not in ("0", "off", "false", ""):
        _sub = None if _rescan.lower() in ("1", "on", "true", "all", "전체") else _rescan
        _wipe = (os.environ.get("WIPE_FIRST", "") or "").lower() in ("1", "on", "true")
        try:
            n = iq.enqueue_all_uploads(_sub, replace=True, wipe=_wipe)
            print(f"전체 다시 읽기: {n}건 대기열 추가"
                  + (" (기존 기록 비우기 포함)" if _wipe else ""))
        except Exception as e:
            print("전체 다시 읽기 실패:", e)

    _pend = iq.pending()
    if _pend:
        print(f"\n════ 스캔 대기열 {len(_pend)}건 (이번 회차 최대 "
              f"{os.environ.get('QUEUE_LIMIT', 20)}건) ════")
        _qlimit = int(os.environ.get("QUEUE_LIMIT", 20))
        _ppj = int(os.environ.get("SCAN_PAGES_PER_JOB", 120))
        for r in iq.run_queue(OPENAI, os.environ.get("OPENAI_SCAN_MODEL"),
                              limit=_qlimit, max_pages=_ppj):
            print("  ", r)
else:
    print("OPENAI_API_KEY 없음 — 스캔 대기열은 건너뜁니다")

summary = []
for subject in subjects:
    print(f"\n════ {subject} ════")
    if cloud.enabled():
        rep = cloud.sync(paths.all_paths(subject))
        print(f"동기화: 내려받음 {len(rep['down'])} · 올림 {len(rep['up'])}")
    try:
        r = run_one(subject, a.fix, MAINT, a.label_limit)
        summary.append((subject, len(r["alerts"]), bool(r.get("retrained"))))
    except Exception as e:
        print(f"⚠️  {subject} 실패: {e}")
        summary.append((subject, -1, False))

print("\n════ 요약 ════")
for s, n, rt in summary:
    print(f"{s}: " + ("실패" if n < 0 else f"경고 {n}건") + (" · 재학습함" if rt else ""))
if cloud.ERRORS:
    print("서버 오류:", cloud.ERRORS)

