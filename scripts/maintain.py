"""
maintain.py — 전 과목 자료 정비 (분할·중복·태그·라벨·재학습).

    python scripts/maintain.py                      # 전 과목 미리보기(변경 없음)
    python scripts/maintain.py --apply              # 실제 적용
    python scripts/maintain.py --subject 국어,수학 --apply
    python scripts/maintain.py --apply --steps fix_tags,dedupe,split

서버 키(SUPABASE_URL/KEY)가 있으면 먼저 내려받고, 끝나면 결과가 자동으로 올라간다.
"""
import os, sys, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import paths, cloud, maintenance as mt

ap = argparse.ArgumentParser()
ap.add_argument("--subject", default=os.environ.get("SUBJECT", "all"))
ap.add_argument("--steps", default=",".join(mt.STEPS))
ap.add_argument("--apply", action="store_true", help="실제로 바꾸기 (없으면 미리보기)")
ap.add_argument("--label-limit", type=int, default=300)
a = ap.parse_args()

steps = tuple(s.strip() for s in a.steps.split(",") if s.strip())
subjects = mt.subject_list(None if a.subject.lower() in ("all", "전체") else a.subject)
if not subjects and cloud.enabled():          # 로컬이 비었으면 서버 목록으로
    subjects = paths.discover_subjects(cloud.list_local_names())
print("대상 과목:", ", ".join(subjects) or "(없음)")

for s in subjects:
    if cloud.enabled():
        rep = cloud.sync(paths.all_paths(s))
        print(f"[{s}] 동기화: 내려받음 {len(rep['down'])} · 올림 {len(rep['up'])}")
    r = mt.run(s, steps=steps, dry_run=not a.apply,
               api_key=os.environ.get("OPENAI_API_KEY"),
               model=os.environ.get("OPENAI_TAG_MODEL", "gpt-4o-mini"),
               label_limit=a.label_limit)
    print(f"\n════ {s} ({'적용' if a.apply else '미리보기'}) ════")
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str)[:2000])

if not a.apply:
    print("\n미리보기였습니다. 실제로 바꾸려면 --apply 를 붙이세요.")
