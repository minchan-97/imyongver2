"""
app.py — 임용 국어 4레이어 Streamlit 앱 (단일 앱·탭 방식)

실행:  streamlit run app.py

탭 구성:
  📚 자료·학습(L2)  : 성취기준/지도서 입력·업로드 → pkl 저장 → 임베딩·SOM 학습
  📈 기출 패턴(L1)  : 기출 입력 → 개념영역별 연도·급별 패턴 분석
  🔎 트렌드(L3)     : 검색어 자동 생성 → Brave 검색 → 가드레일 필터 → pkl 저장
  📝 문제 풀기(L4)  : 개념영역별 연습문제(해설 잠금) → '해설 보기' 버튼으로만 공개

데이터는 전부 과목별 pkl로 data/ 아래 저장된다.
"""
import sys, os, json, time, re
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer1_pattern"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer2_corpus"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer3_trend"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer4_generate"))

from schema import Record, save_records_pkl, load_records_pkl, LEVELS, DOC_TYPES
from embedding import FrozenEmbedding, train_embedding
from som import SOM
from korean_tokenizer import tokenize
import paths
import cloud
import hmac

st.set_page_config(page_title="임용 4레이어", layout="wide")

def api_key(label, secret_name, widget_key):
    """secrets에 키가 있으면 그걸 쓰고 입력칸은 숨김. 없으면 예전처럼 입력칸."""
    v = cloud.cfg(secret_name)
    if v:
        st.caption(f"🔑 {label}: secrets에서 불러옴")
        return str(v)
    return st.text_input(label, type="password", key=widget_key)

def page_scan_ui(prefix, ups, concept_names=None, handwriting=False):
    """
    파일들(pdf·사진·docx·txt, 여러 개 가능) → 쪽 통째로 스캔 → 쪽 표로 검토.
    반환: (edited_df, results) 또는 (None, None) (아직 스캔 전)
    """
    import pandas as pd
    from page_scan import build_pages, scan_pages, render_source, file_sha, ScanCache
    files = [(u.name, u.getvalue()) for u in ups]
    pages = build_pages(files)
    needs_key = any(p["kind"] != "text" for p in pages)
    okey = api_key("OpenAI Key(스캔용)", "OPENAI_API_KEY", f"{prefix}_scankey") if needs_key else ""
    sig = file_sha("|".join(file_sha(r) for _, r in files).encode() + bytes([handwriting]))
    state = f"{prefix}_scan_{sig[:16]}"

    if state not in st.session_state:
        st.caption(f"{len(files)}개 파일 · {len(pages)}쪽"
                   + (" · 손글씨 모드(고해상도·정밀 모델)" if handwriting else ""))
        if needs_key and not okey:
            st.caption("키가 없으면 PDF 텍스트층만 읽어요(사진·스캔본·손글씨는 못 읽음).")
        if st.button("🔍 스캔 시작", type="primary", key=f"{prefix}_scanbtn"):
            bar = st.progress(0.0, text="스캔 준비…")
            t0 = time.time()
            res = scan_pages(pages, okey or None, concept_names or [],
                             cache=ScanCache(paths.scan_cache_path()),
                             model=cloud.cfg("OPENAI_HANDWRITING_MODEL" if handwriting
                                             else "OPENAI_SCAN_MODEL"),
                             handwriting=handwriting,
                             progress=lambda d, n: bar.progress(d / max(n, 1),
                                                                text=f"{d}/{n}쪽"))
            st.session_state[state] = res
            st.session_state[state + "_sec"] = time.time() - t0
            st.rerun()
        return None, None

    res = st.session_state[state]
    n_err = sum(1 for r in res if r.get("error"))
    st.caption(f"{len(res)}쪽 · 캐시 {sum(r.get('cached') for r in res)} · "
               f"건너뜀 {sum(r['skip'] for r in res)} · 오류 {n_err} · "
               f"{st.session_state.get(state + '_sec', 0):.0f}초")
    if handwriting:
        st.caption("✍️ 손글씨: 확신 없는 낱말엔 (?)가 붙어요 — 그 부분만 확인해서 고치세요.")
    if n_err:
        st.warning("오류 난 쪽은 체크 해제돼 있어요. '다시 스캔'하면 그 쪽만 다시 읽어요.")
    multi = len(files) > 1
    df = pd.DataFrame([{
        "넣기": (not r["skip"]) and not r.get("error"),
        "쪽": r["page"] + 1,
        "파일": r["name"] if multi else "",
        "코드": ", ".join(r["codes"]),
        "영역": r["area"],
        "본문": r["text"] or r.get("note", "") or r.get("error", ""),
    } for r in res])
    if not multi:
        df = df.drop(columns=["파일"])
    edited = st.data_editor(
        df, hide_index=True, use_container_width=True, disabled=["쪽", "파일"],
        column_config={"본문": st.column_config.TextColumn(width="large"),
                       "넣기": st.column_config.CheckboxColumn(width="small"),
                       "쪽": st.column_config.NumberColumn(width="small")},
        key=f"{prefix}_editor_{sig[:8]}")
    if st.button("🔄 다시 스캔", key=f"{prefix}_rescan"):
        st.session_state.pop(state, None)
        st.rerun()
    if needs_key:
        with st.expander("🖼️ 원본 페이지와 비교"):
            pg = st.number_input("쪽", 1, max(1, len(res)), 1, key=f"{prefix}_pv")
            img = render_source(pages[pg - 1])
            if img:
                st.image(img, use_container_width=True)
            st.markdown(res[pg - 1]["text"] or "_(내용 없음)_")
    return edited, res


def merge_new(existing, new_records):
    """같은 rec_id(같은 출처+본문)는 다시 넣지 않음 → 같은 파일 두 번 저장해도 중복 없음."""
    have = {r.rec_id for r in existing}
    added = [r for r in new_records if r.rec_id not in have]
    return existing + added, len(added)


# ── core 파일 버전 확인 (바뀐 파일만 올리다 빠뜨린 경우 방지) ──
import version as _ver
_stale = _ver.check()
if _stale:
    st.error(f"core 파일이 app.py(v{_ver.VERSION})와 안 맞아요. 아래 파일을 올려주세요:\n\n"
             + "\n".join("- " + s for s in _stale))
    st.stop()

# ── 접속 비밀번호 (배포 시 URL만 알면 누구나 자료를 바꿀 수 있으므로) ──
_APP_PW = cloud.cfg("APP_PASSWORD")
if _APP_PW and not st.session_state.get("_authed"):
    st.title("🔒 임용 4레이어")
    _pw_in = st.text_input("비밀번호", type="password")
    if _pw_in:
        if hmac.compare_digest(_pw_in, str(_APP_PW)):
            st.session_state["_authed"] = True
            st.rerun()
        st.error("비밀번호가 다릅니다")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_engine(subject, _emb_mtime, _som_mtime):
    """학습 산출물 로드(파일 수정시각을 키로 캐시)."""
    emb = FrozenEmbedding.load(paths.emb_path(subject)) if paths.exists(paths.emb_path(subject)) else None
    som = SOM.load(paths.som_path(subject)) if paths.exists(paths.som_path(subject)) else None
    return emb, som

def _mtime(p):
    return os.path.getmtime(p) if paths.exists(p) else 0

# ── 사이드바: 과목 선택 (지금은 국어, 복제 시 확장) ──────────────
SUBJECT_LIST = ["국어", "영어", "수학", "사회", "과학", "미술", "음악",
                "체육", "실과", "도덕", "총론", "창의적체험활동", "통합교과"]
st.sidebar.title("⚙️ 설정")
subject = st.sidebar.selectbox("과목", SUBJECT_LIST, index=0)
st.sidebar.caption("한 과목으로 검증 후, 같은 앱에서 과목만 바꿔 확장")

# ── 서버(Supabase) 동기화: 과목별로 세션당 1회, 더 최근 쪽 기준 ──────
if cloud.enabled() and st.session_state.get("_synced_subject") != subject:
    with st.spinner("서버에서 자료·분석결과 불러오는 중…"):
        _rep = cloud.sync(paths.all_paths(subject))
    st.session_state["_synced_subject"] = subject
    st.session_state["_sync_report"] = _rep

st.sidebar.caption(f"버전 v{_ver.VERSION}")
st.sidebar.markdown("---")
st.sidebar.write("**🩺 자기검증**")
import selfcheck
_hist = selfcheck.load_history(subject)
_last = _hist[-1] if _hist else None
if _last:
    st.sidebar.caption(f"마지막 {_last['when']} · 경고 {len(_last['alerts'])}건")
_sc1, _sc2 = st.sidebar.columns(2)
_do_fix = st.sidebar.checkbox("기준 미달이면 재학습까지", key="sc_fix")
if _sc1.button("지금 점검", use_container_width=True, key="sc_run"):
    with st.spinner("자기검증 중…"):
        _last = selfcheck.run(subject, fix=_do_fix)
    if _do_fix and _last.get("retrained"):
        load_engine.clear()
    st.rerun()
if _last:
    with st.sidebar.expander(f"결과 보기 ({len(_last['alerts'])}건)",
                             expanded=bool(_last["alerts"])):
        for _a in _last["alerts"]:
            st.warning(_a)
        if not _last["alerts"]:
            st.success("이상 없음")
        _m, _h, _g = _last["model"], _last["hygiene"], _last["regression"]
        if _m.get("trained"):
            st.caption(f"자료 {_m['records']}건 · 벡터화 {_m['coverage']:.0%} · "
                       f"양자화오차 {_m['qe']:.3f}"
                       + (f" ({_m['qe_delta']:+.0%})" if _m.get("qe_prev") else "")
                       + f" · 빈 노드 {_m['dead_nodes']:.0%}")
        else:
            st.caption(_m.get("note", "모델 없음"))
        st.caption(f"근거 실재율 {1 - _g['ungrounded_rate']:.0%} "
                   f"(코드 {_g['checked_codes']} · 노드 {_g['checked_nodes']})")
        if _last.get("retrained"):
            st.caption(f"재학습: {_last['retrained']}")
        if _h["uncertain_scan"]:
            st.write("**재스캔 후보 (판독 불안)**")
            for _x in _h["uncertain_scan"][:10]:
                st.caption(f"· {_x['source']} — (?) {_x['marks']}개")
        if _h["duplicate"]:
            st.write("**중복 쪽**")
            for _x in _h["duplicate"][:10]:
                st.caption(f"· {_x['source']} ≡ {_x['same_as']}")
        if _h["weak_source"]:
            st.caption(f"출처 부실 {len(_h['weak_source'])}건")

st.sidebar.markdown("---")
st.sidebar.write("**☁️ 서버 백업**")
if cloud.enabled():
    _r = st.session_state.get("_sync_report", {})
    st.sidebar.caption(f"Supabase 연결됨 · 내려받음 {len(_r.get('down', []))} · "
                       f"올림 {len(_r.get('up', []))}  (저장할 때마다 자동 백업)")
    if st.sidebar.button("⬆️ 지금 전체 버전 백업", use_container_width=True, key="cloud_snap"):
        _n = sum(cloud.push(p, force_history=True) for p in paths.all_paths(subject)
                 if paths.exists(p))
        st.sidebar.success(f"{_n}개 파일 버전 백업 완료")
    with st.sidebar.expander("🕘 서버 버전으로 되돌리기"):
        _choices = {os.path.basename(p): p for p in paths.all_paths(subject)}
        _pick = st.selectbox("파일", list(_choices), key="hist_file")
        _vers = cloud.list_history(_pick)
        if _vers:
            _v = st.selectbox("버전(UTC)", _vers, key="hist_ver")
            if st.button("이 버전으로 복원", key="hist_restore"):
                if cloud.restore_history(_choices[_pick], _v):
                    st.success(f"{_pick} ← {_v}")
                    load_engine.clear()
                    st.rerun()
                else:
                    st.error("복원 실패 (아래 오류 확인)")
        else:
            st.caption("아직 버전 백업이 없어요")
    if cloud.ERRORS:
        with st.sidebar.expander(f"⚠️ 서버 오류 {len(cloud.ERRORS)}건"):
            st.caption("로컬 저장은 됐고, 서버 반영만 실패한 것. 다음 저장·재접속 때 다시 올라감.")
            for _e in cloud.ERRORS[-10:]:
                st.code(_e)
else:
    st.sidebar.caption("⚪ 미연결 — 로컬 pkl만 사용 중 (secrets에 SUPABASE_URL/KEY 넣으면 켜짐)")

# 학습 상태 표시
_has_som = paths.exists(paths.som_path(subject))
_has_emb = paths.exists(paths.emb_path(subject))
st.sidebar.markdown("---")
st.sidebar.write("**학습 상태**")
st.sidebar.write(f"임베딩: {'✅' if _has_emb else '❌ 미학습'}")
st.sidebar.write(f"SOM: {'✅' if _has_som else '❌ 미학습'}")

# ── pkl 다운로드/백업 ──────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.write("**💾 pkl 다운로드**")
_pkl_targets = [
    ("자료(L2)", paths.l2_path(subject)),
    ("기출(L1)", paths.l1_path(subject)),
    ("트렌드(L3)", paths.l3_path(subject)),
    ("임베딩", paths.emb_path(subject)),
    ("SOM", paths.som_path(subject)),
    ("학습상태(study)", paths.study_path(subject)),
]
for label, p in _pkl_targets:
    if paths.exists(p):
        with open(p, "rb") as _f:
            st.sidebar.download_button(
                f"⬇️ {label}", data=_f.read(),
                file_name=os.path.basename(p), mime="application/octet-stream",
                key=f"dl_{label}", use_container_width=True)

# 전체를 zip 하나로 백업
import io, zipfile
_existing = [p for _, p in _pkl_targets if paths.exists(p)]
if _existing:
    _buf = io.BytesIO()
    with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
        for p in _existing:
            _z.write(p, arcname=os.path.basename(p))
    _buf.seek(0)
    st.sidebar.download_button(
        f"📦 {subject} 전체 백업(zip)", data=_buf.getvalue(),
        file_name=f"{subject}_backup.zip", mime="application/zip",
        key="dl_all", use_container_width=True)

# ── pkl 복원(업로드) ───────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("📥 pkl 복원(업로드)"):
    st.caption("다운받은 pkl 또는 백업 zip을 올려 data/에 복원 "
               "(모든 파일 선택 가능 — pkl이 안 보이면 파일앱/전체보기에서 선택)")
    # type 제한 없음 → 사진 라이브러리·파일앱 등 모든 파일 선택 가능
    _ups = st.file_uploader("pkl 또는 zip (여러 개 가능)", type=None,
                            accept_multiple_files=True, key="restore")
    if _ups and st.button("복원 실행", key="restore_btn"):
        import zipfile, io
        done, skipped = [], []
        for _up in _ups:
            name = _up.name
            try:
                if name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(_up.read())) as z:
                        for inner in z.namelist():
                            if inner.endswith(".pkl"):
                                with open(os.path.join(paths.BASE, os.path.basename(inner)), "wb") as f:
                                    f.write(z.read(inner))
                                done.append(os.path.basename(inner))
                                cloud.push(os.path.join(paths.BASE, os.path.basename(inner)),
                                           force_history=True)
                elif name.lower().endswith(".pkl"):
                    with open(os.path.join(paths.BASE, os.path.basename(name)), "wb") as f:
                        f.write(_up.read())
                    done.append(name)
                    cloud.push(os.path.join(paths.BASE, os.path.basename(name)),
                               force_history=True)
                else:
                    skipped.append(name)
            except Exception as e:
                st.error(f"{name} 복원 실패: {e}")
        if done:
            st.success("복원 완료: " + ", ".join(done))
        if skipped:
            st.warning("pkl/zip 아니라 건너뜀: " + ", ".join(skipped))
        load_engine.clear()
        st.rerun()


emb, som = load_engine(subject, _mtime(paths.emb_path(subject)), _mtime(paths.som_path(subject)))

st.title(f"📖 임용 4레이어 — {subject}")

tabday, tabin, tab2, tab1, tablab, tab3, tab4, tab5, tabp, tabc = st.tabs(
    ["📖 오늘의 자료집", "📥 한 번에 넣기", "📚 자료·학습 (L2)", "📈 기출 패턴 (L1)",
     "🧪 경향 랩",
     "🔎 트렌드 (L3)",
     "📝 문제 풀기 (L4)", "🎯 수능형 연습 (L5)", "📜 지문 학습", "🕸️ 개념 지도"])

# ══════════════════════════════════════════════════════════════
# 탭 — 오늘의 자료집
# ══════════════════════════════════════════════════════════════
with tabday:
    import daily_digest as dd
    _today = dd.today_str()
    _store = dd.load_store(subject)
    _dg = _store["days"].get(_today)
    st.subheader(f"오늘의 자료집 · {subject}")
    st.caption("무엇을 읽을지는 약점·복습 주기·출제 예상으로 정해요. "
               "원문을 출처와 함께 붙이고, 요약도 그 원문만 근거로 만들어요. "
               "새벽 워커가 미리 만들어 두면 여기서 바로 보여요.")

    dc1, dc2, dc3 = st.columns([2, 2, 1])
    _n_items = dc1.number_input("항목 수", 3, 12, 5, key="dg_n")
    _okey_dg = api_key("OpenAI Key(요약용, 선택)", "OPENAI_API_KEY", "dg_key")
    if dc3.button("🔁 다시 편성" if _dg else "📖 오늘 것 만들기", type="primary", key="dg_make"):
        try:
            bar = st.progress(0.0, text="자료집 만드는 중…")
            _dg, _store = dd.build(subject, int(_n_items), _okey_dg or None,
                                   cloud.cfg("OPENAI_DIGEST_MODEL") or "gpt-4o-mini",
                                   force=True,
                                   progress=lambda i, n: bar.progress(i / max(n, 1),
                                                                      text=f"요약 {i}/{n}"))
            st.rerun()
        except Exception as e:
            st.error(f"편성 실패: {e}")

    if not _dg:
        st.info("아직 오늘 자료집이 없어요. 위 버튼을 누르거나, 새벽 워커가 만들어 두면 자동으로 떠요.")
        _n_l2 = len(load_records_pkl(paths.l2_path(subject)))
        _n_l1 = len(load_records_pkl(paths.l1_path(subject)))
        st.caption(f"현재 {subject} 보유: 자료 {_n_l2}건 · 기출 {_n_l1}건 "
                   f"(기출만 있어도 편성돼요)")
    else:
        _done = sum(i["read"] for i in _dg["items"])
        st.progress(_done / max(len(_dg["items"]), 1),
                    text=f"{_done}/{len(_dg['items'])} 읽음 · 총 {_dg['chars']:,}자")
        if len(_dg["items"]) < _n_items:
            st.caption(f"묶을 수 있는 자료가 {_dg['pool']}개라 {len(_dg['items'])}항목만 나왔어요. "
                       "자료를 더 넣으면 늘어나요.")
        for _i, it in enumerate(_dg["items"]):
            with st.expander(("✅ " if it["read"] else "") + f"{it['title']}"
                             + (f"  ·  {it['why'][0]}" if it["why"] else ""),
                             expanded=not it["read"] and _i == _done):
                st.caption(" · ".join(it["why"]))
                if it["summary"]:
                    for line in it["summary"]:
                        st.write("• " + line)
                elif it.get("summary_error"):
                    st.caption(f"요약 실패: {it['summary_error']}")
                for r in it["reads"]:
                    st.markdown(f"**{r['source']}**"
                                + (f"  ·  {r['doc_type']}" if r.get("doc_type") else ""))
                    st.write(r["text"])
                if it["questions"]:
                    st.write("**확인 질문**")
                    for q in it["questions"]:
                        st.write("- " + q)
                b1, b2 = st.columns(2)
                if not it["read"]:
                    if b1.button("✅ 읽음", key=f"dg_ok_{_i}"):
                        dd.mark_read(subject, _today, it["key"], True)
                        st.rerun()
                else:
                    if b1.button("↩️ 읽음 취소", key=f"dg_no_{_i}"):
                        dd.mark_read(subject, _today, it["key"], False)
                        st.rerun()
                b2.caption("읽으면 다음 복습일이 1→3→7→14→30일로 밀려요")

    # ── AI의 질문 ─────────────────────────────────────────────
    import ask_box as ab
    st.markdown("---")
    _asum = ab.summary(subject)
    st.subheader(f"❓ AI의 질문 {_asum['open']}개")
    st.caption("제가 모르는 걸 물어요. 지어낸 궁금증이 아니라 실제로 막힌 지점이에요 — "
               "근거를 못 찾은 개념, 판독이 흐린 글자, 분류가 안 되는 자료, "
               "기출 통계에서 보이는 변화요. 답해 주시면 자료로 들어가서 "
               "다음부터 근거로 쓰이고, 그 답이 실제로 도움이 됐는지도 확인해요.")
    if _asum["answered"]:
        st.caption(f"지금까지 답변 {_asum['answered']}개 · 그중 구멍을 메운 것 "
                   f"{_asum['helped']}개")

    _okey_ab = api_key("OpenAI Key(선택)", "OPENAI_API_KEY", "ab_key")
    if st.button("🔄 질문 다시 뽑기", key="ab_gen"):
        n = ab.generate(subject, api_key=_okey_ab or None,
                        model=cloud.cfg("OPENAI_TAG_MODEL") or "gpt-4o-mini",
                        log=lambda *a: None)
        st.success(f"새 질문 {n}개") if n else st.info("새로 물을 게 없어요")
        st.rerun()

    _qs = ab.pending(subject)
    if not _qs:
        st.caption("지금은 물어볼 게 없어요. 워커가 돌면 다시 쌓여요.")
    else:
        from ask_box import KINDS
        for _q in _qs[:5]:
            with st.expander(f"[{KINDS.get(_q['kind'], _q['kind'])}] {_q['question'][:60]}…",
                             expanded=(_q is _qs[0])):
                st.write(_q["question"])
                if _q.get("context"):
                    st.caption("관련 내용")
                    st.code(_q["context"][:400])
                st.caption(f"묻는 이유: {_q.get('why', '')}")
                _ans = st.text_area("답변", key=f"ab_txt_{_q['id']}", height=120,
                                    placeholder="아는 만큼만 적어도 돼요. 자료로 저장돼요.")
                b1, b2 = st.columns(2)
                if b1.button("✅ 답변 저장", key=f"ab_save_{_q['id']}", type="primary"):
                    if ab.answer(subject, _q["id"], _ans):
                        st.success("자료로 저장했어요 — 다음부터 근거로 씁니다")
                        st.rerun()
                    else:
                        st.warning("답변을 적어주세요")
                if b2.button("건너뛰기", key=f"ab_skip_{_q['id']}"):
                    ab.skip(subject, _q["id"])
                    st.rerun()

    _hist = [d for d in dd.recent(subject, 7) if d["date"] != _today]
    if _hist:
        with st.expander(f"🗓️ 지난 자료집 {len(_hist)}일"):
            for d in reversed(_hist):
                st.caption(f"{d['date']} · " + ", ".join(i["title"] for i in d["items"])
                           + f" ({sum(i['read'] for i in d['items'])}/{len(d['items'])} 읽음)")


# ══════════════════════════════════════════════════════════════
# 탭 — 한 번에 넣기 (자동 분류)
# ══════════════════════════════════════════════════════════════
with tabin:
    st.subheader("한 번에 넣기 — 올리면 알아서 분류")
    st.caption("자료·기출·필기 사진을 섞어서 한꺼번에 올려요. 읽기 → 종류·과목·이름 자동 태깅 → "
               "파일당 한 줄로 확인 → 저장. 기출은 쪽마다 과목을 판정해 과목별 기출로 나눠 들어가요.")
    if st.session_state.get("in_summary"):
        st.success("저장 완료 · " + " · ".join(st.session_state.pop("in_summary")))

    # ── 전체 현황 ──────────────────────────────────────────────
    import inventory as inv
    with st.expander("📊 전체 현황 — 자료가 어디에 얼마나 있나", expanded=False):
        # 앱은 고른 과목만 동기화하므로, 전체를 세기 전에 모든 과목을 받아온다
        if (cloud.enabled() and hasattr(inv, "sync_all")
                and not st.session_state.get("_inv_synced")):
            with st.spinner("모든 과목 자료 받아오는 중…"):
                _r = inv.sync_all()
            st.session_state["_inv_synced"] = True
            if _r["받음"]:
                st.caption(f"서버에서 {_r['받음']}개 파일 내려받음 "
                           f"({', '.join(_r['과목'][:6])})")
        if st.button("🔄 서버와 다시 맞추기", key="inv_resync"):
            st.session_state.pop("_inv_synced", None)
            st.rerun()
        _s = inv.summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("과목", _s["과목수"])
        c2.metric("자료", _s["총자료"])
        c3.metric("기출", _s["총기출"])
        c4.metric("문서", _s["문서수"])
        st.caption(f"총 {_s['총글자']:,}자")

        st.write("**과목별**")
        st.dataframe(inv.subjects(), use_container_width=True, hide_index=True)
        st.caption("임베딩·SOM이 비어 있으면 그 과목은 자료집·자가시험이 안 돌아요 "
                   "(워커가 재학습하면 채워집니다).")

        st.write("**출처(문서)별** — 한 문서가 여러 과목에 흩어졌는지 보여요")
        st.dataframe(inv.sources(), use_container_width=True, hide_index=True)

        _q = inv.queue()
        st.write(f"**스캔 대기열** — 대기 {_q['대기']} · 완료 {_q['완료']} · 실패 {_q['실패']}")
        if _q.get("진행중"):
            st.caption("이어서 처리 중 (큰 파일은 회차를 나눠 읽어요)")
            st.dataframe(_q["진행중"], use_container_width=True, hide_index=True)
        if _q["실패목록"]:
            st.error("들어가지 못한 파일")
            st.dataframe(_q["실패목록"], use_container_width=True, hide_index=True)
        if _q["최근"]:
            st.caption("최근 처리")
            st.dataframe(_q["최근"], use_container_width=True, hide_index=True)

        if cloud.enabled():
            _f = inv.files()
            if _f:
                _miss = [r for r in _f if not r["들어간 쪽"]]
                st.write(f"**원본 파일 대조** — 보관 {len(_f)}개 중 "
                         f"아직 안 들어간 것 {len(_miss)}개")
                st.dataframe(_f, use_container_width=True, hide_index=True)
                if _miss:
                    st.caption("⚠️ 표시된 파일만 아직 안 들어왔어요. "
                               "'저장된 출처'가 비어 있으면 대기열에 건 적이 없는 거예요.")

    # ── 과목 재분류 (사후 자기검증) ────────────────────────────
    import resubject as rsj
    with st.expander("🧭 과목 재분류 · 태그 채우기 — 잘못 들어간 자료 정리"):
        st.caption("판정 단위는 **출처(문서)**예요. 한 지도서에서 코드가 붙은 쪽이 3쪽뿐이어도, "
                   "문서 전체의 근거를 합쳐 판정하고 나머지 쪽도 같이 옮겨요. "
                   "쪽 단위로만 보면 근거 없는 쪽은 계속 미분류로 남아요.")
        rs1, rs2, rs3 = st.columns([2, 1, 1])
        _conf = rs1.slider("최소 확신도", 0.4, 0.95, 0.6, 0.05, key="rs_conf")
        _by_src = rs2.checkbox("출처 단위", value=True, key="rs_bysrc")
        _scope_all = rs3.checkbox("전 과목", value=True, key="rs_all")
        if st.button("🔍 다시 판정", key="rs_run"):
            try:
                st.session_state["rs_rows"] = rsj.audit(
                    None if _scope_all else [subject], min_conf=_conf, by_source=_by_src)
            except TypeError:
                st.error("core/resubject.py가 이전 버전이에요. app.py와 core/ 파일을 "
                         "같은 버전으로 함께 올려주세요.")
        _rows = st.session_state.get("rs_rows")
        if _rows is not None:
            if not _rows:
                st.success("과목이 다르게 판정된 자료가 없어요.")
            else:
                st.warning(f"{len(_rows)}개 {'문서' if _by_src else '쪽'}이 다른 과목으로 판정됐어요.")
                st.dataframe([{"이동": f"{m['from']} → {m['to']}", "문서": m["docs"],
                               "쪽": m["pages"]} for m in rsj.summary(_rows)],
                             use_container_width=True, hide_index=True)
                import pandas as pd
                _df = pd.DataFrame([{
                    "#": i, "옮기기": True, "지금": r["current"], "제안": r["proposed"],
                    "확신": int(r["conf"] * 100), "쪽": r["pages"], "출처": r["source"],
                    "근거": ", ".join(r["evidence"]), "본문": r["text"]}
                    for i, r in enumerate(_rows)])
                _ed = st.data_editor(
                    _df, hide_index=True, use_container_width=True,
                    disabled=["#", "지금", "확신", "쪽", "출처", "근거", "본문"],
                    column_config={"#": None,
                                   "제안": st.column_config.SelectboxColumn(
                                       options=SUBJECT_LIST, required=True),
                                   "확신": st.column_config.ProgressColumn(
                                       min_value=0, max_value=100, format="%d%%"),
                                   "본문": st.column_config.TextColumn(width="large")},
                    key="rs_editor")
                if st.button("✅ 체크한 것 옮기기", type="primary", key="rs_apply"):
                    picks = []
                    for _, row in _ed.iterrows():
                        if row["옮기기"]:
                            r = dict(_rows[int(row["#"])])
                            r["proposed"] = row["제안"]
                            picks.append(r)
                    n = rsj.apply_moves(picks)
                    st.session_state.pop("rs_rows", None)
                    load_engine.clear()
                    st.success(f"{n}쪽 이동 완료 — 옮긴 과목에서 다시 학습(임베딩·SOM)을 권해요")
                    st.rerun()

        st.markdown("---")
        st.caption("**🤖 애매한 문서 LLM 판정** — 마인드맵·개념도·필기처럼 낱말만 남은 자료는 "
                   "규칙(코드·고유어휘)으로 못 갈라요. 문서 단위로 LLM에 한 번씩만 물어봐요.")
        _okey_rs = api_key("OpenAI Key", "OPENAI_API_KEY", "rs_key")
        _undet = rsj.undetermined_docs(None if _scope_all else [subject])
        st.caption(f"규칙으로 못 가른 문서 {len(_undet)}개 "
                   f"({sum(d['pages'] for d in _undet)}쪽)")
        if _undet and _okey_rs and st.button("🤖 LLM으로 판정", key="rs_llm"):
            bar = st.progress(0.0, text="판정 중…")
            st.session_state["rs_llm_rows"] = rsj.judge_docs_llm(
                _undet, _okey_rs, cloud.cfg("OPENAI_TAG_MODEL") or "gpt-4o-mini",
                progress=lambda d, n: bar.progress(d / max(n, 1), text=f"{d}/{n} 문서"))
            st.rerun()
        _lrows = st.session_state.get("rs_llm_rows")
        if _lrows:
            import pandas as pd
            _ldf = pd.DataFrame([{
                "#": i, "적용": r["conf"] >= 0.6, "문서": r["source"], "쪽": r["pages"],
                "지금": r["current"], "과목": r["proposed"] or r["current"],
                "영역": r["area"], "종류": r["doc_type"] or "",
                "확신": int(r["conf"] * 100), "근거": r["evidence"][0]}
                for i, r in enumerate(_lrows)])
            _led = st.data_editor(
                _ldf, hide_index=True, use_container_width=True,
                disabled=["#", "문서", "쪽", "지금", "확신", "근거"],
                column_config={"#": None,
                               "과목": st.column_config.SelectboxColumn(
                                   options=SUBJECT_LIST, required=True),
                               "종류": st.column_config.SelectboxColumn(
                                   options=[""] + sorted(DOC_TYPES)),
                               "확신": st.column_config.ProgressColumn(
                                   min_value=0, max_value=100, format="%d%%"),
                               "근거": st.column_config.TextColumn(width="large")},
                key="rs_llm_editor")
            if st.button("✅ 체크한 문서 적용", type="primary", key="rs_llm_apply"):
                picks = []
                for _, row in _led.iterrows():
                    if row["적용"]:
                        r = dict(_lrows[int(row["#"])])
                        r["proposed"] = row["과목"]
                        r["area"] = str(row["영역"] or "").strip()
                        r["doc_type"] = row["종류"] or None
                        picks.append(r)
                mv, tg = rsj.apply_doc_decisions(picks)
                st.session_state.pop("rs_llm_rows", None)
                load_engine.clear()
                st.success(f"{mv}쪽 이동 · {tg}쪽 영역 채움")
                st.rerun()

        st.markdown("---")
        _gaps = rsj.tag_gaps(subject)
        st.caption("**태그 채우기** — 같은 출처의 다른 쪽에는 있는데 이 쪽에는 없는 "
                   "영역·자료종류·학년군·단원을 다수결로 채워요. "
                   "없는 값을 만들지 않고, 이미 있는 값은 안 건드려요. "
                   "(성취기준 코드는 쪽마다 달라서 제외)")
        if _gaps:
            st.write({k: f"{v}쪽" for k, v in _gaps.items()})
            if st.button(f"🏷️ {subject} 태그 채우기", key="rs_fill"):
                done = rsj.propagate_tags(subject)
                load_engine.clear()
                st.success("채움: " + (", ".join(f"{k} {v}쪽" for k, v in done.items())
                                      or "없음"))
                st.rerun()
        else:
            st.caption(f"{subject}에는 채울 게 없어요.")

    # ── 자료 정비 (전 과목) ────────────────────────────────────
    import maintenance as mt
    with st.expander("🧹 자료 정비 — 긴 쪽 분할 · 중복 · 잘못된 태그 · 영역 라벨 · 재학습"):
        st.caption("한 기록이 수천 자면 임베딩 한 벡터에 개념이 수십 개 뭉개져서 지도가 흐려져요. "
                   "문단 단위로 나누고, 중복을 빼고, 일괄로 잘못 붙은 태그를 정리한 뒤 "
                   "비어 있는 영역을 라벨링하고 다시 학습해요. 먼저 미리보기로 확인하세요.")
        mc1, mc2 = st.columns([2, 1])
        _subjects = mt.subject_list(None)
        _pick = mc1.multiselect("대상 과목", _subjects, default=_subjects, key="mt_subj")
        _steps = mc2.multiselect("단계", list(mt.STEPS), default=list(mt.APP_STEPS),
                                 key="mt_steps")
        st.caption("⏱️ 재학습(retrain)은 자료가 많으면 몇 분씩 걸려요. "
                   "여기서는 빼두고 새벽 워커에 맡기는 걸 권해요 "
                   "(워커는 기준 미달일 때 알아서 다시 학습해요).")
        _okey_mt = api_key("OpenAI Key(영역 라벨용)", "OPENAI_API_KEY", "mt_key")
        _limit = st.number_input("라벨 한도(과목당 쪽)", 50, 2000, 300, 50, key="mt_lim")
        m1, m2 = st.columns(2)
        st.caption("대상 과목은 서버 기준이에요. 실행하면 그 과목 파일을 먼저 받아온 뒤 정비해요. "
                   "파일명 규칙 재배치(seed)는 과목과 무관하게 전체를 한 번 훑어요.")

        def _pull(subj):
            if cloud.enabled():
                cloud.sync(paths.all_paths(subj))

        if m1.button("🔍 미리보기", key="mt_dry"):
            for _s in _pick:
                _pull(_s)
            st.session_state["mt_rep"] = {
                s: mt.run(s, tuple(_steps), True, _okey_mt or None,
                          cloud.cfg("OPENAI_TAG_MODEL") or "gpt-4o-mini", int(_limit))
                for s in _pick}
        if m2.button("⚙️ 실제로 정비", type="primary", key="mt_go"):
            bar = st.progress(0.0, text="정비 중…")
            out = {}
            for i, s in enumerate(_pick):      # 정비 전에 전부 받아둔다
                bar.progress(i / max(len(_pick), 1) * 0.3, text=f"{s} 받아오는 중…")
                _pull(s)
            for i, s in enumerate(_pick):
                bar.progress(0.3 + i / max(len(_pick), 1) * 0.7, text=f"{s} 정비 중…")
                out[s] = mt.run(s, tuple(_steps), False, _okey_mt or None,
                                cloud.cfg("OPENAI_TAG_MODEL") or "gpt-4o-mini",
                                int(_limit))
            st.session_state["mt_rep"] = out
            load_engine.clear()
            st.success("정비 완료 — 자료집을 다시 편성하면 개념 단위로 올라와요")
            st.rerun()
        _rep = st.session_state.get("mt_rep")
        if _rep:
            rows = []
            for s, r in _rep.items():
                sp = r.get("split", {})
                rows.append({"과목": s, "미리보기" if r["dry_run"] else "적용": "○",
                             "태그정리": len(r.get("fix_tags", [])),
                             "깨진글자": r.get("garbage", {}).get("n", 0),
                             "영역정리": r.get("clean_areas", 0),
                             "규칙 재배치": (r.get("seed", {}) or {}).get("pages", 0),
                             "기출 문항분리": (r.get("exam_split", {}) or {}).get("문항", 0),
                             "중복": r.get("dedupe", 0),
                             "분할 대상": sp.get("split", 0),
                             "조각": f"{sp.get('before', 0)}→{sp.get('after', 0)}",
                             "영역 라벨": r.get("label", 0),
                             "재학습": ("함" if r.get("retrain") else "")})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            for s, r in _rep.items():
                for e in r.get("split", {}).get("examples", [])[:3]:
                    st.caption(f"{s}: {e['source'][:40]} — {e['len']:,}자 → {e['parts']}조각")
            if any(r["dry_run"] for r in _rep.values()):
                st.info("미리보기예요. '실제로 정비'를 눌러야 바뀌어요. "
                        "바꾸기 전 로컬 백업(data/backup/)과 서버 버전 백업이 남아요.")

    # ── 구글 드라이브에서 가져오기 ─────────────────────────────
    import drive as gdrive
    _folders = [x for x in re.split(r"[\n,]+", str(cloud.cfg("DRIVE_FOLDERS") or
                                                  cloud.cfg("DRIVE_FOLDER_URL") or ""))
                if x.strip()]
    _gkey, _gsa = cloud.cfg("GOOGLE_API_KEY"), cloud.cfg("GOOGLE_SERVICE_ACCOUNT")
    with st.expander(f"📁 구글 드라이브에서 가져오기 ({len(_folders)}개 폴더)",
                     expanded=bool(_folders) and not (st.session_state.get("in_files") or [])):
        if not _folders:
            st.caption("secrets에 DRIVE_FOLDERS = \"폴더 링크\" 를 넣으면 여기서 바로 가져올 수 있어요 "
                       "(여러 개면 줄바꿈으로 구분).")
        else:
            if not (_gkey or _gsa):
                st.caption("키 없이 공개 폴더 모드로 읽어요 — 폴더 공유가 "
                           "'링크가 있는 모든 사용자 - 뷰어'여야 해요. "
                           "(구글이 임베드 경로를 바꾸면 안 될 수 있어요. "
                           "무료 GOOGLE_API_KEY를 넣으면 안정적이에요.)")
            _force = st.checkbox("이미 가져온 파일도 다시 가져오기 (재스캔용)",
                                 key="dr_force")
            if st.button("🔄 폴더 훑어보기", key="dr_scan"):
                try:
                    _all = []
                    for u in _folders:
                        fid = gdrive.folder_id(u)
                        if not fid:
                            st.warning(f"폴더 링크를 못 알아봤어요: {u[:60]}")
                            continue
                        _all += gdrive.list_folder(fid, _gkey, _gsa)
                    st.session_state["dr_list"] = gdrive.dedupe(_all)
                except Exception as e:
                    st.error(f"드라이브 조회 실패: {e}")
            _lst = st.session_state.get("dr_list")
            if _lst is not None:
                dstate = gdrive.DriveState(paths.drive_state_path())
                _new = _lst if _force else [f for f in _lst if dstate.is_new(f)]
                st.caption(f"폴더 안 파일 {len(_lst)}개 · "
                           + ("전부 다시 가져오기 모드" if _force
                              else f"새 파일 {len(_new)}개"))
                if _force:
                    st.caption("깨진 PDF(CID 글꼴)는 OpenAI 키를 넣고 다시 읽으면 살아나요. "
                               "다시 넣기 전에 '자료 정비 → garbage'로 옛 기록을 지우세요.")
                _sel = st.multiselect("가져올 파일", [f["path"] for f in _new],
                                      default=[f["path"] for f in _new], key="dr_sel")
                if _sel and st.button(f"⬇️ {len(_sel)}개 가져오기", type="primary", key="dr_get"):
                    bar, got = st.progress(0.0, text="내려받는 중…"), []
                    for i, f in enumerate([x for x in _new if x["path"] in _sel]):
                        try:
                            got.append(gdrive.download(f, _gkey, _gsa))
                            dstate.mark(f)
                        except Exception as e:
                            st.warning(str(e))
                        bar.progress((i + 1) / len(_sel), text=f"{i + 1}/{len(_sel)}")
                    dstate.save()
                    st.session_state["in_drive_files"] = got
                    st.rerun()
    # ── 워커에게 스캔 맡기기 ───────────────────────────────────
    import ingest_queue as iq
    with st.expander("🤖 워커에게 스캔 맡기기 (다시 읽기 예약)"):
        st.caption("원본을 고르고 걸어두면 새벽 워커가 내려받아 비전으로 다시 읽고 "
                   "분류해서 저장해요. 깨진 PDF(CID 글꼴)를 살릴 때 쓰세요. "
                   "폰에서 큰 파일을 붙들고 있을 필요가 없어요.")
        _q = iq.load()
        _pend = iq.pending(_q)
        if _pend:
            st.info(f"대기 중 {len(_pend)}건: " + ", ".join(j["source"] for j in _pend[:5]))
        _done = [j for j in _q["jobs"] if j["status"] in ("done", "error")][-5:]
        if _done:
            st.caption("최근 처리: " + " · ".join(
                f"{j['source']}→" + (f"{j['result'].get('added', 0)}쪽"
                                     if j["status"] == "done" else "실패")
                for j in _done))
        st.markdown("**전체 다시 읽기**")
        ra, rb, rc = st.columns(3)
        _all_sub = ra.selectbox("대상", ["이 과목만", "전 과목"], key="q_allsub")
        _all_hw = rb.checkbox("손글씨로", key="q_allhw")
        _all_wipe = rc.checkbox("기존 기록 비우기", key="q_allwipe")
        if cloud.enabled() and st.button("🔄 보관된 원본 전부 다시 읽기 예약", key="q_all"):
            n = iq.enqueue_all_uploads(None if _all_sub == "전 과목" else subject,
                                       replace=True, handwriting=_all_hw,
                                       wipe=_all_wipe, log=lambda *a: None)
            st.success(f"{n}건 예약 — 워커가 한 회차에 20건씩 처리해요 "
                       "(Actions에서 queue_limit을 올리면 한 번에 더 많이).")
            st.rerun()
        if _all_wipe:
            st.warning("'기존 기록 비우기'는 그 과목의 L1·L2를 통째로 지운 뒤 다시 채워요. "
                       "백업은 남지만, 앱에서 직접 입력한 기록도 함께 사라져요.")
        st.markdown("---")

        if not cloud.enabled():
            st.caption("Supabase 연결이 있어야 원본을 워커가 내려받을 수 있어요.")
        else:
            _ups2 = cloud.list_uploads()
            if not _ups2:
                st.caption("보관된 원본이 없어요.")
            else:
                # 지금 깨져 있는 출처를 힌트로 보여줌
                try:
                    import maintenance as _mt2
                    _bad = {}
                    for g in _mt2.drop_garbage(subject, dry_run=True):
                        b = g["source"].rsplit(" (", 1)[0].rsplit(" p.", 1)[0]
                        _bad[b] = _bad.get(b, 0) + 1
                    if _bad:
                        st.warning("깨진 자료: " + ", ".join(f"{k} ({v}쪽)"
                                                          for k, v in _bad.items()))
                except Exception:
                    _bad = {}
                _lbl2 = {f"{u['original_name']} · {u['subject']}": u for u in _ups2}
                qa, qb = st.columns(2)
                _file = qa.selectbox("원본 파일", list(_lbl2), key="q_file")
                _src = qb.text_input("저장될 출처 이름",
                                     value=(list(_bad)[0] if _bad else
                                            os.path.splitext(_file)[0] if _file else ""),
                                     key="q_src")
                qc, qd, qe = st.columns(3)
                _qsub = qc.selectbox("과목(비우면 자동)", ["자동"] + SUBJECT_LIST, key="q_sub")
                _qhw = qd.checkbox("손글씨", key="q_hw")
                _qrep = qe.checkbox("옛 기록 교체", value=True, key="q_rep")
                if st.button("📌 다시 읽기 예약", type="primary", key="q_add"):
                    u = _lbl2[_file]
                    iq.add("rescan", source=_src.strip() or u["original_name"],
                           subject=None if _qsub == "자동" else _qsub,
                           storage_path=u["storage_path"], filename=u["original_name"],
                           handwriting=_qhw, replace=_qrep)
                    st.success("예약 완료 — 새벽 워커가 처리해요. "
                               "지금 바로 처리하려면 GitHub Actions에서 수동 실행하세요.")
                    st.rerun()

    # ── 서버에 보관된 원본 다시 가져오기 ───────────────────────
    with st.expander("☁️ 서버에 보관된 원본 다시 가져오기"):
        if not cloud.enabled():
            st.caption("Supabase 연결이 있어야 보관된 원본을 볼 수 있어요.")
        else:
            _ups = cloud.list_uploads()
            if not _ups:
                st.caption("보관된 원본이 없어요. (앱으로 올린 파일은 자동 보관돼요)")
            else:
                st.caption(f"보관된 원본 {len(_ups)}개 — 깨져서 다시 읽어야 하는 파일을 고르세요. "
                           "가져오면 아래 목록에 합쳐져 스캔·분류로 이어져요.")
                _lbl = {f"{u['original_name']}  ·  {u['subject']}/{u.get('kind', '')}"
                        f"  ·  {u.get('size_bytes', 0) // 1024}KB": u for u in _ups}
                _sel_up = st.multiselect("원본 파일", list(_lbl), key="up_sel")
                if _sel_up and st.button(f"⬇️ {len(_sel_up)}개 다시 가져오기", key="up_get"):
                    bar, got = st.progress(0.0, text="내려받는 중…"), []
                    for i, k in enumerate(_sel_up):
                        u = _lbl[k]
                        raw = cloud.download_upload(u["storage_path"])
                        if raw:
                            got.append((u["original_name"], raw))
                        bar.progress((i + 1) / len(_sel_up))
                    st.session_state["in_drive_files"] = (
                        st.session_state.get("in_drive_files") or []) + got
                    st.rerun()

    _drive_files = st.session_state.get("in_drive_files") or []
    if _drive_files:
        c_d1, c_d2 = st.columns([3, 1])
        c_d1.info(f"가져온 파일 {len(_drive_files)}개가 아래 목록에 포함돼요 "
                  "(드라이브·서버 보관본).")
        if c_d2.button("비우기", key="dr_clear"):
            st.session_state.pop("in_drive_files", None)
            st.session_state.pop("dr_list", None)
            st.rerun()

    ups_in = st.file_uploader(
        "파일 여러 개 (pdf · 사진 · docx · txt)",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "webp", "heic"],
        accept_multiple_files=True, key="in_files")
    okey_in = api_key("OpenAI Key(읽기·분류용)", "OPENAI_API_KEY", "in_key")

    if ups_in and len(ups_in) >= 1:
        _tot = sum(len(u.getvalue()) for u in ups_in) / 1e6
        st.caption(f"올린 파일 {len(ups_in)}개 · {_tot:.1f}MB")
        if len(ups_in) >= 6 or _tot > 20:
            st.warning("파일이 많아요. 여기서 한 번에 스캔하면 폰 화면이 꺼지거나 "
                       "메모리 한도로 앱이 멈출 수 있어요. 아래 '워커에 맡기기'를 권해요.")
        if cloud.enabled() and st.button(
                f"📌 {len(ups_in)}개 전부 워커에 맡기기 (스캔 없이 예약)", key="in_queue"):
            import ingest_queue as iq2
            bar, n = st.progress(0.0, text="원본 보관 중…"), 0
            for i, u in enumerate(ups_in):
                raw = u.getvalue()
                sp = cloud.archive_upload(subject, "inbox", u.name, raw)
                if sp:
                    iq2.add("upload", source=os.path.splitext(u.name)[0], subject=None,
                            storage_path=sp, filename=u.name, replace=False)
                    n += 1
                bar.progress((i + 1) / len(ups_in), text=f"{i + 1}/{len(ups_in)}")
            st.success(f"{n}개 예약 완료 — 새벽 워커가 스캔·분류해서 넣어요. "
                       "급하면 GitHub Actions에서 수동 실행하세요.")

    if ups_in or _drive_files:
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor
        from page_scan import build_pages, scan_pages, ScanCache, file_sha, IMAGE_EXT
        from auto_tag import classify, CATEGORIES
        from concept_dict import ConceptDict
        _cnames = ConceptDict.load(paths.concept_dict_path(subject), subject).names()
        files_in = list(_drive_files) + [(u.name, u.getvalue()) for u in (ups_in or [])]
        sig_in = file_sha("|".join(file_sha(r) for _, r in files_in).encode())[:16]
        skey = f"in_{sig_in}"

        def _scan_one(name, raw, hw, cache, prog=None):
            res = scan_pages(build_pages([(name, raw)]), okey_in or None, _cnames,
                             cache=cache, handwriting=hw,
                             model=cloud.cfg("OPENAI_HANDWRITING_MODEL" if hw
                                             else "OPENAI_SCAN_MODEL"),
                             progress=prog)
            for r in res:
                r["page_in_file"] = r["page"]
            return res

        if skey not in st.session_state:
            st.caption(f"{len(files_in)}개 파일 · 사진은 손글씨 모드로 읽어요")
            if st.button("🔍 읽고 분류하기", type="primary", key="in_go"):
                bar = st.progress(0.0, text="읽는 중…")
                cache = ScanCache(paths.scan_cache_path())
                t0, out = time.time(), []
                for fi, (name, raw) in enumerate(files_in):
                    cloud.archive_upload(subject, "inbox", name, raw)
                    is_img = name.lower().endswith(IMAGE_EXT)
                    res = _scan_one(name, raw, is_img, cache, prog=lambda d, n, fi=fi, name=name:
                                    bar.progress((fi + d / max(n, 1)) / len(files_in),
                                                 text=f"읽는 중 {fi + 1}/{len(files_in)} · "
                                                      f"{name} {d}/{n}쪽"))
                    out.append({"name": name, "raw": raw, "res": res, "hw": is_img})
                _keep = sum(len(f["raw"]) for f in out) < 20_000_000
                if not _keep:                       # 메모리 한도(앱 멈춤) 예방
                    for f in out:
                        f["raw"] = b""
                bar.progress(1.0, text="분류 중…")
                tag_model = cloud.cfg("OPENAI_TAG_MODEL") or "gpt-4o-mini"
                def _tag(f):
                    errs = [r.get("error") for r in f["res"] if r.get("error")]
                    if errs and len(errs) == len(f["res"]):   # 통째로 못 읽은 파일
                        return {"category": "지도서_각론", "subject": None, "title": "",
                                "year": None, "level": None, "grade_band": "",
                                "area": "", "unit": "", "confidence": 0.0,
                                "reason": "⚠️ " + errs[0], "page_subjects": {},
                                "failed": True}
                    return classify(f["name"], f["res"], okey_in or None, tag_model,
                                    is_image=f["hw"])
                with ThreadPoolExecutor(max_workers=6) as ex:
                    tags = list(ex.map(_tag, out))
                for f, t in zip(out, tags):
                    f["tag"] = t
                st.session_state[skey] = out
                st.session_state[skey + "_sec"] = time.time() - t0
                st.rerun()
        else:
            out = st.session_state[skey]
            st.caption(f"{len(out)}개 파일 · {sum(len(f['res']) for f in out)}쪽 · "
                       f"{st.session_state.get(skey + '_sec', 0):.0f}초 · "
                       "확신 낮은 파일이 위에 있어요. 종류·과목·이름만 확인하세요.")
            rows = []
            for i, f in enumerate(out):
                t = f["tag"]
                stem = os.path.splitext(f["name"])[0]
                # 분류기가 과목을 못 집으면 규칙 판정으로 한 번 더 시도.
                # (예전엔 바로 '지금 과목'으로 떨어져서 전부 국어로 쌓였음)
                _auto_subj = t["subject"]
                if not _auto_subj:
                    _txt = " ".join((r.get("text") or "")[:400] for r in f["res"][:6])
                    _g, _gc, _ = rsj.judge(_txt)
                    _auto_subj = _g if (_g and _gc >= 0.6) else None
                rows.append({
                    "#": i, "넣기": not t.get("failed"), "파일": f["name"],
                    "종류": t["category"],
                    "과목": ("쪽별 자동" if t["category"] == "기출" and t["page_subjects"]
                             else (_auto_subj or subject)),
                    "이름": t["title"] or stem,
                    "연도": t["year"], "급": t["level"] or "",
                    "학년군": t["grade_band"], "영역": t["area"], "단원": t["unit"],
                    "손글씨": f["hw"], "확신": int(round(t["confidence"] * 100)),
                    "쪽": len(f["res"]),
                    "근거": t["reason"] + ("" if (t["subject"] or _auto_subj)
                                           else "  ⚠️ 과목 불명 — 현재 과목으로 둠"),
                })
            _failed = [f["name"] for f in out if f["tag"].get("failed")]
            if _failed:
                st.error(f"못 읽은 파일 {len(_failed)}개 (저장에서 제외됨): " + ", ".join(_failed)
                         + " — 표의 '근거' 칸에 이유가 있어요.")
            df_in = pd.DataFrame(rows).sort_values("확신", kind="stable")
            ed_in = st.data_editor(
                df_in, hide_index=True, use_container_width=True,
                disabled=["#", "파일", "확신", "쪽", "근거"],
                column_config={
                    "#": None,
                    "종류": st.column_config.SelectboxColumn(options=CATEGORIES, required=True),
                    "과목": st.column_config.SelectboxColumn(
                        options=["쪽별 자동"] + SUBJECT_LIST, required=True),
                    "급": st.column_config.SelectboxColumn(options=["", "초등", "중등", "특수"]),
                    "연도": st.column_config.NumberColumn(min_value=2000, max_value=2035,
                                                        step=1, format="%d"),
                    "확신": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                          format="%d%%"),
                    "근거": st.column_config.TextColumn(width="large"),
                },
                key=f"in_editor_{sig_in}")
            st.caption("'쪽별 자동'은 기출에서만 의미가 있어요(쪽마다 판정된 과목으로 분배). "
                       "손글씨 체크를 바꾸면 저장할 때 그 파일만 다시 읽어요.")
            b1, b2 = st.columns(2)
            if b2.button("🔄 처음부터 다시", key="in_reset"):
                st.session_state.pop(skey, None)
                st.rerun()
            if b1.button("✅ 확인한 파일 전부 저장", type="primary", key="in_save"):
                cache = ScanCache(paths.scan_cache_path())
                bucket, warns = {}, []
                for _, row in ed_in.iterrows():
                    if not row["넣기"]:
                        continue
                    f = out[int(row["#"])]
                    res = f["res"]
                    if bool(row["손글씨"]) != f["hw"] and okey_in and f["raw"]:
                        res = _scan_one(f["name"], f["raw"], bool(row["손글씨"]), cache)
                    cat, title = row["종류"], str(row["이름"]).strip() or f["name"]
                    pick = row["과목"]
                    main_subj = (pick if pick != "쪽별 자동"
                                 else (f["tag"]["subject"] or subject))
                    level = row["급"] if row["급"] in ("초등", "중등", "특수") else None
                    for r in res:
                        if r["skip"] or r.get("error") or not r["text"]:
                            continue
                        n = r["page_in_file"] + 1
                        try:
                            if cat == "기출":
                                if pd.isna(row["연도"]):
                                    warns.append(f"{f['name']}: 기출인데 연도가 없어 건너뜀")
                                    break
                                subj = (f["tag"]["page_subjects"].get(n, main_subj)
                                        if pick == "쪽별 자동" else pick)
                                rec = Record(text=r["text"], layer="L1_pattern", subject=subj,
                                             source=f"{title} p.{n}", year=int(row["연도"]),
                                             level=level or "초등",
                                             code=r["codes"][0] if r["codes"] else None,
                                             area=r["area"] or None)
                                path = paths.l1_path(subj)
                            else:
                                is_cg = (cat == "교육과정_총론")
                                subj = "공통" if is_cg else main_subj
                                rec = Record(text=r["text"], layer="L2_corpus", subject=subj,
                                             source=f"{title} p.{n}", doc_type=cat,
                                             code=r["codes"][0] if r["codes"] else None,
                                             concepts=r.get("concepts") or None,
                                             grade_band=str(row["학년군"] or "") or None,
                                             area=(r["area"] or str(row["영역"] or "") or None),
                                             unit=str(row["단원"] or "") or None)
                                path = (paths.common_chongron_path() if is_cg
                                        else paths.l2_path(subj))
                            bucket.setdefault(path, []).append(rec)
                        except Exception as e:
                            warns.append(f"{f['name']} p.{n}: {e}")
                summary = []
                for path, recs in bucket.items():
                    merged, added = merge_new(load_records_pkl(path), recs)
                    if added:
                        save_records_pkl(merged, path)
                    summary.append(f"{os.path.basename(path).replace('.pkl', '')} +{added}쪽")
                for w in warns[:10]:
                    st.warning(w)
                if summary:
                    st.session_state["in_summary"] = summary
                    st.session_state.pop(skey, None)
                    st.rerun()


# ══════════════════════════════════════════════════════════════
# 탭 L2 — 자료 입력·학습
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("성취기준·지도서 자료 (L2)")
    l2 = load_records_pkl(paths.l2_path(subject))
    st.write(f"현재 저장된 자료: **{len(l2)}건**")

    # ── 파일 업로드(pdf/docx/txt) + 자료종류·개념 태깅 ────────
    with st.expander("📄 파일로 자료 넣기 (pdf · 사진 · 손글씨 · docx · txt)", expanded=False):
        from schema import DOC_TYPES
        from concept_dict import ConceptDict

        cdict = ConceptDict.load(paths.concept_dict_path(subject), subject)

        dtc1, dtc2 = st.columns(2)
        doc_type = dtc1.selectbox(
            "자료 종류(필수)",
            ["교육과정_성취기준", "지도서_각론", "지도서_총론", "교육과정_총론", "개인_필기"],
            key="l2_doctype")
        fsrc = dtc2.text_input("출처(필수)", key="l2_fsrc",
                               placeholder="국어 지도서 각론 3-1")

        # 교육과정 총론은 과목 공통으로 저장됨을 안내
        save_subject = subject
        if doc_type == "교육과정_총론":
            st.info("교육과정 총론은 전 교과 공통이라 '공통_교육과정총론.pkl'에 저장됩니다.")
            save_subject = "공통"

        # 각론이면 개념/단원/학년군/영역/학습모형 태깅 필드
        is_gakron = (doc_type == "지도서_각론")
        gb = ar = unit = model = ""
        chosen_concepts = []
        if is_gakron:
            st.markdown("**📌 각론 태깅** (개념이 핵심 — 직접 분류하며 공부)")
            gc1, gc2, gc3 = st.columns(3)
            gb = gc1.text_input("학년군", key="l2_gb", placeholder="3~4학년군")
            ar = gc2.text_input("영역", key="l2_ar", placeholder="읽기")
            unit = gc3.text_input("단원", key="l2_unit", placeholder="이야기를 간추려요")
            model = st.text_input("학습모형(선택)", key="l2_model",
                                  placeholder="반응중심학습")
            # 개념 사전에서 고르거나 새로 추가
            existing = cdict.names()
            chosen_concepts = st.multiselect("개념 태그(사전에서 선택)", existing,
                                             key="l2_concepts")
            new_concept = st.text_input("새 개념 추가(사전에 없으면)", key="l2_newconcept")
            if st.button("개념 사전에 추가", key="l2_addconcept") and new_concept.strip():
                if cdict.add(new_concept.strip(), area=ar or None):
                    cdict.save(paths.concept_dict_path(subject))
                    st.success(f"개념 '{new_concept}' 사전에 추가됨")
                    st.rerun()
                else:
                    st.warning("이미 있는 개념이거나 빈 값")

        ups = st.file_uploader(
            "자료 파일 (여러 개 가능 · 폰으로 찍은 필기 사진도 OK)",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "webp", "heic"],
            accept_multiple_files=True, key="l2_file")
        _has_img = any(u.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".heic"))
                       for u in (ups or []))
        hw = st.checkbox("✍️ 손글씨 포함 (한글 필기·여백 메모까지 정밀 판독)",
                         value=(doc_type == "개인_필기" or _has_img), key="l2_hw")
        if ups:
            try:
                for _u in ups:
                    cloud.archive_upload(subject, "L2_corpus", _u.name, _u.getvalue())
                edited, res = page_scan_ui("l2", ups, cdict.names(), handwriting=hw)
                if edited is not None and st.button("체크한 쪽 저장", type="primary",
                                                    key="l2_file_save"):
                    if not fsrc.strip():
                        st.error("출처는 필수입니다.")
                    else:
                        target_path = (paths.common_chongron_path()
                                       if doc_type == "교육과정_총론"
                                       else paths.l2_path(subject))
                        new = []
                        for _, row in edited.iterrows():
                            text = str(row["본문"]).strip()
                            if not row["넣기"] or not text:
                                continue
                            pg = int(row["쪽"])
                            codes = [c.strip() for c in str(row["코드"]).split(",") if c.strip()]
                            _pl = (f"{res[pg - 1]['name']} " if len(ups) > 1 else "")
                            concepts = sorted(set(chosen_concepts) |
                                              set(res[pg - 1].get("concepts") or []))
                            try:
                                new.append(Record(
                                    text=text, layer="L2_corpus", subject=save_subject,
                                    source=f"{fsrc.strip()} {_pl}p.{pg}",
                                    code=codes[0] if codes else None,
                                    doc_type=doc_type, concepts=concepts or None,
                                    grade_band=gb or None,
                                    area=(str(row["영역"]).strip() or ar or None),
                                    unit=unit or None, model=model or None))
                            except Exception as e:
                                st.warning(f"{pg}쪽 거부 — {e}")
                        merged, added = merge_new(load_records_pkl(target_path), new)
                        save_records_pkl(merged, target_path)
                        st.success(f"{added}쪽 저장 → {os.path.basename(target_path)}")
            except Exception as e:
                st.error(f"스캔 실패: {e}")

    with st.expander("➕ 직접 입력", expanded=(len(l2) == 0)):
        c1, c2 = st.columns([3, 1])
        txt = c1.text_area("자료 문장(한 줄에 하나)", height=120,
                           placeholder="이야기를 읽고 인물의 마음을 짐작하며 감상한다")
        src = c2.text_input("출처(필수)", placeholder="2022개정 국어과 성취기준")
        code = c2.text_input("성취기준 코드(선택)", placeholder="[4국05-04]")
        if st.button("자료 저장", type="primary"):
            if not src.strip():
                st.error("출처는 필수입니다.")
            else:
                added = 0
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        l2.append(Record(text=line, layer="L2_corpus",
                                         subject=subject, source=src.strip(),
                                         code=code.strip() or None))
                        added += 1
                    except Exception as e:
                        st.warning(f"거부됨: {line[:20]} — {e}")
                save_records_pkl(l2, paths.l2_path(subject))
                st.success(f"{added}건 저장 → {subject}_L2.pkl")
                st.rerun()

    if l2:
        st.dataframe([{"내용": r.text[:40], "출처": r.source, "코드": r.code}
                      for r in l2[-20:]], use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 임베딩 + SOM 학습")
    st.caption("임베딩은 한 번 학습해 고정. 자료를 크게 바꿨을 때만 다시 학습.")
    cc1, cc2, cc3 = st.columns(3)
    dim = cc1.number_input("임베딩 차원", 16, 128, 32, step=16)
    min_count = cc2.number_input("최소 등장(min_count)", 1, 5, 1)
    grid = cc3.number_input("SOM 격자(한 변)", 6, 20, 10)
    force_emb = st.checkbox("임베딩도 다시 학습(자료 크게 바뀜)", value=not _has_emb)

    if st.button("학습 시작", type="primary"):
        if len(l2) < 3:
            st.error("자료가 너무 적습니다(최소 3건).")
        else:
            texts = [r.text for r in l2]
            with st.spinner("임베딩 학습 중..." if force_emb else "임베딩 로드..."):
                if force_emb or not _has_emb:
                    e = train_embedding(texts, dim=int(dim),
                                        min_count=int(min_count), epochs=30)
                    e.save(paths.emb_path(subject))
                else:
                    e = FrozenEmbedding.load(paths.emb_path(subject))
            X, kept = [], []
            for r in l2:
                v = e.embed_tokens(tokenize(r.text))
                if v is not None:
                    X.append(v); kept.append(r)
            if not X:
                st.error("벡터화 실패(모르는 단어뿐). min_count를 1로.")
            else:
                with st.spinner("SOM 학습 중..."):
                    s = SOM(grid=(int(grid), int(grid)), dim=e.dim)
                    s.train(np.array(X), iters=4000)
                    s.assign(np.array(X), kept)
                    s.save(paths.som_path(subject))
                st.success(f"학습 완료! 어휘 {len(e.word2idx)}개, 벡터화 {len(kept)}/{len(l2)}건")
                load_engine.clear()
                st.rerun()

    # 정합성 판정 데모
    if som is not None and emb is not None:
        st.markdown("---")
        st.subheader("✅ 자료 정합성 판정 (가드레일)")
        from layer2 import judge
        q = st.text_input("판정할 문장", placeholder="학생이 인물의 마음을 짐작하여 표현한다")
        if q:
            by_id = {r.rec_id: r for r in l2}
            res = judge(q, emb, som, by_id)
            v = res.get("verdict")
            if v == "정합":
                st.success(f"정합 (유사도 {res['score']})")
            elif v == "판정불가":
                st.warning(res.get("reason"))
            else:
                st.error(v + f" (유사도 {res.get('score')})")
            if res.get("concept_sources"):
                st.write("**근거 개념영역 자료(출처):**")
                st.json(res["concept_sources"])

# ══════════════════════════════════════════════════════════════
# 탭 L1 — 기출 패턴
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("기출 시계열 패턴 (L1)")
    st.caption("⚠️ 예측기 아님 — 어느 개념이 어느 해에 몰렸나 보는 '경향 분석'")
    l1 = load_records_pkl(paths.l1_path(subject))
    st.write(f"저장된 기출: **{len(l1)}건**")

    # ── 기출 파일 업로드 (초등: 통짜 시험지 → 문항 분할) ──────
    with st.expander("📄 기출 시험지 통째로 넣기 (pdf · 사진)", expanded=False):
        st.caption("시험지를 나누지 않고 통째로 올려요. 검토 단계 없이 바로 저장되고, "
                   "분석용으로만 내부에서 쪽 단위로 쓰입니다. 스캔본·사진도 OK.")
        upqs = st.file_uploader("기출 시험지 (PDF 1개, 또는 시험지 사진 여러 장)",
                                type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "webp", "heic"],
                                accept_multiple_files=True, key="l1_file")
        fc1, fc2, fc3 = st.columns(3)
        fyear = fc1.number_input("출제연도", 2000, 2030, 2023, key="l1_fy")
        flevel = fc2.selectbox("급", ["초등", "중등", "특수", "공통"], key="l1_fl")
        fqsrc = fc3.text_input("시험 이름(필수)", key="l1_fs", placeholder="2023 초등임용 1교시")
        fhw = st.checkbox("✍️ 손글씨 있음 (내가 푼 흔적·메모가 적힌 시험지)", key="l1_hw")
        okey_l1 = api_key("OpenAI Key(스캔용)", "OPENAI_API_KEY", "l1_scankey")
        if upqs and st.button("📥 기출 통째로 저장", type="primary", key="l1_file_save"):
            if not fqsrc.strip():
                st.error("시험 이름(출처)은 필수입니다.")
            elif any(r.source.startswith(fqsrc.strip() + " p.") for r in l1):
                st.error(f"'{fqsrc.strip()}'은 이미 있어요. 다시 넣으려면 아래 목록에서 먼저 삭제하세요.")
            else:
                from page_scan import build_pages, scan_pages, ScanCache
                files = [(u.name, u.getvalue()) for u in upqs]
                for _n, _r in files:
                    cloud.archive_upload(subject, "L1_exam", _n, _r)
                bar = st.progress(0.0, text="스캔 중…")
                res = scan_pages(build_pages(files), okey_l1 or None, [],
                                 cache=ScanCache(paths.scan_cache_path()),
                                 model=cloud.cfg("OPENAI_HANDWRITING_MODEL" if fhw
                                                 else "OPENAI_SCAN_MODEL"),
                                 handwriting=fhw,
                                 progress=lambda d, n: bar.progress(d / max(n, 1),
                                                                    text=f"{d}/{n}쪽"))
                new = []
                for r in res:
                    if r["skip"] or r.get("error") or not r["text"]:
                        continue
                    try:
                        new.append(Record(
                            text=r["text"], layer="L1_pattern", subject=subject,
                            source=f"{fqsrc.strip()} p.{r['page'] + 1}",
                            year=int(fyear), level=flevel,
                            code=r["codes"][0] if r["codes"] else None,
                            area=r["area"] or None))
                    except Exception as e:
                        st.warning(f"{r['page'] + 1}쪽 거부 — {e}")
                bad = [r["page"] + 1 for r in res if r.get("error") or r.get("note")]
                if new:
                    l1, _ = merge_new(l1, new)
                    save_records_pkl(l1, paths.l1_path(subject))
                    st.success(f"'{fqsrc.strip()}' 저장 ({len(new)}쪽)")
                if bad:
                    st.warning(f"못 읽은 쪽: {bad} — 키 확인 후 같은 이름으로 삭제·재업로드")

    # ── 저장된 기출: 시험 단위로 통째로 보기 ───────────────────
    _exams = {}
    for r in l1:
        name = r.source.rsplit(" p.", 1)[0] if " p." in r.source else r.source
        _exams.setdefault(name, []).append(r)
    if _exams:
        with st.expander(f"🗂️ 저장된 기출 ({len(_exams)}개 시험)", expanded=False):
            for name, rs in sorted(_exams.items(), key=lambda x: (-(x[1][0].year or 0), x[0])):
                def _pg(r):
                    t = r.source.rsplit(" p.", 1)
                    return int(t[1]) if len(t) == 2 and t[1].isdigit() else 0
                rs = sorted(rs, key=_pg)
                st.markdown(f"**{name}** · {rs[0].year} {rs[0].level or ''} · {len(rs)}쪽")
                v1, v2 = st.columns(2)
                if v1.toggle("원문 보기", key=f"l1_view_{name}"):
                    st.markdown("\n\n---\n\n".join(r.text for r in rs))
                if v2.button("🗑️ 이 시험 삭제", key=f"l1_del_{name}"):
                    ids = {r.rec_id for r in rs}
                    l1 = [r for r in l1 if r.rec_id not in ids]
                    save_records_pkl(l1, paths.l1_path(subject))
                    st.rerun()

    with st.expander("➕ 직접 입력", expanded=(len(l1) == 0)):
        qtxt = st.text_area("기출 문항", height=80,
                            placeholder="인물의 마음을 짐작하는 지도 방법을 서술하시오")
        d1, d2, d3 = st.columns(3)
        year = d1.number_input("출제연도(필수)", 2000, 2030, 2023)
        level = d2.selectbox("급(필수)", ["초등", "중등", "특수", "공통"])
        qsrc = d3.text_input("출처(필수)", placeholder="2023 초등임용 국어")
        if st.button("기출 저장", type="primary", key="save_l1"):
            if not qtxt.strip() or not qsrc.strip():
                st.error("문항·출처는 필수입니다.")
            else:
                try:
                    l1.append(Record(text=qtxt.strip(), layer="L1_pattern",
                                     subject=subject, source=qsrc.strip(),
                                     year=int(year), level=level))
                    save_records_pkl(l1, paths.l1_path(subject))
                    st.success(f"저장 → {subject}_L1.pkl")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if som is None or emb is None:
        st.info("먼저 L2 탭에서 학습을 완료하세요(기출을 얹을 지도가 필요).")
    elif l1:
        from layer1 import map_exams_to_som, concept_year_report, cross_level_flow
        node_exams = map_exams_to_som(l1, emb, som)
        min_hits = st.slider("패턴 최소 반복", 1, 5, 2)
        st.write("**개념영역별 출제 연도**")
        rep = concept_year_report(node_exams, min_hits=min_hits)
        if rep:
            st.dataframe([{"개념샘플": r["sample"], "출제수": r["hit_count"],
                           "연도": r["years"], "급별": r["levels"]} for r in rep],
                         use_container_width=True)
        else:
            st.caption("아직 반복 패턴 없음 — 기출을 더 넣으면 같은 개념에 연도가 쌓임")
        flows = cross_level_flow(node_exams)
        if flows:
            st.write("**급별 교차 흐름(초/중/특 등장 시점)**")
            st.json(flows)

        # ── 속기사: 패턴 수치 → LLM이 경향 문장으로 옮김 ──────
        st.markdown("---")
        st.write("**🖋️ 경향 해설(속기사)** — 판단은 시스템(SOM)이, LLM은 문장 번역만")
        scribe_key = api_key("OpenAI Key(속기용)", "OPENAI_API_KEY", "l1_scribe")
        if st.button("경향 해설 생성", key="l1_scribe_btn"):
            from layer1 import scribe_trends
            if not scribe_key:
                st.warning("OpenAI 키가 필요합니다.")
            else:
                with st.spinner("패턴 수치를 문장으로 옮기는 중..."):
                    text, facts = scribe_trends(node_exams, som, emb, scribe_key,
                                                min_hits=min_hits)
                st.markdown(text)
                with st.expander("근거가 된 계산 수치(속기사가 옮긴 원본)"):
                    st.json(facts)

# ══════════════════════════════════════════════════════════════
# 탭 — 경향 랩 (예측·채점 원장 + LLM 라벨 → 로컬 태거)
# ══════════════════════════════════════════════════════════════
with tablab:
    import trend_lab as tl
    import labeler as lb
    st.subheader("경향 랩")
    st.caption("LLM은 해석만, 숫자는 여기서 계산해요. 예측은 기계가 채점할 수 있는 형태로만 "
               "받아 적중률을 쌓아요. 성적 좋은 규칙이 다음 예측에 다시 들어가요.")
    okey_lab = api_key("OpenAI Key", "OPENAI_API_KEY", "lab_key")
    lab_model = cloud.cfg("OPENAI_TREND_MODEL") or "gpt-4o-mini"
    _lab = tl.load_lab(subject)
    _years = sorted({r.year for r in l1 if r.year})

    lt1, lt2, lt3, lt4 = st.tabs(["📊 통계", "🔮 예측·백테스트", "🏷️ 라벨 → 로컬 태거",
                                  "🧠 자가 학습"])

    with lt1:
        if not _years:
            st.info("연도가 있는 기출이 없어요. 먼저 기출을 넣어주세요.")
        else:
            _st = tl.stats(l1)
            st.caption(f"기출 {_st['pages']}쪽 · 연도 {_years[0]}~{_years[-1]}")
            st.bar_chart({"쪽수": {str(y): _st["per_year"][y]["pages"] for y in _years}})
            cA, cB = st.columns(2)
            cA.write("**최근 3년 영역**")
            cA.write(_st["areas"]["recent"] or "—")
            cB.write("**최근 3년 핵심어**")
            cB.write(dict(list(_st["keywords"]["recent"].items())[:10]) or "—")
            st.caption(f"새로 등장: {', '.join(_st['codes']['new'][:8]) or '—'} · "
                       f"사라짐: {', '.join(_st['codes']['gone'][:8]) or '—'}")

    with lt2:
        _sc = tl.summary(_lab["predictions"])
        if _sc.get("scored"):
            m1, m2, m3 = st.columns(3)
            m1.metric("채점된 예측", _sc["scored"])
            m2.metric("적중률", f"{_sc['hit_rate']:.0%}")
            m3.metric("Brier", f"{_sc['brier']:.3f}",
                      f"기준선 {_sc['brier_baseline']:.3f}",
                      delta_color="inverse")
            st.caption("Brier는 낮을수록 좋아요. 기준선(항상 평균 확률로 찍기)보다 "
                       "낮아야 예측에 의미가 있어요.")
        else:
            st.caption("아직 채점된 예측이 없어요. 백테스트를 돌리면 바로 성적이 나와요.")

        if not okey_lab:
            st.info("예측에는 OpenAI 키가 필요해요.")
        elif len(_years) < 3:
            st.info("백테스트는 연도가 3개 이상일 때 의미가 있어요.")
        else:
            bt1, bt2 = st.columns(2)
            if bt1.button("🧪 백테스트 실행", type="primary", key="lab_bt"):
                bar = st.progress(0.0, text="예측 중…")
                try:
                    res, _lab = tl.backtest(
                        subject, l1, okey_lab, lab_model,
                        progress=lambda i, n, y: bar.progress(i / max(n, 1),
                                                              text=f"{y}년 예측 중… ({i}/{n})"))
                    st.session_state["lab_bt_res"] = res
                    st.rerun()
                except Exception as e:
                    st.error(f"백테스트 실패: {e}")
            _ty = bt2.number_input("예측할 연도", 2000, 2035,
                                   (max(_years) + 1) if _years else 2026, key="lab_ty")
            if bt2.button("🔮 다음 시험 예측", key="lab_pred"):
                try:
                    preds, meta = tl.predict(subject, l1, int(_ty), okey_lab, lab_model,
                                             _lab, cutoff=int(_ty))
                    tl.score(subject, l1, preds, _lab)   # 그 해 기출이 이미 있으면 바로 채점
                    _lab["predictions"] += preds
                    tl.save_lab(subject, _lab)
                    st.success(f"{len(preds)}개 예측 저장 "
                               + (f"(형식 안 맞아 버린 것 {meta['dropped']}개)"
                                  if meta["dropped"] else ""))
                except Exception as e:
                    st.error(f"예측 실패: {e}")

        _bt = st.session_state.get("lab_bt_res")
        if not isinstance(_bt, list):          # 앱을 껐다 켜도 마지막 결과는 원장에서 복구
            _runs = [r for r in _lab.get("runs", []) if r.get("kind") == "backtest"]
            _bt = _runs[-1]["result"] if _runs else None
        if isinstance(_bt, list) and _bt:
            st.write("**백테스트 결과 (연도별)**")
            st.dataframe([{"연도": r.get("year"), "예측수": r.get("n"),
                           "적중률": (f"{r['hit_rate']:.0%}" if r.get("hit_rate") is not None else "—"),
                           "Brier": (f"{r['brier']:.3f}" if r.get("brier") is not None else "—"),
                           "기준선": (f"{r['brier_baseline']:.3f}" if r.get("brier_baseline") is not None else "—"),
                           "오류": r.get("error", "")}
                          for r in _bt],
                         use_container_width=True, hide_index=True)

        _rules = tl.top_rules(_lab, n=10, min_n=1)
        if _rules:
            st.write("**규칙 성적 (좋은 순)**")
            st.dataframe([{"규칙": r["rule"][:70], "표본": r["n"],
                           "적중률": f"{r['hit_rate']:.0%}", "Brier": f"{r['brier']:.3f}"}
                          for r in _rules], use_container_width=True, hide_index=True)
            st.caption("표본이 쌓일수록 신뢰도가 올라가요. 상위 규칙은 다음 예측에 자동으로 들어가요.")

        _pending = [p for p in _lab["predictions"] if not p["scored"]]
        if _pending:
            st.write(f"**채점 대기 {len(_pending)}건** (그 해 기출이 들어오면 자동 채점)")
            st.dataframe([{"대상": p["target_year"], "범위": p["scope"],
                           "예측": p["target"], "확률": f"{p['prob']:.0%}",
                           "규칙": p["rule"][:40]} for p in _pending[-20:]],
                         use_container_width=True, hide_index=True)
            if st.button("지금 채점 시도", key="lab_score"):
                n = tl.score(subject, l1, _lab["predictions"], _lab)
                tl.save_lab(subject, _lab)
                st.success(f"{n}건 채점" if n else "아직 채점할 수 있는 게 없어요")
                st.rerun()

    with lt3:
        st.caption("LLM이 쪽마다 영역·발문유형·난도를 붙이고, 그 라벨로 로컬 분류기를 학습해요. "
                   "학습 뒤에는 키 없이도 태깅이 돼요.")
        _store = lb.load_labels(subject)
        _pool = list(l1) + list(l2)
        _todo = [r for r in _pool if r.rec_id not in _store]
        st.caption(f"라벨 {len(_store)}건 · 아직 없는 자료 {len(_todo)}건")
        if okey_lab and _todo:
            # 남은 자료가 10건 미만이어도 오류가 나지 않게 범위를 자료 수에 맞춘다
            _max_lab = max(1, min(500, len(_todo)))
            n_lab = st.number_input("한 번에 라벨 붙일 개수", 1, _max_lab,
                                    min(100, _max_lab), step=10, key="lab_n")
            if st.button("🏷️ 라벨 붙이기", type="primary", key="lab_go"):
                bar = st.progress(0.0, text="라벨 중…")
                _store, done = lb.label_records(
                    subject, _todo, okey_lab, lab_model, limit=int(n_lab),
                    progress=lambda d, n: bar.progress(d / max(n, 1), text=f"{d}/{n}"))
                st.success(f"{done}건 라벨 완료")
                st.rerun()
        elif not okey_lab:
            st.info("라벨링에는 OpenAI 키가 필요해요.")

        if _store:
            _ok = [v for v in _store.values() if "error" not in v]
            st.dataframe([{"영역": v.get("area") or "—", "유형": v.get("qtype"),
                           "난도": v.get("level"), "출처": v.get("source", "")[:30]}
                          for v in _ok[-15:]], use_container_width=True, hide_index=True)
            if st.button("🧠 로컬 태거 학습", key="lab_train"):
                try:
                    rep = lb.train_tagger(subject, _pool)
                    st.session_state["tagger_rep"] = rep
                    st.rerun()
                except Exception as e:
                    st.error(f"학습 실패: {e}")
        _rep = st.session_state.get("tagger_rep")
        if _rep:
            st.write("**홀드아웃 성적** (기준선 = 가장 흔한 라벨로 전부 찍기)")
            st.dataframe([{"항목": k,
                           "정확도": (f"{v['accuracy']:.0%}" if "accuracy" in v else "—"),
                           "기준선": (f"{v['baseline']:.0%}" if "baseline" in v else "—"),
                           "학습/검증": (f"{v['train']}/{v['test']}" if "train" in v else ""),
                           "비고": v.get("skip", "")} for k, v in _rep.items()],
                         use_container_width=True, hide_index=True)
            st.caption("기준선보다 높아야 배운 게 있는 거예요. 낮으면 라벨을 더 모으세요.")
        if lb.load_tagger(subject):
            _q = st.text_input("로컬 태거 시험 (문장 붙여넣기)", key="lab_try")
            if _q:
                _r = lb.tag(subject, _q)
                st.write(_r or "벡터화 실패 — 임베딩에 없는 단어뿐이에요")



    with lt4:
        import self_exam as se, gap_search as gs
        st.caption("워커가 밤에 스스로 문제를 내고 풉니다. 채점은 LLM이 아니라 "
                   "자료 자체가 해요 — 같은 성취기준 자료를 찾아왔는지, 가린 낱말을 "
                   "복원했는지로 봅니다. 그래서 자기가 자기를 칭찬하는 구간이 없어요.")
        _lab = se.load(subject)
        _runs = _lab["runs"]
        if _runs:
            _last = _runs[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("근거 검색", f"{_last['retrieval'].get('ok', 0)}/{_last['retrieval'].get('n', 0)}")
            m2.metric("빈칸 복원", f"{_last['cloze'].get('ok', 0)}/{_last['cloze'].get('n', 0)}")
            _fixed = sum(1 for g in _lab["gaps"].values() if g.get("fixed"))
            m3.metric("못 찾은 곳", len(_lab["gaps"]) - _fixed,
                      f"메움 {_fixed}" if _fixed else None)
            st.caption(f"마지막 실행 {time.strftime('%m-%d %H:%M', time.localtime(_last['at']))}")
        else:
            st.info("아직 자가 시험 기록이 없어요. 워커가 돌면 채워져요.")

        st.write("**전략 성적** (근거를 어떻게 찾을지 — 성적 좋은 쪽을 더 씁니다)")
        st.dataframe([{"전략": r["전략"], "시행": r["시행"],
                       "성공률": (f"{r['성공률']:.0%}" if r["성공률"] is not None else "—")}
                      for r in se.arm_table(_lab)],
                     use_container_width=True, hide_index=True)
        st.caption("code=성취기준 코드 · node=SOM 개념영역 · keyword=낱말 겹침 · embed=임베딩 유사도")

        _gaps = se.top_gaps(_lab, n=12)
        if _gaps:
            st.write("**근거를 못 찾는 곳** (자료가 비어 있다는 신호)")
            st.dataframe([{"항목": g["key"], "실패": g["misses"],
                           "이유": g["why"], "검색함": "○" if g.get("searched") else "",
                           "본문": (g.get("sample") or "")[:50]} for g in _gaps],
                         use_container_width=True, hide_index=True)

        st.markdown("---")
        st.write("**🌐 웹 수집함** — 워커가 구멍을 메우려고 찾아둔 후보예요. "
                 "채택해야 자료가 됩니다.")
        _bkey = api_key("Brave API Key", "BRAVE_API_KEY", "se_brave")
        if _gaps and _bkey and st.button("🔎 지금 구멍 검색", key="se_search"):
            n = gs.collect(subject, [g for g in _gaps if not g.get("searched")][:5],
                           _bkey, log=lambda *a: None)
            se.save(subject, _lab)
            st.success(f"후보 {n}건 수집")
            st.rerun()
        _cands = gs.pending(subject)
        if not _cands:
            st.caption("검토할 후보가 없어요.")
        else:
            import pandas as pd
            _cdf = pd.DataFrame([{
                "#": i, "채택": c["trust"] >= 0.8, "신뢰": int(c["trust"] * 100),
                "도메인": c["domain"], "제목": c["title"][:60],
                "메우려는 곳": c["gap_key"], "발췌": c["text"][:120]}
                for i, c in enumerate(_cands[:40])])
            _ced = st.data_editor(
                _cdf, hide_index=True, use_container_width=True,
                disabled=["#", "신뢰", "도메인", "제목", "메우려는 곳", "발췌"],
                column_config={"#": None,
                               "신뢰": st.column_config.ProgressColumn(
                                   min_value=0, max_value=100, format="%d%%"),
                               "발췌": st.column_config.TextColumn(width="large")},
                key="se_cands")
            ca, cb = st.columns(2)
            if ca.button("✅ 체크한 것 자료로 채택", type="primary", key="se_acc"):
                ids = [_cands[int(r["#"])]["id"] for _, r in _ced.iterrows() if r["채택"]]
                n = gs.accept(subject, ids)
                st.success(f"{n}건 자료로 편입 — 다음 자가 시험이 도움이 됐는지 확인해요")
                st.rerun()
            if cb.button("🗑️ 체크 안 한 것 버리기", key="se_rej"):
                ids = [_cands[int(r["#"])]["id"] for _, r in _ced.iterrows() if not r["채택"]]
                st.info(f"{gs.reject(subject, ids)}건 버림")
                st.rerun()
        _dt = gs.domain_table(subject)
        if _dt:
            st.write("**출처 성적** (채택한 자료가 실제로 구멍을 메웠는지)")
            st.dataframe([{**r, "도움률": (f"{r['도움률']:.0%}" if r["도움률"] is not None else "—")}
                          for r in _dt[:10]], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# 탭 L3 — 트렌드 검색
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("최신 트렌드/논문 검색 (L3)")
    if som is None or emb is None:
        st.info("먼저 L2 학습을 완료하세요.")
    else:
        from layer3 import make_queries, brave_search, prepare_trend_candidates
        st.write("**‘자료가 성긴 영역’ 기반 자동 검색어**")
        qs = make_queries(emb, som, subject=subject)
        st.dataframe([{"검색어": q["query"], "QE": round(q["qe"], 3)} for q in qs],
                     use_container_width=True)
        bkey = api_key("Brave API Key", "BRAVE_API_KEY", "l3_brave")
        okey3 = api_key("OpenAI Key(속기 요약용, 선택)", "OPENAI_API_KEY", "l3_openai")
        pass_thr = st.slider("도메인 필터 강도", 0.2, 0.7, 0.40, 0.05)

        if st.button("검색 실행(후보 만들기)") and bkey:
            cands = []
            for q in qs[:5]:
                try:
                    results = brave_search(q["query"], bkey)
                    cands.extend(prepare_trend_candidates(
                        results, emb, som, subject,
                        api_key=okey3 or None, pass_thr=pass_thr))
                except Exception as e:
                    st.warning(f"검색 실패: {q['query'][:20]} — {e}")
            st.session_state["trend_cands"] = cands
            st.success(f"후보 {len(cands)}건 — 아래에서 검토 후 반영")

        # 후보 검토: 반영/제거
        cands = st.session_state.get("trend_cands", [])
        if cands:
            st.markdown("**검토** — 반영할 것만 체크하고 저장(제거는 체크 해제)")
            import pandas as pd
            df = pd.DataFrame([{"반영": c["keep"], "요약": c["summary"],
                                "내용": c["text"][:80], "출처": c["source"],
                                "정합도": c["score"]} for c in cands])
            edited = st.data_editor(df, use_container_width=True, key="l3_editor")
            if st.button("반영한 것만 저장", type="primary", key="l3_save"):
                l3 = load_records_pkl(paths.l3_path(subject))
                added = 0
                for idx, row in edited.iterrows():
                    if not row["반영"]:
                        continue
                    try:
                        l3.append(Record(
                            text=cands[idx]["text"], layer="L3_trend",
                            subject=subject, source=cands[idx]["source"]))
                        added += 1
                    except Exception as e:
                        st.warning(f"거부: {e}")
                save_records_pkl(l3, paths.l3_path(subject))
                st.success(f"반영 {added}건 저장 → {subject}_L3.pkl")
                st.session_state["trend_cands"] = []
                st.rerun()

# ══════════════════════════════════════════════════════════════
# 탭 L4 — 문제 풀기 (해설 잠금 → 버튼으로만 공개)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("연습문제 (L4)")
    st.caption("문제 풀기 → 내 답 채점(제안) → 내가 확정 → 약점지도 학습. 문제·해설 검토로 함께 발전.")
    if som is None or emb is None:
        st.info("먼저 L2 학습을 완료하세요.")
    else:
        from layer4 import pick_concept_nodes, gather_grounding, \
            build_generation_prompt, generate_with_llm, \
            grade_answer_llm, grade_answer_offline, \
            build_stem_only_prompt, generate_stem_only, generate_answer
        from study_state import StudyState
        l2 = load_records_pkl(paths.l2_path(subject))
        l1 = load_records_pkl(paths.l1_path(subject))
        by_id = {r.rec_id: r for r in (l2 + l1)}
        study = StudyState.load(paths.study_path(subject), subject)

        # ── 약점 지도 표시 ──────────────────────────────────
        with st.expander("📊 내 약점 지도", expanded=False):
            wc = study.weak_by_code()
            wn = study.weak_by_node()
            if wc:
                st.write("**성취기준 코드별 (정답률 낮은 순)**")
                st.dataframe(wc, use_container_width=True)
            if wn:
                st.write("**개념영역별**")
                st.dataframe(wn, use_container_width=True)
            if not wc and not wn:
                st.caption("아직 채점 기록 없음 — 문제를 풀고 답을 확정하면 쌓입니다.")

        okey = api_key("OpenAI API Key (문제·채점·해설용)", "OPENAI_API_KEY", "l4_openai")
        n = st.slider("문제 수", 1, 8, 3)
        target_weak = st.checkbox("내 약점 영역 위주로 출제", value=bool(study.weak_by_node()))

        if st.button("문제 생성", type="primary"):
            from layer1 import map_exams_to_som
            l1_nodes = map_exams_to_som(l1, emb, som) if l1 else None
            if target_weak and study.weak_by_node():
                nodes = study.weak_nodes_for_targeting(top=n)
            else:
                nodes = pick_concept_nodes(som, l1_nodes, topn=n)
            probs = []
            for node in nodes:
                grounding = gather_grounding(node, som, by_id)
                if okey:
                    prompt = build_stem_only_prompt(grounding)
                    try:
                        stem = generate_stem_only(prompt, okey)
                    except Exception as e:
                        stem = f"(생성 실패: {e})"
                else:
                    stem = f"[개념영역 {node} — OpenAI 키 넣으면 실제 문제 생성]"
                codes = som.node_codes.get(node, [])
                probs.append({"node": node, "stem": stem,
                              "grounding": grounding, "codes": list(set(codes))})
            st.session_state["problems"] = probs
            st.session_state["revealed"] = set()
            st.session_state["graded"] = {}
            st.session_state["answers_gen"] = {}   # reveal 때 생성된 해설 캐시

        # ── 문제별: 풀기 → 채점 → 확정 → 검토 ──────────────
        for i, p in enumerate(st.session_state.get("problems", [])):
            st.markdown(f"### 문제 {i+1}  ·  개념영역 {p['node']}")
            st.write(p["stem"])

            my_ans = st.text_area("✍️ 내 답", key=f"ans_{i}", height=100)

            cc1, cc2, cc3 = st.columns(3)
            # 1) 자동 채점 제안
            if cc1.button("🤖 채점 제안", key=f"grade_{i}"):
                if not my_ans.strip():
                    st.warning("답을 먼저 쓰세요.")
                else:
                    if okey:
                        g = grade_answer_llm(p["stem"], my_ans, p["grounding"], okey)
                    else:
                        g = grade_answer_offline(my_ans, emb, som, p["node"])
                    st.session_state.setdefault("graded", {})[i] = g
                    st.rerun()

            g = st.session_state.get("graded", {}).get(i)
            if g:
                label = {"correct": "✅ 맞을 것 같음", "partial": "🟡 부분/애매",
                         "wrong": "❌ 틀린 것 같음"}.get(g.get("suggest"), "🟡")
                st.info(f"**자동 제안: {label}** (참고용)\n\n{g.get('feedback','')}")

            # 2) 내가 최종 확정 (이것만 약점지도 반영)
            st.write("**내 확정** (이게 약점지도에 반영됨)")
            fc1, fc2 = st.columns(2)
            if fc1.button("맞음으로 확정", key=f"ok_{i}"):
                study.record_answer(p["node"], p["codes"], my_ans,
                                    (g or {}).get("suggest"), "correct")
                study.save(paths.study_path(subject))
                st.success("맞음으로 기록됨 → 약점지도 갱신")
            if fc2.button("틀림으로 확정", key=f"no_{i}"):
                study.record_answer(p["node"], p["codes"], my_ans,
                                    (g or {}).get("suggest"), "wrong")
                study.save(paths.study_path(subject))
                st.error("틀림으로 기록됨 → 약점지도 갱신")

            # 3) 해설 보기 (잠금 → 버튼 → 그때 생성)
            revealed = st.session_state.get("revealed", set())
            if i in revealed:
                st.success("**해설**")
                # reveal 시점에 생성한 해설 캐시 확인
                gen = st.session_state.get("answers_gen", {}).get(i)
                if gen is None:
                    if okey:
                        try:
                            ma, expl = generate_answer(p["stem"], p["grounding"], okey)
                            gen = {"model_answer": ma, "explanation": expl}
                        except Exception as e:
                            gen = {"model_answer": "", "explanation": f"(해설 생성 실패: {e})"}
                    else:
                        # 채점 제안에서 나온 모범답안이 있으면 그거라도
                        gen = {"model_answer": (g or {}).get("model_answer", ""),
                               "explanation": "(OpenAI 키 없음 — 해설 미생성)"}
                    st.session_state.setdefault("answers_gen", {})[i] = gen
                if gen.get("model_answer"):
                    st.write("**모범답안**"); st.write(gen["model_answer"])
                if gen.get("explanation"):
                    st.write("**해설**"); st.write(gen["explanation"])
                st.write("**📎 근거 출처**")
                st.json(p["grounding"])
                # 4) 문제·해설 검토 (자료 신뢰도 조정 + 복구)
                st.write("**🔧 이 문제·해설 검토** (자료 개선에 반영)")
                rc1, rc2, rc3 = st.columns(3)
                if rc1.button("👍 좋은 문제", key=f"good_{i}"):
                    for g_ in p["grounding"]:
                        pass  # grounding엔 rec_id 없음 → 노드 자료로 반영
                    for rid in som.node_rec_ids.get(p["node"], [])[:3]:
                        study.review_feedback(rid, good=True, reason=f"문제{i+1} 좋음")
                    study.save(paths.study_path(subject))
                    st.success("좋은 자료로 반영")
                if rc2.button("👎 이상한 문제/해설", key=f"bad_{i}"):
                    for rid in som.node_rec_ids.get(p["node"], [])[:3]:
                        study.review_feedback(rid, good=False, reason=f"문제{i+1} 이상")
                    study.save(paths.study_path(subject))
                    st.warning("해당 개념영역 자료 신뢰도 하향(복구 가능)")
                if rc3.button("↩️ 방금 검토 복구", key=f"undo_{i}"):
                    u = study.undo_last_review()
                    study.save(paths.study_path(subject))
                    st.info(f"복구됨: {u['rec_id'][:8] if u else '없음'}")
            else:
                if st.button("🔓 해설 보기", key=f"reveal_{i}"):
                    st.session_state["revealed"].add(i)
                    st.rerun()
            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# 탭 L5 — 수능형(기타형) 순수 연습 (개념 매핑 없음)
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("수능형(기타형) 연습")
    st.caption("출처 없이 제시문 해석해 바로 푸는 문제. 개념 지도와 독립 — 유형별 약점만 쌓음.")
    from exam_practice import ExamBank, ExamWeakness
    bank = ExamBank.load(paths.exam_path(subject), subject)
    weak = ExamWeakness.load(paths.exam_weak_path(subject))

    with st.expander("➕ 수능형 문제 추가", expanded=(len(bank.items) == 0)):
        q = st.text_area("문제(제시문+발문)", height=140, key="l5_q")
        ec1, ec2 = st.columns(2)
        etype = ec1.text_input("유형(자유 태그)", key="l5_type",
                               placeholder="제시문 분석형")
        esrc = ec2.text_input("출처", key="l5_src", placeholder="자작/문제집명")
        ans = st.text_area("정답(선택)", height=60, key="l5_ans")
        expl = st.text_area("해설(선택)", height=60, key="l5_expl")
        if st.button("문제 저장", type="primary", key="l5_save"):
            if q.strip():
                bank.add(q.strip(), ans.strip(), expl.strip(),
                         etype.strip(), esrc.strip() or "수능형")
                bank.save(paths.exam_path(subject))
                st.success("저장됨"); st.rerun()

    # 유형별 약점
    wt = weak.weak_types()
    if wt:
        with st.expander("📊 유형별 약점"):
            st.dataframe(wt, use_container_width=True)

    # 풀기
    if bank.items:
        types = ["(전체)"] + bank.types()
        sel = st.selectbox("유형 필터", types, key="l5_filter")
        pool = bank.items if sel == "(전체)" else bank.by_type(sel)
        st.write(f"문제 {len(pool)}개")
        for i, it in enumerate(pool):
            st.markdown(f"**문제 {i+1}** · {it['exam_type'] or '(미분류)'}")
            st.write(it["question"])
            my = st.text_area("내 답", key=f"l5_myans_{it['id']}", height=80)
            rk = f"l5_reveal_{it['id']}"
            cc1, cc2, cc3 = st.columns(3)
            if cc1.button("해설 보기", key=f"l5_rev_{it['id']}"):
                st.session_state[rk] = True
            if st.session_state.get(rk):
                if it["answer"]:
                    st.success("정답"); st.write(it["answer"])
                if it["explanation"]:
                    st.write("**해설**"); st.write(it["explanation"])
            # 맞음/틀림 확정 → 유형별 약점
            if cc2.button("맞음", key=f"l5_ok_{it['id']}"):
                weak.record(it["exam_type"], True)
                weak.save(paths.exam_weak_path(subject))
                st.success("맞음 기록")
            if cc3.button("틀림", key=f"l5_no_{it['id']}"):
                weak.record(it["exam_type"], False)
                weak.save(paths.exam_weak_path(subject))
                st.error("틀림 기록")
            st.markdown("---")


# ══════════════════════════════════════════════════════════════
# 탭 개념지도 — 각론 개념이 총론/교육과정에서 어떻게 응용되나
# ══════════════════════════════════════════════════════════════
with tabc:
    st.subheader("개념 지도 — 각론 개념의 응용 흐름")
    st.caption("각론(개념 원천) → 지도서 총론·교육과정에서 같은 개념이 어디에 응용되나.")
    l2 = load_records_pkl(paths.l2_path(subject))
    common = load_records_pkl(paths.common_chongron_path())  # 교육과정 총론(공통)

    # 개념별로 어느 자료종류에 등장하는지 모으기
    from collections import defaultdict
    concept_map = defaultdict(lambda: defaultdict(list))
    for r in (l2 + common):
        for c in (r.concepts or []):
            concept_map[c][r.doc_type or "미분류"].append(
                {"text": r.text[:60], "source": r.source, "unit": r.unit})

    if not concept_map:
        st.info("각론 자료에 개념 태그를 달면, 여기서 개념별 응용 흐름이 보입니다. "
                "(L2 탭 → 지도서_각론 → 개념 태깅)")
    else:
        concept = st.selectbox("개념 선택", sorted(concept_map.keys()))
        st.markdown(f"### 개념: {concept}")
        buckets = concept_map[concept]
        # 각론(원천) 먼저, 그다음 응용처
        order = ["지도서_각론", "지도서_총론", "교육과정_성취기준", "교육과정_총론"]
        for dt in order:
            if dt in buckets:
                label = {"지도서_각론": "📗 각론(개념 원천)",
                         "지도서_총론": "📘 지도서 총론(응용)",
                         "교육과정_성취기준": "📕 교육과정 성취기준(응용)",
                         "교육과정_총론": "📙 교육과정 총론(상위 응용)"}[dt]
                st.write(f"**{label}**")
                for item in buckets[dt]:
                    u = f" · {item['unit']}" if item.get("unit") else ""
                    st.write(f"- {item['text']} *(출처: {item['source']}{u})*")


# ══════════════════════════════════════════════════════════════
# 탭 지문 학습 — 기출 지문 세트 + 유사 클러스터링 + 유형 확정
# ══════════════════════════════════════════════════════════════
with tabp:
    st.subheader("지문 학습 — 제시문 통으로 익히기")
    st.caption("기출 지문을 통으로 보관(외우기용). 시스템이 유사 지문끼리 묶어주면 "
               "→ 내가 유형을 확정(이름 붙이기/나누기). 생성 없음, 실제 지문만.")
    from passage_cluster import PassageBank, cluster_passages, group_by_cluster
    from file_ingest import extract_text, split_questions_rule, split_passage_question

    pbank = PassageBank.load(paths.passage_path(subject), subject)
    st.write(f"저장된 지문 세트: **{len(pbank.sets)}개** "
             f"(유형 확정 {len(pbank.sets)-len(pbank.untyped())} / 미분류 {len(pbank.untyped())})")

    # ── 지문 세트 추가 (파일 or 직접) ───────────────────────
    with st.expander("➕ 지문 세트 넣기", expanded=(len(pbank.sets) == 0)):
        upp = st.file_uploader("기출 파일(지문+문제)", type=["pdf", "docx", "txt"], key="p_file")
        pc1, pc2, pc3 = st.columns(3)
        pyear = pc1.number_input("연도", 2000, 2030, 2023, key="p_year")
        plevel = pc2.selectbox("급", ["초등", "중등", "특수", "공통"], key="p_level")
        psrc = pc3.text_input("출처(필수)", key="p_src")
        if upp is not None:
            try:
                _raw = upp.getvalue()
                cloud.archive_upload(subject, "passage", upp.name, _raw)
                raw = extract_text(upp.name, _raw)
                chunks = split_questions_rule(raw)
                st.write(f"**{len(chunks)}개 덩어리** — 각 덩어리를 지문/발문으로 분리해 확인")
                import pandas as pd
                rows = []
                for ch in chunks:
                    p, q = split_passage_question(ch)
                    rows.append({"넣기": True, "지문": p, "발문": q})
                df = pd.DataFrame(rows)
                edited = st.data_editor(df, use_container_width=True,
                                        num_rows="dynamic", key="p_editor")
                if st.button("지문 세트 저장", type="primary", key="p_save"):
                    if not psrc.strip():
                        st.error("출처는 필수입니다.")
                    else:
                        added = 0
                        for _, row in edited.iterrows():
                            if not row["넣기"]:
                                continue
                            passage = str(row["지문"]).strip()
                            quest = str(row["발문"]).strip()
                            if not passage and not quest:
                                continue
                            # 지문 없으면 발문만이라도(수능형 아닌 기출)
                            pbank.add(passage or quest,
                                      [quest] if quest else [],
                                      year=int(pyear), level=plevel,
                                      source=psrc.strip())
                            added += 1
                        pbank.save(paths.passage_path(subject))
                        st.success(f"{added}개 지문 세트 저장")
                        st.rerun()
            except Exception as e:
                st.error(f"처리 실패: {e}")

    # ── 유사 지문 클러스터링 → 유형 확정 ────────────────────
    if len(pbank.sets) >= 4 and emb is not None:
        st.markdown("---")
        st.write("**🔗 유사 지문 자동 묶기 → 유형 확정**")
        k = st.slider("묶음 개수", 2, min(12, len(pbank.sets)), 
                      min(4, len(pbank.sets)), key="p_k")
        if st.button("유사 지문 묶기", key="p_cluster"):
            passages = [s["passage"] for s in pbank.sets]
            labels, valid = cluster_passages(passages, emb, k=k)
            groups = group_by_cluster(passages, labels, valid)
            st.session_state["p_groups"] = {str(c): idxs for c, idxs in groups.items()}

        groups = st.session_state.get("p_groups", {})
        for cid, idxs in groups.items():
            with st.expander(f"묶음 {cid} — {len(idxs)}개 지문"):
                for i in idxs:
                    st.write(f"- {pbank.sets[i]['passage'][:80]}")
                tname = st.text_input(f"이 묶음의 유형 이름", key=f"p_tname_{cid}",
                                      placeholder="수업대화록형")
                if st.button(f"이 묶음에 유형 적용", key=f"p_apply_{cid}"):
                    if tname.strip():
                        for i in idxs:
                            pbank.set_type(pbank.sets[i]["id"], tname.strip())
                        pbank.save(paths.passage_path(subject))
                        st.success(f"묶음 {cid} → '{tname}' 유형 적용")
                        st.rerun()
    elif len(pbank.sets) < 4:
        st.info(f"지문이 4개 이상 쌓이면 자동 묶기가 활성화됩니다(현재 {len(pbank.sets)}개). "
                "데이터가 적으면 묶음이 엉성하니, 좀 모은 뒤 묶는 걸 권해요.")

    # ── 유형별 지문 모아보기(외우기) ────────────────────────
    if pbank.types():
        st.markdown("---")
        st.write("**📖 유형별 지문 모아보기(통으로 익히기)**")
        sel = st.selectbox("유형", pbank.types(), key="p_typesel")
        for s in pbank.by_type(sel):
            with st.expander(f"[{s['year']} {s['level']}] {s['passage'][:40]}..."):
                st.write("**지문**"); st.write(s["passage"])
                if s["questions"]:
                    st.write("**딸린 문제**")
                    for q in s["questions"]:
                        st.write(f"- {q}")
                st.caption(f"출처: {s['source']}")
