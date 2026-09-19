-- 임용 4레이어 · Supabase 초기 설정
-- Supabase 대시보드 → SQL Editor 에 통째로 붙여넣고 Run 한 번.

-- 1) pkl 최신본 메타 (동기화 판단용)
create table if not exists public.artifacts (
  key                 text primary key,          -- 예: gukeo_L2, gukeo_som
  local_name          text not null,             -- 예: 국어_L2.pkl
  sha256              text not null,
  size_bytes          bigint,
  updated_epoch       double precision not null,
  last_history_epoch  double precision
);

-- 2) 자료 미러 (L1 기출 / L2 자료 / L3 트렌드) — 대시보드에서 조회·필터용
create table if not exists public.records (
  file_key    text not null,                     -- 예: gukeo_L2
  rec_id      text not null,
  text        text not null,
  layer       text,
  subject     text,
  source      text not null,                     -- 출처 강제 원칙 그대로
  year        int,
  level       text,
  code        text,
  qtype       text,
  doc_type    text,
  concepts    jsonb default '[]'::jsonb,
  grade_band  text,
  area        text,
  unit        text,
  model       text,
  exam_type   text,
  updated_epoch double precision,
  primary key (file_key, rec_id)
);
create index if not exists records_subject_layer on public.records (subject, layer);
create index if not exists records_code on public.records (code);
create index if not exists records_year on public.records (year);

-- 3) 원본 업로드 파일 목록
create table if not exists public.uploads (
  id             bigserial primary key,
  subject        text not null,
  kind           text,                            -- L2_corpus / L1_exam / passage
  original_name  text,
  storage_path   text,
  sha256         text not null,
  size_bytes     bigint,
  created_epoch  double precision,
  unique (subject, sha256)
);

-- 4) 보안: RLS 켜고 정책은 만들지 않음
--    → anon(공개) 키로는 아무것도 못 읽고 못 씀.
--    → 앱은 서버에서 service_role 키로 접근(RLS 우회).
alter table public.artifacts enable row level security;
alter table public.records   enable row level security;
alter table public.uploads   enable row level security;

-- 5) 비공개 Storage 버킷
insert into storage.buckets (id, name, public)
values ('artifacts', 'artifacts', false), ('uploads', 'uploads', false)
on conflict (id) do nothing;
