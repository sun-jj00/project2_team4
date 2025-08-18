# streamlit_app.py
# -*- coding: utf-8 -*-
"""
대구시 금융 인프라 · 접근성 통합 대시보드 (Streamlit)
- 데이터 예시 파일: 클러스터링용.csv
- 컬럼: [은행id, 은행, 지점명, 주소, 구군, 읍면동, 위도, 경도, 복지스코어, 교통스코어, 전체인구, 65세이상, 고령인구비율, 동별지점수, 포화도]

기능 요약
1) 개요: KPI, 지도, 상/하위 랭킹
2) 지역 분석: 구군/읍면동 필터, 상세 지표, 테이블
3) 은행 분석: 은행별 점유/포화 관계, 구군 내 점유율
4) 정책 제안: 기회지역(신규 지점), 취약지역(모바일/찾아가는 서비스)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="대구시 금융 접근성 대시보드",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# 유틸
# =====================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file: str | None):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            try:
                return pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
                return pd.DataFrame()
    # 기본 경로 시도
    default_paths = [
        "./클러스터링용.csv",
        "./data/클러스터링용.csv",
        "/mnt/data/클러스터링용.csv",
    ]
    for p in default_paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                try:
                    return pd.read_excel(p)
                except Exception:
                    pass
    st.warning("데이터 파일을 업로드하거나 기본 경로(./클러스터링용.csv)가 있어야 합니다.")
    return pd.DataFrame()


def _num(x):
    try:
        return pd.to_numeric(x)
    except Exception:
        return np.nan


def normalize_minmax(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# =====================
# 사이드바
# =====================
st.sidebar.title("🔧 필터")
up = st.sidebar.file_uploader("데이터 파일 업로드 (csv/xlsx)", type=["csv", "xlsx"])
df = load_data(up)

if df.empty:
    st.stop()

# 컬럼 표준화 (영문/한글 혼용 대비)
rename_map = {
    "si_gun_gu": "구군", "sigungu": "구군",
    "eup_myeon_dong": "읍면동", "emd": "읍면동",
    "lat": "위도", "latitude": "위도",
    "lon": "경도", "lng": "경도", "longitude": "경도",
    "senior_ratio": "고령인구비율",
    "population": "전체인구"
}
for k, v in rename_map.items():
    if k in df.columns and v not in df.columns:
        df.rename(columns={k: v}, inplace=True)

# 타입 캐스팅
for c in ["위도", "경도", "복지스코어", "교통스코어", "전체인구", "65세이상", "고령인구비율", "동별지점수", "포화도"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 기본 필터 값들
gu_list = sorted([g for g in df["구군"].dropna().astype(str).unique()]) if "구군" in df else []
bank_list = sorted([b for b in df["은행"].dropna().astype(str).unique()]) if "은행" in df else []

sel_gu = st.sidebar.multiselect("구군 선택", gu_list, default=gu_list)
sel_bank = st.sidebar.multiselect("은행 선택", bank_list, default=bank_list[:5] if len(bank_list) > 5 else bank_list)

min_pop, max_pop = (int(np.nanmin(df["전체인구"]) if "전체인구" in df else 0), int(np.nanmax(df["전체인구"]) if "전체인구" in df else 0))
sel_pop = st.sidebar.slider("전체 인구 범위", min_value=min_pop, max_value=max_pop if max_pop>0 else 100000, value=(min_pop, max_pop if max_pop>0 else 100000), step=100)

min_sat, max_sat = (float(np.nanmin(df["포화도"])) if "포화도" in df else 0.0, float(np.nanmax(df["포화도"])) if "포화도" in df else 1.0)
sel_sat = st.sidebar.slider("포화도 범위", min_value=float(min_sat), max_value=float(max_sat if max_sat>min_sat else min_sat+1), value=(float(min_sat), float(max_sat if max_sat>min_sat else min_sat+1)))


# 필터 적용
q = df.copy()
if "구군" in q and sel_gu:
    q = q[q["구군"].astype(str).isin(sel_gu)]
if "은행" in q and sel_bank:
    q = q[q["은행"].astype(str).isin(sel_bank)]
if "전체인구" in q:
    q = q[(q["전체인구"].fillna(0).between(sel_pop[0], sel_pop[1]))]
if "포화도" in q:
    q = q[(q["포화도"].fillna(q["포화도"].median()).between(sel_sat[0], sel_sat[1]))]

st.sidebar.markdown("---")
st.sidebar.caption("지도 확대/축소, 범례 클릭으로 토글 가능합니다.")

# =====================
# 상단 제목 & KPI
# =====================
st.title("🏦 대구시 금융 접근성 대시보드")
st.caption("은행 지점·인구·고령화·복지·교통 지표를 통합한 분석 뷰")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("표시 지점 수", f"{len(q):,}")
with col2:
    if "전체인구" in q:
        st.metric("평균 인구(표시)", f"{int(np.nanmean(q['전체인구'])):,}")
with col3:
    if "고령인구비율" in q:
        st.metric("평균 고령인구비율", f"{np.nanmean(q['고령인구비율']):.1f}%")
with col4:
    if "복지스코어" in q:
        st.metric("평균 복지스코어", f"{np.nanmean(q['복지스코어']):.2f}")
with col5:
    if "교통스코어" in q:
        st.metric("평균 교통스코어", f"{np.nanmean(q['교통스코어']):.2f}")

# =====================
# 탭 구성
# =====================
tab1, tab2, tab3, tab4 = st.tabs(["개요", "지역 분석", "은행 분석", "정책 제안"])

# ---------------------
# 탭 1: 개요
# ---------------------
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("지도: 지점 분포 · 포화도 색상")
        if {"위도", "경도"}.issubset(q.columns):
            color_col = "포화도" if "포화도" in q.columns else None
            size_col = "전체인구" if "전체인구" in q.columns else None
            hover = [c for c in ["은행", "지점명", "주소", "구군", "읍면동", "포화도", "복지스코어", "교통스코어", "고령인구비율", "전체인구"] if c in q.columns]
            fig = px.scatter_mapbox(
                q.dropna(subset=["위도", "경도"]),
                lat="위도",
                lon="경도",
                color=color_col,
                size=size_col,
                size_max=18,
                hover_data=hover,
                zoom=10,
                height=650,
                mapbox_style="open-street-map",
            )
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("지도 표시를 위해 '위도, 경도' 컬럼이 필요합니다.")
    with c2:
        st.subheader("상/하위 랭킹")
        if "포화도" in q.columns and "읍면동" in q.columns:
            top5 = q.sort_values("포화도", ascending=False).head(5)[["읍면동", "포화도", "은행"]]
            bot5 = q.sort_values("포화도", ascending=True).head(5)[["읍면동", "포화도", "은행"]]
            st.markdown("**포화도 상위 5**")
            st.dataframe(top5.reset_index(drop=True))
            st.markdown("**포화도 하위 5**")
            st.dataframe(bot5.reset_index(drop=True))
        if "고령인구비율" in q.columns:
            st.markdown("**고령인구비율 상위 5**")
            st.dataframe(q.sort_values("고령인구비율", ascending=False).head(5)[[c for c in ["읍면동", "구군", "고령인구비율"] if c in q.columns]].reset_index(drop=True))

# ---------------------
# 탭 2: 지역 분석
# ---------------------
with tab2:
    st.subheader("지역 상세 분석")
    c1, c2 = st.columns([1, 2])
    with c1:
        sel_gu_single = st.selectbox("구군 선택", ["(전체)"] + gu_list, index=0)
        if sel_gu_single != "(전체)":
            q2 = q[q["구군"].astype(str) == sel_gu_single].copy()
        else:
            q2 = q.copy()
        sel_emd = ["(전체)"] + sorted(q2["읍면동"].dropna().astype(str).unique()) if "읍면동" in q2 else ["(전체)"]
        sel_emd_single = st.selectbox("읍면동 선택", sel_emd, index=0)
        if sel_emd_single != "(전체)" and "읍면동" in q2:
            q2 = q2[q2["읍면동"].astype(str) == sel_emd_single]

        # 지표 미니 카드
        k1, k2, k3 = st.columns(3)
        with k1:
            if "전체인구" in q2:
                st.metric("평균 인구", f"{int(np.nanmean(q2['전체인구'])):,}")
        with k2:
            if "고령인구비율" in q2:
                st.metric("평균 고령인구비율", f"{np.nanmean(q2['고령인구비율']):.1f}%")
        with k3:
            if {"복지스코어", "교통스코어"}.issubset(q2.columns):
                st.metric("평균 복지/교통", f"{np.nanmean(q2['복지스코어']):.2f} / {np.nanmean(q2['교통스코어']):.2f}")

    with c2:
        # 막대/히트맵
        if {"읍면동", "포화도"}.issubset(q2.columns):
            st.markdown("**읍면동별 포화도**")
            fig2 = px.bar(q2.groupby("읍면동", as_index=False)["포화도"].mean().sort_values("포화도", ascending=False), x="읍면동", y="포화도")
            st.plotly_chart(fig2, use_container_width=True)
        if {"복지스코어", "교통스코어", "읍면동"}.issubset(q2.columns):
            st.markdown("**복지·교통 스코어 산점도**")
            fig3 = px.scatter(q2, x="복지스코어", y="교통스코어", color="포화도" if "포화도" in q2 else None, hover_data=[c for c in ["은행", "지점명", "읍면동"] if c in q2])
            st.plotly_chart(fig3, use_container_width=True)

        # 데이터 테이블
        st.markdown("**지점 상세 테이블**")
        show_cols = [c for c in ["은행", "지점명", "주소", "구군", "읍면동", "전체인구", "65세이상", "고령인구비율", "복지스코어", "교통스코어", "동별지점수", "포화도", "위도", "경도"] if c in q2.columns]
        st.dataframe(q2[show_cols].sort_values(by=["읍면동", "은행"] if "읍면동" in q2 and "은행" in q2 else show_cols).reset_index(drop=True), use_container_width=True)

# ---------------------
# 탭 3: 은행 분석
# ---------------------
with tab3:
    st.subheader("은행별 경쟁도 및 점유")

    if "은행" in q:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**은행별 지점 수**")
            fig4 = px.bar(q.groupby("은행", as_index=False).size().sort_values("size", ascending=False), x="은행", y="size", labels={"size": "지점 수"})
            st.plotly_chart(fig4, use_container_width=True)
        with c2:
            if {"포화도", "은행"}.issubset(q.columns):
                st.markdown("**은행별 평균 포화도**")
                fig5 = px.box(q, x="은행", y="포화도")
                st.plotly_chart(fig5, use_container_width=True)

        st.markdown("**포화도 vs 복지/교통 (버블)**")
        size_col = normalize_minmax(q["전체인구"]) * 40 + 10 if "전체인구" in q else None
        fig6 = px.scatter(
            q,
            x="복지스코어" if "복지스코어" in q else None,
            y="교통스코어" if "교통스코어" in q else None,
            size=size_col,
            color="은행" if "은행" in q else None,
            hover_data=[c for c in ["은행", "지점명", "읍면동", "포화도", "전체인구"] if c in q.columns],
        )
        st.plotly_chart(fig6, use_container_width=True)

# ---------------------
# 탭 4: 정책 제안
# ---------------------
with tab4:
    st.subheader("정책/전략 제안: 기회지역 & 취약지역")
    # 지표 정규화
    cols_need = [c for c in ["복지스코어", "교통스코어", "포화도", "전체인구", "고령인구비율"] if c in q.columns]
    qq = q.dropna(subset=cols_need).copy()
    if not qq.empty and {"복지스코어", "교통스코어", "포화도"}.issubset(qq.columns):
        qq["N_복지"] = normalize_minmax(qq["복지스코어"])
        qq["N_교통"] = normalize_minmax(qq["교통스코어"])
        qq["N_포화"] = 1 - normalize_minmax(qq["포화도"])  # 낮을수록 기회 ↑
        if "전체인구" in qq:
            qq["N_인구"] = normalize_minmax(qq["전체인구"])
        else:
            qq["N_인구"] = 0.5
        if "고령인구비율" in qq:
            qq["N_고령"] = normalize_minmax(qq["고령인구비율"])
        else:
            qq["N_고령"] = 0.5

        w_welfare = st.slider("복지 가중치", 0.0, 2.0, 1.0, 0.1)
        w_trans   = st.slider("교통 가중치", 0.0, 2.0, 1.0, 0.1)
        w_sat_inv = st.slider("포화(역가중) 가중치", 0.0, 2.0, 1.2, 0.1)
        w_pop     = st.slider("인구 가중치", 0.0, 2.0, 1.0, 0.1)
        w_senior  = st.slider("고령비중 가중치", 0.0, 2.0, 1.0, 0.1)

        qq["기회지수"] = (
            w_welfare * qq["N_복지"] +
            w_trans   * qq["N_교통"] +
            w_sat_inv * qq["N_포화"] +
            w_pop     * qq["N_인구"] +
            w_senior  * qq["N_고령"]
        ) / (w_welfare + w_trans + w_sat_inv + w_pop + w_senior + 1e-9)

        st.markdown("**신규 지점 기회지역 TOP 10**")
        top10 = qq.sort_values("기회지수", ascending=False).head(10)
        cols_show = [c for c in ["구군", "읍면동", "은행", "지점명", "기회지수", "포화도", "복지스코어", "교통스코어", "전체인구", "고령인구비율"] if c in top10.columns]
        st.dataframe(top10[cols_show].reset_index(drop=True))

        st.markdown("**고령비중 높고 접근성 낮은 취약지역 TOP 10**")
        if {"고령인구비율", "복지스코어", "교통스코어"}.issubset(qq.columns):
            vuln = qq.copy()
            vuln["vulner_index"] = normalize_minmax(vuln["고령인구비율"]) * (1 - 0.5*normalize_minmax(vuln["복지스코어"]) - 0.5*normalize_minmax(vuln["교통스코어"]))
            v10 = vuln.sort_values("vulner_index", ascending=False).head(10)
            cols_v = [c for c in ["구군", "읍면동", "은행", "지점명", "고령인구비율", "복지스코어", "교통스코어", "포화도"] if c in v10.columns]
            st.dataframe(v10[cols_v].reset_index(drop=True))

        st.markdown("**기회지수 지도**")
        if {"위도", "경도"}.issubset(qq.columns):
            fig7 = px.scatter_mapbox(
                qq.dropna(subset=["위도", "경도"]),
                lat="위도",
                lon="경도",
                color="기회지수",
                size=(normalize_minmax(qq["전체인구"]) * 20 + 8) if "전체인구" in qq else None,
                hover_data=[c for c in ["은행", "지점명", "구군", "읍면동", "포화도", "복지스코어", "교통스코어", "전체인구", "고령인구비율", "기회지수"] if c in qq.columns],
                zoom=10,
                height=650,
                mapbox_style="open-street-map",
            )
            fig7.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig7, use_container_width=True)

    else:
        st.info("정책 제안 계산을 위해 복지·교통·포화도 지표가 필요합니다.")

# 푸터
st.markdown("""
---
**Tip**: 좌측 필터로 관심 지역·은행을 좁혀 본 뒤, `정책 제안` 탭에서 가중치를 조절해 다양한 시나리오를 비교하세요.
""")
