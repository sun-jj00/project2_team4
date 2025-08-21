# main_3tabs_template.py — One-page, 3-tab Shiny app template
# -----------------------------------------------------------------------------
# How to use
# 1) Keep your current app1.py, app2.py, app3.py open for reference.
# 2) In each TAB_* section below, paste the corresponding UI pieces and server
#    logic from your original files. The template already reserves the same
#    output/input IDs so you can drop code in with minimal changes.
# 3) Put every static asset (folium HTMLs, icons, etc.) into ./www .
# 4) Run:  shiny run --reload main_3tabs_template.py
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import os, re, json

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
from branca.colormap import LinearColormap
from math import radians
from shinywidgets import output_widget, render_widget
from html import escape

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from shiny import App, ui, render, reactive, module

# === App-level static dir (shared by all tabs) ===
APP_DIR = Path(__file__).parent
APP_WWW = APP_DIR / "www"
APP_WWW.mkdir(exist_ok=True)

# === Global CSS (shared) ===
GLOBAL_CSS = """
.container-fluid { max-width: 100% !important; }
.card { width: 100% !important; }
/* A subtle helper card look */
.app-note { background:#fafafa; border:1px dashed #cbd5e1; border-radius:12px; padding:10px; }
"""

# =============================================================================
# TAB 1 — app1: 클러스터 대시보드 (지도 + 레이더 + 테이블)
# =============================================================================
# ---- Data load (T1_) ----
try:
    T1_DF = pd.read_csv('./data/클러스터포함_전체.csv', encoding='utf-8-sig')
except UnicodeDecodeError:
    T1_DF = pd.read_csv('./data/클러스터포함_전체.csv', encoding='cp949')

try:
    T1_GDF = gpd.read_file('./data/대구_행정동_군위포함.shp', encoding='utf-8')
    T1_GDF = T1_GDF.to_crs(epsg=4326)
    T1_BOUNDARY = json.loads(T1_GDF.to_json())
except Exception as e:
    print(f"[Tab1] GeoJSON load failed: {e}")
    T1_BOUNDARY = None

T1_DF['은행id'] = pd.to_numeric(T1_DF.get('은행id'), errors='coerce')
T1_DF['정책제안클러스터'] = pd.to_numeric(T1_DF.get('정책제안클러스터'), errors='coerce')

T1_CLUSTER_NAMES = {
    0: "교통·복지 취약 고령지역 지점",
    5: "교통우수 초고령지역 지점",
    6: "교통·복지 우수 고령밀집지역 지점",
}
T1_CLUSTER_COLORS = {
    0: {'line': 'blue',  'fill': 'rgba(0, 0, 255, 0.1)'},
    5: {'line': 'green', 'fill': 'rgba(0, 128, 0, 0.1)'},
    6: {'line': 'red',   'fill': 'rgba(255, 0, 0, 0.1)'}
}

T1_METRICS = ["교통스코어", "복지스코어", "고령인구비율", "포화도"]
for _c in T1_METRICS:
    T1_DF[_c] = pd.to_numeric(T1_DF.get(_c), errors='coerce')

T1_QUARTILES: dict[str, dict[str, float]] = {}
for _m in T1_METRICS:
    _s = T1_DF[_m].dropna().astype(float).values
    if len(_s) == 0:
        T1_QUARTILES[_m] = {"Q1": 0.0, "Q2": 0.0, "Q3": 1.0}
        continue
    _q1 = float(np.quantile(_s, 0.25))
    _q2 = float(np.quantile(_s, 0.50))
    _q3 = float(np.quantile(_s, 0.75)) or 1e-9
    T1_QUARTILES[_m] = {"Q1": _q1, "Q2": _q2, "Q3": _q3}

T1_Q1_BAR = float(np.mean([T1_QUARTILES[m]["Q1"]/T1_QUARTILES[m]["Q3"] for m in T1_METRICS]))
T1_Q2_BAR = float(np.mean([T1_QUARTILES[m]["Q2"]/T1_QUARTILES[m]["Q3"] for m in T1_METRICS]))
T1_Q3_BAR = 1.0

T1_CLUSTER_MEANS = T1_DF.groupby('클러스터')[T1_METRICS].mean(numeric_only=True)

def T1_normalize_row_to_q3(row: pd.Series) -> list[float]:
    vals: list[float] = []
    for _m in T1_METRICS:
        _v = row.get(_m)
        _q3 = T1_QUARTILES[_m]["Q3"]
        vals.append(min(max(float(_v)/_q3, 0.0), 1.0) if pd.notna(_v) else T1_QUARTILES[_m]["Q2"]/_q3)
    return vals

def T1_make_square_radar(cluster_ids: list[int]) -> go.Figure:
    fig = go.Figure()
    # Q-rings
    fig.add_trace(go.Scatterpolar(r=[T1_Q3_BAR]*361, theta=np.linspace(0,360,361), mode="lines",
                                  line=dict(color="darkgrey", width=2.5, dash='dot'), name="Q3", legendgroup="1"))
    fig.add_trace(go.Scatterpolar(r=[T1_Q1_BAR]*361, theta=np.linspace(0,360,361), mode="lines",
                                  line=dict(color="grey", width=2, dash="dash"), name="Q1", legendgroup="1"))
    fig.add_trace(go.Scatterpolar(r=[T1_Q2_BAR]*361, theta=np.linspace(0,360,361), mode="lines",
                                  line=dict(color="dimgray", width=3.0), name="Median", legendgroup="1"))

    angles = np.linspace(0, 360, len(T1_METRICS), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    for cid in cluster_ids:
        if cid not in T1_CLUSTER_MEANS.index:
            continue
        v01 = T1_normalize_row_to_q3(T1_CLUSTER_MEANS.loc[cid])
        r_vals = v01 + [v01[0]]
        colors = T1_CLUSTER_COLORS.get(cid, {'line':'gray','fill':'rgba(128,128,128,0.1)'})
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=angles_closed, mode="lines+markers",
            line=dict(width=2, color=colors['line']), marker=dict(size=6, color=colors['line']),
            name=T1_CLUSTER_NAMES.get(cid, str(cid)), fill="toself", fillcolor=colors['fill'], legendgroup="2"
        ))

    fig.update_polars(radialaxis=dict(range=[0,1], showline=False, ticks="", showticklabels=False,
                                      showgrid=True, gridcolor="rgba(0,0,0,0.2)"),
                      angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                       tickvals=angles, ticktext=T1_METRICS), gridshape="linear")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5,
                                  traceorder="grouped"), margin=dict(l=0,r=150,t=50,b=100), height=480,
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="black"))
    return fig

@module.ui
def tab_app1_ui():
    return ui.page_fluid(
        ui.tags.style(GLOBAL_CSS),
        # Bootstrap Icons (아이콘)
        ui.head_content(
            ui.tags.link(
                rel="stylesheet",
                href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
            ),
            ui.tags.style(".btn-group { display: flex; gap: 10px; margin-top: 10px; margin-bottom: 10px; } .color-box { display: inline-block; width: 12px; height: 12px; margin-right: 8px; vertical-align: middle; border: 1px solid #ccc; } .shiny-input-checkboxgroup .shiny-input-container { display: flex; flex-direction: column; } .checkbox { display: flex; align-items: center; } .checkbox label { flex-grow: 1; }")
        ),
        ui.h5("은행 입지 클러스터 대시보드"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    ui.input_action_button("select_all", "전체선택", class_="btn-sm btn-outline-primary"),
                    ui.input_action_button("deselect_all", "전체해제", class_="btn-sm btn-outline-secondary"),
                    class_="btn-group"
                ),
                ui.input_checkbox_group(
                    "selected_clusters", "", # 제목 제거
                    {
                        "0": ui.span(
                            ui.HTML(f'<span class="color-box" style="background-color: blue;"></span> {T1_CLUSTER_NAMES[0]}'),
                            ui.tooltip(ui.tags.i(class_="bi bi-question-circle-fill ms-2"), ui.HTML("교통 불편<br>노인복지 시너지 낮음<br>고령비율 높음"), placement="right")
                        ),
                        "5": ui.span(
                            ui.HTML(f'<span class="color-box" style="background-color: green;"></span> {T1_CLUSTER_NAMES[5]}'),
                            ui.tooltip(ui.tags.i(class_="bi bi-question-circle-fill ms-2"), ui.HTML("노인복지 시너지 중간<br>교통 좋음<br>고령비율 매우 높음"), placement="right")
                        ),
                        "6": ui.span(
                            ui.HTML(f'<span class="color-box" style="background-color: red;"></span> {T1_CLUSTER_NAMES[6]}'),
                            ui.tooltip(ui.tags.i(class_="bi bi-question-circle-fill ms-2"), ui.HTML("노인복지, 교통 둘다 좋은편<br>고령비율 높음<br>지점당 인구수 높음"), placement="right")
                        ),
                    },
                    selected=[], inline=False
                ),
                ui.input_switch("policy_switch", "정책 제안만 보기", value=False),
                ui.input_action_button("show_policy", "정책 설명", class_="btn-sm btn-info w-100 mt-2"),
                width="350px", open="always"
            ),
            ui.div(
                ui.row(
                    ui.column(7,
                        ui.card(ui.card_header("은행 지점 지도"), ui.output_ui("map_widget"), style="height: 60vh;")
                    ),
                    ui.column(5,
                        ui.card(
                            ui.card_header("특징 비교 (레이더 차트)"),
                            output_widget("radar_chart"),
                            style="height: 60vh;"
                        )
                    )
                ),
                ui.card(
                    ui.card_header("데이터 테이블", ui.download_button("download_csv", "CSV 저장",
                               class_="btn-sm btn-outline-primary float-end")),
                    ui.output_data_frame("data_table")
                ),
                ui.download_button("download_map", "지도 저장 (HTML)", class_="btn-primary w-100 mt-3")
            )
        )
    )

@module.server
def tab_app1_server(input, output, session):
    @reactive.Effect
    @reactive.event(input.select_all)
    def _sel_all():
        ui.update_checkbox_group("selected_clusters", selected=["0","5","6"])  # same IDs

    @reactive.Effect
    @reactive.event(input.deselect_all)
    def _desel_all():
        ui.update_checkbox_group("selected_clusters", selected=[])

    @reactive.Effect
    @reactive.event(input.show_policy)
    def _show_policy():
        m = ui.modal(
            ui.h4("클러스터별 정책 제안"), ui.tags.hr(),
            ui.h5(T1_CLUSTER_NAMES[0]),
            ui.p(ui.HTML('<b>찾아가는 금융서비스 시행 지점 제안</b><br>취지: 도내 시외지역 및 복지관 이용 어르신들의 금융편의\
<br>제안: 외곽지역(군위군, 달성군) 지점들을 <span style="background-color: rgba(0,0,255,0.2);">찾아가는 금융서비스</span> 거점으로 선정')),
            ui.h5(T1_CLUSTER_NAMES[5]),
            ui.p(ui.HTML("<b>시니어 금융코너 확장 전략 실행에 최적</b><br>복지 시너지 강화를 위해, <span style='background-color: rgba(0,128,0,0.2);'>디지털 금융 교육존 및 은행 공동 커뮤니티 센터</span> 등 운영 가능")),
            ui.h5(T1_CLUSTER_NAMES[6]),
            ui.p(ui.HTML("<b>신규 점포 수요 및 시너지, 접근성 우수</b><br><span style='background-color: rgba(255,0,0,0.2);'>'시니어 특화점포 개설'</span>에 가장 최적의 군집으로 선정\
<br><span style='display: inline-block; width: 12px; height: 12px; background-color: yellow; border-radius: 50%; border: 1px solid #ccc; margin-right: 5px;'></span>대구 시니어특화점포 : iM뱅크 대봉브라보점")),
            title="정책 제안 상세 설명", easy_close=True, footer=ui.modal_button("닫기")
        )
        ui.modal_show(m)

    @reactive.Calc
    def T1_filtered_df_full():
        base_df = T1_DF[T1_DF['클러스터'].isin([0,5,6])]
        if not input.selected_clusters():
            return base_df
        selected = [int(c) for c in input.selected_clusters()]
        filtered = T1_DF[T1_DF['클러스터'].isin(selected)].copy()
        if input.policy_switch():
            filtered = filtered[filtered['정책제안클러스터'] == filtered['클러스터']]
        return filtered

    T1_CURRENT_MAP = reactive.Value(None)

    @output
    @render.ui
    def map_widget():
        map_data = T1_filtered_df_full()
        _map = folium.Map(location=[35.8714, 128.6014], zoom_start=11, tiles="cartodbpositron")
        if T1_BOUNDARY:
            tooltip = folium.GeoJsonTooltip(fields=['ADM_DR_NM'], aliases=[''],
                      style=('background-color: white; color: black; font-family: sans-serif; font-size: 10px; padding: 5px;'))
            folium.GeoJson(T1_BOUNDARY, style_function=lambda x: {'color': '#808080','weight':1.0,'fillOpacity':0,'opacity':0.7},
                           tooltip=tooltip).add_to(_map)
        if not map_data.empty:
            for _, row in map_data.iterrows():
                color = 'yellow' if row.get('은행id') == 31 else T1_CLUSTER_COLORS.get(row.get('클러스터'), {'line':'#888'})['line']
                tooltip_text = f"{row.get('은행','-')}<br>{row.get('지점명','-')}<br>{row.get('읍면동','-')}"
                folium.Circle(location=[row['위도'], row['경도']], radius=500, color=color, fill=True,
                              fill_color=color, fill_opacity=0.15).add_to(_map)
                folium.CircleMarker(location=[row['위도'], row['경도']], radius=4, color=color, fill=True,
                                    fill_color=color, fill_opacity=0.8, tooltip=tooltip_text).add_to(_map)
        T1_CURRENT_MAP.set(_map)
        return ui.HTML(_map._repr_html_())

    @output
    @render_widget
    def radar_chart():
        if not input.selected_clusters():
            fig = go.Figure().update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text="클러스터를 선택하여<br>특성을 비교하세요.", showarrow=False, font=dict(size=16, color="grey"))]
            )
            return fig
        cids = [int(c) for c in input.selected_clusters()]
        return T1_make_square_radar(cids)

    @output
    @render.data_frame
    def data_table():
        cols = ['은행id','은행','지점명','주소','구군','읍면동','복지스코어','교통스코어','고령인구비율','포화도']
        data = T1_filtered_df_full()
        return data[[c for c in cols if c in data.columns]]

    @session.download(filename="filtered_data.csv")
    def download_csv():
        cols = ['은행id','은행','지점명','주소','구군','읍면동','복지스코어','교통스코어','고령인구비율','포화도']
        data = T1_filtered_df_full()
        yield data[[c for c in cols if c in data.columns]].to_csv(index=False, encoding='utf-8-sig')

    @session.download(filename="daegu_bank_map.html")
    def download_map():
        _map = T1_CURRENT_MAP.get()
        temp_file = APP_DIR / "temp_map.html"
        _map.save(str(temp_file))
        with open(temp_file, "rb") as f:
            yield f.read()


# -----------------------------------------------------------------------------
# TAB 2 — Clone of app2.py (교통/복지 스코어 맵 + Top5 막대)
# -----------------------------------------------------------------------------

# ====== Matplotlib/Seaborn 기본 스타일 + 한글 폰트 설정 ======
sns.set_theme(style="whitegrid")

def _set_korean_font():
    candidates = [
        "Malgun Gothic",     # Windows
        "AppleGothic",       # macOS
        "NanumGothic",       # Linux
        "Noto Sans CJK KR",  # Google Noto
        "Noto Sans KR",
        "DejaVu Sans"
    ]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in avail:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

_set_korean_font()

# =========================
# 0) 파일 경로 설정
# =========================
BANKS_FILE          = "./data/통합추가.csv"
SENIOR_CENTER_FILE  = "./data/노인복지센터.csv"
SENIOR_HALL_FILE    = "./data/대구_경로당_구군동추가.csv"
BUS_FILE            = "./data/대구_버스정류소_필터.csv"
SUBWAY_FILE         = "./data/대구_지하철_주소_좌표추가.csv"

# 시각화 파라미터
CENTER_DAEGU = (35.8714, 128.6014)
RADIUS_BANK  = 4.0
RADIUS_INFRA = RADIUS_BANK * 0.90
OP_FILL_INFRA = 0.50
OP_LINE_INFRA = 0.80
MAX_BUS_POINTS = 8000
MAX_SUB_POINTS = 3000
H500_M = 500.0

IR_REVERSE = False  # 값↑ → 더 ‘빨강’

# =========================
# 1) 유틸
# =========================
def read_csv_safe(path):
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

LAT_CANDS = ["위도","lat","latitude","LAT","Latitude"]
LON_CANDS = ["경도","lon","lng","longitude","LON","Longitude"]
BANK_NAME_CANDS = ["은행","은행명","bank","bank_name"]
BRANCH_CANDS    = ["지점명","점포명","branch","branch_name","점명"]
ADDR_CANDS      = ["주소","도로명주소","address","addr"]

# 복지/교통 지표
HALL_CNT_CANDS      = ["반경500m_경로당수","경로당수","hall_count","count_hall_500m"]
CENTER_CNT_CANDS    = ["반경500m_노인복지회관수","노인복지회관수","center_count","count_center_500m"]
WELFARE_SCORE_CANDS = ["복지스코어","welfare_score","score_welfare"]

BUS_COUNT_CANDS     = ["반경500m_버스정류장수","버스정류장수","bus_count_500m"]
SUBWAY_COUNT_CANDS  = ["반경500m_지하철역수","지하철역수","subway_count_500m"]
ROUTES_SUM_CANDS    = ["반경500m_경유노선합","경유노선합","반경500m_버스_sqrt(경유노선수)_합","bus_routes_sqrt_sum_500m"]
TRAFFIC_SCORE_CANDS = ["교통스코어","traffic_score","교통_스코어"]

# 행정동 컬럼은 '읍면동'으로 고정(요청사항)
ADMIN_COL = "읍면동"

def find_col(df, candidates, required=True, label=""):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"'{label}' 컬럼 후보 {candidates} 중 발견되지 않았습니다.")
    return None

def series_minmax_num(s):
    v = pd.to_numeric(s, errors="coerce")
    vmin = float(np.nanmin(v)) if np.isfinite(v).any() else 0.0
    vmax = float(np.nanmax(v)) if np.isfinite(v).any() else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9
    return vmin, vmax

def ir_color(cm, val, vmin, vmax, reverse=False, alpha_when_nan="#999999"):
    if pd.isna(val):
        return alpha_when_nan
    x = float(val)
    if reverse:
        x = vmin + (vmax - (x - vmin))
    x = min(max(x, vmin), vmax)
    return cm(x)

def pick_coords_center(df, lat_c, lon_c):
    try:
        lat0 = float(df[lat_c].astype(float).mean())
        lon0 = float(df[lon_c].astype(float).mean())
        if np.isfinite(lat0) and np.isfinite(lon0):
            return (lat0, lon0)
    except Exception:
        pass
    return CENTER_DAEGU

EARTH_R = 6371000.0  # m
def haversine_vec(lat1, lon1, lat2_vec, lon2_vec):
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2 = np.radians(lat2_vec); lon2 = np.radians(lon2_vec)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_R * (2 * np.arcsin(np.sqrt(a)))

def filter_points_within_radius(points_df, p_lat, p_lon, banks_xy, radius_m=H500_M):
    if len(points_df) == 0 or len(banks_xy[0]) == 0:
        return points_df.iloc[0:0].copy()
    lat_arr = points_df[p_lat].to_numpy(dtype=float, copy=False)
    lon_arr = points_df[p_lon].to_numpy(dtype=float, copy=False)
    keep_mask = np.zeros(len(points_df), dtype=bool)
    for bl, bo in zip(banks_xy[0], banks_xy[1]):
        d = haversine_vec(bl, bo, lat_arr, lon_arr)
        keep_mask |= (d <= radius_m)
        if keep_mask.all():
            break
    return points_df[keep_mask].copy()

def percentile_filter(df, score_col, lo_pct, hi_pct):
    s = pd.to_numeric(df[score_col], errors="coerce")
    lo_v = np.nanpercentile(s, lo_pct) if np.isfinite(s).any() else -np.inf
    hi_v = np.nanpercentile(s, hi_pct) if np.isfinite(s).any() else np.inf
    return df[(s >= lo_v) & (s <= hi_v)]

def discrete_legend_html(title: str, vmin: float, vmax: float, cm, reverse: bool, n_bins: int = 5) -> str:
    bins = np.linspace(vmin, vmax, n_bins + 1)
    items = []
    for i in range(n_bins):
        a, b = bins[i], bins[i+1]
        mid = (a + b) / 2.0
        color = ir_color(cm, mid, vmin, vmax, reverse=reverse)
        items.append(
            f"""
            <div style="display:flex; align-items:center; margin:2px 8px;">
              <div style="width:22px; height:12px; background:{color}; border:1px solid #888; margin-right:6px;"></div>
              <div style="font-size:12px; color:#222;">{a:.3f} – {b:.3f}</div>
            </div>
            """
        )
    return f"""
    <div style="margin-top:6px; padding:8px 10px; border:1px solid #ddd; border-radius:8px; background:#fff;">
      <div style="font-weight:600; font-size:13px; margin-bottom:6px;">{title}</div>
      <div style="display:flex; flex-wrap:wrap;">{''.join(items)}</div>
      <div style="font-size:11px; color:#666; margin-top:4px;">높음=빨강, 낮음=노랑 (진할수록 점수 높음)</div>
    </div>
    """

# === 하단 그래프: 읍면동 Top5 막대 ===
def make_top5_admin_fig(df_filtered: pd.DataFrame, title: str, n_top: int = 5):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    if ADMIN_COL not in df_filtered.columns or df_filtered.empty:
        ax.text(0.5, 0.5, "표시할 데이터가 없습니다.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        return fig

    counts = (
        df_filtered[ADMIN_COL]
        .astype(str)
        .value_counts()
        .head(n_top)
        .sort_values(ascending=False)
    )

    x = counts.index.tolist()
    y = counts.values.tolist()

    # YlOrRd 팔레트에서 5색 (연한→진한). Top값이 왼쪽이므로 진한색부터 주고 싶으면 reversed 사용.
    YLORRD_5 = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]
    colors = list(reversed(YLORRD_5[:len(y)]))  # 큰 값일수록 진한색

    #bars = sns.barplot(x=x, y=y, palette=colors, ax=ax, edgecolor="black", linewidth=1.0)

    # 기존
# bars = sns.barplot(x=x, y=y, palette=colors, ax=ax, edgecolor="black", linewidth=1.0)

# 변경
    palette_map = {lab: col for lab, col in zip(x, colors)}
    bars = sns.barplot(
        x=x, y=y,
        hue=x,                     # <- hue 추가
        palette=palette_map,       # <- 라벨별 색 매핑
        dodge=False,
        legend=False,              # <- 범례 숨김
        ax=ax,
        edgecolor="black", linewidth=1.0
    )
    ax.bar_label(bars.containers[0], fmt="%.0f", padding=3, fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("행정동(읍면동)")
    ax.set_ylabel("은행 지점 수")
    ax.set_ylim(0, max(y) * 1.15 if y else 1)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig

# =========================
# 2) 데이터 로드(앱 시작 시 1회)
# =========================
banks   = read_csv_safe(BANKS_FILE)
centers = read_csv_safe(SENIOR_CENTER_FILE)
halls   = read_csv_safe(SENIOR_HALL_FILE)
bus_df  = read_csv_safe(BUS_FILE)
sub_df  = read_csv_safe(SUBWAY_FILE)

for d in (banks, centers, halls, bus_df, sub_df):
    d.columns = d.columns.map(lambda x: x.strip() if isinstance(x, str) else x)

# 은행 컬럼 탐지
b_lat  = find_col(banks, LAT_CANDS, True, "은행 위도")
b_lon  = find_col(banks, LON_CANDS, True, "은행 경도")
b_bank = find_col(banks, BANK_NAME_CANDS, required=False, label="은행명")
b_br   = find_col(banks, BRANCH_CANDS,    required=False, label="지점명")
b_addr = find_col(banks, ADDR_CANDS,      required=False, label="주소")

# 지표 컬럼
b_hcnt   = find_col(banks, HALL_CNT_CANDS,     required=False, label="반경500m_경로당수")
b_ccnt   = find_col(banks, CENTER_CNT_CANDS,   required=False, label="반경500m_노인복지회관수")
b_wsc    = find_col(banks, WELFARE_SCORE_CANDS,required=False, label="복지스코어")
b_buscnt = find_col(banks, BUS_COUNT_CANDS,    required=False, label="반경500m_버스정류장수")
b_subcnt = find_col(banks, SUBWAY_COUNT_CANDS, required=False, label="반경500m_지하철역수")
b_routes = find_col(banks, ROUTES_SUM_CANDS,   required=False, label="반경500m_경유노선합")
b_tsc    = find_col(banks, TRAFFIC_SCORE_CANDS,required=False, label="교통스코어")

# 좌표 숫자화 & 결측 제거
for df, la, lo in [(banks,b_lat,b_lon),
                   (centers,find_col(centers,LAT_CANDS),find_col(centers,LON_CANDS)),
                   (halls,find_col(halls,LAT_CANDS),find_col(halls,LON_CANDS)),
                   (bus_df,find_col(bus_df,LAT_CANDS),find_col(bus_df,LON_CANDS)),
                   (sub_df,find_col(sub_df,LAT_CANDS),find_col(sub_df,LON_CANDS))]:
    df[la] = pd.to_numeric(df[la], errors="coerce")
    df[lo] = pd.to_numeric(df[lo], errors="coerce")
    df.dropna(subset=[la, lo], inplace=True)

# 스코어 숫자화
if b_wsc: banks[b_wsc] = pd.to_numeric(banks[b_wsc], errors="coerce")
if b_tsc: banks[b_tsc] = pd.to_numeric(banks[b_tsc], errors="coerce")

# vmin/vmax & 컬러맵 (YlOrRd)
vmin_w, vmax_w = series_minmax_num(banks[b_wsc]) if b_wsc else (0.0, 1.0)
vmin_t, vmax_t = series_minmax_num(banks[b_tsc]) if b_tsc else (0.0, 1.0)

YLORRD = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c",
    "#fc4e2a", "#e31a1c", "#bd0026", "#800026"
]
welfare_cm = LinearColormap(colors=YLORRD, vmin=vmin_w, vmax=vmax_w)
traffic_cm = LinearColormap(colors=YLORRD, vmin=vmin_t, vmax=vmax_t)

# =========================
# 3) 맵 빌더 (교통/복지)
# =========================
def _add_corner_legend_transport(m: folium.Map):
    # 교통 범례(좌하단)
    html = f"""
    <div style="
        position:absolute; left:12px; bottom:12px; z-index:9999;
        background:rgba(255,255,255,0.95); border:1px solid #ccc;
        border-radius:8px; padding:8px 10px; font-size:12px; box-shadow:0 2px 6px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:600; margin-bottom:6px;">표시 범례</div>
      <div style="display:flex; align-items:center; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%;
                     background:rgba(144,238,144,0.50); border:2px solid rgba(120,200,70,0.80); margin-right:8px;"></span>
        버스정류장
      </div>
      <div style="display:flex; align-items:center; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%;
                     background:rgba(20,70,140,0.55); border:2px solid rgba(20,70,140,0.85);
                     box-shadow:0 0 6px rgba(20,70,140,0.25); margin-right:8px;"></span>
        지하철역
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))

def _add_corner_legend_welfare(m: folium.Map):
    # 복지 범례(좌하단)
    html = f"""
    <div style="
        position:absolute; left:12px; bottom:12px; z-index:9999;
        background:rgba(255,255,255,0.95); border:1px solid #ccc;
        border-radius:8px; padding:8px 10px; font-size:12px; box-shadow:0 2px 6px rgba(0,0,0,0.15);
    ">
      <div style="font-weight:600; margin-bottom:6px;">표시 범례</div>
      <div style="display:flex; align-items:center; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%;
                     background:rgba(0,0,0,0.35); border:2px solid rgba(0,0,0,0.45); margin-right:8px;"></span>
        경로당
      </div>
      <div style="display:flex; align-items:center; margin:4px 0;">
        <span style="display:inline-block; width:14px; height:14px; border-radius:50%;
                     background:rgba(148,0,211,0.55); border:2px solid rgba(128,0,128,0.75);
                     box-shadow:0 0 6px rgba(128,0,128,0.25); margin-right:8px;"></span>
        노인복지회관
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))

def build_welfare_map(only_within: bool, pct_range: tuple[int, int]) -> folium.Map:
    m = folium.Map(
        location=pick_coords_center(banks, b_lat, b_lon),
        zoom_start=12, tiles="CartoDB positron",
        height=492, width="100%"
    )

    banks_f = percentile_filter(banks, b_wsc, pct_range[0], pct_range[1]) if b_wsc else banks.copy()

    # 초기표시: 은행 지점만 True, 나머지 False
    fg_r500  = folium.FeatureGroup(name="반경 500m", show=False)
    fg_banks = folium.FeatureGroup(name="은행 지점", show=True)
    cluster  = MarkerCluster(name="클러스터(복지 IR)", show=False,
                             options={"spiderfyOnMaxZoom": True, "disableClusteringAtZoom": 16})
    fg_halls = folium.FeatureGroup(name="경로당", show=False)
    fg_cent  = folium.FeatureGroup(name="노인복지회관", show=False)

    for _, row in banks_f.iterrows():
        lat, lon = float(row[b_lat]), float(row[b_lon])
        w_val = float(row.get(b_wsc)) if (b_wsc and pd.notna(row.get(b_wsc))) else np.nan
        color = ir_color(welfare_cm, w_val, vmin_w, vmax_w, reverse=IR_REVERSE)
        alpha = 0.65 if pd.isna(w_val) else (0.65 + 0.30 * ((w_val - vmin_w) / (vmax_w - vmin_w + 1e-12)))

        bank_name = (str(row.get(b_bank)) if b_bank and pd.notna(row.get(b_bank)) else "-")
        branch    = (str(row.get(b_br))   if b_br   and pd.notna(row.get(b_br))   else "-")
        addr      = (str(row.get(b_addr)) if b_addr and pd.notna(row.get(b_addr)) else "-")
        hall_cnt  = (int(row.get(b_hcnt)) if b_hcnt and pd.notna(row.get(b_hcnt)) else 0)
        cent_cnt  = (int(row.get(b_ccnt)) if b_ccnt and pd.notna(row.get(b_ccnt)) else 0)

        tooltip_html = f"""
        <div style="font-size:12px;">
          <b>은행</b>: {bank_name}<br>
          <b>지점명</b>: {branch}<br>
          <b>복지스코어</b>: {('-' if pd.isna(w_val) else f'{w_val:.3f}') }<br>
          <b>반경500m_경로당수</b>: {hall_cnt}<br>
          <b>반경500m_노인복지회관수</b>: {cent_cnt}<br>
          <hr style='margin:4px 0;'>
          <b>주소</b>: {addr}
        </div>
        """

        # 반경 링(초기 비표시 그룹)
        folium.Circle(location=(lat, lon), radius=H500_M,
                      color="rgba(30,144,255,0.8)", weight=1,
                      fill=True, fill_color="rgba(30,144,255,0.5)", fill_opacity=0.06,
                      tooltip=folium.Tooltip(tooltip_html, sticky=False), opacity=0.9).add_to(fg_r500)

        # 은행 글로우 + 본 마커(표시)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_BANK*2.6,
                            color=None, weight=0, fill=True,
                            fill_color=color, fill_opacity=0.18).add_to(fg_banks)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_BANK*1.6,
                            color=None, weight=0, fill=True,
                            fill_color=color, fill_opacity=0.28).add_to(fg_banks)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_BANK,
                            color=color, weight=2, fill=True, fill_color=color,
                            fill_opacity=alpha,
                            tooltip=folium.Tooltip(tooltip_html, sticky=False)).add_to(fg_banks)

        folium.Marker(location=(lat, lon),
                      tooltip=folium.Tooltip(tooltip_html, sticky=False),
                      icon=folium.DivIcon(html="<div style='font-size:18px; line-height:18px;'>🏦</div>",
                                          class_name="bank-emoji")).add_to(cluster)

    banks_xy = (banks_f[b_lat].to_numpy(), banks_f[b_lon].to_numpy())

    # 경로당: 검정(너무 진하지 않게 투명도 완화)
    hl_la = find_col(halls, LAT_CANDS); hl_lo = find_col(halls, LON_CANDS)
    halls_plot = filter_points_within_radius(halls, hl_la, hl_lo, banks_xy) if only_within else halls
    for _, r in halls_plot.iterrows():
        lat, lon = float(r[hl_la]), float(r[hl_lo])
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA,
                            color="rgba(0,0,0,0.45)", weight=1,
                            fill=True, fill_color="rgba(0,0,0,0.35)",
                            fill_opacity=OP_FILL_INFRA, opacity=OP_LINE_INFRA).add_to(fg_halls)

    # 노인복지회관: 더 진하게 + 글로우
    ce_la = find_col(centers, LAT_CANDS); ce_lo = find_col(centers, LON_CANDS)
    centers_plot = filter_points_within_radius(centers, ce_la, ce_lo, banks_xy) if only_within else centers
    for _, r in centers_plot.iterrows():
        lat, lon = float(r[ce_la]), float(r[ce_lo])
        # 글로우(두 겹)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA*1.9,
                            color=None, weight=0, fill=True,
                            fill_color="rgba(148,0,211,0.18)").add_to(fg_cent)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA*1.3,
                            color=None, weight=0, fill=True,
                            fill_color="rgba(148,0,211,0.28)").add_to(fg_cent)
        # 본 마커(더 진하게)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA,
                            color="rgba(128,0,128,0.75)", weight=1,
                            fill=True, fill_color="rgba(148,0,211,0.55)",
                            fill_opacity=OP_FILL_INFRA, opacity=OP_LINE_INFRA).add_to(fg_cent)

    fg_r500.add_to(m); fg_banks.add_to(m); cluster.add_to(m)
    fg_halls.add_to(m); fg_cent.add_to(m)
    MiniMap(toggle_display=True, minimized=True).add_to(m)
    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    _add_corner_legend_welfare(m)
    return m


def build_traffic_map(only_within: bool, pct_range: tuple[int, int]) -> folium.Map:
    m = folium.Map(
        location=pick_coords_center(banks, b_lat, b_lon),
        zoom_start=12, tiles="CartoDB positron",
        height=492, width="100%"
    )

    banks_f = percentile_filter(banks, b_tsc, pct_range[0], pct_range[1]) if b_tsc else banks.copy()

    # 초기표시: 은행 지점만 True, 나머지 False
    fg_r500  = folium.FeatureGroup(name="반경 500m", show=False)
    fg_banks = folium.FeatureGroup(name="은행 지점", show=True)
    cluster  = MarkerCluster(name="클러스터(교통 IR)", show=False,
                             options={"spiderfyOnMaxZoom": True, "disableClusteringAtZoom": 16})
    fg_bus   = folium.FeatureGroup(name="버스정류장", show=False)
    fg_sub   = folium.FeatureGroup(name="지하철역", show=False)

    for _, row in banks_f.iterrows():
        lat, lon = float(row[b_lat]), float(row[b_lon])
        t_val = float(row.get(b_tsc)) if (b_tsc and pd.notna(row.get(b_tsc))) else np.nan
        color = ir_color(traffic_cm, t_val, vmin_t, vmax_t, reverse=IR_REVERSE)
        alpha = 0.65 if pd.isna(t_val) else (0.65 + 0.30 * ((t_val - vmin_t) / (vmax_t - vmin_t + 1e-12)))

        bank_name = (str(row.get(b_bank)) if b_bank and pd.notna(row.get(b_bank)) else "-")
        branch    = (str(row.get(b_br))   if b_br   and pd.notna(row.get(b_br))   else "-")
        addr      = (str(row.get(b_addr)) if b_addr and pd.notna(row.get(b_addr)) else "-") if b_addr else "-"
        bus_cnt   = (int(row.get(b_buscnt)) if b_buscnt and pd.notna(row.get(b_buscnt)) else 0)
        sub_cnt   = (int(row.get(b_subcnt)) if b_subcnt and pd.notna(row.get(b_subcnt)) else 0)
        routes    = (f"{float(row.get(b_routes)):.3f}" if b_routes and pd.notna(row.get(b_routes)) else "-")

        tooltip_html = f"""
        <div style="font-size:12px;">
          <b>은행</b>: {bank_name}<br>
          <b>지점명</b>: {branch}<br>
          <b>교통스코어</b>: {('-' if pd.isna(t_val) else f'{t_val:.3f}') }<br>
          <b>반경500m_버스정류장수</b>: {bus_cnt}<br>
          <b>반경500m_지하철역수</b>: {sub_cnt}<br>
          <b>반경500m_경유노선합</b>: {routes}<br>
          <hr style='margin:4px 0;'>
          <b>주소</b>: {addr}
        </div>
        """

        # 반경 링(초기 비표시)
        folium.Circle(location=(lat, lon), radius=H500_M,
                      color="rgba(30,144,255,0.8)", weight=1,
                      fill=True, fill_color="rgba(30,144,255,0.5)", fill_opacity=0.06,
                      tooltip=folium.Tooltip(tooltip_html, sticky=False), opacity=0.9).add_to(fg_r500)

        # 은행 글로우 + 본 마커(표시)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_BANK*2.6,
                            color=None, weight=0, fill=True,
                            fill_color=color, fill_opacity=0.18).add_to(fg_banks)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_BANK*1.6,
                            color=None, weight=0, fill=True,
                            fill_color=color, fill_opacity=0.28).add_to(fg_banks)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_BANK,
                            color=color, weight=2, fill=True, fill_color=color,
                            fill_opacity=alpha,
                            tooltip=folium.Tooltip(tooltip_html, sticky=False)).add_to(fg_banks)

        folium.Marker(location=(lat, lon),
                      tooltip=folium.Tooltip(tooltip_html, sticky=False),
                      icon=folium.DivIcon(html="<div style='font-size:18px; line-height:18px;'>🏦</div>",
                                          class_name="bank-emoji")).add_to(cluster)

    banks_xy = (banks_f[b_lat].to_numpy(), banks_f[b_lon].to_numpy())

    # 버스(연두)
    bs_lat = find_col(bus_df, LAT_CANDS); bs_lon = find_col(bus_df, LON_CANDS)
    bus_use = bus_df.sample(MAX_BUS_POINTS, random_state=42) if len(bus_df) > MAX_BUS_POINTS else bus_df
    bus_plot = filter_points_within_radius(bus_use, bs_lat, bs_lon, banks_xy) if only_within else bus_use
    for _, r in bus_plot.iterrows():
        lat, lon = float(r[bs_lat]), float(r[bs_lon])
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA,
                            color="rgba(120,200,70,0.80)", weight=1,
                            fill=True, fill_color="rgba(144,238,144,0.55)",
                            fill_opacity=OP_FILL_INFRA, opacity=OP_LINE_INFRA).add_to(fg_bus)

    # 지하철(더 어두운 파랑 + 글로우)
    su_lat = find_col(sub_df, LAT_CANDS); su_lon = find_col(sub_df, LON_CANDS)
    sub_use = sub_df.sample(MAX_SUB_POINTS, random_state=42) if len(sub_df) > MAX_SUB_POINTS else sub_df
    sub_plot = filter_points_within_radius(sub_use, su_lat, su_lon, banks_xy) if only_within else sub_use
    for _, r in sub_plot.iterrows():
        lat, lon = float(r[su_lat]), float(r[su_lon])
        # 글로우
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA*1.9,
                            color=None, weight=0, fill=True,
                            fill_color="rgba(20,70,140,0.18)").add_to(fg_sub)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA*1.3,
                            color=None, weight=0, fill=True,
                            fill_color="rgba(20,70,140,0.28)").add_to(fg_sub)
        # 본 마커(더 어둡고 진하게)
        folium.CircleMarker(location=(lat, lon), radius=RADIUS_INFRA,
                            color="rgba(20,70,140,0.85)", weight=1,
                            fill=True, fill_color="rgba(20,70,140,0.60)",
                            fill_opacity=OP_FILL_INFRA, opacity=OP_LINE_INFRA).add_to(fg_sub)

    fg_r500.add_to(m); fg_banks.add_to(m); cluster.add_to(m)
    fg_bus.add_to(m); fg_sub.add_to(m)
    MiniMap(toggle_display=True, minimized=True).add_to(m)
    Fullscreen().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    _add_corner_legend_transport(m)
    return m

# =========================
# 4) Shiny UI — 상단 탭 + 사이드 (맵 + Top5 막대)
# =========================
explain_transport = """
<div style='max-width:420px; font-size:12.5px; line-height:1.5;'>
  <b>1) 각 은행 지점 별 반경 500m 이내 버스정류장 수 및 지하철 역수 도출</b><br>
  - 고령층 보행속도(0.8~0.9 m/s)고려 → 도보 10분 ≈ 480~540m ⇒ 반경 500m<br>
  - 위/경도 기반 하버사인 거리로 반경 내 대중교통 인프라 집계
  <br><br>
  <b>2) 대중교통 접근성 합산 지수</b><br>
  지수 = Scaling(sqrt(경유노선수)) + Scaling(지하철역 수)<br>
  - 경유노선수에 제곱근 적용(큰 정류장 영향 확대)<br>
  - (0~1) 스케일링 후 합산
  <br><br>
  <b>3) 1~10 스케일로 리스케일(스코어화)</b>
</div>
"""

explain_welfare = """
<div style='max-width:420px; font-size:12.5px; line-height:1.5;'>
  <b>1) 반경 500m 이내 경로당·노인복지센터 집계</b><br>
  - 사회복지시설 계획 지침(근린생활권 5~10분) 참고 → 500m 기준
  <br><br>

  <b>2) KDE(커널밀도추정) 기반 스코어링</b><br>
  - 각 지점을 중심으로 500m 커널함수를 씌워 연속 밀도 표면 생성<br>
  - 복지센터에 경로당 대비 가중치 10 적용, 밀집 효과 반영
  <br><br>

  <b>기법 선정 근거</b><br>
  - 시니어 시설이 많고, 가까이 있을수록 높은 점수 부여<br>
  - 밀집이 높을수록 인프라 시설 간 네트워크 시너지 반영
  <br><br>

  <b>3) 1~10 스케일로 리스케일</b>
</div>
"""

@module.ui
def tab_app2_ui():
    # Keep structure similar to app2.py: left options + right map/plot cards
    return ui.page_fluid(
    ui.tags.style("""
    .container-fluid { max-width: 100% !important; }

    /* 맵/그래프 높이 492px */
    .card iframe { height: 492px !important; width: 100% !important; border: 0; }
    .leaflet-container, .folium-map, .html-widget {
        min-height: 492px !important;
        width: 100% !important;
    }
    .card { width: 100% !important; }

    /* 사이드바 카드 내부 컨텐츠 폭 90% */
    .sidebar-card .card-body { width: 90%; margin: 0 auto; }
    """),

    ui.h5("지점 별 스코어 - 대중교통 및 노인복지 인프라"),

    ui.navset_tab(
        ui.nav_panel(
            "교통스코어 맵",
            ui.layout_columns(
                # 좌측 사이드
                ui.card(
                    ui.card_header("교통 · 옵션"),
                    ui.input_checkbox("only_within_t", "반경 이내 요소만 표시", True),
                    ui.input_slider("traffic_pct", "은행 지점 교통스코어 분위(%)", 0, 100, (0, 100)),
                    ui.input_action_button("apply_t", "선택 구간만 표시"),
                    ui.br(),
                    ui.input_action_button("btn_explain_t", "교통스코어 설명 보기"),
                    ui.output_ui("popup_t"),
                    style="min-height: 492px;",
                    class_="sidebar-card"
                ),
                # 우측(맵 + Top5 막대)
                ui.div(
                    ui.card(
                        ui.card_header("교통 스코어 맵"),
                        ui.div(ui.output_ui("traffic_map_ui"), style="height: 492px;"),
                        ui.output_ui("traffic_legend_ui"),
                        full_screen=True
                    ),
                    ui.card(
                        ui.card_header("행정동 Top5 (선택 구간 기준)"),
                        ui.output_plot("traffic_top5_plot", height="492px"),
                        full_screen=True
                    ),
                    style="display:flex; flex-direction:column; gap:0.75rem; width:100%;"
                ),
                col_widths=[3, 7],
                gap="0.75rem"
            )
        ),
        ui.nav_panel(
            "복지스코어 맵",
            ui.layout_columns(
                # 좌측 사이드
                ui.card(
                    ui.card_header("복지 · 옵션"),
                    ui.input_checkbox("only_within_w", "반경 이내 요소만 표시", True),
                    ui.input_slider("welfare_pct", "은행 지점 복지스코어 분위(%)", 0, 100, (0, 100)),
                    ui.input_action_button("apply_w", "선택 구간만 표시"),
                    ui.br(),
                    ui.input_action_button("btn_explain_w", "복지스코어 설명 보기"),
                    ui.output_ui("popup_w"),
                    style="min-height: 492px;",
                    class_="sidebar-card"
                ),
                # 우측(맵 + Top5 막대)
                ui.div(
                    ui.card(
                        ui.card_header("복지 스코어 맵"),
                        ui.div(ui.output_ui("welfare_map_ui"), style="height: 492px;"),
                        ui.output_ui("welfare_legend_ui"),
                        full_screen=True
                    ),
                    ui.card(
                        ui.card_header("행정동 Top5 (선택 구간 기준)"),
                        ui.output_plot("welfare_top5_plot", height="492px"),
                        full_screen=True
                    ),
                    style="display:flex; flex-direction:column; gap:0.75rem; width:100%;"
                ),
                col_widths=[3, 7],
                gap="0.75rem"
            )
        )
    )
)

@module.server
def tab_app2_server(input, output, session):
# 설명 팝업 토글
    show_t = reactive.Value(False)
    show_w = reactive.Value(False)

    # 적용된 분위 구간(버튼 클릭으로만 갱신)
    applied_range_t = reactive.Value((0, 100))
    applied_range_w = reactive.Value((0, 100))

    @reactive.Effect
    @reactive.event(input.btn_explain_t)
    def _toggle_t():
        show_t.set(not show_t())

    @reactive.Effect
    @reactive.event(input.btn_explain_w)
    def _toggle_w():
        show_w.set(not show_w())

    # 구간 적용 버튼
    @reactive.Effect
    @reactive.event(input.apply_t)
    def _apply_t():
        lo, hi = input.traffic_pct()
        applied_range_t.set((lo, hi))

    @reactive.Effect
    @reactive.event(input.apply_w)
    def _apply_w():
        lo, hi = input.welfare_pct()
        applied_range_w.set((lo, hi))

    def popup_html(inner_html: str):
        # 팝업 15% 확대 + 스크롤 대비
        return f"""
        <div style="
            position:fixed; right:18px; bottom:18px; z-index:9999;
            background:#ffffff; border:1px solid #ddd; border-radius:10px;
            padding:12px 14px; box-shadow:0 2px 10px rgba(0,0,0,0.15);
            transform: scale(1.15); transform-origin: bottom right;
            max-width: 520px; max-height: 70vh; overflow:auto;
        ">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <div style="font-weight:600; color:#333;">설명</div>
            <button onclick="this.parentElement.parentElement.style.display='none';"
                    style="border:none; background:transparent; font-size:16px;">✕</button>
          </div>
          {inner_html}
        </div>
        """

    @output
    @render.ui
    def popup_t():
        return ui.HTML(popup_html(explain_transport) if show_t() else "")

    @output
    @render.ui
    def popup_w():
        return ui.HTML(popup_html(explain_welfare) if show_w() else "")

    # ----- 맵 (적용된 구간에만 의존) -----
    @output
    @render.ui
    def traffic_map_ui():
        lo, hi = applied_range_t()
        m = build_traffic_map(
            only_within=input.only_within_t(),
            pct_range=(lo, hi),
        )
        return ui.HTML(m._repr_html_())

    @output
    @render.ui
    def welfare_map_ui():
        lo, hi = applied_range_w()
        m = build_welfare_map(
            only_within=input.only_within_w(),
            pct_range=(lo, hi),
        )
        return ui.HTML(m._repr_html_())

    # ----- 범례 -----
    @output
    @render.ui
    def traffic_legend_ui():
        has_col = (b_tsc is not None) and banks[b_tsc].notna().any()
        if not has_col:
            return ui.HTML("<div style='margin-top:6px; font-size:12px; color:#666;'>교통스코어 컬럼이 없어 범례를 표시할 수 없습니다.</div>")
        html = discrete_legend_html("교통스코어 색상 구간 (YlOrRd)", vmin_t, vmax_t, traffic_cm, IR_REVERSE, n_bins=5)
        return ui.HTML(html)

    @output
    @render.ui
    def welfare_legend_ui():
        has_col = (b_wsc is not None) and banks[b_wsc].notna().any()
        if not has_col:
            return ui.HTML("<div style='margin-top:6px; font-size:12px; color:#666;'>복지스코어 컬럼이 없어 범례를 표시할 수 없습니다.</div>")
        html = discrete_legend_html("복지스코어 색상 구간 (YlOrRd)", vmin_w, vmax_w, welfare_cm, IR_REVERSE, n_bins=5)
        return ui.HTML(html)

    # ----- 하단 Top5 막대 (선택 구간 기준, 행정동=읍면동) -----
    @output
    @render.plot
    def traffic_top5_plot():
        lo, hi = applied_range_t()
        df = percentile_filter(banks, b_tsc, lo, hi) if b_tsc else banks.iloc[0:0]
        return make_top5_admin_fig(df, "행정동 Top5 (교통스코어 선택 구간)")

    @output
    @render.plot
    def welfare_top5_plot():
        lo, hi = applied_range_w()
        df = percentile_filter(banks, b_wsc, lo, hi) if b_wsc else banks.iloc[0:0]
        return make_top5_admin_fig(df, "행정동 Top5 (복지스코어 선택 구간)")


# -----------------------------------------------------------------------------
# TAB 3 — Clone of app3.py (행정동 선택 지도 + 2개 Plotly 그래프)
# -----------------------------------------------------------------------------

# 정적 폴더 설정
WWW_DIR = Path(__file__).parent / "www"
WWW_DIR.mkdir(exist_ok=True)

# ---------- 파일 경로 ----------
SHAPE_PATH = "./data/대구_행정동_군위포함.shp"
CSV_PATH   = "./data/클러스터포함_전체.csv"

# ---------- 상수 ----------
NAN_COLOR       = "#BDBDBD"   # 초기/결측 채움색
BASE_OPACITY    = 0.4         # 초기 회색 채움 불투명도
DEFAULT_PALETTE = "YlGnBu"    # folium ColorBrewer 팔레트
DEFAULT_BINMODE = "quantile"  # "quantile" 또는 "equal"
DEFAULT_K       = 7           # 구간 수
DEFAULT_OPACITY = 0.75        # 선택 영역 채움 불투명도

# 창 높이 → 지도 높이 비율(필요시 이 값만 조정)
MAP_VH_RATIO    = 0.72
MIN_MAP_HEIGHT  = 480  # 너무 작아지지 않도록 최소값
RIGHT_TRIM_PX = 85  # ← 지도 하단 빈공간만큼 줄이기(필요시 10~20 사이 미세 조정)

# ---------- 유틸 ----------
def norm_name(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("（", "(").replace("）", ")")
    return s

def guess_and_to_wgs84(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if g.crs is None:
        g = g.set_crs(epsg=5179, allow_override=True)
    minx, miny, maxx, maxy = g.total_bounds
    looks_like_lonlat = (120 <= minx <= 140) and (20 <= miny <= 50)
    if looks_like_lonlat and ("4326" in str(g.crs)):
        return g
    try:
        return g.to_crs(epsg=4326)
    except Exception:
        g = g.set_crs(epsg=5179, allow_override=True)
        return g.to_crs(epsg=4326)

def compute_bins(series: pd.Series, mode: str, k: int) -> np.ndarray:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([0.0, 0.33, 0.66, 1.0])
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if np.isclose(vmin, vmax):
        eps = (1e-9 if vmax == vmin else (vmax - vmin) * 1e-6) or 1e-9
        return np.array([vmin - 2*eps, vmin - eps, vmin, vmin + eps])
    k = max(int(k), 3)
    if mode == "equal":
        bins = np.linspace(vmin, vmax, k + 1)
    else:
        qs = np.linspace(0, 1, k + 1)
        bins = np.unique(np.nanquantile(v, qs))
        if len(bins) < 4:
            bins = np.linspace(vmin, vmax, 4)
    if len(bins) < 4:
        bins = np.linspace(vmin, vmax, 4)
    return bins

# ---------- 데이터 로드 ----------
gdf = gpd.read_file(SHAPE_PATH)
if "동" not in gdf.columns and "ADM_DR_NM" in gdf.columns:
    gdf = gdf.rename(columns={"ADM_DR_NM": "동"})
if "행정동코드" not in gdf.columns and "ADM_DR_CD" in gdf.columns:
    gdf = gdf.rename(columns={"ADM_DR_CD": "행정동코드"})
gdf["동"] = gdf["동"].map(norm_name)
gdf = guess_and_to_wgs84(gdf)
try:
    gdf["geometry"] = gdf.buffer(0)
except Exception:
    pass

GDF_UNION = gpd.GeoDataFrame(geometry=[gdf.unary_union], crs=gdf.crs)

def read_metrics(path: str) -> pd.DataFrame:
    try:
        m = pd.read_csv(path, encoding="utf-8")
    except Exception:
        m = pd.read_csv(path, encoding="cp949")
    if "동" not in m.columns and "읍면동" in m.columns:
        m = m.rename(columns={"읍면동": "동"})
    if "동" not in m.columns and "ADM_DR_NM" in m.columns:
        m = m.rename(columns={"ADM_DR_NM": "동"})
    if "동" not in m.columns:
        raise ValueError("CSV 병합 키가 없습니다. '읍면동' 또는 '동' 컬럼이 필요합니다.")
    m["동"] = m["동"].map(norm_name)
    rename_map = {}
    for col in m.columns:
        c = str(col).strip()
        if c in ["포화도(%)", "포화_%", "saturation", "포화"]:
            rename_map[col] = "포화도"
        if c in ["고령인구비율(%)", "노령인구비율", "elderly_ratio", "고령화인구비율"]:
            rename_map[col] = "고령인구비율"
    if rename_map:
        m = m.rename(columns=rename_map)
    for c in ["포화도", "고령인구비율"]:
        if c in m.columns:
            m[c] = (
                m[c].astype(str)
                  .str.replace("%", "", regex=False)
                  .str.replace(",", "", regex=False)
            )
            m[c] = pd.to_numeric(m[c], errors="coerce")
    return m

metrics = read_metrics(CSV_PATH)
gdf = gdf.merge(metrics, on="동", how="left")

# ---------- UI ----------
all_dongs = sorted(gdf["동"].dropna().unique().tolist())
available_metrics = [c for c in ["포화도", "고령인구비율"] if c in gdf.columns]
metric_choices = ["(없음)"] + available_metrics
default_metric = "포화도" if "포화도" in available_metrics else (
    "고령인구비율" if "고령인구비율" in available_metrics else "(없음)"
)

@module.ui
def tab_app3_ui():
    return ui.page_sidebar(
    ui.sidebar(
        # ▼ 모두선택/모두해제 2버튼 (요청 스타일)
        ui.div(
            {"class": "btn-row"},
            ui.input_action_button("select_all_", "☑ 모두선택"),
            ui.input_action_button("clear_all", "☐ 모두해제"),
        ),

        ui.tags.details(
            {"id": "dong_details", "open": ""},  # ← 이 부분 추가
            ui.tags.summary("읍·면·동 선택"),
            ui.div(
                {"id": "dong_list_container",
                 "style": "max-height: 40vh; overflow:auto; border:1px solid #eee; padding:6px; border-radius:8px;"},
                ui.input_checkbox_group("dongs", None, choices=all_dongs, selected=[])
            )
        ),
        ui.hr(),
        ui.input_select("metric", "채색 지표", choices=metric_choices, selected=default_metric),
        ui.hr(),
        ui.input_action_button("btn_glossary", "ℹ️ 용어 설명"),

        # ui.p(ui.code(SHAPE_PATH), " → WGS84 변환 후 표시"),
        # ui.p(ui.code(CSV_PATH), " 의 ", ui.code("읍면동/동, 포화도, 고령인구비율"), " 사용"),
    ),

    # 본문 1행 2열 (6:6)
    ui.layout_columns(
        # [좌] 지도 카드
        ui.div(
            {"class": "card-box"},
            ui.div({"class": "card-title"}, "대구시 읍·면·동 선택 영역 지도"),
            ui.div({"class": "card-divider"}),   # ← 제목 아래 구분선
            ui.output_ui("map_container_dyn"),
        ),
        # [우] 그래프 카드 2개
        ui.div(
            {"style": "display:flex; flex-direction:column; gap:12px;"},
            ui.div(
                {"class": "card-box"},
                ui.div({"class": "card-title"}, "동별 고령인구비율"),
                ui.div({"class": "card-divider"}),
                ui.output_ui("plot_elderly"),
            ),
            ui.div(
                {"class": "card-box"},
                ui.div({"class": "card-title"}, "동별 지점당 인구수"),
                ui.div({"class": "card-divider"}),
                ui.output_ui("plot_saturation"),
            ),
        ),
        col_widths=[6, 6]
    ),

    # 스타일 + viewport 높이 전달 스크립트
    ui.tags.style("""
      /* --- 카드/제목/구분선 --- */
      .card-box {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 10px 12px 12px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
      }
      .card-title {
        background: #f7f8fa;       /* 제목 배경 연회색 */
        border: 1px solid #eef0f2;  /* 살짝 테두리 */
        border-radius: 8px;
        padding: 6px 10px;
        font-weight: 700;
        display: inline-block;      /* 내용 길이만큼 배경 */
        margin: 2px 0 8px 2px;
      }
      .card-divider {
        height: 1px;
        background: #e8eaee;       /* 제목 아래 구분선 */
        margin: 4px 0 10px 0;
      }

      /* --- 지도 컨테이너 --- */
      #map_container { min-height: 300px; position: relative; z-index: 0; overflow: hidden; }
      #map_container .folium-map,
      #map_container .leaflet-container,
      #map_container iframe,
      #map_container > div {
        height: 100% !important; width: 100% !important;
        position: relative !important; z-index: 0 !important; display: block;
      }

        /* --- 사이드바 --- */
        details > summary { cursor: pointer; font-weight: 600; margin: 0 0 6px 0; }
        #dong_list_container { background: #fff; }

        /* --- '모두선택/모두해제' 버튼 스타일 --- */
        .btn-row { display:flex; gap:10px; align-items:center; margin: 0 0 8px 0; }
        #select_all_, #clear_all {
            padding: 8px 14px !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 12px;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.08);
            min-width: 93px;
        }
        #select_all_ { background: #2196f3 !important; }         /* 파란색 */
        #select_all_:hover { background: #1e88e5 !important; }
        #clear_all { background: #f44336 !important; }          /* 빨간색 */
        #clear_all:hover { background: #e53935 !important; }
      
        #btn_glossary {
            padding: 8px 14px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            color: #fff !important;
            background: #64748b !important; /* slate */
            border: none !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.08);
            margin-bottom: 8px;
        }
        #btn_glossary:hover { background:#475569 !important; }
    """),
    ui.tags.script("""
        (function(){
            var lastH = -1, timer = null, DEBOUNCE_MS = 180;

            function nowH(){ return (window.innerHeight || document.documentElement.clientHeight || 0); }

            // 모듈 네임스페이스 자동 추적 (예: 'p1')
            function getNamespace(){
                // 모듈 내부에 확실히 존재하는 입력 id를 기준으로 추적 (여기선 체크박스 'dongs')
                var el = document.querySelector('[id$="-dongs"]');
                if (!el || !el.id) return "";
                return el.id.replace(/-dongs$/, ""); // 'p1-dongs' -> 'p1'
            }

            function sendVH(force){
                var h = nowH();
                if (!force && h === lastH) return;
                lastH = h;

                // 1) CSS 변수 --vh 업데이트
                document.documentElement.style.setProperty('--vh', (h * 0.01) + 'px');

                // 2) Shiny 입력 전송 (모듈 ns에 맞게)
                var ns = getNamespace();
                if (window.Shiny && Shiny.setInputValue){
                // 모듈 입력
                if (ns) Shiny.setInputValue(ns + '-viewport_h', h, {priority:'event'});
                // 혹시를 위해 루트 입력도 같이 보냄(무해)
                Shiny.setInputValue('viewport_h', h, {priority:'event'});
                }
            }

            function onResize(){
                clearTimeout(timer);
                timer = setTimeout(function(){ sendVH(false); }, DEBOUNCE_MS);
            }

            window.addEventListener('resize', onResize, {passive:true});
            window.addEventListener('orientationchange', function(){ setTimeout(function(){ sendVH(true); }, 200); }, {passive:true});
            document.addEventListener('DOMContentLoaded', function(){ sendVH(true); });
            setTimeout(function(){ sendVH(true); }, 150);
            })();
    """),

    title="대구시 고령인구비율 및 은행 지점당 인구수"
)

@module.server
def tab_app3_server(input, output, session):
    # --- 알림 헬퍼 (ms -> 초로 변환) ---
    def notify(msg: str, type_: str = "warning", duration_ms: int | None = 3500):
        # duration_ms=None 이면 사용자가 닫을 때까지 유지
        dur_sec = None if duration_ms is None else max(0.5, float(duration_ms) / 1000.0)
        try:
            ui.notification_show(msg, type=type_, duration=dur_sec)
        except Exception:
            pass


    # 현재 지표에서 값이 있는 동만 허용
    def allowed_dongs_for_metric(metric_name: str) -> list[str]:
        if metric_name in ["포화도", "고령인구비율"] and metric_name in gdf.columns:
            s = pd.to_numeric(gdf[metric_name], errors="coerce")
            return sorted(gdf.loc[s.notna(), "동"].unique().tolist())
        return sorted(gdf["동"].dropna().unique().tolist())

    # 지표 변경 시 체크박스 choices/selected 갱신 (결측 제외)
    @reactive.Effect
    def _refresh_dong_choices():
        metric = input.metric()
        allowed = allowed_dongs_for_metric(metric)
        current_selected = [d for d in (input.dongs() or []) if d in allowed]
        try:
            ui.update_checkbox_group("dongs", choices=allowed, selected=current_selected)
        except Exception:
            session.send_input_message("dongs", {"choices": allowed, "selected": current_selected})

    # ▶ 모두선택
    @reactive.Effect
    @reactive.event(input.select_all_)
    def _select_all():
        allowed = allowed_dongs_for_metric(input.metric())
        try:
            ui.update_checkbox_group("dongs", selected=allowed)
        except Exception:
            session.send_input_message("dongs", {"selected": allowed})

    # ▶ 모두해제
    @reactive.Effect
    @reactive.event(input.clear_all)
    def _clear_all():
        try:
            ui.update_checkbox_group("dongs", selected=[])
        except Exception:
            session.send_input_message("dongs", {"selected": []})
    
        # ▶ 지도 클릭 → 선택 토글
     # ▶ 지도 클릭 → 선택 토글 (값 없는 동 클릭 시 경고)
    @reactive.Effect
    @reactive.event(input.map_clicked_dong)
    def _toggle_from_map():
        evt = input.map_clicked_dong() or {}
        dong = evt.get("dong") if isinstance(evt, dict) else None
        if not dong:
            return

        metric = input.metric()
        allowed = allowed_dongs_for_metric(metric)  # 현재 지표에서 값이 있는 동만

        # 값이 없는 동을 클릭한 경우 → 알림 후 중단
        if dong not in allowed and metric in ["포화도", "고령인구비율"] and metric in gdf.columns:
            notify(f"'{dong}'에는 '{metric}' 값이 없어 선택할 수 없습니다.", type_="warning", duration_ms=3500)
            return

        # 정상 토글
        current = input.dongs() or []
        selected = set(current)
        if dong in selected:
            selected.remove(dong)
        else:
            selected.add(dong)

        # 허용 목록 내에서만 유지 + 원래 순서 보존
        selected = [d for d in allowed if d in selected] if (metric in ["포화도","고령인구비율"] and metric in gdf.columns) \
                   else [d for d in (sorted(gdf["동"].dropna().unique().tolist())) if d in selected]

        try:
            ui.update_checkbox_group("dongs", selected=selected)
        except Exception:
            session.send_input_message("dongs", {"selected": selected})

    # ▶ 용어 설명 모달 열기
    @reactive.Effect
    @reactive.event(input.btn_glossary)
    def _open_glossary():
        # 통상적 정의(간단 요약)
        desc = ui.tags.dl(
            ui.tags.dt("고령인구비율"),
            ui.tags.dd(
                "활용 데이터: 대구 광역시 동 별 고령인구 현황(전체 인구, 고령인구 수 → 고령인구 비율) 및 은행 지점별 주소",
                ui.tags.br(),
                "1. 은행 지점 별 주소 값에서 ‘구군’ 및 ‘읍면동’ 추출",
                ui.tags.br(),
                "2. 동 별 고령인구 현황 데이터의 행정구역에 매핑",
                ui.tags.br(),
                "3. 각 은행 지점에 대해 ‘구군’ , ‘행정동’, ‘고령인구비율’ 도출 ",
            ),
            ui.tags.dt("지점당 인구수"),
            ui.tags.dd(
                "활용 데이터: 은행 지점별 주소(’구군’, ‘읍면동’), 대구광역시 동 별 인구수",
                ui.tags.br(),
                "1. ‘고령인구비율’ 도출 시 추가했던 각 은행 지점의 행정동에 대해 행정동 별 은행 수 집계",
                ui.tags.br(),
                "2. 각 행정동 별로   → 전체 인구수/ 은행 수   로 도출",
                ui.tags.br(),
                "(해당 지수가 높을 경우 은행 수 대비 인구 수가 많아, 은행 방문 시 대기 시간이 길어지는 등 포화될 가능성이 높은 것으로 판단) "
            ),
        )

        ui.modal_show(
            ui.modal(
                ui.div({"style":"max-width:760px"},  # 팝업 폭 소형
                    desc,
                ),
                title="용어 설명",
                easy_close=True,
                footer=ui.div(
                    ui.input_action_button("glossary_close", "닫기")
                ),
                size="l"
            )
        )

    # ▶ 모달 닫기
    @reactive.Effect
    @reactive.event(input.glossary_close)
    def _close_glossary():
        try:
            ui.modal_remove()
        except Exception:
            pass

    def subset_by_dong(geo: gpd.GeoDataFrame, selected: list) -> gpd.GeoDataFrame:
        if not selected:
            return geo.iloc[0:0].copy()
        return geo[geo["동"].isin(selected)].copy()

    # -------- 동적 높이 계산 --------
    def current_map_height() -> int:
        raw = input.viewport_h()  # JS에서 setInputValue('viewport_h', ...)로 들어옴
        # 문자열/None 등 어떤 값이 와도 안전하게 숫자로 변환
        try:
            vh = float(raw) if raw is not None else None
        except Exception:
            vh = None
        base = vh if (vh is not None and vh > 0) else 900.0
        h = int(max(base * MAP_VH_RATIO, MIN_MAP_HEIGHT))
        return h
    
    def map_height_safe() -> int:
        # 존재하지 않을 수도 있는 입력을 안전하게 읽기
        raw = None
        try:
            # 모듈 네임스페이스에 있는 입력
            raw = input.viewport_h()
        except Exception:
            try:
                # 혹시 루트 입력으로 들어온 경우 (위 스크립트가 둘 다 보냄)
                raw = session.root_input.get("viewport_h")()  # 없으면 또 예외
            except Exception:
                raw = None

        # 숫자로 변환 + 폴백
        try:
            vh = float(raw) if raw is not None else None
        except Exception:
            vh = None

        base = vh if (vh is not None and vh > 0) else 900.0
        return int(max(base * MAP_VH_RATIO, MIN_MAP_HEIGHT))
        
    # -------- 지도 컨테이너(동적 높이) --------
    @output
    @render.ui
    def map_container_dyn():
        pct = int(MAP_VH_RATIO * 100)  # 예: 72
        # CSS max()로 최소 높이 보장 + 뷰포트 비율 적용
        return ui.div(
            {"id": "map_container",
            "style": f"height: max({MIN_MAP_HEIGHT}px, calc(var(--vh, 1vh) * {pct}));"},
            ui.output_ui("map_html")
    )
    # -------- 지도 생성 (folium → srcdoc) --------
    def build_map_html(selected: list[str]) -> str:
        metric   = input.metric()
        palette  = DEFAULT_PALETTE
        binmode  = DEFAULT_BINMODE
        k        = DEFAULT_K
        opacity  = DEFAULT_OPACITY

        m = folium.Map(location=[35.8714, 128.6014], zoom_start=11,
                       tiles="cartodbpositron", width="100%", height="100%")

        folium.GeoJson(
            data=GDF_UNION.__geo_interface__,
            name="기본(균일 연회색, 단일 다각형)",
            style_function=lambda f: {
                "fillColor": NAN_COLOR, "color": NAN_COLOR,
                "weight": 0, "fillOpacity": BASE_OPACITY
            },
            tooltip=None,
        ).add_to(m)

        folium.GeoJson(
            data=gdf.__geo_interface__,
            name="읍면동 경계",
            style_function=lambda f: {"fillOpacity": 0.0, "color": "#808080", "weight": 1.0},
            tooltip=folium.GeoJsonTooltip(fields=["동"], aliases=["동"]),
        ).add_to(m)

        gsel = subset_by_dong(gdf, selected)
        tb_src = (gsel if len(gsel) > 0 else gdf).total_bounds
        minx, miny, maxx, maxy = tb_src
        m.fit_bounds([[miny, minx], [maxy, maxx]])

        if len(gsel) > 0 and metric in gsel.columns and metric in ["포화도", "고령인구비율"]:
            s = pd.to_numeric(gsel[metric], errors="coerce")
            if s.notna().sum() > 0:
                bins = compute_bins(s, binmode, k)
                df_val = gsel[["동", metric]].copy()
                df_val[metric] = pd.to_numeric(df_val[metric], errors="coerce")
                ch = folium.Choropleth(
                    geo_data=gsel.__geo_interface__,
                    data=df_val,
                    columns=["동", metric],
                    key_on="feature.properties.동",
                    fill_color=palette,
                    fill_opacity=opacity,
                    line_opacity=0.8,
                    nan_fill_color=NAN_COLOR,
                    nan_fill_opacity=opacity,
                    bins=bins.tolist(),
                    legend_name=str(metric),
                    highlight=True,
                    name=f"{metric} (선택 영역)",
                )
                ch.add_to(m)
                ch.geojson.add_child(
                    folium.features.GeoJsonTooltip(
                        fields=["동", metric],
                        aliases=["동", metric],
                        localize=True,
                        sticky=True,
                        labels=True
                    )
                )
                # --- 전체 선택 시 상위 3개 빨간 테두리 강조 ---
                selected_list = selected or []
                try:
                    allowed_all = allowed_dongs_for_metric(metric)
                except Exception:
                    allowed_all = sorted(gdf["동"].dropna().unique().tolist())

                is_all_selected = bool(selected_list) and (set(selected_list) == set(allowed_all))

                if is_all_selected and metric in ["포화도", "고령인구비율"]:
                    # 중복/NaN 대비: 동별 평균 후 상위 3
                    df_rank = (
                        gsel[["동", metric]].copy()
                        .assign(**{metric: pd.to_numeric(gsel[metric], errors="coerce")})
                        .dropna(subset=[metric])
                        .groupby("동", as_index=False)[metric].mean()
                        .sort_values(metric, ascending=False)
                    )
                    if not df_rank.empty:
                        TOPN = 3
                        top_names = df_rank.head(TOPN)["동"].tolist()
                        g_top = gsel[gsel["동"].isin(top_names)].copy()

                        folium.GeoJson(
                            data=g_top.__geo_interface__,
                            name=f"{metric} 상위 {TOPN} 강조",
                            style_function=lambda f: {
                                "fillOpacity": 0.0,
                                "color": "#e53935",  # 빨강
                                "weight": 3.0,
                                "dashArray": None,
                            },
                            tooltip=None,
                        ).add_to(m)
                # --- 끝 ---

                # === [안전 라벨 블록] 선택한 동을 정확히 라벨링 (+진단 HUD) ===
                # === [안전 라벨 블록] 선택한 동 라벨링 (+진단 HUD) ===
                MAX_LABELS = 10           # 일부 선택 시 상한
                TOPN_ALL   = 10           # 전체 선택 시 정확히 이 개수만 라벨

                def _fix_geom(geom):
                    """가능하면 geometry를 고쳐서 반환 (buffer(0) 등)"""
                    if geom is None:
                        return None
                    try:
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                    except Exception:
                        pass
                    return geom

                def _biggest_poly(geom):
                    """MultiPolygon이면 가장 큰 폴리곤 선택"""
                    try:
                        if geom.geom_type == "MultiPolygon" and len(geom.geoms) > 0:
                            return max([g for g in geom.geoms if not g.is_empty], key=lambda g: g.area)
                    except Exception:
                        pass
                    return geom

                def _safe_rep_point_from(geom):
                    g = _fix_geom(geom)
                    if g is None or g.is_empty:
                        return None
                    try:
                        g = _biggest_poly(g)
                        return g.representative_point()
                    except Exception:
                        try:
                            return g.centroid
                        except Exception:
                            return None

                # 1) 전용 pane
                folium.map.CustomPane("labels").add_to(m)
                m.get_root().html.add_child(folium.Element("""
                <style>.leaflet-labels-pane { z-index: 700 !important; pointer-events: none; }</style>
                """))

                # 2) 전체선택 여부 판단
                selected_list = selected or []
                try:
                    allowed_all = allowed_dongs_for_metric(metric)
                except Exception:
                    allowed_all = sorted(gdf["동"].dropna().unique().tolist())
                is_all_selected = bool(selected_list) and (set(selected_list) == set(allowed_all))

                # 3) 라벨 대상으로 사용할 동 이름 목록(target_names) 결정
                if is_all_selected and metric in ["포화도", "고령인구비율"] and metric in gdf.columns:
                    # 전체선택이면 지표 기준 상위 TOPN_ALL만
                    df_all = (
                        gdf[gdf["동"].isin(allowed_all)][["동", metric]].copy()
                        .assign(**{metric: pd.to_numeric(gdf[metric], errors="coerce")})
                        .dropna(subset=[metric])
                        .groupby("동", as_index=False)[metric].mean()
                        .sort_values(metric, ascending=False)
                    )
                    target_names = df_all.head(TOPN_ALL)["동"].tolist()
                else:
                    # 일부 선택: 선택한 순서 유지, 최대 MAX_LABELS
                    target_names = (selected_list[:MAX_LABELS]) if selected_list else []

                # 4) 각 동별로 대표 지오메트리 확보 (gdf에서 가장 큰 폴리곤 1개)
                rows = gdf[gdf["동"].isin(target_names)].copy()
                # 같은 동이 여러 행이면 면적 큰 것 하나만
                rows["_area"] = rows["geometry"].apply(lambda g: getattr(g, "area", 0.0))
                rows = rows.sort_values("_area", ascending=False).drop_duplicates(subset=["동"], keep="first")

                # 5) 대표점 계산 (없으면 gsel에서 보완 시도)
                label_points = []
                missing_for = []
                for _, r in rows.iterrows():
                    nm = r["동"]
                    pt = _safe_rep_point_from(r["geometry"])
                    if (pt is None or pt.is_empty) and len(gsel) > 0:
                        # 선택 영역에서 해당 동 geometry 보완
                        try:
                            gg = gsel.loc[gsel["동"] == nm, "geometry"].values
                            if len(gg) > 0:
                                pt = _safe_rep_point_from(gg[0])
                        except Exception:
                            pass
                    if pt is None or pt.is_empty:
                        missing_for.append(nm)
                        continue
                    label_points.append((nm, pt))

                # 6) 라벨 폰트(선택 있으면 크게)
                label_font_px = 15 if selected_list else 12
                label_font_wt = 800 if selected_list else 700

                # 7) 라벨 생성 (선택한 동 이름의 순서를 최대한 유지)
                name_to_pt = {nm: pt for nm, pt in label_points}
                ordered = [nm for nm in target_names if nm in name_to_pt]
                for nm in ordered:
                    pt = name_to_pt[nm]
                    folium.Marker(
                        location=[pt.y, pt.x],
                        pane="labels",
                        icon=folium.DivIcon(html=f'''
                            <div style="
                                display:inline-block;
                                transform: translate(-50%, -100%);
                                font: {label_font_wt} {label_font_px}px/1.25 Malgun Gothic,AppleGothic,NanumGothic,'Noto Sans KR',Arial,sans-serif;
                                color:#111; background:rgba(255,255,255,.9);
                                padding:3px 7px; border-radius:5px; border:1px solid rgba(0,0,0,.18);
                                white-space:nowrap; pointer-events:none;
                                text-shadow:-1px -1px 0 #fff, 1px -1px 0 #fff,
                                            -1px  1px 0 #fff, 1px  1px 0 #fff;">
                                {nm}
                            </div>
                        ''')
                    ).add_to(m)

                # 8) HUD: 선택/라벨/보강실패
                sel_cnt = len(selected_list)
                lbl_cnt = len(ordered)
                miss_cnt = len(missing_for)
                hud_html = f"""
                <div class="map-hud">선택 {sel_cnt} / 라벨 {lbl_cnt}{' · 실패 '+str(miss_cnt)+'개' if miss_cnt else ''}</div>
                <style>
                .map-hud {{
                position:absolute; top:8px; right:8px; z-index:1000;
                background:rgba(255,255,255,.92); border:1px solid #e0e0e0;
                border-radius:6px; padding:4px 8px;
                font:600 11px/1.2 'Noto Sans KR', Arial; pointer-events:none;
                }}
                </style>
                """
                m.get_root().html.add_child(folium.Element(hud_html))

        # ===== [여기서부터 교체] 지도 HTML 추출 + 클릭 바인딩 주입 =====
        try:
            base_html = m.get_root().render()  # folium이 만든 HTML
        except Exception:
            base_html = ""  # 혹시 렌더 실패해도 안전하게

        inject_js = r"""
<style>
/* 폴리곤 위에 포인터 커서 */
.leaflet-interactive { cursor: pointer; }
/* 툴팁을 커서에서 살짝 띄워 라벨과 겹침 최소화 */
.leaflet-tooltip { 
  pointer-events: none;
  margin-left: 12px;   /* 오른쪽으로 */
  margin-top:  -12px;  /* 위로 */
  box-shadow: 0 1px 2px rgba(0,0,0,.25);
  opacity: .96;
  font-weight: 600;
}
.leaflet-tooltip-left  { margin-left: -12px; } /* Leaflet 방향 클래스 보정 */
.leaflet-tooltip-right { margin-left:  12px; }
.leaflet-tooltip-top   { margin-top:  -12px; }
.leaflet-tooltip-bottom{ margin-top:   12px; }
</style>
<script>
(function(){
  function getMap(){{
    var fmap=null;
    for (var k in window) {{
      if (k.indexOf('map_')===0 && window[k] && typeof window[k].eachLayer==='function') {{
        fmap = window[k];
      }}
    }}
    return fmap;
  }}

  function bindClicks(){{
    var m=getMap(); if(!m) return;

    function onClickFeature(e){{
      try {{
        var dong = e && e.target && e.target.feature && e.target.feature.properties && e.target.feature.properties["동"];
        if (!dong) return;
        var payload = {{ dong: dong, nonce: Date.now() }};
        if (window.parent && window.parent.Shiny && window.parent.Shiny.setInputValue) {{
          window.parent.Shiny.setInputValue("{ns_clicked}", payload, {{priority:"event"}});
        }}
      }} catch(err) {{ console.warn("click handler error:", err); }}
    }}

    function walk(layer){{
      if (!layer) return;
      if (layer.feature && typeof layer.on === 'function') layer.on('click', onClickFeature);
      if (typeof layer.eachLayer === 'function') layer.eachLayer(walk);
    }}

    m.eachLayer(walk);
    m.on('layeradd', function(ev){{ walk(ev.layer); }});
  }}

  if (document.readyState === 'complete') setTimeout(bindClicks, 0);
  else window.addEventListener('load', function(){{ setTimeout(bindClicks, 0); }});
})();
</script>
"""
        full_html = (base_html or "") + inject_js.replace("{ns_clicked}", session.ns("map_clicked_dong"))

        # www/folium_map.html 로 저장
        # fpath = WWW_DIR / "folium_map.html"
        # with open(fpath, "w", encoding="utf-8") as f:
        #     f.write(full_html)

        # # 캐시 방지용 쿼리스트링
        # import time as _time
        # nonce = int(_time.time() * 1000)

        # srcdoc 대신 src 사용 (★ 슬래시 없이 상대경로 권장)
        # return f'<iframe src="folium_map.html?v={nonce}" style="width:100%;height:100%;border:none;"></iframe>'
        safe = escape(full_html)
        return f'<iframe style="width:100%;height:100%;border:none;" srcdoc="{safe}"></iframe>'

    @output
    @render.ui
    def map_html():
        selected = input.dongs() or []
        return ui.HTML(build_map_html(selected))

    # -------- Plotly: 세로 막대 Top10 (동적 높이) --------
    def build_plotly_topN(metric_col: str, title_prefix: str, ylabel: str, height_px: int, selected: list[str], topn: int = 10):
        try:
            # ---------- 상수 ----------
            BAR_TEXT_SIZE = 18  # 막대 내부 수치 글자 크기
            if metric_col not in gdf.columns:
                fig = go.Figure()
                fig.update_layout(
                    title=f"'{metric_col}' 컬럼이 없습니다.",
                    height=height_px, margin=dict(l=10, r=10, t=48, b=10),
                    font=dict(family="Malgun Gothic, AppleGothic, NanumGothic, Noto Sans KR, Arial")
                )
                return fig

            geo = gdf[["동", metric_col]].copy()
            geo[metric_col] = pd.to_numeric(geo[metric_col], errors="coerce")
            geo = geo.dropna(subset=[metric_col])

            # ▼ 선택된 동만 사용 (선택이 없으면 전체)
            if selected:
                geo = geo[geo["동"].isin(selected)]

            if geo.empty:
                msg = "선택된 동에 유효한 데이터가 없습니다." if selected else "유효한 데이터가 없습니다."
                fig = go.Figure()
                fig.update_layout(
                    title=msg,
                    height=height_px, margin=dict(l=10, r=10, t=48, b=10),
                    font=dict(family="Malgun Gothic, AppleGothic, NanumGothic, Noto Sans KR, Arial")
                )
                return fig

            # 동 이름 중복 대비 → 평균 집계
            geo = geo.groupby("동", as_index=False)[metric_col].mean()

            # 0~1.5면 비율 판단 → % 표시
            s = geo[metric_col]
            is_ratio = (s.min() >= 0) and (s.max() <= 1.5)
            scale = 100.0 if is_ratio else 1.0
            disp_col = f"{metric_col}__disp"
            geo[disp_col] = s * scale

            # 상위 N만 남기기
            geo = geo.sort_values(disp_col, ascending=False)
            N = min(topn, len(geo))
            top = geo.head(N).reset_index(drop=True)

            # --- 전체 선택 여부 판단(해당 지표 기준) ---
            try:
                allowed_all = allowed_dongs_for_metric(metric_col)  # 현재 지표에서 유효한 전체 동
            except Exception:
                allowed_all = sorted(gdf["동"].dropna().unique().tolist())

            is_all_selected = bool(selected) and (set(selected) == set(allowed_all))
            is_none_selected = not selected

            highlight = is_all_selected or is_none_selected  # ⬅️ 두 경우 모두 강조
            TOP_HILITE = min(3, len(top))  # 데이터가 3 미만이면 있는 만큼만

            # 막대 색/외곽선 구성
            DEFAULT_BAR = "#636EFA"
            HILITE_FILL = "#e53935"
            HILITE_LINE = "#b71c1c"

            bar_colors = []
            line_colors = []
            line_widths = []
            for i in range(len(top)):
                if highlight and i < TOP_HILITE:
                    bar_colors.append(HILITE_FILL)
                    line_colors.append(HILITE_LINE)
                    line_widths.append(2.0)
                else:
                    bar_colors.append(DEFAULT_BAR)
                    line_colors.append("rgba(0,0,0,0)")
                    line_widths.append(0)

            # 동적 제목
            if selected:
                title = f"{title_prefix} (선택 {len(set(selected))}개 중 상위 {N})"
            else:
                title = f"{title_prefix} (전체 중 상위 {N})"

            fig = px.bar(
                top, x="동", y=disp_col, title=title,
                labels={"동": "동", disp_col: ylabel},
                text=top[disp_col].round(1)
            )
            fig.update_traces(textposition="inside")  # 텍스트를 막대 내부에 고정
            fig.update_traces(
                insidetextfont=dict(size=BAR_TEXT_SIZE, color="white"),
                outsidetextfont=dict(size=BAR_TEXT_SIZE, color="#111"),
            )
            fig.update_layout(uniformtext_minsize=BAR_TEXT_SIZE-2, uniformtext_mode="hide")
            fig.update_traces(
                marker_color=bar_colors,
                marker_line_color=line_colors,
                marker_line_width=line_widths,
            )
            fig.update_traces(
                hovertemplate="동=%{x}<br>"+ylabel+"=%{y:.1f}"+("%" if is_ratio else "")+"<extra></extra>",
                texttemplate="%{text:.1f}"+("%" if is_ratio else ""),
                cliponaxis=False
            )
            fig.update_layout(
                height=height_px,
                margin=dict(l=10, r=10, t=56, b=10),
                xaxis=dict(tickangle=-35, categoryorder="array", categoryarray=top["동"].tolist()),
                yaxis=dict(rangemode="tozero", ticksuffix=("%" if is_ratio else "")),
                font=dict(family="Malgun Gothic, AppleGothic, NanumGothic, Noto Sans KR, Arial"),
            )
            return fig

        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"그래프 오류: {e}",
                            height=height_px, margin=dict(l=10, r=10, t=48, b=10))
            return fig

    @output
    @render.ui
    def plot_elderly():
        map_h = map_height_safe()
        gap_between_cards = 12  # 우측 카드 사이 간격 (CSS와 일치)
        total_for_right = max(map_h - RIGHT_TRIM_PX, 0)
        height = max(int((total_for_right - gap_between_cards) / 2), 220)

        selected = input.dongs() or []
        fig = build_plotly_topN("고령인구비율", "동별 고령인구비율", "스코어", height, selected, topn=10)
        html = fig.to_html(full_html=False, include_plotlyjs="inline", config={"responsive": True})
        return ui.HTML(f'<div style="width:100%;height:{height}px;">{html}</div>')

    @output
    @render.ui
    def plot_saturation():
        map_h = map_height_safe()
        gap_between_cards = 12
        total_for_right = max(map_h - RIGHT_TRIM_PX, 0)
        height = max(int((total_for_right - gap_between_cards) / 2), 220)

        selected = input.dongs() or []
        fig = build_plotly_topN("포화도", "동별 지점당 인구수", "스코어", height, selected, topn=10)
        html = fig.to_html(full_html=False, include_plotlyjs="inline", config={"responsive": True})
        return ui.HTML(f'<div style="width:100%;height:{height}px;">{html}</div>')

# -----------------------------------------------------------------------------
# TAB 4 — 부록
# -----------------------------------------------------------------------------

# ====== UI ======
@module.ui
def tab_app4_ui():
    return ui.page_fluid(
        # 1행
        ui.layout_columns(
            ui.card(ui.card_header("1. 주제 선정 배경 및 필요성"),
                    ui.div(ui.output_ui("appendix_1")),   # scroll-body 제거
                    style="height:460px;"),
            ui.card(ui.card_header("2. 분석 개요"),
                    ui.div(ui.output_ui("appendix_2")),   # scroll-body 제거
                    style="height:460px;"),
            col_widths=[6,6]
        ),

        # 2행
        ui.layout_columns(
            ui.card(ui.card_header("3. 데이터 설명(출처, 데이터명, 비고)"),
                    ui.div(ui.output_ui("appendix_3")),   # scroll-body 제거
                    style="height:460px;"),
            ui.card(ui.card_header("4. Feature 4개 지표 산정식"),
                    ui.div(ui.output_image("appendix_4_img"), class_="img-box"), # scroll-body 제거
                    style="height:460px;"),
            col_widths=[6,6]
        ),

        # 3행
        ui.layout_columns(
            ui.card(ui.card_header("5. 타겟클러스터 선정 기준"),
                    ui.div(ui.output_ui("appendix_5")),   # scroll-body 제거
                    style="height:460px;"),
            ui.card(ui.card_header("6. 각 클러스터 별 정책제안 지점 도출 기준"),
                    ui.div(ui.output_ui("appendix_6")),   # scroll-body 제거
                    style="height:460px;"),
            col_widths=[6,6]
        ),
    )
# ====== Server ======
@module.server
def tab_app4_server(input, output, session):
    # --- 한글 폰트 자동 지정 ---
    def set_korean_font():
        from matplotlib import font_manager
        candidates = [
            "Malgun Gothic","MalgunGothic","AppleGothic",
            "NanumGothic","Noto Sans CJK KR","Noto Sans KR",
            "Arial Unicode MS"
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break
        else:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = candidates + list(
                matplotlib.rcParams.get("font.sans-serif", [])
            )
        matplotlib.rcParams["axes.unicode_minus"] = False
    set_korean_font()

    # --- 공통 집계 함수 (값은 1~10 스케일 그대로 사용) ---
    def _compute_cluster_stats(path="./data/통합추가.csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, encoding="cp949")

        s = (df["고령인구비율"].astype(str)
             .str.replace("%", "", regex=False)
             .str.replace(",", "", regex=False))
        df["고령인구비율"] = pd.to_numeric(s, errors="coerce")

        cluster_avg = (df.groupby("클러스터", dropna=True)["고령인구비율"]
                         .mean()
                         .reset_index()
                         .sort_values("클러스터"))
        median_value = df["고령인구비율"].median()
        target_clusters = cluster_avg.loc[
            cluster_avg["고령인구비율"] > median_value, "클러스터"
        ].tolist()
        return cluster_avg, median_value, target_clusters

    # ---- 1번 카드: 주제 선정 배경 및 필요성 ----
    @output
    @render.ui
    def appendix_1():
        return ui.markdown(
            """
### 배경
1. **전국 은행들의 오프라인 지점·ATM 감소 추세**  
   - *시사점:* 비대면 확산 속 오프라인 채널 축소 → **고령·저소득층 금융 접근성 리스크**

2. **대구의 고령화 현실(광역시 중 2위)**  
   - *시사점:* 대구는 고령화가 뚜렷 → **디지털 전환의 사각지대**가 생기기 쉬움

3. **세대간 디지털 금융 이용 격차**  
   - 최근 1개월 모바일 금융 이용경험: **20~40대 95%+ vs 60대+ 53.8% (2025년 3월 기준)**  
   - *시사점:* 세대 간 격차에 따른 **취약계층 맞춤 보완 채널** 필요

### 전국적 대응 현황
1) 일부 은행이 **시니어 특화 점포**를 운영 중이나, **서울·수도권 편중**  
2) 실제로 시니어 특화 점포가 절실한 곳은 **인구소멸/고령화 지역**에 더 많이 존재

### 분석 필요성
- 고령화가 가속화 중인 **대구시 고령층 금융 소외 해소** 및 **포용성 제고**  
- 은행과 지자체의 **정책적 협력**을 위한 **참고 기준**과 **수립 방안** 필요
            """
        )

    # ---- 2번 카드: 분석 개요 ----
    @output
    @render.ui
    def appendix_2():
        return ui.markdown(
            """
- ‘**시니어 특화 은행 서비스**’에 초점을 맞춰, 
**대구시 내 기존 은행 지점**의 입지 특성 변수 도출  
- 도출 변수 기반 **군집화(Clustering)** 및 **타겟 군집 선정**  
- 타겟 군집의 **입지 특징 도출** 및 특징에 따른 
**시니어 금융 서비스 전략** 설정  
- 각 타겟 군집 별 **전략 기반 벤치마킹 지점 도출**
(신규 입지 제안 아님)

> *입지 제안은 신규 위치 제안이 아닌, **기존 지점**을 분석해 벤치마킹 지점을 도출하는 방식*
            """
        )

    # ---- 3번 카드: (건드리지 않음, 그대로) ----
    @output
    @render.ui
    def appendix_3():
        rows = [
            {"src":"각 은행 지점 별 사이트", "src_sub":"(국민, 신한, 우리, 하나, 농협, DGB대구)",
             "name":"대구광역시 은행 지점", "name_sub":"(총 236개 지점)", "use":"은행명, 지점명, 주소"},
            {"src":"공공데이터포털", "src_sub":"", 
             "name":"대구광역시 시내버스 정류소 위치정보", "name_sub":"", "use":"버스정류소 행정코드, GPS, 경유노선"},
            {"src":"공공데이터포털", "src_sub":"", 
             "name":"국가철도공단-대구_지하철-주소데이터", "name_sub":"", "use":"지하철역명, 주소"},
            {"src":"공공데이터포털", "src_sub":"", 
             "name":"대구광역시_경로당", "name_sub":"", "use":"경로당명, 주소"},
            {"src":"대구광역시청", "src_sub":"", 
             "name":"대구광역시 노인여가복지시설", "name_sub":"", "use":"복지회관 기관명, 주소"},
            {"src":"행정안전부", "src_sub":"", 
             "name":"대구광역시 동별 고령인구 현황", "name_sub":"", "use":"행정기관(동), 전체인구, 65세이상인구"},
        ]

        header = ui.tags.thead(ui.tags.tr(
            ui.tags.th("출처"), ui.tags.th("데이터명"), ui.tags.th("활용 정보")
        ))

        def create_cell(main_text, sub_text=""):
            return ui.tags.td(main_text, ui.tags.br(), sub_text) if sub_text else ui.tags.td(main_text)

        body_rows = []
        for r in rows:
            src_cell  = create_cell(r["src"],  r.get("src_sub", ""))
            name_cell = create_cell(r["name"], r.get("name_sub", ""))
            use_cell  = create_cell(r["use"],  r.get("use_sub", ""))
            body_rows.append(ui.tags.tr(src_cell, name_cell, use_cell))

        table = ui.tags.table({"class":"nice-table"},
                              ui.tags.caption(ui.tags.span("데이터 설명", class_="chip"),
                                              "출처 · 데이터명 · 활용정보 요약"),
                              header, ui.tags.tbody(*body_rows))
        footnote = ui.tags.div(
            ui.tags.strong("※ 참고: "),
            "카카오맵 API를 활용하여 주소 정보를 위도·경도 좌표로 일괄 변환",
            class_="note"
        )
        return ui.div(table, footnote)

    # ---- 4번 카드: (건드리지 않음, 그대로) ----
    @output
    @render.image
    def appendix_4_img():
        return {
            "src": "./www/feature.png",
            "alt": "지표 산정식",
            "delete_file": False,
        }

    # ---- 5번 카드: (건드리지 않음, 그대로) ----
    @output
    @render.ui
    def appendix_5():
        try:
            cluster_avg, median_value, target_clusters = _compute_cluster_stats()
        except Exception as e:
            return ui.div(f"데이터 파일을 읽을 수 없습니다: {e}")

        explanation = ui.div(
            ui.tags.h4("타겟 클러스터 선정 기준", style="margin-top: 12px; color: #374151;"),
            ui.tags.p(
                f"'고령인구비율'이 기준값(중앙값: {median_value:.2f})을 초과하는 군집을 타겟으로 설정",
                style="line-height: 1.6; margin-bottom: 6px;"
            ),
            ui.tags.p(
                ui.tags.strong(
                    "타겟 클러스터: " +
                    (", ".join([f"{c}번" for c in target_clusters]) if target_clusters else "없음")
                ),
                style="color: #dc2626; font-size: 1.02em;"
            ),
            ui.tags.ul(
                *[
                    ui.tags.li(
                        f"클러스터 {int(c) if str(c).isdigit() else c}번: "
                        f"{cluster_avg.loc[cluster_avg['클러스터']==c,'고령인구비율'].iloc[0]:.2f}"
                    )
                    for c in target_clusters
                ],
                style="margin-top: 8px; line-height: 1.7;"
            ),
            style=("background: #f8fafc; padding: 12px; border-radius: 8px; "
                   "border-left: 4px solid #3b82f6; margin-top: 10px;")
        )

        return ui.div(
            ui.output_plot("cluster_age_plot", width="100%", height="340px"),
            explanation
        )

    @output
    @render.plot(alt="클러스터별 고령인구비율 평균")
    def cluster_age_plot():
        import matplotlib.pyplot as plt

        cluster_avg, median_value, target_clusters = _compute_cluster_stats()

        x = list(range(len(cluster_avg)))
        y = cluster_avg["고령인구비율"].to_list()
        labels = [f"{str(c)}번" for c in cluster_avg["클러스터"]]
        colors = ["#ff6b6b" if c in target_clusters else "#74c0fc"
                  for c in cluster_avg["클러스터"]]

        fig, ax = plt.subplots(figsize=(7.4, 3.2))
        bars = ax.bar(x, y, color=colors, edgecolor="#1f2937", linewidth=0.6)

        ax.axhline(median_value, linestyle="--", color="red", linewidth=1.3,
                   label=f"중앙값: {median_value:.2f}")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("고령인구비율")
        ax.set_xlabel("클러스터")
        ax.set_title("클러스터별 고령인구비율 평균", pad=10)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=9)

        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.20)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        return fig

    # ---- 6번 카드: 각 클러스터 별 정책제안 지점 도출 기준 ----
    @output
    @render.ui
    def appendix_6():
        return ui.markdown(
            """
### 0번 군집
- **특징:** 교통 불편, 노인복지 시너지 낮음, 고령비율 높음  
- **전략 및 근거:** *찾아가는 금융 서비스* 시행 지점 제안  
  - (전국적 방향성 참고) 시외지역 및 복지관 이용 어르신의 금융 편의 제고  
- **기준:** 대구 0번 클러스터 중 **외곽지역**(북부 군위군 / 남부 달성군) 거점 선정  
- **지점:**  
  - 북부(군위군) — *NH 농협은행 군위군지부*, *군위군(출)*  
  - 남부(달성군) — *NH농협은행 달성군청*, *iM뱅크 달성군청(출)*

---

### 5번 군집
- **특징:** 복지 시너지 **중간**, 교통 **좋음**, 고령비율 **매우 높음**, **지점당 인구수 낮음**  
- **전략 및 근거:** 현 거점 기준 **시니어 금융코너 확장** + **디지털 금융 교육존/공동 커뮤니티** 운영  
- **기준:** 복지 낮고, 교통 좋고, **지점당 인구수 ≤ 2.2**인 지역 타겟  
- **지점:** *우리은행 대구 3공단(비산 7동)*, *iM뱅크 성명(대명 10동)*

---

### 6번 군집
- **특징:** **복지·교통 우수**, 고령비율 높음, **지점당 인구수 높음**  
- **전략 및 근거:** **시니어 특화점포 개설**에 최적 (수요·접근성·복지 시너지 우수)  
- **기준:** 복지·교통·지점당 인구수·고령비율 **동일 가중치**, 상위 **3개** 선정  
- **지점:** *iM뱅크 방촌*, *iM뱅크 동구청*, *iM뱅크 달서구청(출)*
            """
        )


# -----------------------------------------------------------------------------
# Main UI/Server — stitch 3 tabs together
# -----------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(rel="icon", href="favicon.ico", type="image/x-icon"),
        ui.tags.link(rel="icon", href="favicon-32x32.png", type="image/png", sizes="32x32"),
        ui.tags.link(rel="icon", href="favicon-16x16.png", type="image/png", sizes="16x16"),
        ui.tags.link(rel="apple-touch-icon", href="apple-touch-icon.png", sizes="180x180"),
        ui.tags.link(rel="manifest", href="site.webmanifest"),
        ui.tags.meta(name="theme-color", content="#ffffff"),
        ui.tags.style("""
            .page-title { display: flex; align-items:center; gap:10px; margin:0 0 8px 0; }
            .page-title img { height:28px; width:auto; }
            @media (min-width:1280px){ .page-title img{ height:32px; } }
        """),
        ui.tags.link(rel="stylesheet", href="common.css"),
        ui.tags.link(rel="stylesheet", href="tab1.css"),
        ui.tags.link(rel="stylesheet", href="tab2.css"),
        ui.tags.link(rel="stylesheet", href="tab3.css"),
        ui.tags.link(rel="stylesheet", href="tab4.css"),
    ),
    ui.div(
        {"class": "page-title"},
        ui.tags.img(src="logo.png", alt="로고", loading="lazy", decoding="async"),
        ui.h2("대구지역 시니어 금융 서비스 전략 및 입지 제안"),
    ),
    ui.navset_tab(
        ui.nav_panel("지점별 서비스 전략 제안", tab_app1_ui("t1")),
        ui.nav_panel("지점별 교통/복지 스코어 비교", tab_app2_ui("t2")),
        ui.nav_panel("고령인구비율 및 은행 지점당 인구수", tab_app3_ui("t3")),
        ui.nav_panel("부록(기준 및 세부설명)", tab_app4_ui("t4")),
        id="main_tabs", selected="지점별 서비스 전략 제안"
    )
)


def server(input, output, session):
    tab_app1_server("t1")
    tab_app2_server("t2")
    tab_app3_server("t3")
    tab_app4_server("t4")


app = App(app_ui, server, static_assets=WWW_DIR)
