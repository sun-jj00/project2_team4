# app.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
from branca.colormap import LinearColormap
from math import radians

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

from shiny import App, Inputs, Outputs, Session, ui, render, reactive

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
BANKS_FILE          = "통합추가.csv"
SENIOR_CENTER_FILE  = "노인복지센터.csv"
SENIOR_HALL_FILE    = "대구_경로당_구군동추가.csv"
BUS_FILE            = "대구_버스정류소_필터.csv"
SUBWAY_FILE         = "대구_지하철_주소_좌표추가.csv"

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

app_ui = ui.page_fluid(
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

    ui.h3("지점 별 스코어 - 대중교통 및 노인복지 인프라"),

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

# =========================
# 5) Shiny 서버
# =========================
def server(input: Inputs, output: Outputs, session: Session):

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

app = App(app_ui, server)
