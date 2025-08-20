# app.py — Responsive height (map & plotly sync), folium map via srcdoc, 1:1 columns
import os, re
import geopandas as gpd
import pandas as pd
import numpy as np
import folium

from shiny import App, ui, render, reactive
from html import escape
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

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

# 카드 + 1:1 레이아웃
app_ui = ui.page_sidebar(
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
                ui.div({"class": "card-title"}, "동별 포화도"),
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
        var lastH = -1;            // 마지막으로 보낸 높이
        var timer = null;
        var DEBOUNCE_MS = 180;     // 필요시 조절

        function nowH(){
            return (window.innerHeight || document.documentElement.clientHeight || 0);
        }

        function sendVH(force){
            var h = nowH();
            // 높이가 바뀌지 않았으면 아무 것도 보내지 않음
            if (!force && h === lastH) return;
            lastH = h;
            if (window.Shiny && Shiny.setInputValue) {
            Shiny.setInputValue('viewport_h', h, {priority: 'event'});
            }
        }

        // 리사이즈를 디바운스해서 전송
        function onResize(){
            clearTimeout(timer);
            timer = setTimeout(function(){ sendVH(false); }, DEBOUNCE_MS);
        }

        window.addEventListener('resize', onResize, {passive:true});
        window.addEventListener('orientationchange', function(){ setTimeout(function(){ sendVH(true); }, 200); }, {passive:true});
        document.addEventListener('DOMContentLoaded', function(){ sendVH(true); });
        setTimeout(function(){ sendVH(true); }, 150);  // 초기 보정 한 번 더
        })();
    """),

    title="대구시 고령인구비율 및 은행 지점별 포화도"
)

# ---------- SERVER ----------
def server(input, output, session):
    
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
            ui.tags.dt("포화도"),
            ui.tags.dd(
                "활용 데이터: 은행 지점별 주소(’구군’, ‘읍면동’), 대구광역시 동 별 인구수",
                ui.tags.br(),
                "1. ‘고령인구비율’ 도출 시 추가했던 각 은행 지점의 행정동에 대해 행정동 별 은행 수 집계",
                ui.tags.br(),
                "2. 각 행정동 별로   → 전체 인구수/ 은행 수   로 포화도 도출",
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

    def subset_by_dong(df: gpd.GeoDataFrame, selected: list) -> gpd.GeoDataFrame:
        if not selected:
            return df.iloc[0:0].copy()
        return df[df["동"].isin(selected)].copy()

    # -------- 동적 높이 계산 --------
    def current_map_height() -> int:
        vh = input.viewport_h() or 900  # 초기 로딩 시 None일 수 있음
        h = int(max(vh * MAP_VH_RATIO, MIN_MAP_HEIGHT))
        return h

    # -------- 지도 컨테이너(동적 높이) --------
    @output
    @render.ui
    def map_container_dyn():
        h = current_map_height()
        return ui.div({"id": "map_container", "style": f"height:{h}px;"},
                      ui.output_ui("map_html"))

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
  // Folium이 만든 전역 map_XXXX를 찾아옴
  function getMap(){
    var fmap = null;
    for (var k in window) {
      if (k.indexOf('map_') === 0 && window[k] && typeof window[k].eachLayer === 'function') {
        fmap = window[k];
      }
    }
    return fmap;
  }

  function bindClicks(){
    var m = getMap();
    if(!m) return;

    // 클릭 시 '동'을 부모 Shiny로 전송
    function onClickFeature(e){
      try {
        var dong = e && e.target && e.target.feature && e.target.feature.properties && e.target.feature.properties["동"];
        if (!dong) return;
        var payload = { dong: dong, nonce: Date.now() }; // 강제 갱신용 nonce
        if (window.parent && window.parent.Shiny && window.parent.Shiny.setInputValue) {
          window.parent.Shiny.setInputValue("map_clicked_dong", payload, {priority:"event"});
        } else if (window.top && window.top.Shiny && window.top.Shiny.setInputValue) {
          window.top.Shiny.setInputValue("map_clicked_dong", payload, {priority:"event"});
        }
      } catch(err) {
        console.warn("click handler error:", err);
      }
    }

    // 레이어를 순회하며 GeoJSON 피처에 클릭 바인딩
    function walk(layer){
      if (!layer) return;
      if (layer.feature && typeof layer.on === 'function') {
        layer.on('click', onClickFeature);
      }
      if (typeof layer.eachLayer === 'function') {
        layer.eachLayer(walk);
      }
    }

    m.eachLayer(walk);
    // 이후 추가되는 레이어에도 적용
    m.on('layeradd', function(ev){ walk(ev.layer); });
  }

  if (document.readyState === 'complete') {
    setTimeout(bindClicks, 0);
  } else {
    window.addEventListener('load', function(){ setTimeout(bindClicks, 0); });
  }
})();
</script>
"""
        full_html = (base_html or "") + inject_js

        # www/folium_map.html 로 저장
        fpath = WWW_DIR / "folium_map.html"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(full_html)

        # 캐시 방지용 쿼리스트링
        import time as _time
        nonce = int(_time.time() * 1000)

        # srcdoc 대신 src 사용 (★ 슬래시 없이 상대경로 권장)
        return f'<iframe src="folium_map.html?v={nonce}" style="width:100%;height:100%;border:none;"></iframe>'


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

            df = gdf[["동", metric_col]].copy()
            df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
            df = df.dropna(subset=[metric_col])

            # ▼ 선택된 동만 사용 (선택이 없으면 전체)
            if selected:
                df = df[df["동"].isin(selected)]

            if df.empty:
                msg = "선택된 동에 유효한 데이터가 없습니다." if selected else "유효한 데이터가 없습니다."
                fig = go.Figure()
                fig.update_layout(
                    title=msg,
                    height=height_px, margin=dict(l=10, r=10, t=48, b=10),
                    font=dict(family="Malgun Gothic, AppleGothic, NanumGothic, Noto Sans KR, Arial")
                )
                return fig

            # 동 이름 중복 대비 → 평균 집계
            df = df.groupby("동", as_index=False)[metric_col].mean()

            # 0~1.5면 비율 판단 → % 표시
            s = df[metric_col]
            is_ratio = (s.min() >= 0) and (s.max() <= 1.5)
            scale = 100.0 if is_ratio else 1.0
            disp_col = f"{metric_col}__disp"
            df[disp_col] = s * scale

            # 상위 N만 남기기
            df = df.sort_values(disp_col, ascending=False)
            N = min(topn, len(df))
            top = df.head(N).reset_index(drop=True)

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
        h_map = current_map_height()
        height = max(int(h_map/2) - 56, 220)  # 카드 헤더/여백 보정
        selected = input.dongs() or []
        fig = build_plotly_topN("고령인구비율", "동별 고령인구비율", "고령인구비율(%)", height, selected, topn=10)
        html = fig.to_html(full_html=False, include_plotlyjs="inline", config={"responsive": True})
        return ui.HTML(f'<div style="width:100%;height:{height}px;">{html}</div>')


    @output
    @render.ui
    def plot_saturation():
        h_map = current_map_height()
        height = max(int(h_map/2) - 56, 220)
        selected = input.dongs() or []
        fig = build_plotly_topN("포화도", "동별 포화도", "포화도(%)", height, selected, topn=10)
        html = fig.to_html(full_html=False, include_plotlyjs="inline", config={"responsive": True})
        return ui.HTML(f'<div style="width:100%;height:{height}px;">{html}</div>')

app = App(app_ui, server, static_assets=WWW_DIR)




#shiny run app.py --reload --reload-excludes www\*