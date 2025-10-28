# =============================================================================
# 부산 외곽 주요 4개 지점 표시 (체크박스 선택 기능 포함)
# 실행 명령어: shiny run busanbank.py
# =============================================================================
from shiny import App, ui, render
import pandas as pd
import geopandas as gpd
import folium
import json

# -----------------------------------------------------------------------------
# 1️⃣ 데이터 로드
# -----------------------------------------------------------------------------
CSV_PATH = "./data/부산_타겟만.csv"
SHP_PATH = "./LSMD_부산/LSMD_ADM_SECT_UMD_26_202510.shp"

try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="cp949")

try:
    gdf = gpd.read_file(SHP_PATH, encoding="EUC-KR").to_crs(epsg=4326)
    BOUNDARY = json.loads(gdf.to_json())
except Exception as e:
    print(f"[ERROR] 부산 행정경계 로드 실패: {e}")
    BOUNDARY = None

# -----------------------------------------------------------------------------
# 2️⃣ 지도 설정
# -----------------------------------------------------------------------------
MAP_CENTER = [35.1796, 129.0756]  # 부산 중심
COLOR_MAIN = "red"
RADIUS_M = 10000

# -----------------------------------------------------------------------------
# 3️⃣ 대상 지점 (은행id 기준)
# -----------------------------------------------------------------------------
ID_NAME_MAP = {
    128: "NH농협은행 서동지점 (금정구)",
    7: "BNK부산은행 기장지점 (기장군)",
    24: "BNK부산은행 모라동지점 (사상구)",
    107: "KB국민은행 당리동지점 (사하구)",
}

# -----------------------------------------------------------------------------
# 4️⃣ UI
# -----------------------------------------------------------------------------
app_ui = ui.page_sidebar(
    # ==== 왼쪽 사이드바 ====
    ui.sidebar(
        ui.h4("지점 선택", class_="mt-3 mb-3"),
        ui.input_checkbox_group(
            "selected_points",
            None,
            choices=ID_NAME_MAP,
            selected=list(ID_NAME_MAP.keys()),  # 기본은 모두 선택
        ),
        ui.hr(),
        ui.p("※ 선택한 지점만 지도에 표시됩니다.", style="font-size:13px; color:#666;"),
        width="350px",
        open="desktop"
    ),

    # ==== 본문 ====
    ui.div(
        {
            "style": (
                "border:1px solid #ccc; border-radius:10px; "
                "box-shadow:0 2px 5px rgba(0,0,0,0.1); padding:6px;"
            )
        },
        ui.output_ui("busan_map")
    ),
)

# -----------------------------------------------------------------------------
# 5️⃣ SERVER
# -----------------------------------------------------------------------------
def server(input, output, session):
    @output
    @render.ui
    def busan_map():
        selected_ids = input.selected_points()
        if not selected_ids:
            selected_ids = []

        m = folium.Map(
            location=MAP_CENTER,
            zoom_start=10,
            tiles="cartodbpositron",
            width="100%",
            height="800px"
        )

        # ==== 행정경계 표시 ====
        if BOUNDARY is not None:
            folium.GeoJson(
                BOUNDARY,
                style_function=lambda x: {
                    'color': '#555',
                    'weight': 1.2,
                    'fillOpacity': 0,
                    'opacity': 0.8
                },
                interactive=False
            ).add_to(m)

        # ==== 선택된 지점만 표시 ====
        target_df = df[df["은행id"].isin([int(i) for i in selected_ids])]

        if not target_df.empty:
            for _, row in target_df.iterrows():
                lat = row.get("위도")
                lon = row.get("경도")
                if pd.isna(lat) or pd.isna(lon):
                    continue

                bank = str(row.get("은행", "") or "").strip()
                name = str(row.get("지점명", "") or "").strip()
                district = str(row.get("구군", "") or "").strip()
                dong = str(row.get("읍면동", "") or "").strip()

                tooltip_text = (
                    f"<b>{bank}</b><br>"
                    f"{name}<br>"
                    f"{district} {dong}"
                )

                folium.Circle(
                    location=[lat, lon],
                    radius=RADIUS_M,
                    color=COLOR_MAIN,
                    fill=True,
                    fill_color=COLOR_MAIN,
                    fill_opacity=0.08,
                    interactive=False
                ).add_to(m)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color=COLOR_MAIN,
                    fill=True,
                    fill_color=COLOR_MAIN,
                    fill_opacity=0.9,
                    tooltip=tooltip_text
                ).add_to(m)

        return ui.HTML(m._repr_html_())

# -----------------------------------------------------------------------------
# 6️⃣ Shiny 앱 객체
# -----------------------------------------------------------------------------
app = App(app_ui, server)
