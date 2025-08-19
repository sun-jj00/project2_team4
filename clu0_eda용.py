import json
import math
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

# =========================
# 0) 파일 경로
# =========================
SHP_FILE     = "emd.shp"                      # 읍면동 경계 (SHP)
BANK_FILE    = "타겟클러스터만.csv"             # 은행 지점(클러스터 포함)
WELFARE_FILE = "노인복지센터.csv"               # 노인복지센터
SENIOR_FILE  = "대구_경로당_구군동추가.csv"      # 경로당
BUS_FILE     = "대구_버스정류소_필터.csv"                 # 버스정류소
SUBWAY_FILE  = "대구_지하철_주소_좌표추가.csv"                   # 지하철역

# =========================
# 유틸: 안전 CSV 로더 + 위경도 표준화
# =========================
def read_csv_safe(path):
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 인코딩 자동 실패 시 기본 로드
    return pd.read_csv(path)

def normalize_latlon(df, lat_candidates=("위도","lat","latitude","LAT","Latitude"),
                          lon_candidates=("경도","lon","lng","longitude","LON","Longitude")):
    lat_name, lon_name = None, None
    for c in lat_candidates:
        if c in df.columns: lat_name = c; break
    for c in lon_candidates:
        if c in df.columns: lon_name = c; break
    if lat_name is None or lon_name is None:
        raise ValueError("위도/경도 컬럼명을 찾지 못했습니다. 파일의 컬럼명을 확인하세요.")
    if lat_name != "위도": df = df.rename(columns={lat_name: "위도"})
    if lon_name != "경도": df = df.rename(columns={lon_name: "경도"})
    return df

# =========================
# 1) 대구 읍면동 경계
# =========================
gdf = gpd.read_file(SHP_FILE, encoding="cp949")
if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)
gdf = gdf.reset_index(drop=True)
gdf["gid"] = gdf.index
geojson = json.loads(gdf.to_json())

# =========================
# 2) 데이터 불러오기
# =========================
bank_df    = normalize_latlon(read_csv_safe(BANK_FILE))
welfare_df = normalize_latlon(read_csv_safe(WELFARE_FILE))
senior_df  = normalize_latlon(read_csv_safe(SENIOR_FILE))
bus_df     = normalize_latlon(read_csv_safe(BUS_FILE))
subway_df  = normalize_latlon(read_csv_safe(SUBWAY_FILE))

# --- 클러스터 0만 필터 (문자/숫자 혼용 대비) ---
if "클러스터" not in bank_df.columns:
    raise KeyError("BANK_FILE에 '클러스터' 컬럼이 없습니다.")
bank0 = bank_df[bank_df["클러스터"].astype(str) == "0"].copy()

# =========================
# 3) 지도 생성
# =========================
fig = go.Figure()

# 읍면동 경계(외곽선만 보이도록 투명 채움)
fig.add_trace(
    go.Choroplethmapbox(
        geojson=geojson,
        locations=gdf["gid"],
        z=[1]*len(gdf),
        featureidkey="properties.gid",
        showscale=False,
        colorscale=[[0,"rgba(0,0,0,0)"], [1,"rgba(0,0,0,0)"]],
        marker_line_width=1.2,
        marker_line_color="yellow",
        zmin=0, zmax=1,
        name="읍면동 경계"
    )
)

# =========================
# 4) 포인트 레이어들
# =========================
# (A) 클러스터 0 은행 지점
fig.add_trace(
    go.Scattermapbox(
        lat=bank0["위도"],
        lon=bank0["경도"],
        mode="markers",
        marker=dict(size=9, color="red"),
        name="은행 지점 (클러스터 0)"
    )
)

# (B) 경로당
fig.add_trace(
    go.Scattermapbox(
        lat=senior_df["위도"],
        lon=senior_df["경도"],
        mode="markers",
        marker=dict(size=7, color="orange"),
        name="경로당"
    )
)

# (C) 노인복지센터
fig.add_trace(
    go.Scattermapbox(
        lat=welfare_df["위도"],
        lon=welfare_df["경도"],
        mode="markers",
        marker=dict(size=8, color="purple"),
        name="노인복지센터"
    )
)

# (D) 버스정류소
fig.add_trace(
    go.Scattermapbox(
        lat=bus_df["위도"],
        lon=bus_df["경도"],
        mode="markers",
        marker=dict(size=6, color="deepskyblue"),
        name="버스정류소"
    )
)

# (E) 지하철역
fig.add_trace(
    go.Scattermapbox(
        lat=subway_df["위도"],
        lon=subway_df["경도"],
        mode="markers",
        marker=dict(size=8, color="green"),
        name="지하철역"
    )
)

# =========================
# 5) 반경 800m 원 (은행 지점별)
# =========================
def create_circle_coords(lat, lon, radius_m=800, num_points=60):
    """
    중심좌표(lat, lon) 기준 반경 radius_m의 원형 경계좌표(위경도) 생성
    - 위도 1도 ≈ 111,320m
    - 경도 1도 ≈ 40075,000m * cos(lat) / 360
    """
    coords = []
    for k in range(num_points+1):
        angle = 2 * math.pi * k / num_points
        dlat = (radius_m / 111320.0) * math.cos(angle)
        dlon = (radius_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)) * math.sin(angle)
        coords.append((lat + dlat, lon + dlon))
    return coords

for _, r in bank0.iterrows():
    latlon = create_circle_coords(r["위도"], r["경도"], radius_m=800, num_points=80)
    lat_list, lon_list = zip(*latlon)
    fig.add_trace(
        go.Scattermapbox(
            lat=lat_list,
            lon=lon_list,
            mode="lines",
            line=dict(width=2, color="rgba(255,0,0,0.35)"),  # 빨강 반투명
            name="은행 반경 800m",
            hoverinfo="skip",
            showlegend=False  # 원이 많으면 범례는 1개만 두는 게 깔끔
        )
    )

# =========================
# 6) 레이아웃
# =========================
fig.update_layout(
    mapbox=dict(
        style="carto-positron",   # 필요시 'open-street-map'로 변경 가능(토큰 불필요)
        zoom=9,
        center={"lat": 35.8714, "lon": 128.6014},
    ),
    title_text="대구광역시: 클러스터 0 은행 지점 · 경로당 · 노인복지센터 · 버스정류소 · 지하철역 (은행 반경 800m)",
    title_x=0.5,
    legend=dict(y=0.5, x=1.02),
    margin=dict(l=10, r=10, t=50, b=10),
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.5, xanchor="center",
            y=1.08, yanchor="top",
            showactive=False,
            buttons=[
                dict(
                    label="도시(광역)",
                    method="relayout",
                    args=[{"mapbox.zoom": 8,  "mapbox.center": {"lat": 35.8714, "lon": 128.6014}}]
                ),
                dict(
                    label="구역(중간)",
                    method="relayout",
                    args=[{"mapbox.zoom": 10, "mapbox.center": {"lat": 35.8714, "lon": 128.6014}}]
                ),
                dict(
                    label="동네(근접)",
                    method="relayout",
                    args=[{"mapbox.zoom": 13, "mapbox.center": {"lat": 35.8714, "lon": 128.6014}}]
                ),
                dict(
                    label="초기화",
                    method="relayout",
                    args=[{"mapbox.zoom": 9,  "mapbox.center": {"lat": 35.8714, "lon": 128.6014}}]
                ),
            ]
        )
    ]
)

# 마우스 스크롤로 확대/축소 활성화
fig.show(config={"scrollZoom": True})