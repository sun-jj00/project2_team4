import pandas as pd
from shiny import App, render, ui, reactive, Session
from shinywidgets import output_widget, render_widget
import folium
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import json
import geopandas as gpd # GeoPandas 라이브러리 추가

# =========================
# 1. 데이터 및 차트 준비
# =========================
# CSV 파일을 안전하게 읽어옵니다.
try:
    df = pd.read_csv('클러스터포함_전체.csv', encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv('클러스터포함_전체.csv', encoding='cp949')

# GeoJSON 데이터 로드 (로컬 Shapefile 사용)
try:
    # GeoPandas로 Shapefile 읽기 (8개 파일이 모두 같은 폴더에 있어야 함)
    gdf = gpd.read_file('대구_행정동_군위포함.shp', encoding='utf-8')
    # 경고 메시지 해결을 위해 표준 좌표계(WGS84)로 변환
    gdf = gdf.to_crs(epsg=4326)
    # GeoPandas DataFrame을 GeoJSON으로 변환
    daegu_boundary = json.loads(gdf.to_json())
except Exception as e:
    print(f"GeoJSON 데이터를 불러오는 데 실패했습니다: {e}")
    daegu_boundary = None

# 데이터 타입 변환
df['은행id'] = pd.to_numeric(df['은행id'], errors='coerce')
df['정책제안클러스터'] = pd.to_numeric(df['정책제안클러스터'], errors='coerce')

# --- 클러스터 이름 및 색상 정의 ---
CLUSTER_NAMES = {
    0: "교통·복지 취약 고령지역 지점",
    5: "교통우수 초고령지역 지점",
    6: "교통·복지 우수 고령밀집지역 지점"
}
CLUSTER_COLORS = {
    0: {'line': 'blue', 'fill': 'rgba(0, 0, 255, 0.1)'},
    5: {'line': 'green', 'fill': 'rgba(0, 128, 0, 0.1)'},
    6: {'line': 'red', 'fill': 'rgba(255, 0, 0, 0.1)'}
}

# --- 레이더 차트 설정 시작 ---
METRICS = ["교통스코어", "복지스코어", "고령인구비율", "포화도"]
for c in METRICS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 사분위수 계산
quartiles = {}
for m in METRICS:
    s = df[m].dropna().astype(float).values
    q1 = float(np.quantile(s, 0.25))
    q2 = float(np.quantile(s, 0.50))
    q3 = float(np.quantile(s, 0.75))
    quartiles[m] = {"Q1": q1, "Q2": q2, "Q3": q3 if q3 != 0 else 1e-9}

q1_bar = float(np.mean([quartiles[m]["Q1"] / quartiles[m]["Q3"] for m in METRICS]))
q2_bar = float(np.mean([quartiles[m]["Q2"] / quartiles[m]["Q3"] for m in METRICS]))
q3_bar = 1.0

# 클러스터별 평균 계산
cluster_means = df.groupby('클러스터')[METRICS].mean(numeric_only=True)

def normalize_row_to_q3(row):
    vals = []
    for m in METRICS:
        v = row[m]
        q3 = quartiles[m]["Q3"]
        vals.append(min(max(float(v) / q3, 0.0), 1.0) if pd.notna(v) else quartiles[m]["Q2"] / q3)
    return vals

# 다중 클러스터를 받는 레이더 차트 생성 함수
def make_square_radar(cids):
    fig = go.Figure()
    
    # Q1, Q2, Q3 링 (회색 및 스타일 변경, legendgroup 추가)
    fig.add_trace(go.Scatterpolar(r=[q3_bar]*361, theta=np.linspace(0, 360, 361), mode="lines", line=dict(color="darkgrey", width=2.5, dash='dot'), name="Q3", legendgroup="1"))
    fig.add_trace(go.Scatterpolar(r=[q1_bar]*361, theta=np.linspace(0, 360, 361), mode="lines", line=dict(color="grey", width=2, dash="dash"), name="Q1", legendgroup="1"))
    fig.add_trace(go.Scatterpolar(r=[q2_bar]*361, theta=np.linspace(0, 360, 361), mode="lines", line=dict(color="dimgray", width=3.0), name="Median", legendgroup="1"))

    angles_deg = np.linspace(0, 360, len(METRICS), endpoint=False).tolist()
    angles_deg_closed = angles_deg + [angles_deg[0]]

    # 선택된 각 클러스터에 대해 반복하여 차트에 추가
    for cid in cids:
        if cid not in cluster_means.index:
            continue

        values01 = normalize_row_to_q3(cluster_means.loc[cid])
        r_vals = values01 + [values01[0]]
        
        colors = CLUSTER_COLORS.get(cid, {'line': 'gray', 'fill': 'rgba(128,128,128,0.1)'})
        
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=angles_deg_closed, mode="lines+markers",
            line=dict(width=2, color=colors['line']),
            marker=dict(size=6, color=colors['line']),
            name=CLUSTER_NAMES.get(cid), fill="toself", fillcolor=colors['fill'],
            legendgroup="2" # 클러스터 범례 그룹 지정
        ))

    fig.update_polars(
        radialaxis=dict(range=[0, 1], showline=False, ticks="", showticklabels=False, showgrid=True, gridcolor="rgba(0,0,0,0.2)"),
        angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=angles_deg, ticktext=METRICS),
        gridshape="linear"
    )
    fig.update_layout(
        # traceorder를 'grouped'로 설정하고 범례 간격 조정
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, traceorder="grouped"),
        # 좌우 여백을 줄이고 높이를 키워 차트를 중앙에 더 크게 표시
        margin=dict(l=50, r=50, t=50, b=100),
        height=480,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="black")
    )
    return fig
# --- 레이더 차트 설정 끝 ---

# =========================
# 2. Shiny UI (사용자 인터페이스) 정의
# =========================
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
        ),
        ui.tags.style(".btn-group { display: flex; gap: 10px; margin-top: 10px; margin-bottom: 10px; } .color-box { display: inline-block; width: 12px; height: 12px; margin-right: 8px; vertical-align: middle; border: 1px solid #ccc; } .shiny-input-checkboxgroup .shiny-input-container { display: flex; flex-direction: column; } .checkbox { display: flex; align-items: center; } .checkbox label { flex-grow: 1; }")
    ),
    ui.h2("은행 입지 클러스터 대시보드"),
    ui.layout_sidebar(
        ui.sidebar(
            # "필터 옵션" 제목 제거
            ui.div(
                ui.input_action_button("select_all", "전체선택", class_="btn-sm btn-outline-primary"),
                ui.input_action_button("deselect_all", "전체해제", class_="btn-sm btn-outline-secondary"),
                class_="btn-group"
            ),
            ui.input_checkbox_group(
                "selected_clusters", "", # 제목 제거
                {
                    "0": ui.span(
                        ui.HTML(f'<span class="color-box" style="background-color: blue;"></span> {CLUSTER_NAMES[0]}'),
                        ui.tooltip(ui.tags.i(class_="bi bi-question-circle-fill ms-2"), ui.HTML("교통 불편<br>노인복지 시너지 낮음<br>고령비율 높음"), placement="right")
                    ),
                    "5": ui.span(
                        ui.HTML(f'<span class="color-box" style="background-color: green;"></span> {CLUSTER_NAMES[5]}'),
                        ui.tooltip(ui.tags.i(class_="bi bi-question-circle-fill ms-2"), ui.HTML("노인복지 시너지 중간<br>교통 좋음<br>고령비율 매우 높음"), placement="right")
                    ),
                    "6": ui.span(
                        ui.HTML(f'<span class="color-box" style="background-color: red;"></span> {CLUSTER_NAMES[6]}'),
                        ui.tooltip(ui.tags.i(class_="bi bi-question-circle-fill ms-2"), ui.HTML("노인복지, 교통 둘다 좋은편<br>고령비율 높음<br>포화도 높음"), placement="right")
                    ),
                },
                selected=[], inline=False
            ),
            ui.input_switch("policy_switch", "정책 제안만 보기", value=False),
            # 정책 설명 버튼 추가
            ui.input_action_button("show_policy", "정책 설명", class_="btn-sm btn-info w-100 mt-2"),
            width="350px",
            open="always"
        ),
        ui.div(
            ui.row(
                ui.column(7,
                    ui.card(
                        ui.card_header("은행 지점 지도"),
                        ui.div(
                            ui.output_ui("map_widget"),
                            style="position: relative; height: 100%;"
                        ),
                        style="height: 60vh;" # 카드 높이 지정
                    )
                ),
                ui.column(5,
                    ui.card(
                        ui.card_header("특징 비교"),
                        output_widget("radar_chart"),
                        style="height: 60vh;" # 카드 높이 지정
                    )
                )
            ),
            ui.card(
                ui.card_header(
                    "데이터 테이블",
                    ui.download_button("download_csv", "CSV 저장", class_="btn-sm btn-outline-primary float-end")
                ),
                ui.output_data_frame("data_table")
            ),
            ui.download_button("download_map", "지도 저장 (HTML)", class_="btn-primary w-100 mt-3")
        )
    )
)

# =========================
# 3. Shiny Server (앱 로직) 정의
# =========================
def server(input, output, session: Session):
    current_map = reactive.Value()

    @reactive.Effect
    @reactive.event(input.select_all)
    def _():
        ui.update_checkbox_group("selected_clusters", selected=["0", "5", "6"])

    @reactive.Effect
    @reactive.event(input.deselect_all)
    def _():
        ui.update_checkbox_group("selected_clusters", selected=[])
        
    # 정책 설명 팝업 로직
    @reactive.Effect
    @reactive.event(input.show_policy)
    def _():
        m = ui.modal(
            ui.h4("클러스터별 정책 제안"),
            ui.tags.hr(),
            ui.h5(CLUSTER_NAMES[0]),
            ui.p(ui.HTML('<b>찾아가는 금융서비스 시행 지점 제안</b><br>취지: 도내 시외지역 및 복지관 이용 어르신들의 금융편의<br>제안: 외곽지역(군위군, 달성군) 지점들을 <span style="background-color: rgba(0,0,255,0.2);">찾아가는 금융서비스</span> 거점으로 선정')),
            ui.h5(CLUSTER_NAMES[5]),
            ui.p(ui.HTML("<b>시니어 금융코너 확장 전략 실행에 최적</b><br>복지 시너지 강화를 위해, <span style='background-color: rgba(0,128,0,0.2);'>디지털 금융 교육존 및 은행 공동 커뮤니티 센터</span> 등 신규 설립 가능")),
            ui.h5(CLUSTER_NAMES[6]),
            ui.p(ui.HTML("<b>신규 점포 수요 및 시너지, 접근성 우수</b><br><span style='background-color: rgba(255,0,0,0.2);'>'시니어 특화점포 개설'</span>에 가장 최적의 군집으로 선정<br><span style='display: inline-block; width: 12px; height: 12px; background-color: yellow; border-radius: 50%; border: 1px solid #ccc; margin-right: 5px;'></span>대구 시니어특화점포 : iM뱅크 대봉브라보점")),
            title="정책 제안 상세 설명",
            easy_close=True,
            footer=ui.modal_button("닫기")
        )
        ui.modal_show(m)

    @reactive.Calc
    def filtered_data_full():
        # 이 함수는 행 필터링만 수행하고 모든 열을 반환합니다.
        base_df = df[df['클러스터'].isin([0, 5, 6])]
        
        if not input.selected_clusters(): 
            return base_df

        selected = [int(c) for c in input.selected_clusters()]
        filtered = df[df['클러스터'].isin(selected)].copy()
        
        if input.policy_switch():
            filtered = filtered[filtered['정책제안클러스터'] == filtered['클러스터']]
        
        return filtered

    @output
    @render.ui
    def map_widget():
        map_data = filtered_data_full()
        
        # 지도 배경(tiles) 변경
        daegu_map = folium.Map(location=[35.8714, 128.6014], zoom_start=11, tiles="cartodbpositron")

        if daegu_boundary:
            tooltip = folium.GeoJsonTooltip(
                fields=['ADM_DR_NM'],
                aliases=[''],
                style=('background-color: white; color: black; font-family: sans-serif; font-size: 10px; padding: 5px;')
            )
            # 경계선 색상(color) 변경
            folium.GeoJson(
                daegu_boundary,
                style_function=lambda x: {'color': '#808080', 'weight': 1.0, 'fillOpacity': 0, 'opacity': 0.7},
                tooltip=tooltip
            ).add_to(daegu_map)

        if not map_data.empty:
            for _, row in map_data.iterrows():
                color = 'yellow' if row['은행id'] == 31 else CLUSTER_COLORS[row['클러스터']]['line']
                tooltip_text = f"{row['은행']}<br>{row['지점명']}<br>{row['읍면동']}"
                folium.Circle(location=[row['위도'], row['경도']], radius=500, color=color, fill=True, fill_color=color, fill_opacity=0.15).add_to(daegu_map)
                folium.CircleMarker(location=[row['위도'], row['경도']], radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.8, tooltip=tooltip_text).add_to(daegu_map)
        
        current_map.set(daegu_map)
        return ui.HTML(daegu_map._repr_html_())
    
    @output
    @render_widget
    def radar_chart():
        if not input.selected_clusters():
            return go.Figure().update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', annotations=[dict(text="클러스터를 선택하여<br>특성을 비교하세요.", showarrow=False, font=dict(size=16, color="grey"))])
        cids = [int(c) for c in input.selected_clusters()]
        return make_square_radar(cids)

    @output
    @render.data_frame
    def data_table():
        # 표시할 칼럼 정의
        cols_to_show = ['은행id', '은행', '지점명', '주소', '구군', '읍면동', '복지스코어', '교통스코어', '고령인구비율', '포화도']
        data_to_display = filtered_data_full()
        return data_to_display[cols_to_show]

    @session.download(filename="filtered_data.csv")
    def download_csv():
        # 다운로드할 때도 동일한 열 선택 로직 적용
        cols_to_show = ['은행id', '은행', '지점명', '주소', '구군', '읍면동', '복지스코어', '교통스코어', '고령인구비율', '포화도']
        data_to_download = filtered_data_full()
        yield data_to_download[cols_to_show].to_csv(index=False, encoding='utf-8-sig')

    @session.download(filename="daegu_bank_map.html")
    def download_map():
        map_to_save = current_map.get()
        temp_file = Path(session.app.root) / "temp_map.html"
        map_to_save.save(str(temp_file))
        with open(temp_file, "rb") as f:
            yield f.read()

app = App(app_ui, server)
