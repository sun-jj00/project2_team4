# app.py
from shiny import App, ui, render
import pandas as pd
import matplotlib

# ====== 전역 CSS ======
global_css = ui.tags.style("""
.scroll-body { height: 420px; overflow: auto; padding-right: 8px; }
.scroll-body::-webkit-scrollbar { width: 8px; }
.scroll-body::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.25); border-radius: 8px; }

.nice-table { width:100%; border-collapse:separate; border-spacing:0; font-size:.95rem; line-height:1.45;
  background:#fff; border:1px solid #e6e6e6; border-radius:14px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,.04);}
.nice-table caption { caption-side:top; text-align:left; padding:14px 16px 6px; font-weight:700; font-size:1.05rem; color:#374151;}
.nice-table thead th { position:sticky; top:0; background:#f8e9f4; color:#4a154b; font-weight:700; letter-spacing:.2px;
  padding:12px 14px; border-bottom:1px solid #e8d2e3; white-space:nowrap;}
.nice-table tbody td { padding:12px 14px; vertical-align:top; border-bottom:1px dashed #efefef; }
.nice-table tbody tr:nth-child(odd) td { background:#fcfcfe; }
.nice-table tbody tr:hover td { background:#f7fbff; }

.note { margin-top:10px; background:#fff8e6; border:1px solid #ffe3a3; color:#7a5d00; padding:10px 12px; border-radius:10px; font-size:.9rem; }
.chip { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:600; font-size:.82rem; margin-right:6px; border:1px solid #e5e7eb; line-height:1.6; }

/* ✅ 4번 카드 이미지가 카드 영역(.scroll-body) 안에 '완전히' 들어오도록 */
.img-box {
  width: 100%;
  height: 100%;           /* .scroll-body 높이(420px)를 꽉 채움 */
  display: flex;
  align-items: center;    /* 세로 중앙 */
  justify-content: center;/* 가로 중앙 */
}
.img-box img {
  max-width: 100%;
  max-height: 100%;       /* 가로·세로 어느 쪽도 넘치지 않게 */
  width: auto;            /* 원본 비율 유지 */
  height: auto;           /* 원본 비율 유지 */
  display: block;
}
""")

# ====== UI ======
app_ui = ui.page_fluid(
    global_css,
    ui.navset_tab(
        ui.nav_panel(
            "부록(기준 및 세부설명)",

            # 1행
            ui.layout_columns(
                ui.card(ui.card_header("1. 주제 선정 배경 및 필요성"),
                        ui.div(ui.output_ui("appendix_1"), class_="scroll-body"),
                        style="height:460px;"),
                ui.card(ui.card_header("2. 분석 개요"),
                        ui.div(ui.output_ui("appendix_2"), class_="scroll-body"),
                        style="height:460px;"),
                col_widths=[6,6]
            ),

            # 2행
            ui.layout_columns(
                ui.card(ui.card_header("3. 데이터 설명(출처, 데이터명, 비고)"),
                        ui.div(ui.output_ui("appendix_3"), class_="scroll-body"),
                        style="height:460px;"),
                ui.card(ui.card_header("4. Feature 4개 지표 산정식"),
                        # ✅ 변경: 이미지 출력물을 .img-box로 감싸 자동 축소/맞춤
                        ui.div(ui.div(ui.output_image("appendix_4_img"), class_="img-box"), class_="scroll-body"),
                        style="height:460px;"),
                col_widths=[6,6]
            ),

            # 3행
            ui.layout_columns(
                ui.card(ui.card_header("5. 타겟클러스터 선정 기준"),
                        ui.div(ui.output_ui("appendix_5"), class_="scroll-body"),
                        style="height:460px;"),
                ui.card(ui.card_header("6. 각 클러스터 별 정책제안 지점 도출 기준"),
                        ui.div(ui.output_ui("appendix_6"), class_="scroll-body"),
                        style="height:460px;"),
                col_widths=[6,6]
            ),
        )
    )
)

# ====== Server ======
def server(input, output, session):

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
    def _compute_cluster_stats(path="통합추가.csv"):
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
            "src": "feature.png",
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
- **특징:** 복지 시너지 **중간**, 교통 **좋음**, 고령비율 **매우 높음**, **포화도 낮음**  
- **전략 및 근거:** 현 거점 기준 **시니어 금융코너 확장** + **디지털 금융 교육존/공동 커뮤니티** 운영  
- **기준:** 복지 낮고, 교통 좋고, **포화도 ≤ 2.2**인 지역 타겟  
- **지점:** *우리은행 대구 3공단(비산 7동)*, *iM뱅크 성명(대명 10동)*

---

### 6번 군집
- **특징:** **복지·교통 우수**, 고령비율 높음, **포화도 높음**  
- **전략 및 근거:** **시니어 특화점포 개설**에 최적 (수요·접근성·복지 시너지 우수)  
- **기준:** 복지·교통·포화도·고령비율 **동일 가중치**, 상위 **3개** 선정  
- **지점:** *iM뱅크 방촌*, *iM뱅크 동구청*, *iM뱅크 달서구청(출)*
            """
        )

# ====== App ======
app = App(app_ui, server)
