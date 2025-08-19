import re
import pandas as pd
from pathlib import Path

# ===== 파일 경로 =====
SUBWAY_CSV = "./data/대구_지하철_주소_좌표추가.csv"
BANK_CSV = "./data/대구은행_지점_위경도.csv"
SENIOR_POP_CSV = "./data/대구광역시 202507 동별 고령 인구현황.csv"  # columns: 동, 전체인구, 65세이상인구
SENIOR_CENTER_CSV = "./data/대구광역시_경로당_20231231.csv"  # columns: 경로당명, 주소, 동, (optional 위도,경도)
BUS_STOP_CSV = "./data/대구광역시_시내버스 정류소 위치정보_20240924.csv"  # columns: 정류소명, 동, 경도, 위도

# ===== 유틸: 인코딩 안전하게 CSV 읽기 =====
def load_csv_safe(path):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV 로드 실패: {path} (인코딩 확인 필요)")

# 1) '대구광역시', '대구시', '대구' 토큰을 먼저 제거
CITY_STRIP = re.compile(r'(?:^|\s)(대구광역시|대구시|대구)(?=\s)')

# 2) 구/군은 단어 경계(뒤에 공백/끝)일 때만 매칭 -> '수성구청' 같은 건 패스
GU_PATTERN = re.compile(r'([가-힣]{1,6}(?:구|군))(?=\s|$)')

# (참고) 동/읍/면/리/…가 패턴은 그대로 사용 가능
DONG_PATTERN = re.compile(r'([가-힣0-9]+?(?:동|[0-9]+가|가|읍|면|리))')

def extract_gu_dong(address: str):
    if not isinstance(address, str) or not address.strip():
        return (pd.NA, pd.NA)

    s = re.sub(r'\s+', ' ', address.strip())

    # === 여기 추가: 시(대구) 토큰 제거 ===
    s = CITY_STRIP.sub(' ', s)

    # 구/군 추출 (이제 '대구'는 건너뜀)
    g = GU_PATTERN.search(s)
    gu = g.group(1) if g else pd.NA

    # 괄호 안 법정동 우선
    dong = pd.NA
    for inner in re.findall(r'\((.*?)\)', s):
        cand = DONG_PATTERN.findall(inner)
        if cand:
            dong = cand[-1]

    # 본문에서 보조로 탐색
    if pd.isna(dong):
        search_area = s[g.end():] if g else s
        cand = DONG_PATTERN.findall(search_area)
        if cand:
            dong = cand[0]

    return (gu, dong)

def add_gu_dong_columns(df: pd.DataFrame, addr_col_candidates=("주소", "address", "도로명주소", "지번주소")):
    """주소 칼럼을 찾아 구/군, 동을 추가. 기존 '동'이 있으면 비어있는 값만 보충."""
    addr_col = next((c for c in addr_col_candidates if c in df.columns), None)
    if addr_col is None:
        # 주소가 없으면 스킵 (버스정류장 데이터처럼)
        return df, None

    out = df.copy()
    gu_dong = out[addr_col].apply(extract_gu_dong).apply(pd.Series)
    gu_dong.columns = ["구군_from_addr", "동_from_addr"]

    # 구군 추가
    if "구군" in out.columns:
        out["구군"] = out["구군"].fillna(gu_dong["구군_from_addr"])
    else:
        out["구군"] = gu_dong["구군_from_addr"]

    # 동 추가/보완
    if "동" in out.columns:
        out["동"] = out["동"].where(out["동"].notna() & (out["동"].astype(str).str.strip() != ""),
                                  gu_dong["동_from_addr"])
    else:
        out["동"] = gu_dong["동_from_addr"]

    return out, addr_col

def save_csv_excel_safe(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 엑셀에서 한글 깨짐 방지를 위해 utf-8-sig 권장
    df.to_csv(path, index=False, encoding="utf-8-sig")

# ===== 1) 지하철 =====
subway = load_csv_safe(SUBWAY_CSV)
subway_enriched, addr_col = add_gu_dong_columns(subway)
print(f"[지하철] 주소칼럼: {addr_col}, 추출결측(구군/동):",
      subway_enriched['구군'].isna().sum(), subway_enriched['동'].isna().sum())
save_csv_excel_safe(subway_enriched, "./out/대구_지하철_구군동추가.csv")

# ===== 2) 은행 지점 =====
bank = load_csv_safe(BANK_CSV)
bank_enriched, addr_col = add_gu_dong_columns(bank)
print(f"[은행] 주소칼럼: {addr_col}, 추출결측(구군/동):",
      bank_enriched['구군'].isna().sum(), bank_enriched['동'].isna().sum())
save_csv_excel_safe(bank_enriched, "./out/대구은행_지점_구군동추가.csv")

# ===== 3) 경로당 (주소, 동 존재) =====
senior_center = load_csv_safe(SENIOR_CENTER_CSV)
senior_center_enriched, addr_col = add_gu_dong_columns(senior_center)
print(f"[경로당] 주소칼럼: {addr_col}, 추출결측(구군/동):",
      senior_center_enriched['구군'].isna().sum(), senior_center_enriched['동'].isna().sum())
save_csv_excel_safe(senior_center_enriched, "./out/대구_경로당_구군동추가.csv")

# ===== 4) 고령인구 현황 (주소 없음: 동만 존재) -> 구군은 여기서 바로 못 만듦
senior_pop = load_csv_safe(SENIOR_POP_CSV)
if "구군" not in senior_pop.columns:
    senior_pop["구군"] = pd.NA  # 필요 시 별도 매핑 이용
save_csv_excel_safe(senior_pop, "./out/대구_고령인구_구군빈칸.csv")

# ===== 5) 버스정류장 (주소 없음: 동만 존재) -> 구군은 별도 매핑 필요
bus_stop = load_csv_safe(BUS_STOP_CSV)
if "구군" not in bus_stop.columns:
    bus_stop["구군"] = pd.NA  # 필요 시 동->구군 매핑 dict로 보완
save_csv_excel_safe(bus_stop, "./out/대구_버스정류장_구군빈칸.csv")
