import pandas as pd
import requests
import time

# ==== 설정 부분 ====
INPUT_FILE = "./out/대구_경로당_구군동추가.csv"  # 원본 CSV 파일명
OUTPUT_FILE = "대구_경로당_구군동추가.csv"                # 결과 CSV 파일명
API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"                # 카카오 REST API 키
ADDRESS_COLUMN = "주소"                               # 도로명 주소가 들어있는 컬럼명
# ===================

# CSV 읽기 (EUC-KR 인코딩)
# df = pd.read_csv(INPUT_FILE, encoding="euc-kr")
df = pd.read_csv(INPUT_FILE)
# df = pd.read_excel(INPUT_FILE)

# 위도, 경도 컬럼 추가
df["위도"] = None
df["경도"] = None

headers = {"Authorization": f"KakaoAK {API_KEY}"}

for idx, addr in enumerate(df[ADDRESS_COLUMN]):
    try:
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        params = {"query": addr}
        res = requests.get(url, headers=headers, params=params)
        
        if res.status_code == 200:
            result = res.json()
            if result["documents"]:
                df.at[idx, "경도"] = result["documents"][0]["x"]
                df.at[idx, "위도"] = result["documents"][0]["y"]
                print(f"[{idx+1}/{len(df)}] 변환 완료: {addr}")
            else:
                print(f"[{idx+1}/{len(df)}] 좌표 없음: {addr}")
        else:
            print(f"[{idx+1}/{len(df)}] 요청 실패: {addr} (HTTP {res.status_code})")

        time.sleep(0.1)  # API 호출 제한 방지 (초당 10회 이내)

    except Exception as e:
        print(f"[{idx+1}/{len(df)}] 오류 발생: {addr} - {e}")

# CSV 저장 (UTF-8 with BOM으로 저장하면 엑셀에서 한글 깨짐 방지)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUTPUT_FILE}")

# 위도, 경도
lat =35.8338511489387,
lon = 128.5706071307

url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"
params = {
    "x": lon,  # 경도
    "y": lat   # 위도
}
headers = {"Authorization": f"KakaoAK {API_KEY}"}


response = requests.get(url, params=params, headers=headers)
data = response.json()

print(data)


import os
import time
import json
import math
import requests
import pandas as pd
from pathlib import Path

# ============ 사용자 설정 ============
INPUT_PATH = "./은행_통합_복지스코어_교통스코어_병합.csv"   # CSV 또는 XLSX 경로
LAT_COL = "위도"   # 위도 컬럼명
LON_COL = "경도"   # 경도 컬럼명
SAVE_CSV = True
SAVE_XLSX = False  # True로 바꾸면 엑셀(xlsx)도 저장

# 카카오 REST API 키: 환경변수 또는 직접 문자열로 입력
KAKAO_API_KEY = "e1bfc7ab682fa7a1cde39d6fece2ac2a"  # 직접 넣으려면 "xxxxxxxxxxxxxxxx"로
SLEEP_SEC = 0.12     # 호출 간 간격(과금/제한 회피를 위해 안전하게 0.1~0.2초 권장)
ROUND_DIGITS = 6     # 좌표 반올림(중복요청 방지용)

# ============ 함수들 ============
def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"입력 파일 없음: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_csv(p)

def to_numeric_latlon(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    return df

def kakao_region_from_coord(lat: float, lon: float, session: requests.Session) -> dict:
    """
    Kakao coord2regioncode API로 좌표 → 행정구역명 조회.
    행정동(H) 우선, 없으면 법정동(B) 사용.
    반환: {"구군": str|None, "읍면동": str|None}
    """
    url = "https://dapi.kakao.com/v2/local/geo/coord2regioncode.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"x": lon, "y": lat, "input_coord": "WGS84"}
    r = session.get(url, headers=headers, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()

    docs = data.get("documents", []) or []
    # 행정동(H) 우선
    doc_h = next((d for d in docs if d.get("region_type") == "H"), None)
    doc_b = next((d for d in docs if d.get("region_type") == "B"), None)
    doc = doc_h or doc_b

    if not doc:
        return {"구군": None, "읍면동": None}

    gu_gun = doc.get("region_2depth_name")
    eup_myeon_dong = doc.get("region_3depth_name")
    return {"구군": gu_gun, "읍면동": eup_myeon_dong}

def main():
    if not KAKAO_API_KEY:
        raise RuntimeError(
            "KAKAO_REST_API_KEY가 설정되지 않았습니다. "
            "환경변수로 설정하거나 코드 상단 KAKAO_API_KEY에 직접 입력하세요."
        )

    df = load_table(INPUT_PATH)
    if LAT_COL not in df.columns or LON_COL not in df.columns:
        raise KeyError(f"'{LAT_COL}', '{LON_COL}' 컬럼을 찾을 수 없습니다. 실제 컬럼명을 확인하세요.")

    df = to_numeric_latlon(df, LAT_COL, LON_COL)

    # 좌표 쿼리 대상(유효 좌표만)
    coords = df[[LAT_COL, LON_COL]].dropna()
    if coords.empty:
        raise ValueError("유효한 위도/경도 값이 없습니다.")

    # 중복 요청 방지: 반올림 좌표로 unique 추출
    coords["lat_r"] = coords[LAT_COL].round(ROUND_DIGITS)
    coords["lon_r"] = coords[LON_COL].round(ROUND_DIGITS)
    unique_pairs = coords.drop_duplicates(subset=["lat_r", "lon_r"])[["lat_r", "lon_r"]].values.tolist()

    # API 호출
    mapping = {}  # (lat_r, lon_r) -> {"구군":..., "읍면동":...}
    sess = requests.Session()
    for i, (lat_r, lon_r) in enumerate(unique_pairs, start=1):
        try:
            res = kakao_region_from_coord(lat_r, lon_r, sess)
        except Exception as e:
            # 실패 시 None 채우고 다음 진행
            res = {"구군": None, "읍면동": None}
            # (선택) print(f"[{i}/{len(unique_pairs)}] 실패: {e}")
        mapping[(lat_r, lon_r)] = res
        time.sleep(SLEEP_SEC)

    # 원본 DF에 붙이기
    df["__lat_r__"] = df[LAT_COL].round(ROUND_DIGITS)
    df["__lon_r__"] = df[LON_COL].round(ROUND_DIGITS)
    df["구군"] = df.apply(lambda r: (mapping.get((r["__lat_r__"], r["__lon_r__"]), {})).get("구군"), axis=1)
    df["읍면동"] = df.apply(lambda r: (mapping.get((r["__lat_r__"], r["__lon_r__"]), {})).get("읍면동"), axis=1)
    df.drop(columns=["__lat_r__","__lon_r__"], inplace=True)

    # 저장
    in_path = Path(INPUT_PATH)
    stem = in_path.stem
    out_csv = in_path.with_name(stem + "_행정구역추가.csv")
    out_xlsx = in_path.with_name(stem + "_행정구역추가.xlsx")

    if SAVE_CSV:
        # 엑셀에서 한글 깨짐 방지: utf-8-sig(BOM)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[저장] CSV: {out_csv}")

    if SAVE_XLSX:
        df.to_excel(out_xlsx, index=False)
        print(f"[저장] XLSX: {out_xlsx}")

    # 간단 요약
    print("샘플 5행:")
    print(df[[LAT_COL, LON_COL, "구군", "읍면동"]].head())

if __name__ == "__main__":
    main()

pd.read_csv("./은행_통합_복지스코어_교통스코어_병합_행정구역추가.csv")