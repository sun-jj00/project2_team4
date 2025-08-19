import os

# 현재 작업 디렉토리 확인
print(os.getcwd())

os.chdir("..")  # 상위 디렉토리로 이동

# 하위 폴더 'plotly'로 이동
os.chdir("plotly")
os.chdir("daegu")

import pandas as pd
import numpy as np
from math import radians

# ===== 파라미터 =====
RADIUS_M   = 500.0    # 반경 500m
ROBUST_PLO = 10       # 버스 로버스트 Min–Max 하한 분위수
ROBUST_PHI = 90       # 버스 로버스트 Min–Max 상한 분위수
SUB_CAP    = 3        # 지하철 CAP (1개/2개 확실히 구분됨: 1/3=0.333, 2/3=0.667)

# ===== 유틸 =====
def read_csv_safely(path):
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            pass
    raise ValueError(f"{path} 불러오기 실패")

def detect_lat_lon(df):
    lat = [c for c in df.columns if "위도" in c or "lat" in c.lower()][0]
    lon = [c for c in df.columns if "경도" in c or "lon" in c.lower()][0]
    return lat, lon

EARTH_R = 6371000.0  # m
def haversine_vec(lat1, lon1, lat2_vec, lon2_vec):
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2 = np.radians(lat2_vec); lon2 = np.radians(lon2_vec)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return EARTH_R * (2 * np.arcsin(np.sqrt(a)))

def robust_minmax_0_1(x, plo=10, phi=90):
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, [plo, phi])
    if hi <= lo:  # 분위수 같은 경우 폴백: 전체 min-max
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx == mn:
            return np.full_like(x, 0.5)
        z = (x - mn) / (mx - mn)
        return np.clip(z, 0, 1)
    z = (x - lo) / (hi - lo)
    return np.clip(z, 0, 1)

# ===== 데이터 로드 =====
banks_df  = read_csv_safely("은행_통합.csv")
bus_df    = read_csv_safely("대구_버스정류소_필터.csv")
subway_df = read_csv_safely("대구_지하철_주소_좌표추가.csv")

bank_lat, bank_lon = detect_lat_lon(banks_df)
bus_lat,  bus_lon  = detect_lat_lon(bus_df)
sub_lat,  sub_lon  = detect_lat_lon(subway_df)

# 경유노선수
route_col = [c for c in bus_df.columns if ("경유노선수" in c) or ("노선수" in c)][0]
bus_df[route_col] = pd.to_numeric(bus_df[route_col], errors="coerce").fillna(0).clip(lower=0)

# ===== 벡터 준비 =====
bus_lat_arr = bus_df[bus_lat].values
bus_lon_arr = bus_df[bus_lon].values
bus_routes  = bus_df[route_col].values
sub_lat_arr = subway_df[sub_lat].values
sub_lon_arr = subway_df[sub_lon].values

# ===== 은행 지점별 원시 스코어 계산 =====
rows = []
for _, r in banks_df.iterrows():
    bl, bo = r[bank_lat], r[bank_lon]

    # 버스: 500m 내 sqrt(경유노선수) 합
    d_bus = haversine_vec(bl, bo, bus_lat_arr, bus_lon_arr)
    mask  = d_bus <= RADIUS_M
    bus_raw = float(np.sqrt(bus_routes[mask]).sum())

    # 지하철: 500m 내 역 개수
    d_sub = haversine_vec(bl, bo, sub_lat_arr, sub_lon_arr)
    sub_raw = int((d_sub <= RADIUS_M).sum())

    rows.append({
        "은행": r.get("은행", ""), "지점명": r.get("지점명", ""),
        "위도": bl, "경도": bo,
        "bus_raw": bus_raw, "sub_raw": sub_raw
    })

df = pd.DataFrame(rows)

# ===== 정규화: 버스=로버스트 Min–Max, 지하철=CAP 스케일 =====
df["bus_norm"] = robust_minmax_0_1(df["bus_raw"].values, ROBUST_PLO, ROBUST_PHI)
df["sub_norm"] = np.clip(df["sub_raw"].values / float(SUB_CAP), 0.0, 1.0)

# ===== 1.5:1 합산 (0~1) → 1~10 변환 =====
df["score_0_1"]  = ((df["bus_norm"]*1.5 + df["sub_norm"]) / 2.5).astype(float)
df["score_1_10"] = 1.0 + 9.0 * df["score_0_1"]

# ===== 저장/확인 =====
cols = ["은행","지점명","위도","경도","bus_raw","sub_raw","bus_norm","sub_norm","score_0_1","score_1_10"]
df[cols].to_csv("은행_교통인프라스코어_equal_CAP3_1to10.csv", index=False, encoding="utf-8-sig")
print(df[cols].head())



df