
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------
# 0) 설정: 파일 경로(필요 시 경로만 바꿔주세요)
# ------------------------------------------------------------
BASE = "."  # 같은 폴더라면 "." , 다르면 절대/상대경로로 수정
BANK_FILE    = os.path.join(BASE, "은행_통합.csv")
SENIOR_FILE  = os.path.join(BASE, "시니어서비스스코어.csv")
TRAFFIC_FILE = os.path.join(BASE, "교통스코어.csv")

OUT_CSV  = os.path.join(BASE, "은행_통합_복지스코어_교통스코어_병합.csv")
OUT_XLSX = os.path.join(BASE, "은행_통합_복지스코어_교통스코어_병합.xlsx")

# ------------------------------------------------------------
# 1) 유틸: 안전한 CSV 로더 (인코딩 자동 시도)
# ------------------------------------------------------------
def read_csv_safe(path):
    tried = []
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[INFO] Read OK: {os.path.basename(path)} (encoding={enc})")
            return df
        except Exception as e:
            tried.append((enc, str(e)[:120]))
    # 마지막 시도: encoding 지정 없이 (판다스 추정)
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Read OK (fallback): {os.path.basename(path)} (encoding=auto)")
        return df
    except Exception as e:
        msg = f"[ERROR] Failed to read {path}\nTried encodings: {tried}\nLast error: {e}"
        raise RuntimeError(msg)

# ------------------------------------------------------------
# 2) 로드
# ------------------------------------------------------------
bank   = read_csv_safe(BANK_FILE)
senior = read_csv_safe(SENIOR_FILE)
traffic= read_csv_safe(TRAFFIC_FILE)

# ------------------------------------------------------------
# 3) 전처리: 불필요 컬럼 제거 / 문자열 트림 / 좌표 반올림키 생성
#    - geometry 컬럼 제거(있을 경우)
#    - (은행, 지점명, 주소) 공백/중복공백 정리
#    - 좌표 반올림 키(위도/경도) -> 부동소수점 미스매치 방지용
# ------------------------------------------------------------
if "geometry" in senior.columns:
    senior = senior.drop(columns=["geometry"])

def strip_cols(df, cols):
    for c in cols:
        if c in df.columns:
            # 문자열로 캐스팅 후 트림 + 중복 공백 1칸으로
            df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

strip_cols(bank,   ["은행", "지점명", "주소"])
strip_cols(senior, ["은행", "지점명", "주소"])
strip_cols(traffic,["은행", "지점명", "주소"])

# 좌표형 컬럼 정리 (있을 때만)
for df in [bank, traffic]:
    for c in ["위도", "경도"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# 좌표 반올림 키 (6~7자리 권장; 동일 소스에서 나온 좌표면 6으로 충분)
ROUND_DIGITS = 6
if {"위도","경도"}.issubset(bank.columns):
    bank["lat_r"] = bank["위도"].round(ROUND_DIGITS)
    bank["lon_r"] = bank["경도"].round(ROUND_DIGITS)
if {"위도","경도"}.issubset(traffic.columns):
    traffic["lat_r"] = traffic["위도"].round(ROUND_DIGITS)
    traffic["lon_r"] = traffic["경도"].round(ROUND_DIGITS)

# ------------------------------------------------------------
# 4) 병합(LEFT JOIN)
#    - 기준: 은행_통합.csv
#    - senior: (은행, 지점명, 주소)
#    - traffic: (은행, 지점명, 위도, 경도) → 반올림키로 매칭 안정화
# ------------------------------------------------------------
# 4-1) 시니어서비스스코어 병합
need_cols_senior = [c for c in ["은행","지점명","주소","복지스코어"] if c in senior.columns]
if not set(["은행","지점명","주소"]).issubset(senior.columns):
    raise KeyError("[ERROR] 시니어서비스스코어.csv 에 '은행','지점명','주소' 중 누락된 컬럼이 있습니다.")
merged = bank.merge(
    senior[need_cols_senior],
    on=["은행","지점명","주소"],
    how="left"
)

# 4-2) 교통스코어 병합 (좌표 반올림 키 포함)
need_cols_traffic = [c for c in ["은행","지점명","위도","경도","score_1_10","lat_r","lon_r"] if c in traffic.columns]
if not set(["은행","지점명","위도","경도"]).issubset(traffic.columns):
    raise KeyError("[ERROR] 교통스코어.csv 에 '은행','지점명','위도','경도' 중 누락된 컬럼이 있습니다.")

# 기준 테이블에도 반올림키가 있어야 한다
if "lat_r" not in merged.columns or "lon_r" not in merged.columns:
    if {"위도","경도"}.issubset(merged.columns):
        merged["lat_r"] = merged["위도"].round(ROUND_DIGITS)
        merged["lon_r"] = merged["경도"].round(ROUND_DIGITS)
    else:
        raise KeyError("[ERROR] 기준 테이블에 '위도','경도'가 없어 좌표키를 만들 수 없습니다.")

merged = merged.merge(
    traffic[need_cols_traffic],
    on=["은행","지점명","lat_r","lon_r"],   # 좌표 반올림키로 매칭(은행/지점명 보강)
    how="left",
    suffixes=("", "_traffic")
)

# ------------------------------------------------------------
# 5) 컬럼 정리: 앞쪽에 핵심 컬럼 배치
# ------------------------------------------------------------
front = [c for c in ["은행","지점명","주소","위도","경도","복지스코어","score_1_10"] if c in merged.columns]
others = [c for c in merged.columns if c not in front and c not in ["lat_r","lon_r"]]
merged = merged[front + others]
merged.head()

# ========================== 추가/수정 파트 시작 ==========================
import re

# ---------- (1) 복지스코어 1~10 스케일링 ----------
# 이미 0~1 범위라도, 최소/최대가 약간 다를 수 있으므로 Min-Max 후 1~10 변환
if "복지스코어" not in merged.columns:
    raise KeyError("[ERROR] 병합 테이블에 '복지스코어' 컬럼이 없습니다. 앞 단계 병합을 확인하세요.")

min_v = merged["복지스코어"].min(skipna=True)
max_v = merged["복지스코어"].max(skipna=True)
denom = (max_v - min_v) if pd.notnull(max_v) and pd.notnull(min_v) and (max_v != min_v) else 1.0

# 0~1 정규화
welfare_0_1 = (merged["복지스코어"] - min_v) / denom
# 1~10 스케일 (NaN은 유지)
merged["복지스코어_1_10"] = welfare_0_1.mul(9).add(1).where(~merged["복지스코어"].isna(), np.nan)

# ---------- (2) score_1_10 -> 교통스코어 ----------
if "score_1_10" in merged.columns:
    merged = merged.rename(columns={"score_1_10": "교통스코어"})
else:
    print("[WARN] 'score_1_10' 컬럼을 찾지 못했습니다. 이미 이름이 변경되어 있거나 입력 데이터에 없습니다.")

# ---------- (3) 주소에서 구/군, 동/읍/면 추출 ----------
def extract_gu_gun(addr: str):
    """주소 문자열에서 '구' 또는 '군' 단위를 우선 추출. 없으면 ''."""
    if not isinstance(addr, str):
        return ""
    # 가장 먼저 일치하는 구/군 토큰 반환
    m = re.search(r'([^\s]+?(구|군))', addr)
    return m.group(1) if m else ""

def extract_dong_eup_myeon(addr: str):
    """주소 문자열에서 '동/읍/면' 단위를 추출. 없으면 ''."""
    if not isinstance(addr, str):
        return ""
    m = re.search(r'([^\s]+?(동|읍|면))', addr)
    return m.group(1) if m else ""

if "주소" not in merged.columns:
    raise KeyError("[ERROR] 병합 테이블에 '주소' 컬럼이 없습니다.")

merged["구군"] = merged["주소"].apply(extract_gu_gun)
merged["동읍면"] = merged["주소"].apply(extract_dong_eup_myeon)

# ---------- (3-추가) '주소' 옆에 구군/동읍면 컬럼 위치시키기 ----------
def insert_after(df, target_col, new_cols):
    cols = list(df.columns)
    if target_col not in cols:
        return df  # 타겟이 없으면 그대로
    idx = cols.index(target_col)
    # 기존 컬럼을 빼고, 타겟 뒤에 새 컬럼을 삽입
    for c in new_cols:
        if c in cols:
            cols.remove(c)
    for i, c in enumerate(new_cols, start=1):
        cols.insert(idx + i, c)
    return df[cols]

merged = insert_after(merged, "주소", ["구군", "동읍면"])

# ---------- (선택) 앞쪽 핵심 컬럼 재정렬 ----------
front = [c for c in ["은행","지점명","주소","구군","동읍면","위도","경도","복지스코어","복지스코어_1_10","교통스코어"] if c in merged.columns]
others = [c for c in merged.columns if c not in front]
merged = merged[front + others]
# ========================== 추가/수정 파트 끝 ==========================
merged.head()
del merged['구군']
del merged['복지스코어']
merged.rename(columns= {'복지스코어_1_10' : '복지스코어'})

del merged['위도_traffic']
del merged['경도_traffic']


# ---------- (4) 결과 저장 ----------
merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
try:
    merged.to_excel(OUT_XLSX, index=False)
except Exception as e:
    print("[WARN] XLSX 저장 중 오류가 발생했습니다. openpyxl 설치가 필요할 수 있습니다.")
    print("       pip install openpyxl")
    print("       오류:", e)

print("\n[완료] 전처리 및 저장이 끝났습니다.")
print(f"- CSV : {OUT_CSV}")
print(f"- XLSX: {OUT_XLSX}")





########동별 고령 전처리


import pandas as pd

# 1. 파일 불러오기
file_path = "동별_고령인구.xlsx"
df = pd.read_excel(file_path)

# 2. '행정기관' 열에서 '대구광역시' 단어 삭제
df['행정기관'] = df['행정기관'].str.replace('대구광역시', '', regex=False).str.strip()
df['행정기관']
# 3. '구군', '읍면동' 분리
df['구군'] = df['행정기관'].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else None)
df['읍면동'] = df['행정기관'].apply(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 1 else None)
df.head()

df.rename(columns= {'65세이상 전체': '65이상'})


# 4. 새로운 파일 저장 (CSV, UTF-8-SIG 인코딩)
output_path = "동별_고령_읍면동추가.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

df2 = pd.read_csv('은행별_전처리용.csv')

df2.iloc[[38], :]['읍면동'] ='서변동'

df2[df2['읍면동'] == '무태조야동']

# 38번, 85번 행의 '읍면동' 값 변경
df2.loc[38, '읍면동'] = '서변동'
df2.loc[85, '읍면동'] = '연경동'

# 변경 내용 확인
print(df2.loc[[38, 85], ['구군', '읍면동']])

del df2['동읍면']

df2.info()

df2 = df2.rename(columns= {'복지스코어_1_10': '복지스코어'})

# 원하는 열 순서 지정
new_order = ['은행', '지점명', '주소', '구군', '읍면동', '위도', '경도', '복지스코어', '교통스코어']

# 열 순서 변경
df2 = df2[new_order]

df2
# 확인
print(df2.head())



df3 = pd.read_csv('동별_고령_읍면동추가.csv')
df3.head()

df3.info()


# df2와 df3 병합 (구군, 읍면동 기준)
df_merged = pd.merge(
    df2,
    df3[['구군', '읍면동', '전체', '65세이상 전체', '고령인구비율']],
    on=['구군', '읍면동'],
    how='left'
)

# 결과 확인
df_merged.head()


#포화도 도출
df_merged.info()


# 1) 첫 번째 열에 '은행id' 추가 (1 ~ n)
df_merged.insert(0, '은행id', range(1, len(df_merged) + 1))

# 2) 읍면동별 '은행id' 개수를 세서 '동별지점수' 열 추가
df_merged['동별지점수'] = (
    df_merged.groupby('읍면동')['은행id']
             .transform('count')
             .astype('int64')
)

df_merged
# 3) '포화도' 계산
tmp_total = (
    df_merged['전체']
      .astype(str)
      .str.replace(',', '', regex=False)
      .str.replace(' ', '', regex=False)
)
df_merged['전체_numeric'] = pd.to_numeric(tmp_total, errors='coerce')
df_merged['포화도'] = df_merged['전체_numeric'] / df_merged['동별지점수'].replace({0: pd.NA})


df_merged.head()


from sklearn.preprocessing import MinMaxScaler

# 스케일러 객체 생성 (1~10 범위)
scaler = MinMaxScaler(feature_range=(1, 10))

# NaN 제외한 값만 스케일링
df_merged['포화도_스케일링'] = scaler.fit_transform(
    df_merged[['포화도']].fillna(0)  # NaN 처리
)

# 결과 확인
print(df_merged[['포화도', '포화도_스케일링']].head())


df_merged['고령인구비율'] = df_merged['고령인구비율'] * 100

df_merged['포화도']

# 1) '전체', '65세이상 전체'를 정수형으로 변환 후 같은 열에 저장
df_merged['전체'] = (
    df_merged['전체']
      .astype(str)
      .str.replace(',', '', regex=False)
      .str.strip()
      .astype('int64')
)

df_merged['65세이상 전체'] = (
    df_merged['65세이상 전체']
      .astype(str)
      .str.replace(',', '', regex=False)
      .str.strip()
      .astype('int64')
)

# 2) 열 이름 변경
df_merged.rename(columns={
    '65세이상 전체': '65세이상',
    '전체': '전체인구'
}, inplace=True)



# 원하는 열 순서 지정
new_order = ['은행', '지점명', '주소', '구군', '읍면동', '위도', '경도', '복지스코어', '교통스코어']

# 열 순서 변경
df2 = df2[new_order]

df2

import pandas as pd

# 1) 숫자화: 콤마/공백 제거 → 숫자로 변환(실패 시 NaN) → nullable 정수형(Int64)
df_merged['전체'] = (
    pd.to_numeric(
        df_merged['전체'].astype(str)
                         .str.replace(',', '', regex=False)
                         .str.strip()
                         .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA}),
        errors='coerce'
    ).astype('Int64')
)

df_merged['65세이상 전체'] = (
    pd.to_numeric(
        df_merged['65세이상 전체'].astype(str)
                               .str.replace(',', '', regex=False)
                               .str.strip()
                               .replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA}),
        errors='coerce'
    ).astype('Int64')
)

# 2) 열 이름 변경
df_merged.rename(columns={
    '전체': '전체인구',
    '65세이상 전체': '65세이상'
}, inplace=True)


df_merged.head()
del df_merged['전체_numeric']


df_merged.to_csv('클러스터링용.csv',  index=False, encoding='utf-8-sig')


# df_merged를 xlsx로 저장
output_path = "클러스터링용.xlsx"
df_merged.to_excel(output_path, index=False)