import pandas as import pandas as pd


data = pd.read_csv("타겟df.csv")


# ============================================================
# (클러스터==5) 커뮤니티 적합도 / (클러스터==6) 특화점포 적합도만 계산
# 각 적합도 타입별 상위 4개 지점 출력
# ============================================================
import pandas as pd
import numpy as np

# 1) 데이터 로드
df = pd.read_csv("타겟df.csv", encoding="utf-8-sig")

# (선택) 숫자 변환 보정
num_cols = ["클러스터", "교통스코어", "유동인구스코어", "복지스코어", "포화도", "인프라성숙도"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 2) 1~10 → 0~1 변환: (x - 1) / 9
to01 = lambda s: (s - 1.0) / 9.0
df["_교통01"]   = to01(df["교통스코어"])
df["_유동01"]   = to01(df["유동인구스코어"])
df["_복지01"]   = to01(df["복지스코어"])
df["_포화01"]   = to01(df["포화도"])
df["_인프라01"] = to01(df["인프라성숙도"])

# 3) 클러스터 마스크
is_c5 = (df["클러스터"] == 5)
is_c6 = (df["클러스터"] == 6)

# 4) 적합도 계산 (요청 로직 + (역) 변수는 부호 반전)
# 커뮤니티 적합도: (클러스터==5)에서만
df["커뮤니티적합도"] = np.where(
    is_c5,
    (
        df["_교통01"]*0.25 +
        df["_유동01"]*0.25 +
        (-df["_복지01"])*0.30 +
        (-df["_포화01"])*0.10 +
        (-df["_인프라01"])*0.10
    ),
    np.nan
)

# 특화점포 적합도: (클러스터==6)에서만
df["특화점포적합도"] = np.where(
    is_c6,
    (
        df["_유동01"]*0.20 +
        df["_인프라01"]*0.25 +
        df["_교통01"]*0.25 +
        (-df["_포화01"])*0.10 +
        df["_복지01"]*0.20
    ),
    np.nan
)

# 5) 타입별 상위 4개만 추출 & 출력
comm_top4 = (
    df[is_c5]
    .sort_values("커뮤니티적합도", ascending=False)
    .head(4)[["은행id", "은행", "지점명", "커뮤니티적합도"]]
)

spec_top4 = (
    df[is_c6]
    .sort_values("특화점포적합도", ascending=False)
    .head(4)[["은행id", "은행", "지점명", "특화점포적합도"]]
)

print("\n====== 커뮤니티 적합도 TOP 4 (클러스터==5) ======")
print(comm_top4.to_string(index=False))

print("\n====== 특화점포 적합도 TOP 4 (클러스터==6) ======")
print(spec_top4.to_string(index=False))
