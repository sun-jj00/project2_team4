import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity

# ===== 1. 데이터 불러오기 =====
# (경로는 실제 파일 경로로 변경)
senior_center = pd.read_csv("노인복지센터.csv")       # 복지센터
senior_hall = pd.read_csv("대구_경로당_구군동추가.csv")  # 경로당
banks = pd.read_csv("은행_통합.csv")                  # 은행 지점

# ===== 2. GeoDataFrame 변환 =====
# 복지센터
gdf_center = gpd.GeoDataFrame(
    senior_center,
    geometry=gpd.points_from_xy(senior_center['경도'], senior_center['위도']),
    crs="EPSG:4326"
)

# 경로당
gdf_hall = gpd.GeoDataFrame(
    senior_hall,
    geometry=gpd.points_from_xy(senior_hall['경도'], senior_hall['위도']),
    crs="EPSG:4326"
)

# 은행 지점
gdf_banks = gpd.GeoDataFrame(
    banks,
    geometry=gpd.points_from_xy(banks['경도'], banks['위도']),
    crs="EPSG:4326"
)

# ===== 3. 좌표계 변환 (미터 단위: EPSG 5186) =====
gdf_center = gdf_center.to_crs(epsg=5186)
gdf_hall = gdf_hall.to_crs(epsg=5186)
gdf_banks = gdf_banks.to_crs(epsg=5186)

# ===== 4. 가중치 적용 =====
gdf_center['weight'] = 10.0  # 복지센터
gdf_hall['weight'] = 1.0     # 경로당

# ===== 5. 시니어 인프라 통합 =====
gdf_senior = pd.concat([
    gdf_center[['geometry', 'weight']],
    gdf_hall[['geometry', 'weight']]
], ignore_index=True)

# ===== NaN 좌표 제거 =====
gdf_senior = gdf_senior[~gdf_senior.geometry.is_empty & gdf_senior.geometry.notna()]
gdf_senior = gdf_senior.dropna(subset=['geometry'])

# 좌표가 POINT인지 확인(혹시 POLYGON이나 다른 형태가 있으면 제거)
gdf_senior = gdf_senior[gdf_senior.geometry.type == 'Point']

# ===== 6. KDE 학습 =====
X = np.vstack([gdf_senior.geometry.x, gdf_senior.geometry.y]).T
weights = gdf_senior['weight'].values

bandwidth = 500
kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde.fit(X, sample_weight=weights)



# 대역폭 500m
bandwidth = 500
kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde.fit(X, sample_weight=weights)

# ===== 7. 은행 지점별 KDE 값 계산 =====
bank_coords = np.vstack([gdf_banks.geometry.x, gdf_banks.geometry.y]).T
log_density = kde.score_samples(bank_coords)
density = np.exp(log_density)  # 로그값 -> 원래값 변환

# ===== 8. 0~1 정규화 =====
score_norm = (density - density.min()) / (density.max() - density.min())

# ===== 9. 결과 저장 =====
gdf_banks['복지스코어'] = score_norm
gdf_banks
gdf_banks[['은행', '지점명', '주소','복지스코어', 'geometry']].to_csv(
    "시니어서비스스코어.csv", index=False, encoding='utf-8-sig'
)

print("은행별 시니어 인프라 스코어 계산 완료!")


