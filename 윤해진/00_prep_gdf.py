# 00_prep_gdf.py
import os, sys, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import geopandas as gpd
from shapely.validation import make_valid as shp_make_valid

SHAPE_PATH = "./data/대구_행정동_군위포함.shp"
OUT_DIR = "./data/processed"
OUT_PKL = os.path.join(OUT_DIR, "daegu_emd.pkl")
OUT_GEOJSON = os.path.join(OUT_DIR, "daegu_emd.geojson")

def try_read(path: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path)              # 보통 자동 인코딩 처리됨
    except Exception as e:
        print(f"[INFO] 기본 읽기 실패 → CP949 재시도: {e}")
        return gpd.read_file(path, encoding="cp949")

def make_valid(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = g.copy()
    g["geometry"] = g["geometry"].apply(lambda geom: shp_make_valid(geom) if geom is not None else None)
    return g

def main():
    if not os.path.exists(SHAPE_PATH):
        print(f"[ERROR] 파일이 없습니다: {SHAPE_PATH}")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)

    g = try_read(SHAPE_PATH)

    # ★ 사용자가 준 info에 맞춘 표준화: ADM_DR_NM, ADM_DR_CD, BASE_DATE 존재 가정
    required = {"ADM_DR_NM", "ADM_DR_CD", "BASE_DATE", "geometry"}
    missing = required - set(g.columns)
    if missing:
        print(f"[ERROR] 기대 컬럼이 없습니다: {missing}")
        sys.exit(1)

    # 좌표계 보정
    if g.crs is None:
        print("[WARN] CRS가 비어 있음 → EPSG:5179(국가TM)로 가정")
        g = g.set_crs(epsg=5179, allow_override=True)
    g = g.to_crs(epsg=4326)

    # 지오메트리 유효화 및 찌꺼기 제거
    g = make_valid(g)
    g = g[~g.geometry.is_empty & g.geometry.area.gt(0)]

    # ★ 컬럼 표준화
    g = g.rename(columns={
        "ADM_DR_NM": "동",           # 읍/면/동 명
        "ADM_DR_CD": "행정동코드",   # 코드
        "BASE_DATE": "기준일자"      # 기준일
    })
    g["동"] = g["동"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    g["행정동코드"] = g["행정동코드"].astype(str).str.strip()
    g["기준일자"] = g["기준일자"].astype(str).str.strip()

    # 저장
    g.to_pickle(OUT_PKL)
    g.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"[DONE] 저장 완료\n - {OUT_PKL}\n - {OUT_GEOJSON}\n행 수: {len(g)}")

if __name__ == "__main__":
    main()
