import pandas as pd
from sklearn.cluster import KMeans

import os

# 현재 작업 디렉토리 확인
print(os.getcwd())

os.chdir("..")  # 상위 디렉토리로 이동

# 하위 폴더 'plotly'로 이동
os.chdir("plotly")
os.chdir("daegu")


# 1. 데이터 불러오기
df2 = pd.read_csv('./클러스터링용_스케일링반영.csv', encoding= 'cp949')

# 2. 군집분석에 사용할 변수 선택
X1 = df2[['복지스코어', '교통스코어', '고령인구비율', '포화도']]
X.describe()


# 3. K-means 모델 생성 및 학습
kmeanss = KMeans(n_clusters=8, init='k-means++', random_state=42, n_init=10)  # n_init=10은 초기화 반복 횟수
df2['클러스터'] = kmeanss.fit_predict(X1)

# 4. 각 클러스터 중심 좌표 확인
print("클러스터 중심:")
print(pd.DataFrame(kmeanss.cluster_centers_, 
                   columns=['복지스코어', '교통스코어', '고령인구비율', '포화도']))

# 클러스터별 개수 확인
cluster_counts = df2['클러스터'].value_counts().sort_index()
print(cluster_counts)

df8 = df2.copy()
df8.to_csv("클러스터포함_전체.csv", index= False, encoding='utf-8-sig')

clu = [0, 5, 6]
get = df8[df['클러스터'].isin(clu)]
get.to_csv("타겟클러스터만.csv", index=False, encoding= 'utf-8-sig')



from sklearn.metrics import silhouette_score
# 1. 데이터 불러오기
df3 = pd.read_csv('./클러스터링용_스케일링반영.csv', encoding= 'cp949')

# 2) 군집 변수 선택
X2 = df3[['복지스코어', '교통스코어', '고령인구비율', '포화도']]
# 선택 사항: 요약 통계 확인
# print(X1.describe())

# 3) k=4~10 실루엣 지수 비교
k_list = range(4, 11)
sil_scores = {}

for k in k_list:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X2)
    # (안전장치) 모든 포인트가 한 클러스터면 실루엣 불가 → 낮은 점수로 처리
    if len(set(labels)) <= 1:
        score = -1.0
    else:
        score = silhouette_score(X2, labels, metric='euclidean')
    sil_scores[k] = score

sil_scores

# 4) 최적 k 선택 (실루엣 최대)
best_k = max(sil_scores, key=sil_scores.get)
print("=== Silhouette scores by k ===")
print(pd.Series(sil_scores).round(4))
print(f"\n[선정] 최적 k (실루엣 최대): {best_k} | score={sil_scores[best_k]:.4f}")

# 5) 최적 k로 최종 학습 + 라벨/중심/개수 출력
kmeanss = KMeans(n_clusters=best_k, init='k-means++', random_state=42, n_init=10)
df3['클러스터'] = kmeanss.fit_predict(X2)

print("\n클러스터 중심:")
centers = pd.DataFrame(kmeanss.cluster_centers_,
                       columns=['복지스코어','교통스코어','고령인구비율','포화도'])
print(centers)

print("\n클러스터별 개수:")
cluster_counts = df3['클러스터'].value_counts().sort_index()
print(cluster_counts)

