## 군집분석 K-means
import pandas as pd
from sklearn.cluster import KMeans

# 1. 데이터 불러오기
df = pd.read_csv('./data/project2/클러스터링용_스케일링반영.csv', encoding='cp949')

# 2. 군집분석에 사용할 변수 선택
X = df[['복지스코어', '교통스코어', '고령인구비율', '포화도']]

# 3. K-means 모델 생성 및 학습
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)  # n_init=10은 초기화 반복 횟수
df['클러스터'] = kmeans.fit_predict(X)

# 4. 각 클러스터 중심 좌표 확인
print("클러스터 중심:")
print(pd.DataFrame(kmeans.cluster_centers_, 
                   columns=['복지스코어', '교통스코어', '고령인구비율', '포화도']))