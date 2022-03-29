import tensorflow as tf
import pandas as pd
import numpy as np

data=pd.read_csv('gpascore.csv')
data = data.dropna()                                 #판다스를 활용한 데이터 전처리
y = data['admit'].values
x= []
for i, rows in data.iterrows():
   x.append([rows['gre'], rows['gpa'], rows['rank']])       




model = tf.keras.models.Sequential([                       #모델 설정
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #모델 컴파일

model.fit(np.array(x), np.array(y), epochs=3000) #모델 학습시키기 ->(x, y, epoch) x데이터에는 학습 데이터, y데이터에는 정답, epoch는 학습횟수

#예측하기
predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predict)