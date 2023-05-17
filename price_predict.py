import pandas as pd
import numpy as np
import math as ma
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

#スタート
print("\n\n---電力価格予測プログラム開始---\n\n")

# データの読み込み
input_data = pd.read_csv("input_data2022.csv")
weather_data = pd.read_csv("weather_data.csv")
pv_predict = pd.read_csv("pv_predict.csv")

#時系列のsin, cosを追加
hourSin = np.sin(weather_data["hour"]/12*(ma.pi))
hourCos = np.cos(weather_data["hour"]/12*(ma.pi))
time_data = pd.concat([hourSin, hourCos], axis=1)
name = ['hourSin', 'hourCos'] # 列名
time_data.columns = name # 列名付与
#元のデータに統合
weather_data = pd.concat([weather_data, time_data], axis=1)

# 使用するパラメータ
#parameters = ['temperature', 'total precipitation', 'u-component of wind', 'v-component of wind',
              #'radiation flux', 'pressure', 'relative humidity', 'yearSin', 'yearCos', 'monthSin',
              #'monthCos', 'hourSin', 'hourCos', 'PVout']
parameters = ['radiation flux', 'PVout', 'temperature', 'hourCos']           
predict_parameters = ['price', 'imbalance']

# データの前処理
scaler = MinMaxScaler()
input_data[parameters] = scaler.fit_transform(input_data[parameters])
pv_predict[parameters] = scaler.transform(pv_predict[parameters])

# 学習データとターゲットデータの作成
X = input_data[parameters].values
y = input_data[predict_parameters].values

# モデルの定義
hidden_units = [64, 64, 64]  # 隠れ層のユニット数
epochs = 100  # エポック数

model = keras.Sequential()
model.add(keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(len(parameters),)))
for units in hidden_units[1:]:
    model.add(keras.layers.Dense(units, activation='relu'))
model.add(keras.layers.Dense(len(predict_parameters)))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mse')

# モデルの学習
model.fit(X, y, epochs=epochs, verbose=0)

# 予測の実行
predictions = model.predict(pv_predict[parameters].values)

# 予測結果の保存
pred_df = pd.DataFrame(columns=["year","month","hour","day","hourSin","hourCos","upper","lower","PVout","price","imbalance"])
pred_df[["price","imbalance"]] = predictions

pred_df[["year","month","hour","day","hourSin","hourCos"]] = weather_data[["year","month","hour","day","hourSin","hourCos"]]
pred_df[["upper","lower","PVout"]] = pv_predict[["upper","lower","PVout"]]
pred_df.to_csv("price_predict.csv", index=False)

# グラフの描画
#plt.figure(figsize=(10, 5))
#plt.plot(pv_predict['hour'], predictions[:, 0], label='price')
#plt.plot(pv_predict['hour'], predictions[:, 1], label='imbalance')
#plt.xlabel('hour')
#plt.ylabel('value')
#plt.title('Price and Imbalance Prediction')#
#plt.legend()
#plt.show()

#終了
print("\n\n---電力価格予測プログラム終了---\n\n")
