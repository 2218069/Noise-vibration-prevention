import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = r'C:\Users\User\Desktop\vibration.csv'
data = pd.read_csv(file_path)

print("데이터 요약 정보:")
print(data.info())

print("\n데이터의 처음 5행:")
print(data.head())

data.hist(bins=20, figsize=(12, 8))
plt.show()

data.dropna(inplace=True)

X = data[['IndependentVariable1', 'IndependentVariable2']]
y = data['DependentVariable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n머신러닝 모델 예측 결과:")
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print("\n평균 제곱 오차 (MSE):", mse)

kfold = KFold(n_splits=5, shuffle=True, random_state=42
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
print("Cross-Validation MSE Scores:", -cv_results)
print("Mean MSE:", -cv_results.mean())
print("Standard Deviation of MSE:", cv_results.std())
