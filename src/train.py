import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import joblib

data = pd.read_csv("diabetes.csv")
print(data.head())
# 🔥 kritik kolonlar
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# 0 → NaN
for col in cols:
    data[col] = data[col].replace(0, np.nan)

# median ile doldur (daha stabil)
for col in cols:
    data[col].fillna(data[col].median(), inplace=True)

# kontrol (çok önemli)
print("NaN var mı?\n", data.isnull().sum())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 🔥 RobustScaler (outlier’a dayanıklı)
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

tahmin = model.predict(X_test)

print("Accuracy skor =", accuracy_score(y_test, tahmin))

joblib.dump(model, "model.pkl")