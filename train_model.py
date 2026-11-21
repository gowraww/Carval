
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

# 1. Load dataset
data = pd.read_csv("usedcar.csv")

# 2. Remove missing values
data = data.dropna()

# 3. Input and output
X = data.drop("selling_price_lakh", axis=1)
y = data["selling_price_lakh"]

# 4. Identify columns
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

# 5. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ]
)

# 6. Linear Regression with log transformation
regression = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,
    inverse_func=np.expm1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", regression)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL EVALUATION =====")
print("Algorithm: Linear Regression")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

joblib.dump(model, "car_model.pkl")
print("\nModel saved successfully as car_model.pkl")
