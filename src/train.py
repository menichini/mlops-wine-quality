
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/winequality-red.csv", sep=';')
X = data.drop("quality", axis=1)
y = data["quality"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow tracking
mlflow.set_experiment("WineQuality-Regression")

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    mlflow.sklearn.log_model(model, "model")
