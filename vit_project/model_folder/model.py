import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("house_cost_prediction_dataset_10000.csv")

# Select features and target
X = df[['Area_m2', 'Rooms', 'Bathrooms']]
y = df['Estimated_Cost_INR']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R^2 Score:", r2)

import joblib
joblib.dump(model, "house_cost_model.pkl")
print("Model saved as house_cost_model.pkl")