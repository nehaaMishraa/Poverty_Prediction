import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

# STEP 1: Load dataset with proper header row
df = pd.read_excel(r'/Users/nehamishra/Developer/ML/ML Proj/MPM-Historical-Data-AM2024-Final.xlsx', header=1)

# STEP 2: Rename unnamed columns to meaningful names
df.rename(columns={
    'Unnamed: 10': 'Education enrollment deprivation rate (share of population)',
    'Unnamed: 11': 'Education attainment deprivation rate (share of population)',
    'Unnamed: 12': 'Electricity deprivation rate (share of population)',
    'Unnamed: 13': 'Sanitation deprivation rate (share of population)',
    'Unnamed: 14': 'Drinking water deprivation rate (share of population)'
}, inplace=True)

# STEP 3: Drop irrelevant columns
columns_to_drop = ['Region', 'Country code', 'Economy', 'Survey name', 'Survey coverage']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

# STEP 4: Select feature and target columns
selected_features = [
    'Deprivation rate (share of population)',
    'Education enrollment deprivation rate (share of population)',
    'Education attainment deprivation rate (share of population)',
    'Electricity deprivation rate (share of population)',
    'Sanitation deprivation rate (share of population)',
    'Drinking water deprivation rate (share of population)'
]
target_col = 'Multidimensional poverty headcount ratio (%)'

# STEP 5: Drop missing values and convert data to float
df = df.dropna(subset=selected_features + [target_col])
df[selected_features] = df[selected_features].astype(float)
df[target_col] = df[target_col].astype(float)

# STEP 6: Train-Test split
X = df[selected_features]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 7: Set up parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# STEP 8: Initialize XGBRegressor
xgb = XGBRegressor(random_state=42)

# STEP 9: Run GridSearchCV to find the best parameters
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,                        # 5-fold cross-validation
    scoring='r2',              
    verbose=1,
    n_jobs=-1         
    )          

grid_search.fit(X_train, y_train)

# STEP 10: Retrieve the best model
best_model = grid_search.best_estimator_
print("✅ Best Parameters Found:", grid_search.best_params_)

# NEW: Save the best model to a file
joblib.dump(best_model, 'poverty_prediction_model.pkl')
print("✅ Model saved as 'poverty_prediction_model.pkl'")

# STEP 11: Evaluate the best model
y_pred = best_model.predict(X_test)
print("📉 Best XGBoost MSE:", mean_squared_error(y_test, y_pred))
print("📈 Best XGBoost R² Score:", r2_score(y_test, y_pred))

# STEP 12: Predict on new data
X_new = np.array([[0.6, 2.3, 6.7, 0.5, 3.0, 0.3]])
prediction = best_model.predict(X_new)
print("🔮 Predicted Multidimensional Poverty Rate:", prediction[0])

# STEP 13: Optional - Description of the target column
print("\n📊 Target Variable Description:")
print(df[target_col].describe())

# STEP 14: 📊 Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[selected_features + [target_col]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("📊 Correlation Heatmap of Features and Target")
plt.tight_layout()
plt.show()

# STEP 15: 📈 Feature Importance Plot from Trained XGBoost Model
importances = best_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('📌 Feature Importance from Trained XGBoost Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()




