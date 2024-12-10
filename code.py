import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from geopy.distance import geodesic

# Load train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Load metadata for taxi stands
metadata_taxistands = pd.read_csv('metaData_taxistandsID_name_GPSlocation.csv')

# Fix any incorrect Latitude/Longitude formatting in the metadata
def split_lat_lon(value):
    if isinstance(value, str) and '-' in value:
        parts = value.split('-')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    return value

# Apply function to correct concatenated latitude/longitude
metadata_taxistands[['Latitude', 'Longitude']] = metadata_taxistands.apply(
    lambda row: split_lat_lon(row['Latitude']), axis=1
)

# Extract pickup coordinates (first point in POLYLINE)
def get_pickup_coordinates(polyline):
    if isinstance(polyline, str) and polyline.startswith('[') and '],[' in polyline:
        first_point = eval(polyline)[0]  # Extract first point (pickup)
        return tuple(first_point)
    return None

# Calculate distance to nearest taxi stand
def calculate_nearest_distance(coords, stands):
    if coords is None:
        return np.nan
    distances = stands.apply(
        lambda stand: geodesic(coords, (stand['Latitude'], stand['Longitude'])).km, axis=1
    )
    return distances.min()

# Preprocess POLYLINE to add pickup coordinates
train_data['pickup_coords'] = train_data['POLYLINE'].apply(get_pickup_coordinates)
test_data['pickup_coords'] = test_data['POLYLINE'].apply(get_pickup_coordinates)

# Add distance to nearest taxi stand as a feature
train_data['nearest_taxi_stand_distance'] = train_data['pickup_coords'].apply(
    lambda coords: calculate_nearest_distance(coords, metadata_taxistands)
)
test_data['nearest_taxi_stand_distance'] = test_data['pickup_coords'].apply(
    lambda coords: calculate_nearest_distance(coords, metadata_taxistands)
)

# Drop intermediate columns
train_data = train_data.drop(['pickup_coords'], axis=1)
test_data = test_data.drop(['pickup_coords'], axis=1)

# Drop unnecessary columns and handle missing values
train_data = train_data.dropna()

# Feature engineering (example for datetime column)
if 'TIMESTAMP' in train_data.columns:
    train_data['TIMESTAMP'] = pd.to_datetime(train_data['TIMESTAMP'], unit='s')
    train_data['hour'] = train_data['TIMESTAMP'].dt.hour
    train_data['day_of_week'] = train_data['TIMESTAMP'].dt.dayofweek
    train_data = train_data.drop('TIMESTAMP', axis=1)

# Split features and target
target = 'trip_duration'  # Update with actual target column name
if target not in train_data.columns:
    raise ValueError(f"Target column '{target}' not found in dataset")

X = train_data.drop(target, axis=1)
y = train_data[target]

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Add nearest_taxi_stand_distance to numerical features
numerical_features.append('nearest_taxi_stand_distance')

# Preprocessing
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define and train models with GridSearchCV for hyperparameter tuning
param_grid_rf = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10]
}

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X, y)

# Print best parameters and model performance
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
best_rf_model = grid_search_rf.best_estimator_

# Predict and evaluate Random Forest
y_pred_rf = best_rf_model.predict(X)
mse_rf = mean_squared_error(y, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print(f"Random Forest RMSE: {rmse_rf}")

# Linear Regression with Splines
spline_transformer = SplineTransformer(degree=3, n_knots=5)
linear_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('spline', spline_transformer),
    ('regressor', LinearRegression())
])

linear_reg_pipeline.fit(X, y)

# Predict and evaluate Linear Regression with Splines
y_pred_splines = linear_reg_pipeline.predict(X)
mse_splines = mean_squared_error(y, y_pred_splines)
rmse_splines = np.sqrt(mse_splines)
print(f"Linear Regression with Splines RMSE: {rmse_splines}")

# Neural Network Regression with Keras
model = Sequential([
    Dense(128, activation='relu', input_dim=X.shape[1]),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train the Neural Network
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Predict and evaluate Neural Network
y_pred_nn = model.predict(X).flatten()
mse_nn = mean_squared_error(y, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
print(f"Neural Network RMSE: {rmse_nn}")

# Gradient Boosting Regression
pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

pipeline_gb.fit(X, y)

# Predict and evaluate Gradient Boosting
y_pred_gb = pipeline_gb.predict(X)
mse_gb = mean_squared_error(y, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
print(f"Gradient Boosting RMSE: {rmse_gb}")

# Compare all models
print("\nModel Performance Comparison:")
print(f"Random Forest RMSE: {rmse_rf}")
print(f"Linear Regression with Splines RMSE: {rmse_splines}")
print(f"Neural Network RMSE: {rmse_nn}")
print(f"Gradient Boosting RMSE: {rmse_gb}")

