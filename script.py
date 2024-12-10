import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.spatial import distance
import geopandas as gpd
import matplotlib.pyplot as plt
import tensorflow as tf

# Load train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Load metadata for taxi stands
metadata_taxistands = pd.read_csv('metaData_taxistandsID_name_GPSlocation.csv')

# Extract pickup coordinates (first point in POLYLINE)
def get_pickup_coordinates(polyline):
    if isinstance(polyline, str) and polyline.startswith('[') and '],[' in polyline:
        first_point = eval(polyline)[0]  # Extract first point (pickup)
        return tuple(first_point)
    return None

# Calculate distance to the nearest taxi stand
def calculate_nearest_distance(coords, stands):
    if coords is None:
        return np.nan
    distances = stands.apply(
        lambda stand: distance.euclidean(coords, (stand['Latitude'], stand['Longitude'])), axis=1
    )
    return distances.min()

# Preprocess POLYLINE to add pickup coordinates
train_data['pickup_coords'] = train_data['POLYLINE'].apply(get_pickup_coordinates)
test_data['pickup_coords'] = test_data['POLYLINE'].apply(get_pickup_coordinates)

# Add distance to the nearest taxi stand as a feature
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

# 1. Random Forest
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# 2. Linear Regression
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 3. Support Vector Machine (SVM)
pipeline_svm = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

# 4. Neural Network Regression with Keras
model_nn = Sequential([
    Dense(128, activation='relu', input_dim=X.shape[1]),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])
model_nn.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and evaluate models
models = {
    'Random Forest': pipeline_rf,
    'Linear Regression': pipeline_lr,
    'SVM': pipeline_svm
}

# Train and evaluate each model
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"{name} RMSE: {rmse}")

# Train Neural Network separately
model_nn.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
y_pred_nn = model_nn.predict(X_test).flatten()
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
print(f"Neural Network RMSE: {rmse_nn}")

# Visualize spatial data (optional)
gdf = gpd.GeoDataFrame(
    metadata_taxistands,
    geometry=gpd.points_from_xy(metadata_taxistands.Longitude, metadata_taxistands.Latitude)
)
gdf.plot(marker='o', color='red', markersize=5)
plt.title("Taxi Stands")
plt.show()

