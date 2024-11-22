import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

#Load data
df = pd.read_csv('df.csv')

#Manipulate data
def preprocess_data(data, is_training=True, training_columns=None):
    data = data.copy()
    
    data['birthdate'] = pd.to_datetime(data['birthdate'])
    data['date1'] = pd.to_datetime(data['date1'])
    data['date2'] = pd.to_datetime(data['date2'])
    data['age_at_race'] = (data['date1'] - data['birthdate']).dt.days / 365.25
    
    data['days_between_races'] = (data['date2'] - data['date1']).dt.days
    
    data['distance_diff'] = data['distance2'] - data['distance1']
    
    data = pd.get_dummies(data, columns=['stadium', 'trap1', 'trap2'], drop_first=True)
    
    if not is_training and training_columns is not None:
        for col in training_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[training_columns]
    
    drop_cols = ['birthdate', 'date1', 'date2', 'comment1']
    if is_training:
        drop_cols.append('time2')
    data = data.drop(columns=drop_cols, errors='ignore')
    
    return data

X = preprocess_data(df)
y = df['time2']

training_columns = X.columns

#Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Define model (using Extreme Gradient Boosting)
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    ))
])


#Define parameters for tuning
params = {
    'regressor__n_estimators': [100, 500, 1000],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [4, 6, 8],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9]
}

#Tune hyperparameters
random_search = RandomizedSearchCV(
    model,
    params,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

#Run the grid search
random_search.fit(X_train, y_train)

#Get the best parameters
best_params = random_search.best_params_

#Evaluate the best model on validation data
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation Mean Squared Error: {mse:.4f}")

#Run model on testing dataset
unseen_df = pd.read_csv('unseendf_example.csv')
X_unseen = preprocess_data(unseen_df, is_training=False, training_columns=training_columns)

unseen_df['time2'] = best_model.predict(X_unseen)

unseen_df.to_csv('./predictions/mypred.csv', index=False)
print("Predictions saved to mypred.csv")