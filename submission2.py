import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


df = pd.read_csv('df.csv')

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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1) 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation Mean Squared Error: {mse:.4f}")

unseen_df = pd.read_csv('unseendf_example.csv')

X_unseen = preprocess_data(unseen_df, is_training=False, training_columns=training_columns)
X_unseen = scaler.transform(X_unseen)

unseen_df['predtime'] = model.predict(X_unseen).flatten()

unseen_df.to_csv('./predictions/mypred2.csv', index=False)
print("Predictions saved to mypred2.csv")

#0.1533