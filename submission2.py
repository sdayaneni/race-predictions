import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout, InputLayer
# from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint




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

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     # tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(32, activation='relu'),
#     # tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(1) 
# ])

# Reshape data to 3D for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# Define the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(shape=(1, X_train.shape[2])))  # Define input shape explicitly
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.LSTM(128, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1))  # Regression output

opt = tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 1e-6)

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.compile(optimizer=opt, loss='mse', metrics=['mae'])


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, verbose=1, callbacks=[early_stopping])


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