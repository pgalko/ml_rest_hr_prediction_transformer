import pandas as pd
import datetime
import numpy as np
from scipy.stats import pearsonr

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow import keras
from keras import layers

from matplotlib import pyplot as plt

data = pd.read_csv("Wellness_Data_All.csv")
#drop all rows that have NaN values in oura_sleep_rmssd column
data = data.dropna(subset=['wellness_resting_heart_rate'])

data = data[['calendar_date','wellness_resting_heart_rate']]

pred_column = 'wellness_resting_heart_rate'

data[pred_column] = data[pred_column].astype(float)

data = data.sort_values('calendar_date', ascending=True)
data = data.reset_index(drop=True)

days_back = 575 # 0 to predict tomorrow's value

today = datetime.datetime.today()-datetime.timedelta(days=days_back)
data['calendar_date']= pd.to_datetime(data['calendar_date'])

SEQUENCE_SIZE = 3

rmssd_train_df = data[data['calendar_date']<today]
rmssd_test_df = data[data['calendar_date']>=today]

rmssd_train_pred, rmssd_test_pred = [df[pred_column].tolist() for df in (rmssd_train_df, rmssd_test_df)]

def to_sequences(seq_size, obs):
    x, y = [], []
    for i in range(len(obs)-seq_size):
        window, after_window = obs[i:(i+seq_size)], obs[i+seq_size]
        # Average of the window
        avg = sum(window) / len(window)
        list_2d = [[avg]]
        x.append(list_2d)
        y.append(after_window)
    return np.array(x), np.array(y)
    
x_train,y_train = to_sequences(SEQUENCE_SIZE,rmssd_train_pred)
x_test,y_test = to_sequences(SEQUENCE_SIZE,rmssd_test_pred)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    num_layers,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
    num_layers=6,# PG:Default is 6
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4)
)
#model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, \
    restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

pred = model.predict(x_test)

# Testing the model

def evaluate(y_test,predictions):
    mape = mean_absolute_percentage_error(y_test,predictions)*100
    mape_man = np.mean(np.abs((y_test-predictions)/y_test))*100
    return mape,mape_man

rmse = np.sqrt(metrics.mean_squared_error(pred,y_test))
mape =  evaluate(y_test, pred)[0]
# Calculate correlation between actual and predicted values
corr, _ = pearsonr(y_test, pred)
# Print the results
print('Model Correlation: {}'.format(corr))
print("RMSE: {}".format(rmse))
print("MAPE: {} %".format(mape))

# PLOTS #

# Plot the predicted vs measured on a scatter plot
fig, ax = plt.subplots()
ax.scatter(y_test, pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# Plot predicted vs Measured on a line plot
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="Measured")
plt.plot(x_ax, pred, label="Predicted")
plt.xlabel('Timeline (Days)')
plt.ylabel('Resting HR (BPM)')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()




