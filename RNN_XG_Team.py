import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU
import pickle
import os
import matplotlib.pyplot as plt
from ML_Functions_LoadArrays import *
import numpy as np
import shap
from tensorflow.keras.layers import Masking
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Masking, GRU, Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import contextlib
import sys
import os
#matplotlib.use('Agg')  # o 'Qt5Agg'
import matplotlib.pyplot as plt

verbose = False
plot = True


# Check if the input file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python RNN_XG_Team.py <input_file>")
    sys.exit(1)

team = os.path.basename(sys.argv[1]).split('_')[2].split('.')[0]  # Extract team name from the input file name

input_file = sys.argv[1]

# Load the dataset from the provided CSV file
tXG_all = pd.read_csv(input_file)


# Drop columns related to speed, time, or half, as they are not needed for modeling
cols_to_drop = [col for col in tXG_all.columns if ('speed' in col or 'Time' in col  or 'Half' in col)]
tXG_all = tXG_all.drop(columns=cols_to_drop)

# Assign a unique shot number (XG_n) to each sequence of XG values
shot = 0 
XG_n = [0]
i = 0
while i < len(tXG_all):
    XG_n.append(shot)
    while i + 1 < len(tXG_all) and abs(tXG_all['XG'][i] - tXG_all['XG'][i + 1]) < 0.0001:
        i += 1
        XG_n.append(shot)
    shot += 1
    i += 1

# Remove the extra first element if necessary
XG_n = XG_n[:len(tXG_all)]
tXG_all['XG_n'] = XG_n

# Remove columns with too many missing values (less than min_presence threshold)
min_presence = 0.20
threshold = int(min_presence * len(tXG_all))
cols_to_keep = [col for col in tXG_all.columns if tXG_all[col].count() >= threshold]
tXG_all = tXG_all[cols_to_keep]
removed_cols = [col for col in tXG_all.columns if col not in cols_to_keep]

# Prepare the features by dropping the first column (likely an index or ID)
train_positions = tXG_all.drop(tXG_all.columns[[0]], axis=1)

# Group the data by shot number (XG_n)
grpous = tXG_all.groupby('XG_n')
xg_groups = {group['XG'].iloc[0]: {'group': group, 'n': name} for name, group in grpous}
train_X = []
train_y = []
appo = []

# For each group (shot), extract sequences of features for modeling
for xg_val, group_info in xg_groups.items():
    group_df = group_info['group']
    appo = []
    # Sample frames from the end of the sequence, stepping backwards
    for i in range(group_df.shape[0] - 1, -1 , -1):
        appo.append(group_df.drop(['XG' , 'XG_n'], axis=1).iloc[i].values)
    appo = np.array(appo)
    appo = np.flip(appo, axis=0)  # Flip to maintain correct temporal order
    train_X.append(appo)
    train_y.append(xg_val)

train_X = np.array(train_X, dtype=object)
train_y = np.array(train_y, dtype=float)


if verbose:
    print("Shape of train_X:", train_X.shape)
    print("Shape of train_y:", train_y.shape)

# Pad sequences to the same length for RNN input
maxlen = max([seq.shape[0] for seq in train_X])
train_X_padded = pad_sequences(train_X, maxlen=maxlen, dtype='float32', padding='post')
train_y = np.nan_to_num(train_y, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize features across all samples and timesteps
num_samples, timesteps, num_features = train_X_padded.shape
scaler = StandardScaler()
train_X_padded = scaler.fit_transform(train_X_padded.reshape(-1, num_features)).reshape(num_samples, timesteps, num_features)
train_X_padded = np.nan_to_num(train_X_padded, nan=-100.0, posinf=0.0, neginf=0.0)

if verbose:
    print("After padding and standardization:")
    print("Shape of train_X_padded:", train_X_padded.shape)
    print("Shape of train_y:", train_y.shape)


# Split data into positive and negative XG samples
X_train_padded_negative = train_X_padded[train_y <= 0]
X_train_padded_positive = train_X_padded[train_y > 0]
train_y_negative = train_y[train_y <= 0]
train_y_positive = train_y[train_y > 0]

# Further split positive and negative samples into training and test sets
X_train_positive, X_test_positive, y_train_positive, y_test_positive = train_test_split(X_train_padded_positive, train_y_positive, test_size=0.15, random_state=42)
X_train_negative, X_test_negative, y_train_negative, y_test_negative = train_test_split(X_train_padded_negative, train_y_negative, test_size=0.15, random_state=42)



# Define the RNN model architecture for positive XG samples
inputs = Input(shape=(maxlen, num_features))
x = Masking(mask_value=-100.0)(inputs)
#x = GRU(64, return_sequences=True)(x)
#x = Dropout(0.1)(x)
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.1)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)
#x = Dense(64, activation='relu')(x)
x = Dense(16, activation='sigmoid')(x)
outputs = Dense(1, activation='linear')(x)

model_positive = Model(inputs, outputs)
model_positive.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])


if verbose:
    print("Trining positive samples size:", X_train_positive.shape)

# Train the model on positive XG samples, with early stopping and periodic logging
model_positive.fit(
    X_train_positive, y_train_positive,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    verbose=0, 
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
                print(f"Epoch {epoch + 1}: Loss = {logs['mae']}, Val Loss = {logs['val_mae']}")
                if (epoch + 1) % 50 == 0 and verbose else None
        )
    ]
)

# Evaluate the trained model on the positive test set
loss, mae = model_positive.evaluate(X_test_positive, y_test_positive)
if verbose:
    print(f"Test Loss (positive): {loss}, Test MAE (positive): {mae}")

# Make predictions on the positive test set and compute absolute errors
predictions = model_positive.predict(X_test_positive)
distances = np.abs(predictions.flatten() - y_test_positive)

# Plot training history and histogram of prediction errors

if plot: 
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training and validation loss over epochs
    history = model_positive.history.history
    axs[0].plot(history['loss'], label='Training Loss')
    axs[0].plot(history['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot histogram of absolute prediction errors
    axs[1].hist(distances, bins=20, color='skyblue', edgecolor='black')
    axs[1].set_xlabel('Absolute Distance |Predicted XG - Actual XG|')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Histogram of distances\nTest MAE Positive: {mae:.4f}')

    plt.tight_layout()
    plt.show()


# Select random samples from the positive test set for SHAP analysis
idx = np.random.choice(X_test_positive.shape[0], 20)
X_sample_seq = X_test_positive[idx]
X_sample_seq_flat = X_sample_seq.reshape(X_sample_seq.shape[0], -1)

# Define a wrapper for model prediction to work with SHAP
def model_predict_flat(x):
    x_seq = x.reshape((x.shape[0], X_sample_seq.shape[1], X_sample_seq.shape[2]))
    return model_positive.predict(x_seq)

# Compute SHAP values for the selected samples (suppressing output)
with open(os.devnull, "w") as fnull, \
    contextlib.redirect_stdout(fnull), \
    contextlib.redirect_stderr(fnull):
    explainer = shap.KernelExplainer(model_predict_flat, X_sample_seq_flat)
    shap_values = explainer.shap_values(X_sample_seq_flat, nsamples=50, silent=True, progress_bar=False)

# Calculate mean SHAP values for each feature across all timesteps
timesteps = X_sample_seq.shape[1]
num_features = X_sample_seq.shape[2]
base_feature_names = train_positions.drop(['XG_n'], axis=1).columns.tolist()

mean_importance = np.mean(shap_values, axis=0)  # shape: (timesteps * num_features,)
mean_importance = mean_importance.reshape(timesteps, num_features)
mean_importance_per_feature = mean_importance.mean(axis=0)  # shape: (num_features,)

# Create a DataFrame of feature importances
importances_df = pd.DataFrame({
    'feature': base_feature_names,
    'mean_shap': mean_importance_per_feature
}).sort_values('mean_shap', ascending=False)

# Extract player identifiers from feature names
players = [col.split('_')[1] for col in importances_df.feature if 'us_' in col]

# Helper function to compute mean SHAP value for each player
def shape_mean(number, df ) : 
    str_to_find = '_' + str(number) + '_'
    raw = [col for col in df.feature if str_to_find in col]
    return df[df.feature.isin(raw)].mean_shap.mean()

# Calculate the mean SHAP value for each player
mean_shap_values = {}
for i in (players):
    mean_shap_values[f'{i}'] = shape_mean(i, importances_df)

mean_shap_values = dict(sorted(mean_shap_values.items(), key=lambda item: item[1], reverse=True))


if plot: 
    # Plot mean SHAP value per player for attacking situations
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x=list(mean_shap_values.keys()),
        y=list(mean_shap_values.values()),
        palette="viridis"
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title(f'Mean SHAP Value per Player {team} for attacking situations', fontsize=18, fontweight='bold')
    plt.ylabel('Mean SHAP value', fontsize=14)
    plt.xlabel('Player', fontsize=14)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

if verbose:
    print("Negative Model")
    print("Training neative size :", X_train_negative.shape)
    print("Testing neative size :", X_test_negative.shape)

# Define the RNN model architecture for negative XG samples
outputs_negative = Dense(1, activation=None)(x)  # linear output
inputs_negative = Input(shape=(maxlen, num_features))

nputs_negative = Input(shape=(maxlen, num_features))
x_neg = Masking(mask_value=-100.0)(inputs_negative)
#x_neg = GRU(64, return_sequences=True)(x_neg)
#x_neg = Dropout(0.1)(x_neg)
x_neg = LSTM(64, return_sequences=False)(x_neg)
x_neg = Dropout(0.1)(x_neg)
x_neg = Dense(64, activation='relu')(x_neg)
x_neg = Dropout(0.1)(x_neg)
#x_neg = Dense(64, activation='relu')(x_neg)
x_neg = Dense(16, activation='sigmoid')(x_neg)
outputs_negative = Dense(1, activation=None)(x_neg)  # linear output

model_negative = Model(inputs_negative, outputs_negative)
model_negative.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

if verbose:
    print("Model summary for negative samples:")
    model_negative.summary()
# Train the model on negative XG samples, with early stopping and periodic logging
model_negative.fit(
    X_train_negative, y_train_negative,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    verbose=0, 
    shuffle=True,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
                print(f"Epoch {epoch + 1}: Loss = {logs['mae']}, Val Loss = {logs['val_mae']}")
                if (epoch + 1) % 50 == 0 and verbose else None
        )
    ]
)

# Evaluate the trained model on the negative test set
loss, mae = model_negative.evaluate(X_test_negative, y_test_negative)
if verbose:
    print(f"Test Loss (negative): {loss}, Test MAE (negative): {mae}")

# Make predictions on the negative test set and compute absolute errors
predictions_negative = model_negative.predict(X_test_negative)
distances_negative = np.abs(predictions_negative.flatten() - y_test_negative)


if plot:
    # Plot training history and histogram of prediction errors for negative samples
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    history_negative = model_negative.history.history
    axs[0].plot(history_negative['loss'], label='Training Loss')
    axs[0].plot(history_negative['val_loss'], label='Validation Loss')
    axs[0].set_title('Training and Validation Loss (Negative)')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].hist(distances_negative, bins=20, color='skyblue', edgecolor='black')
    axs[1].set_xlabel('Absolute Distance |Predicted XG - Actual XG| (Negative)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(f'Histogram of distances (Negative)\nTest MAE Negative: {mae:.4f}')
    plt.tight_layout()
    plt.show()

# Find the test sample with the largest prediction error
distances_negative = np.abs(predictions_negative.flatten() - y_test_negative)
idx_max = np.argmax(distances_negative)

# Prepare negative test samples for SHAP analysis
idx = np.random.choice(X_test_negative.shape[0])
X_sample_seq_negative = X_test_negative
X_sample_seq_flat_negative = X_sample_seq_negative.reshape(X_sample_seq_negative.shape[0], -1)

# Define a wrapper for model prediction to work with SHAP (negative samples)
def model_predict_flat_negative(x):
    x_seq = x.reshape((x.shape[0], X_sample_seq_negative.shape[1], X_sample_seq_negative.shape[2]))
    return model_negative.predict(x_seq)

# Compute SHAP values for negative samples (suppressing output)
with open(os.devnull, "w") as fnull, \
     contextlib.redirect_stdout(fnull), \
     contextlib.redirect_stderr(fnull):
    explainer_negative = shap.KernelExplainer(model_predict_flat_negative, X_sample_seq_flat_negative, silent=True, progress_bar=False)
    shap_values_negative = explainer_negative.shap_values(X_sample_seq_flat_negative, nsamples=50, silent=True, progress_bar=False)

# Calculate mean SHAP values for each feature across all timesteps (negative samples)
mean_importance_negative = np.mean(shap_values_negative, axis=0)  
mean_importance_negative = mean_importance_negative.reshape(timesteps, num_features)
mean_importance_per_feature_negative = mean_importance_negative.mean(axis=0)  
importances_df_negative = pd.DataFrame({
    'feature': base_feature_names,
    'mean_shap': mean_importance_per_feature_negative.astype(float) 
}).sort_values('mean_shap', ascending=False)

# Extract player identifiers from feature names (negative samples)
players_negative = [col.split('_')[1] for col in importances_df_negative.feature if 'us_' in col]

# Helper function to compute mean SHAP value for each player (negative samples)
def shape_mean_negative(number, df ) : 
    str_to_find = '_' + str(number) + '_'
    raw = [col for col in df.feature if str_to_find in col]
    return df[df.feature.isin(raw)].mean_shap.mean()

# Calculate the mean SHAP value for each player (negative samples)
mean_shap_values_negative = {}
for i in (players_negative):
    mean_shap_values_negative[f'{i}'] = shape_mean_negative(i, importances_df_negative)
mean_shap_values_negative = dict(sorted(mean_shap_values_negative.items(), key=lambda item: item[1], reverse=True))


if plot:
    # Plot mean SHAP value per player for defensive situations
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x=list(mean_shap_values_negative.keys()),
        y=list(mean_shap_values_negative.values()),
        palette="viridis"
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title(f'Mean SHAP Value per Player {team} for defensive situations', fontsize=18, fontweight='bold')
    plt.ylabel('Mean SHAP value', fontsize=14)
    plt.xlabel('Player', fontsize=14)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


# Combine mean SHAP values for all players from both positive and negative samples
average_shap_total = {}
for player in mean_shap_values.keys():
    if player in mean_shap_values_negative:
        average_shap_total[player] = (mean_shap_values[player] + mean_shap_values_negative[player]) / 2
    else:
        average_shap_total[player] = mean_shap_values[player]

average_shap_total = dict(sorted(average_shap_total.items(), key=lambda item: item[1], reverse=True))
if plot: 
    plt.figure(figsize=(14, 7))
    sns.barplot(
        x=list(average_shap_total.keys()),
        y=list(average_shap_total.values()),
        palette="viridis"
    )
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title(f'Mean SHAP Value per Player {team} for all situations', fontsize=18, fontweight='bold')
    plt.ylabel('Mean SHAP value', fontsize=14)
    plt.xlabel('Player', fontsize=14)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()



#Print the final average SHAP values for all players
pd.DataFrame(list(average_shap_total.items()), columns=['player', 'mean_shap']).to_csv(f'{team}_shap_total.csv', index=False)