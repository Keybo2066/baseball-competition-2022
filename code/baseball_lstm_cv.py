import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold



def process_inn_time(df_input: pd.DataFrame):
    df_input = pd.merge(df_input, df_game[["game_id", "game_delayed_time", "game_started_time", "game_ended_time", "game_end_status"]], on="game_id")
    df_input["game_started_time"] = pd.to_datetime(df_input["game_started_time"])
    df_input["game_ended_time"] = pd.to_datetime(df_input["game_ended_time"])
    df_input["game_end"] = (df_input["game_ended_time"] - df_input["game_started_time"]).dt.total_seconds()
    # コールドの試合は削除
    df_input = df_input[df_input["game_end_status"].isna()]
    df_input = df_input[df_input["game_delayed_time"].isna()]
    return df_input

def create_dataset(df_train: pd.DataFrame, df_test: pd.DataFrame):

    def _get_lag_X(col: str, lag: int):
        X_game_train = []
        X_game_test = []
        for idx in range(0, len(df_X_game_train), lag):
            X_game_train.append(list(df_X_game_train[idx:idx+lag][col].values))
        for idx in range(0, len(df_X_game_test), lag):
            X_game_test.append(list(df_X_game_test[idx:idx+lag][col].values))
        scaler = MinMaxScaler()
        X_game_train = scaler.fit_transform(X_game_train)
        X_game_test = scaler.transform(X_game_test)
        return np.array(X_game_train).reshape(len(X_game_train), lag, 1), np.array(X_game_test).reshape(len(X_game_test), lag, 1)
    
    
    df_X_game_train = df_train[df_train["inn_top_bottom"] <= 62]
    df_X_game_test = df_test[df_test["inn_top_bottom"] <= 62]

    X_train, X_test = _get_lag_X(col="inn_end", lag=12)
    y_train = df_train.groupby("game_id")["game_end"].max().reset_index(drop=True)
    y_test = df_test.groupby("game_id")["game_end"].max().reset_index(drop=True)
    
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test, scaler_y
    
def create_model():
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

    model = Sequential()
    # model.add(Embedding(12, 1024, mask_zero = True))
    model.add(LSTM(256, batch_input_shape=(None, 12, 1), return_sequences=True)) # LSTM 128層
    model.add(LSTM(128)) # LSTM 128層
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='relu'))

    model.compile(loss=root_mean_squared_error, optimizer=Adam(learning_rate=.001) , metrics = ['mae'])
    return model

def plot_history(hist):
    # 損失値(Loss)の遷移のプロット
    plt.plot(hist.history['loss'],label="train set")
    plt.plot(hist.history['val_loss'],label="test set")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("../fig/loss.png")
    plt.show()

def fit_lstm(X, y, cv, batch_size, n_epoch, scaler_y):
    oof_pred = np.zeros(len(X), dtype=np.float32)
    models = []
    scores = []

    for i, (idx_train, idx_valid) in enumerate(cv):
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]
        
        model = create_model()
        # 学習
        hist = model.fit(X_train, y_train,
                        epochs=n_epoch,
                        validation_data=(X_valid, y_valid),
                        verbose=1,
                        batch_size=batch_size)
        
        plot_history(hist)

        oof = scaler_y.inverse_transform(model.predict(X_valid))
        oof_pred[idx_valid] = oof.flatten()
        models.append(model)

        print("-"*50)
        print(f"score {i+1}:\t {mean_absolute_error(y_valid, oof)}")
    
    print("*"*50)
    print(f"score: {mean_absolute_error(y, oof_pred)}")
    return models, oof_pred

# ====================================== fit
fold = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv = fold.split(X_train, y_train)
models, oof_pred = fit_lstm(X_train, y_train, cv, 128, 50, scaler_y)

# k 個のモデルの予測確率 (predict_proba) を作成. shape = (k, N_test, n_classes).
pred_prob = np.array([scaler_y.inverse_transform(model.predict(X_test)) for model in models])
print(f"1. shape: {pred_prob.shape}")

# k 個のモデルの平均を計算
pred_prob = np.mean(pred_prob, axis=0) # axis=0 なので shape の `k` が潰れる 
print(f"2. shape: {pred_prob.shape}")