import os
import joblib
import csv
from src_elena.training_module import *
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#Creation sequential model and Functions
def create_model_Seq (input):
    sequen_model = Sequential([
        # keras.Input(shape=input),
        keras.layers.Dense(2048, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ])
    return sequen_model

def _ensure_2d(input_model):
    return input_model.reshape(input_model.shape[0], -1) if input_model.ndim == 3 else input_model

def train_and_test(fingerprints ,model_path, scaler_path):
    """
    Trains and test the sequential model.
    :return:
    """
    # Obtain data
    X, y = data_from_csv(fingerprints)
    scaler = RobustScaler()
    # Set division
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=10,
        restore_best_weights=True,
        mode='min'
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Model training and testing
    model = create_model_Seq(X)
    model.compile(
        optimizer="adam",
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=1,
        callbacks=[early_stop, reduce_lr]
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    test_model(X_test, y_test, scaler_path, model_path, os.path.join("results"))
    mae_graphic(history, os.path.join("results"))


