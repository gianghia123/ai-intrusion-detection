# Deep Learning Model Architectures (Tier 1 & Tier 2 Neural Networks)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def build_tier1_binary_model(window_size, num_features):
    """Tầng 1: Phân biệt Normal vs Attack (Binary Classification)"""
    model = Sequential([
        Input(shape=(window_size, num_features)),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid', name='binary_output')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_tier2_multiclass_model(window_size, num_features, num_classes):
    """Tầng 2: Phân loại chi tiết loại mã độc (Multi-class Classification)"""
    model = Sequential([
        Input(shape=(window_size, num_features)),
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax', name='multiclass_output')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
