import os
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense
from keras.api.callbacks import TensorBoard, ReduceLROnPlateau

# Load training data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Set up log directory for TensorBoard
log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)

# Initialize the model
model = Sequential()

# First LSTM Layer
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))

# Second LSTM Layer
model.add(LSTM(128, return_sequences=True, activation='relu'))

# Third LSTM Layer
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Dense Layer
model.add(Dense(64, activation='relu'))

# Another Dense Layer
model.add(Dense(32, activation='relu'))

# Output Layer with 3 units (one for each class)
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback, lr_scheduler])

# Save the trained model
model_dir = 'trained_model'
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, 'asl_model.h5'))
