import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.api.utils import to_categorical

# Path to the collected data
DATA_PATH = os.path.join('MP_Data')
actions = ['A', 'B', 'C']
sequence_length = 30  # Number of frames per sequence

# Label mapping (action to numerical value)
label_map = {label: num for num, label in enumerate(actions)}

# Load and preprocess data
def load_data():
    sequences, labels = [], []
    
    for action in actions:
        for sequence in range(30):
            window = []
            for frame_num in range(sequence_length):
                file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                if os.path.exists(file_path):
                    res = np.load(file_path)
                    window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    
    return np.array(sequences), np.array(labels)

# Main preprocessing function
def preprocess():
    # Load the sequences and labels
    X, y = load_data()

    # Normalize the data
    X = X / np.max(X)

    # Convert labels to categorical (one-hot encoding)
    y = to_categorical(y).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Save the preprocessed data for later use
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    print(f'Data Preprocessing Complete. Train data shape: {X_train.shape}, Test data shape: {X_test.shape}')

if __name__ == "__main__":
    preprocess()
