import numpy as np
from keras.api.models import load_model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# Load the saved model
model = load_model('trained_model/asl_model.h5')

# Re-compile the model if necessary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Load the test data and labels
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Perform predictions
yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()  # Convert to a list of true labels
yhat = np.argmax(yhat, axis=1).tolist()  # Convert to a list of predicted labels

# Compute and print the multilabel confusion matrix and accuracy score
print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))
