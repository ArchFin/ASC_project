import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap

# Step 1: Generate or load your EEG-like data
n_samples_per_state = 250
n_features = 3

# Create artificial patterns for each state
state_1 = np.random.normal(loc=1, scale=0.2, size=(n_samples_per_state, n_features))  # Low values
state_2 = np.random.normal(loc=[3, 1, 2], scale=0.2, size=(n_samples_per_state, n_features))  # Mixed values
state_3 = np.random.normal(loc=5, scale=0.2, size=(n_samples_per_state, n_features))  # High values
state_4 = np.sin(np.linspace(0, 10, n_samples_per_state)).reshape(-1, 1) * np.random.normal(1, 0.1, (n_samples_per_state, n_features))  # Oscillating

# Concatenate states to form the dataset
data = np.vstack([state_1, state_2, state_3, state_4])

# Create corresponding labels
labels = np.array([0] * n_samples_per_state + 
                  [1] * n_samples_per_state + 
                  [2] * n_samples_per_state + 
                  [3] * n_samples_per_state)

# Visualize the data
plt.figure(figsize=(10, 6))
for i in range(n_features):
    plt.plot(data[:, i], label=f"Feature {i+1}")
plt.xlabel("Samples")
plt.ylabel("Feature Values")
plt.title("Artificial Data with Patterns")
plt.legend()
plt.show()

# Step 2: Preprocess the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)

# Step 3: Define the neural network
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer
    keras.layers.Dense(64, activation='relu'),  # Second hidden layer
    keras.layers.Dense(4, activation='softmax')  # Output layer for classification (4 classes)
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 6: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 7: Visualize training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Step 8: Confusion Matrix
# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()


