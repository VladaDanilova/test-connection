import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create a simple dataset
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
Y_train = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)  # Y = 2*X

# Define a simple sequential model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100)

# Predict Y values
Y_pred = model.predict(X_train)

# Plot actual vs predicted values
plt.scatter(X_train, Y_train, color='blue', label='Actual')
plt.plot(X_train, Y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
