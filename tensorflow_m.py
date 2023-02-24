import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Dummy Testing Model
def generate_dataset(num_samples, test_size):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)

    # Nueral network consisting of 2,5,1 nodes in subsequent layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Model compiler
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss="MSE")

    # Train model
    model.fit(x_train, y_train, epochs=100)

    # Model evaluation
    print("\nModel Evaluation:")
    model.evaluate(x_test, y_test, verbose=1)

    # Prediction run
    data = np.array([[0.2, 0.4], [0.3, 0.5]])
    predictions = model.predict(data)
    print(predictions) #([0.59627783], [0.7523067593574524])

    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))

