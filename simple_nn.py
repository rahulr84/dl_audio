# Creating a simple NN which can execute the following function
"""
float function(float x){
    float y = (3 * x) - 5;
    return y;
}
"""

# Import tensorflow and other libs
import tensorflow as tf
import numpy as np
from tensorflow import keras

if __name__ == "__main__":

    # create train/test sets
    X_train = np.random.randn(1000) # 1000 training examples
    y_train = (3 * X_train) - 5
    X_test = np.random.randn(400) # 400 test examples
    y_test = (3 * X_test) - 5

    # define model with NN parameters
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mean_squared_error'])

    # train/fit the model
    hist = model.fit(X_train, y_train, epochs=100)

    # Evaluate the model
    loss, loss = model.evaluate(X_test, y_test)

    # predict using the trained model and test set
    y_pred = model.predict([10.0])

    print("3 * " + str(10.0) + " - 5 = " + str(y_pred) + " with loss of " + str(loss))



