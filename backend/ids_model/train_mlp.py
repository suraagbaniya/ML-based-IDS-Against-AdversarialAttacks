import joblib
import tensorflow as tf


#Load the data
x_train = joblib.load("data/processed/x_train.pkl")
y_train = joblib.load("data/processed/y_train.pkl")
x_test  = joblib.load("data/processed/x_test.pkl")
y_test  = joblib.load("data/processed/y_test.pkl")


#Train MLP model (neural network model)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=256, validation_split=0.2)

#Save MLP model 
model.save("backend/ids_model/mlp_model.keras")

