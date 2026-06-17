import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# cargar dataset
X = np.load("dataset/X.npy")
y = np.load("dataset/y.npy")

# normalizar
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "dataset/scaler.pkl")
np.save("dataset/mean.npy", scaler.mean_)
np.save("dataset/std.npy", scaler.scale_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20, batch_size=32)

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

model.save("dataset/model_tf.h5")

model = tf.keras.models.load_model("dataset/model_tf.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("dataset/model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido a TFLite")