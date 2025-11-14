import tensorflow as tf

# Load the original trained model (.h5)
model = tf.keras.models.load_model("F:/apple_dataset/apple_pest_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save output file
with open("F:/apple_dataset/apple_pest_model.tflite", "wb") as f:
    f.write(tflite_model)

print("DONE: apple_pest_model.tflite created successfully!")
