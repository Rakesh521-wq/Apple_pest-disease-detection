import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Path to your dataset on F: drive ---
train_dir = "F:/apple_dataset/"  # <-- updated path

# --- Image generator ---
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# --- Simple CNN model ---
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(train_generator.num_classes,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Train ---
model.fit(train_generator, validation_data=val_generator, epochs=10)

# --- Save model on F: drive ---
model.save("F:/apple_pest_model.h5")  # <-- updated save path
print("Model training complete and saved at F:/apple_pest_model.h5")
