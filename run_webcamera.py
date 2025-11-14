import cv2
import numpy as np
import tensorflow as tf
import time

# ------------------------------
# MODEL & LABEL PATH
# ------------------------------
MODEL_PATH = "F:/apple_dataset/apple_pest_model.tflite"
LABEL_PATH = "F:/apple_dataset/labels.txt"

# ------------------------------
# LOAD MODEL
# ------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------
# LOAD LABELS
# ------------------------------
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

print("Labels:", labels)

# ------------------------------
# CAMERA DETECTOR
# ------------------------------
def get_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        time.sleep(1)
        if cap.isOpened():
            print(f"📸 Camera detected at index {i}")
            return cap
        cap.release()
    return None

cap = get_camera()

if cap is None:
    print("❌ No camera found. Close other apps using the camera.")
    exit()

print("Camera started... Press 'q' to quit")

# ------------------------------
# MAIN LOOP
# ------------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠ Frame error... retrying...")
        time.sleep(0.3)
        continue

    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    img = cv2.resize(frame, (width, height))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    # Run model
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Multi-class mode
    pred_idx = int(np.argmax(output[0]))

    # Safe label
    label = labels[pred_idx] if pred_idx < len(labels) else "Unknown"

    # ONLY LABEL — no percentage
    cv2.putText(frame, f"{label}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Apple Pest Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
