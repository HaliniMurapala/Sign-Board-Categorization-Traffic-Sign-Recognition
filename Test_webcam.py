# ─── test_webcam.py ────────────────────────────────────────
# Real time traffic sign detection using webcam

import numpy as np
import cv2
import tensorflow as tf
from labels import LABELS

# ─── Load Model ─────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model("model/traffic_sign_model.h5")
print("✅ Model loaded — Starting webcam...")

IMAGE_SIZE  = (32, 32)
CONFIDENCE_THRESHOLD = 70.0  # Only show if confidence > 70%

# ─── Start Webcam ───────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit()

print("✅ Webcam started!")
print("   Press Q to quit")
print("   Press S to save screenshot")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame!")
        break

    # Process every 5 frames for performance
    frame_count += 1
    if frame_count % 5 == 0:

        # Preprocess frame for prediction
        img_resized    = cv2.resize(frame, IMAGE_SIZE)
        img_normalized = img_resized.astype("float32") / 255.0
        img_expanded   = np.expand_dims(img_normalized, axis=0)

        # Predict
        predictions = model.predict(img_expanded, verbose=0)
        class_id    = np.argmax(predictions[0])
        confidence  = predictions[0][class_id] * 100
        sign_name   = LABELS.get(class_id, "Unknown")

        # Show prediction on frame
        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{sign_name}"
            conf  = f"Confidence: {confidence:.1f}%"
            color = (0, 255, 0)  # Green for high confidence
        else:
            label = "No sign detected"
            conf  = f"Confidence: {confidence:.1f}%"
            color = (0, 0, 255)  # Red for low confidence

        # Draw results on frame
        cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, conf, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Print to terminal
        if confidence >= CONFIDENCE_THRESHOLD:
            print(f"Detected: {sign_name} — {confidence:.1f}%")

    # Show frame
    cv2.imshow("Traffic Sign Detection — Press Q to quit", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('s'):
        screenshot_name = f"screenshot_{frame_count}.png"
        cv2.imwrite(screenshot_name, frame)
        print(f"✅ Screenshot saved: {screenshot_name}")

cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed!")
