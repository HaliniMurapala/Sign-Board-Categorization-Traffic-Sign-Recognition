# ─── test_image.py ─────────────────────────────────────────
# Test trained model on a single image

import numpy as np
import cv2
import tensorflow as tf
from labels import LABELS

# ─── Load Model ─────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model("model/traffic_sign_model.h5")
print("✅ Model loaded!")

IMAGE_SIZE = (32, 32)

def predict_sign(image_path):
    """Predict traffic sign from image"""

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return

    # Preprocess
    img_resized   = cv2.resize(img, IMAGE_SIZE)
    img_normalized = img_resized.astype("float32") / 255.0
    img_expanded  = np.expand_dims(img_normalized, axis=0)

    # Predict
    predictions   = model.predict(img_expanded, verbose=0)
    class_id      = np.argmax(predictions[0])
    confidence    = predictions[0][class_id] * 100

    # Get sign name
    sign_name = LABELS.get(class_id, "Unknown Sign")

    # Show results in terminal
    print("\n" + "=" * 50)
    print("  Traffic Sign Prediction Result")
    print("=" * 50)
    print(f"  Image      : {image_path}")
    print(f"  Sign       : {sign_name}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  Class ID   : {class_id}")
    print("=" * 50)

    # Show top 3 predictions
    print("\n  Top 3 Predictions:")
    top3 = np.argsort(predictions[0])[::-1][:3]
    for i, idx in enumerate(top3):
        print(f"  {i+1}. {LABELS.get(idx, 'Unknown')} "
              f"— {predictions[0][idx]*100:.2f}%")

    # Show image with prediction
    label_text = f"{sign_name} ({confidence:.1f}%)"
    cv2.putText(
        img, label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2
    )
    cv2.imshow("Traffic Sign Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ─── Run prediction ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ")

    predict_sign(image_path)
