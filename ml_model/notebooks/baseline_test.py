# ml_model/notebooks/baseline_test.py

"""
RouteX - FINAL Baseline YOLOv8 Test (No Network)
This script verifies that YOLO can be loaded and run successfully
by generating its own test image locally.
"""

from ultralytics import YOLO
import numpy as np
from PIL import Image


def run_local_baseline_test():
    """
    Loads a pre-trained YOLOv8 model and runs inference on a
    locally generated test image to verify the environment.
    """
    print("--- RouteX ML Baseline Test (Local Network-Free Mode) ---")

    # 1. Load Model (it should be cached locally from the last run)
    print("[1/3] Loading pre-trained YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("✅ Model loaded successfully.")

    # 2. Create a Local Test Image
    print("[2/3] Generating a local test image in memory...")
    # Create a random 640x480 RGB image
    img_array = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    print("✅ Local test image created.")

    # 3. Run Inference
    print("[3/3] Running inference on the local image...")
    results = model(img)

    # Save the result image so you can see it
    results[0].save(filename='baseline_result.jpg')
    print("✅ Result image saved to 'baseline_result.jpg'.")

    # Check the result
    if len(results[0].boxes) == 0:
        print("\nSUCCESS: The model ran correctly and found no objects in the random image, as expected.")
    else:
        # This can sometimes happen if random noise looks like a pattern
        print("\nSUCCESS: The model ran correctly and found some random patterns.")

    print("\n--- ✅✅✅ Your ML Environment is Ready for RouteX! ---")


if __name__ == "__main__":
    run_local_baseline_test()
