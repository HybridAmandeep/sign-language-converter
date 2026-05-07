"""
=============================================================================
  STEP 2: PREPROCESSING SCRIPT
  Sign Language to Text/Speech Converter
=============================================================================
  This script reads all the gesture images you collected in Step 1,
  extracts hand landmark coordinates using MediaPipe, and saves them
  as a CSV file that the ML model will learn from.

  WHAT IT DOES (simple explanation):
  - Opens each image from the data/ folder
  - Finds the hand in the image using MediaPipe
  - Extracts 21 key points on the hand (fingertips, knuckles, wrist, etc.)
  - Each point has x, y, z coordinates = 21 x 3 = 63 numbers per image
  - Saves all these numbers + gesture label into a CSV file

  HOW TO USE:
  1. Make sure you've collected data using collect_data.py first
  2. Run: python preprocess.py
  3. Wait for it to process all images
  4. Output: data/landmarks.csv
=============================================================================
"""

import os
import csv
import cv2
import mp_patch
import mediapipe as mp
import numpy as np

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "landmarks.csv")

# ─── SETUP MEDIAPIPE ────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # Image mode (not video)
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Number of landmarks per hand
NUM_LANDMARKS = 21
# Each landmark has x, y, z = 3 values
FEATURES_PER_LANDMARK = 3
TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63


def extract_landmarks(image_path):
    """
    Extract hand landmarks from a single image.

    Returns:
        list of 63 floats (normalized landmarks) or None if no hand detected
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert BGR to RGB (MediaPipe expects RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image
    results = hands.process(rgb_image)

    # Check if any hand was detected
    if not results.multi_hand_landmarks:
        return None

    # Get the first hand's landmarks
    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract all landmark coordinates
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    # ─── NORMALIZE relative to wrist (landmark 0) ───────────────
    # This makes the model work regardless of where the hand is on screen
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized = []
    for i in range(0, len(landmarks), 3):
        normalized.append(landmarks[i] - wrist_x)      # x relative to wrist
        normalized.append(landmarks[i + 1] - wrist_y)   # y relative to wrist
        normalized.append(landmarks[i + 2] - wrist_z)   # z relative to wrist

    return normalized


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("  SIGN LANGUAGE DATA PREPROCESSING")
    print("=" * 60)
    print()

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print("[ERROR] Data directory not found!")
        print(f"  Expected: {DATA_DIR}")
        print("  → Run collect_data.py first to capture gesture images.")
        return

    # Get all gesture folders
    gesture_folders = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not gesture_folders:
        print("[ERROR] No gesture folders found in data/")
        print("  → Run collect_data.py first to capture gesture images.")
        return

    print(f"[INFO] Found {len(gesture_folders)} gesture classes: {gesture_folders}")
    print()

    # ─── CREATE CSV HEADER ──────────────────────────────────────
    # Column names: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
    header = []
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    # ─── PROCESS ALL IMAGES ─────────────────────────────────────
    all_data = []
    total_processed = 0
    total_skipped = 0

    for gesture in gesture_folders:
        gesture_path = os.path.join(DATA_DIR, gesture)
        image_files = sorted([
            f for f in os.listdir(gesture_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        processed = 0
        skipped = 0

        print(f"[PROCESSING] Gesture '{gesture}': {len(image_files)} images...", end=" ")

        for img_file in image_files:
            img_path = os.path.join(gesture_path, img_file)
            landmarks = extract_landmarks(img_path)

            if landmarks is not None:
                # Add the label and save
                row = landmarks + [gesture]
                all_data.append(row)
                processed += 1
            else:
                skipped += 1

        success_rate = (processed / len(image_files) * 100) if image_files else 0
        print(f"✓ {processed} OK, {skipped} skipped ({success_rate:.0f}% success)")

        total_processed += processed
        total_skipped += skipped

    # ─── SAVE TO CSV ────────────────────────────────────────────
    print()
    print(f"[SAVING] Writing {total_processed} samples to {OUTPUT_CSV}...")

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)

    print()
    print("=" * 60)
    print("  PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"  Total samples processed: {total_processed}")
    print(f"  Total samples skipped:   {total_skipped}")
    print(f"  Success rate:            {total_processed / (total_processed + total_skipped) * 100:.1f}%")
    print(f"  Output file:             {OUTPUT_CSV}")
    print(f"  Features per sample:     {TOTAL_FEATURES}")
    print(f"  Gesture classes:         {gesture_folders}")
    print("=" * 60)
    print()
    print("  ✅ You can now run: python train_model.py")
    print()

    # Cleanup
    hands.close()


if __name__ == "__main__":
    main()
