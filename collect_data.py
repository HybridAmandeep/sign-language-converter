"""
=============================================================================
  STEP 1: DATA COLLECTION SCRIPT
  Sign Language to Text/Speech Converter
=============================================================================
  This script opens your webcam and lets you capture hand gesture images.
  You will show a hand sign, press a key to label it, then save multiple
  frames of that gesture into a folder.

  HOW TO USE:
  1. Run: python collect_data.py
  2. A webcam window will open showing your hand with landmarks drawn
  3. Press the letter key you want to collect (e.g., press 'A' for gesture A)
  4. Press 'S' to start saving frames (it will save 200 frames automatically)
  5. Repeat for each gesture you want to collect
  6. Press 'Q' to quit when done

  TIPS:
  - Make sure your hand is clearly visible and well-lit
  - Try different angles and positions for each gesture
  - Collect at least 200 images per gesture for good accuracy
  - Use a plain background if possible
=============================================================================
"""

import cv2
import os
import time
import mp_patch
import mediapipe as mp

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMAGES_PER_GESTURE = 200       # Number of images to capture per gesture
COUNTDOWN_SECONDS = 3          # Countdown before capture begins
FRAME_DELAY_MS = 50            # Delay between frame captures (milliseconds)

# ─── SETUP MEDIAPIPE HANDS ──────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,    # Video mode (faster)
    max_num_hands=1,            # Detect only one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def create_data_folder(label):
    """Create a folder for the given gesture label if it doesn't exist."""
    folder_path = os.path.join(DATA_DIR, str(label))
    os.makedirs(folder_path, exist_ok=True)
    # Count existing images to avoid overwriting
    existing = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    return folder_path, existing


def draw_info_text(frame, text, position=(30, 40), color=(0, 255, 0), scale=1.0):
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, 2)[0]
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_size[1] - 10),
                  (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


def main():
    """Main data collection loop."""
    print("=" * 60)
    print("  SIGN LANGUAGE DATA COLLECTION TOOL")
    print("=" * 60)
    print()
    print("  CONTROLS:")
    print("  - Press any letter (A-Z) or number (0-9) to set gesture label")
    print("  - Press 'S' to start capturing frames")
    print("  - Press 'Q' to quit")
    print()
    print("  Make sure your hand is visible to the camera!")
    print("=" * 60)
    print()

    # Create main data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        print("  → Make sure your webcam is connected and not in use by another app.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_label = None    # The gesture label we're collecting
    collecting = False      # Whether we're actively saving frames
    frames_saved = 0        # Counter for saved frames
    countdown_start = None  # When countdown started
    save_folder = None      # Path to save images
    start_index = 0         # Starting file number

    print("[INFO] Webcam opened successfully. Waiting for input...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam!")
            break

        # Flip frame horizontally (mirror effect — more natural)
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks if detected
        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # ─── COUNTDOWN MODE ────────────────────────────────────────
        if countdown_start is not None and not collecting:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            if remaining > 0:
                draw_info_text(frame, f"Starting in {remaining}...",
                               (200, 240), (0, 255, 255), 2.0)
            else:
                # Countdown finished — start collecting
                collecting = True
                frames_saved = 0
                countdown_start = None
                print(f"[INFO] Capturing started for gesture '{current_label}'...")

        # ─── COLLECTING MODE ───────────────────────────────────────
        if collecting and hand_detected:
            # Save the frame
            filename = f"{start_index + frames_saved:04d}.jpg"
            filepath = os.path.join(save_folder, filename)
            cv2.imwrite(filepath, frame)
            frames_saved += 1

            # Show progress
            progress = f"Saving '{current_label}': {frames_saved}/{IMAGES_PER_GESTURE}"
            draw_info_text(frame, progress, (30, 40), (0, 255, 0), 0.8)

            # Draw progress bar
            bar_width = int(400 * frames_saved / IMAGES_PER_GESTURE)
            cv2.rectangle(frame, (30, 55), (430, 75), (50, 50, 50), -1)
            cv2.rectangle(frame, (30, 55), (30 + bar_width, 75), (0, 255, 0), -1)

            if frames_saved >= IMAGES_PER_GESTURE:
                collecting = False
                print(f"[DONE] Saved {frames_saved} images for gesture '{current_label}'")
                print(f"       Location: {save_folder}")
                print()

        # ─── STATUS DISPLAY ────────────────────────────────────────
        if not collecting and countdown_start is None:
            # Show current status
            status = f"Label: '{current_label}'" if current_label else "Press A-Z or 0-9 to set label"
            draw_info_text(frame, status, (30, 40), (255, 255, 255), 0.7)

            if current_label:
                draw_info_text(frame, "Press 'S' to start capture",
                               (30, 80), (0, 200, 255), 0.6)

            # Hand detection status
            hand_status = "Hand: DETECTED" if hand_detected else "Hand: NOT DETECTED"
            hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            draw_info_text(frame, hand_status, (30, 450), hand_color, 0.6)

        # Show the frame
        cv2.imshow("Sign Language Data Collection", frame)

        # ─── KEYBOARD INPUT ────────────────────────────────────────
        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("[INFO] Quitting data collection.")
            break
        elif key == ord('s') or key == ord('S'):
            if current_label and not collecting:
                # Start countdown
                save_folder, start_index = create_data_folder(current_label)
                countdown_start = time.time()
                print(f"[INFO] Starting countdown for gesture '{current_label}'...")
            elif not current_label:
                print("[WARNING] Set a label first! Press A-Z or 0-9.")
        elif (ord('a') <= key <= ord('z')) or (ord('0') <= key <= ord('9')):
            if not collecting:
                current_label = chr(key).upper()
                print(f"[INFO] Label set to: '{current_label}'")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Print summary
    print()
    print("=" * 60)
    print("  DATA COLLECTION SUMMARY")
    print("=" * 60)
    if os.path.exists(DATA_DIR):
        gestures = [d for d in os.listdir(DATA_DIR)
                    if os.path.isdir(os.path.join(DATA_DIR, d))]
        if gestures:
            for g in sorted(gestures):
                count = len(os.listdir(os.path.join(DATA_DIR, g)))
                print(f"  Gesture '{g}': {count} images")
        else:
            print("  No gestures collected yet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
