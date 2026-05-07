"""
=============================================================================
  STEP 4: MAIN DESKTOP APPLICATION
  Sign Language to Text/Speech Converter
=============================================================================
  This is the main desktop application that brings everything together:
  - Opens webcam and detects your hand in real-time
  - Recognizes the sign language gesture using the trained ML model
  - Displays the detected letter on screen
  - Builds words and sentences from individual letters
  - Speaks the sentence using text-to-speech

  CONTROLS:
  - Show hand gestures to the camera — letters appear on screen
  - Press SPACEBAR to add a space between words
  - Press BACKSPACE to delete last character
  - Press ENTER to speak the current sentence aloud
  - Press 'C' to clear the entire sentence
  - Press 'Q' to quit

  HOW TO USE:
  1. Make sure you've trained the model (python train_model.py)
  2. Run: python app.py
  3. Show hand gestures to the camera
  4. Press ENTER to hear the sentence spoken aloud
=============================================================================
"""

import os
import cv2
import time
import numpy as np
import mp_patch
import mediapipe as mp
import threading

# ─── NVIDIA MAGPIE TTS ──────────────────────────────────────────────────────
try:
    from nvidia_tts import NvidiaTTS
    _tts = NvidiaTTS()
    print(f"[INFO] TTS Engine: {'NVIDIA Magpie (Cloud)' if _tts.api_available else 'pyttsx3 (Offline Fallback)'}")
except ImportError:
    import pyttsx3
    _tts = None
    _pyttsx3_engine = pyttsx3.init()
    _pyttsx3_engine.setProperty('rate', 150)
    _pyttsx3_engine.setProperty('volume', 1.0)
    voices = _pyttsx3_engine.getProperty('voices')
    for voice in voices:
        if 'english' in voice.name.lower() or 'zira' in voice.name.lower():
            _pyttsx3_engine.setProperty('voice', voice.id)
            break
    print("[INFO] TTS Engine: pyttsx3 (nvidia_tts module not found)")

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.tflite")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.npy")

CONFIDENCE_THRESHOLD = 0.75    # Only accept predictions above 75% confidence
LETTER_COOLDOWN = 1.5          # Seconds between accepting same letter
STABLE_FRAMES = 10             # Frames a gesture must be stable to be accepted
FPS_DISPLAY = True             # Show FPS on screen

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────
# These variables are shared across the app and the web dashboard
current_letter = ""
current_confidence = 0.0
current_sentence = ""
detection_history = []
frame_for_web = None


# ─── SETUP MEDIAPIPE ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def speak_text(text):
    """Speak text using NVIDIA Magpie TTS (or pyttsx3 fallback) in a separate thread."""
    if _tts is not None:
        _tts.speak(text)
    else:
        def _speak():
            try:
                _pyttsx3_engine.say(text)
                _pyttsx3_engine.runAndWait()
            except Exception as e:
                print(f"[WARNING] TTS error: {e}")
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()


def load_model():
    """Load the TFLite model and label map."""
    print("[INFO] Loading model...")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  → Run train_model.py first!")
        return None, None

    if not os.path.exists(LABEL_MAP_PATH):
        print(f"[ERROR] Label map not found: {LABEL_MAP_PATH}")
        print("  → Run train_model.py first!")
        return None, None

    # Load TFLite model
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load label map
    class_names = np.load(LABEL_MAP_PATH, allow_pickle=True)
    print(f"  → Model loaded successfully!")
    print(f"  → Gestures: {list(class_names)}")

    return interpreter, class_names


def extract_landmarks(hand_landmarks):
    """Extract and normalize hand landmarks for prediction."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    # Normalize relative to wrist (same as preprocessing)
    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized = []
    for i in range(0, len(landmarks), 3):
        normalized.append(landmarks[i] - wrist_x)
        normalized.append(landmarks[i + 1] - wrist_y)
        normalized.append(landmarks[i + 2] - wrist_z)

    return np.array(normalized, dtype=np.float32)


def predict_gesture(interpreter, landmarks, class_names):
    """Run inference on the landmarks and return predicted gesture + confidence."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input
    input_data = landmarks.reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get best prediction
    predicted_idx = np.argmax(output_data)
    confidence = output_data[predicted_idx]
    predicted_label = class_names[predicted_idx]

    return predicted_label, confidence


def draw_ui(frame, letter, confidence, sentence, hand_detected, fps):
    """Draw the user interface overlay on the camera frame."""
    h, w = frame.shape[:2]

    # ─── TOP BAR ────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    # Title
    cv2.putText(frame, "Sign Language Converter", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS
    if FPS_DISPLAY:
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Hand status
    status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # ─── DETECTED LETTER (large display) ────────────────────────
    if letter and confidence > CONFIDENCE_THRESHOLD:
        # Background box
        cv2.rectangle(frame, (w - 150, 80), (w - 10, 220), (30, 30, 30), -1)
        cv2.rectangle(frame, (w - 150, 80), (w - 10, 220), (0, 255, 0), 2)

        # Letter
        cv2.putText(frame, letter, (w - 120, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 4)

        # Confidence bar
        bar_x = w - 145
        bar_width = int(130 * confidence)
        cv2.rectangle(frame, (bar_x, 205), (bar_x + 130, 215), (50, 50, 50), -1)
        bar_color = (0, 255, 0) if confidence > 0.9 else (0, 255, 255)
        cv2.rectangle(frame, (bar_x, 205), (bar_x + bar_width, 215), bar_color, -1)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (bar_x + 45, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # ─── SENTENCE BAR (bottom) ──────────────────────────────────
    cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)

    cv2.putText(frame, "Sentence:", (15, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Show sentence with cursor
    display_sentence = sentence + "_"
    cv2.putText(frame, display_sentence, (15, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Controls hint
    controls = "[SPACE] Space  [BACKSPACE] Delete  [ENTER] Speak  [C] Clear  [Q] Quit"
    cv2.putText(frame, controls, (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    return frame


def main():
    """Main application loop."""
    global current_letter, current_confidence, current_sentence
    global detection_history, frame_for_web

    print("=" * 60)
    print("  SIGN LANGUAGE TO TEXT/SPEECH CONVERTER")
    print("=" * 60)
    print()

    # Load model
    interpreter, class_names = load_model()
    if interpreter is None:
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print()
    print("[INFO] Application started! Show hand gestures to the camera.")
    print("  → Press ENTER to speak the sentence")
    print("  → Press Q to quit")
    print()

    sentence = ""
    last_letter = ""
    last_letter_time = 0
    stable_count = 0
    prev_prediction = ""
    fps_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Calculate FPS
        fps = 1.0 / (time.time() - fps_time + 0.001)
        fps_time = time.time()

        # Process hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_letter = ""
        confidence = 0.0
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmarks and predict
            landmarks = extract_landmarks(hand_landmarks)
            predicted_label, conf = predict_gesture(interpreter, landmarks, class_names)

            if conf > CONFIDENCE_THRESHOLD:
                detected_letter = predicted_label
                confidence = conf

                # Stability check — same prediction for STABLE_FRAMES frames
                if predicted_label == prev_prediction:
                    stable_count += 1
                else:
                    stable_count = 0
                    prev_prediction = predicted_label

                # Accept letter if stable and cooldown passed
                if stable_count >= STABLE_FRAMES:
                    current_time = time.time()
                    if (detected_letter != last_letter or
                            current_time - last_letter_time > LETTER_COOLDOWN):
                        sentence += detected_letter
                        last_letter = detected_letter
                        last_letter_time = current_time
                        detection_history.append({
                            'letter': detected_letter,
                            'confidence': float(conf),
                            'time': current_time
                        })
                        print(f"  ✓ Detected: {detected_letter} ({conf * 100:.0f}%) → Sentence: {sentence}")
                        stable_count = 0

        # Update global state (for web dashboard)
        current_letter = detected_letter
        current_confidence = confidence
        current_sentence = sentence

        # Draw UI
        frame = draw_ui(frame, detected_letter, confidence, sentence, hand_detected, fps)

        # Save frame for web streaming
        frame_for_web = frame.copy()

        # Show frame
        cv2.imshow("Sign Language Converter", frame)

        # ─── KEYBOARD INPUT ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == 13:  # ENTER — speak sentence
            if sentence.strip():
                print(f"  🔊 Speaking: \"{sentence}\"")
                speak_text(sentence)
        elif key == 8:  # BACKSPACE — delete last character
            sentence = sentence[:-1]
            print(f"  ← Deleted. Sentence: \"{sentence}\"")
        elif key == 32:  # SPACEBAR — add space
            sentence += " "
            last_letter = ""
            print(f"  [SPACE] Sentence: \"{sentence}\"")
        elif key == ord('c') or key == ord('C'):  # Clear sentence
            sentence = ""
            last_letter = ""
            detection_history = []
            print("  🗑️  Sentence cleared.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()
