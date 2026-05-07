"""
=============================================================================
  Sign Language Translator Dashboard
  ─────────────────────────────────────────────────────
  A communication system that converts
  sign language gestures into text and speech in real-time.

  Features:
  ● Dual-hand detection (both left & right hands tracked simultaneously)
  ● Live camera feed with colour-coded hand landmarks
  ● Chat-style message history with timestamps
  ● NVIDIA Magpie TTS (Cloud API) with multiple voices
  ● Continuous sentence builder with auto-speak
  ● Word suggestions & quick phrases
  ● Beautiful dark sci-fi dashboard

  HOW TO USE:
  1. Train your model first:  python train_model.py  (or train_slt_model.py)
  2. Run:  python web/server.py
  3. Open browser:  http://localhost:5000
=============================================================================
"""

import os
import sys
import cv2
import json
import time
import base64
import numpy as np

# Add parent directory to path so mp_patch and nvidia_tts can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mp_patch
import mediapipe as mp
import threading
from flask import Flask, render_template, Response, jsonify, request, send_file
from gesture_rules import recognize_gesture, get_word_suggestions

# ─── NVIDIA TTS IMPORT ──────────────────────────────────────────────────────
try:
    from nvidia_tts import NvidiaTTS, VOICES
    nvidia_tts_module = True
except ImportError:
    nvidia_tts_module = False
    print("[WARNING] nvidia_tts module not found. Browser TTS will be used.")

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.tflite")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.npy")

# ─── FLASK APP ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────
state = {
    "current_letter": "",
    "confidence": 0.0,
    "sentence": "",
    "history": [],
    "chat_messages": [],     # Chat-style messages
    "hand_detected": False,
    "hands_count": 0,        # Number of hands detected (0, 1, or 2)
    "left_hand": False,
    "right_hand": False,
    "fps": 0,
    "is_running": False,
    "model_loaded": False,
    "auto_speak": True,      # Auto-speak completed words
    "speaking": False,
    "mode": "word",           # 'word' or 'letter' mode
    "suggestions": [],       # Word suggestions
    "camera_index": 0,
    "switch_camera": False,
    "tts_engine": "nvidia",  # 'nvidia' or 'browser'
    "tts_voice": "aria",     # Current voice key
    "tts_available": False,  # Whether NVIDIA TTS API is available
}

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
LETTER_COOLDOWN = 1.2
STABLE_FRAMES = 8

# ─── MEDIAPIPE SETUP ─────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── NVIDIA TTS SETUP ───────────────────────────────────────────────────────
tts_instance = None
tts_lock = threading.Lock()

def init_tts():
    """Initialize NVIDIA Magpie TTS."""
    global tts_instance
    if nvidia_tts_module:
        try:
            tts_instance = NvidiaTTS()
            state["tts_available"] = tts_instance.api_available
            state["tts_engine"] = "nvidia" if tts_instance.api_available else "browser"
            print(f"[INFO] TTS Engine: {'NVIDIA Magpie (Cloud)' if tts_instance.api_available else 'Browser Fallback'}")
        except Exception as e:
            print(f"[WARNING] NVIDIA TTS init failed: {e}")
            state["tts_engine"] = "browser"
    else:
        state["tts_engine"] = "browser"
        print("[INFO] TTS Engine: Browser-based (nvidia_tts module not found)")


def speak_text(text):
    """Speak text using NVIDIA Magpie TTS or browser fallback."""
    if tts_instance and tts_instance.api_available:
        tts_instance.speak(text, voice_key=state.get("tts_voice", "aria"))
    # Browser TTS is handled client-side as fallback


# ─── GLOBALS FOR VIDEO ───────────────────────────────────────────────────────
output_frame = None
lock = threading.Lock()


def load_model():
    """Load TFLite model and label map."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("[WARNING] Model files not found. Dashboard will run in demo mode.")
        print(f"  Expected: {MODEL_PATH}")
        print("  -> Run train_model.py or train_slt_model.py first!")
        return None, None

    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        class_names = np.load(LABEL_MAP_PATH, allow_pickle=True)
        print(f"[INFO] Model loaded! Gestures: {list(class_names)}")
        return interpreter, class_names
    except ImportError:
        print("[WARNING] TensorFlow not installed. Dashboard will run in demo mode.")
        return None, None


def extract_landmarks(hand_landmarks):
    """Extract and normalize landmarks."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
    normalized = []
    for i in range(0, len(landmarks), 3):
        normalized.append(landmarks[i] - wrist_x)
        normalized.append(landmarks[i + 1] - wrist_y)
        normalized.append(landmarks[i + 2] - wrist_z)

    return np.array(normalized, dtype=np.float32)


def camera_loop():
    """Background thread that processes camera frames with dual-hand detection."""
    global output_frame, state

    interpreter, class_names = load_model()
    state["model_loaded"] = interpreter is not None

    # Initialize hand detector for BOTH hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,        # Detect BOTH hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        state["is_running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state["is_running"] = True
    last_letter = ""
    last_letter_time = 0
    stable_count = 0
    prev_prediction = ""
    fps_time = time.time()

    print("[INFO] Camera loop started - dual-hand detection active!")

    while state["is_running"]:
        if state.get("switch_camera"):
            cap.release()
            state["camera_index"] += 1
            new_cap = cv2.VideoCapture(state["camera_index"])
            if not new_cap.isOpened() and state["camera_index"] > 0:
                state["camera_index"] = 0
                new_cap = cv2.VideoCapture(0)
            
            if new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap = new_cap
            state["switch_camera"] = False

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_time_val = time.time()
        fps = 1.0 / (current_time_val - fps_time + 0.001)
        fps_time = current_time_val
        state["fps"] = round(fps, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_letter = ""
        confidence = 0.0
        hand_detected = False
        hands_count = 0
        left_hand = False
        right_hand = False

        if results.multi_hand_landmarks:
            hand_detected = True
            hands_count = len(results.multi_hand_landmarks)

            # Process each detected hand
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine handedness
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx]
                    label = handedness.classification[0].label
                    # Mirror: camera shows mirrored, so Left in camera = Right hand
                    if label == 'Left':
                        right_hand = True
                    elif label == 'Right':
                        left_hand = True

                # Draw landmarks with hand-specific colour
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                    hand_index=idx
                )

                # Predict gesture (use first hand for prediction)
                if idx == 0:
                    predicted_label = ""
                    conf = 0.0

                    if interpreter is not None:
                        # Use trained ML model
                        landmarks_arr = extract_landmarks(hand_landmarks)
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()

                        input_data = landmarks_arr.reshape(1, -1)
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                        predicted_idx_val = np.argmax(output_data)
                        conf = float(output_data[predicted_idx_val])
                        predicted_label = str(class_names[predicted_idx_val])
                    else:
                        # Use rule-based gesture recognition (no model needed)
                        current_mode = state.get("mode", "word")
                        predicted_label, conf = recognize_gesture(
                            hand_landmarks.landmark, mode=current_mode
                        )

                    if conf > CONFIDENCE_THRESHOLD:
                        detected_letter = str(predicted_label)
                        confidence = float(conf)

                        if predicted_label == prev_prediction:
                            stable_count += 1
                        else:
                            stable_count = 0
                            prev_prediction = predicted_label

                        # Word mode needs fewer stable frames for faster response
                        current_mode = state.get("mode", "word")
                        needed_frames = 6 if current_mode == "word" else STABLE_FRAMES

                        if stable_count >= needed_frames:
                            current_time = time.time()
                            cooldown = 1.8 if current_mode == "word" else LETTER_COOLDOWN
                            if (detected_letter != last_letter or
                                    current_time - last_letter_time > cooldown):
                                # In word mode, add space before each word
                                if current_mode == "word" and state["sentence"]:
                                    state["sentence"] += " " + detected_letter
                                else:
                                    state["sentence"] += detected_letter
                                last_letter = detected_letter
                                last_letter_time = current_time
                                state["history"].append({
                                    "letter": detected_letter,
                                    "confidence": round(confidence * 100, 1),
                                    "time": time.strftime("%H:%M:%S")
                                })
                                if len(state["history"]) > 100:
                                    state["history"] = state["history"][-100:]
                                # Update word suggestions
                                state["suggestions"] = get_word_suggestions(
                                    state["sentence"]
                                )
                                stable_count = 0

        state["current_letter"] = detected_letter
        state["confidence"] = round(confidence * 100, 1)
        state["hand_detected"] = hand_detected
        state["hands_count"] = hands_count
        state["left_hand"] = left_hand
        state["right_hand"] = right_hand

        with lock:
            output_frame = frame.copy()

    cap.release()
    hands.close()
    print("[INFO] Camera loop stopped.")


def generate_frames():
    """Generator that yields MJPEG frames for the video feed."""
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main Sign Language Translator dashboard."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def get_status():
    """Return current detection status."""
    return jsonify(state)


@app.route('/api/speak', methods=['POST'])
def speak():
    """Speak the current sentence or custom text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", state.get("sentence", "")).strip()
    if text:
        speak_text(text)
        return jsonify({"status": "speaking", "text": text})
    return jsonify({"status": "empty"})


@app.route('/api/tts', methods=['POST'])
def tts_synthesize():
    """
    Synthesize text to audio and return WAV bytes.
    Used by the frontend for high-quality NVIDIA Magpie TTS playback.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    voice = data.get("voice", state.get("tts_voice", "aria"))

    if not text:
        return jsonify({"status": "empty"}), 400

    if tts_instance and tts_instance.api_available:
        audio_bytes = tts_instance.synthesize(text, voice_key=voice)
        if audio_bytes:
            # Return audio as base64 for easy browser playback
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                "status": "ok",
                "audio": audio_b64,
                "format": "wav",
                "voice": voice,
                "engine": "nvidia-magpie"
            })

    return jsonify({
        "status": "fallback",
        "text": text,
        "engine": "browser"
    })


@app.route('/api/voices')
def get_voices():
    """Return available TTS voices."""
    voices = {}
    if nvidia_tts_module:
        voices = {k: v["label"] for k, v in VOICES.items()}
    return jsonify({
        "voices": voices,
        "current": state.get("tts_voice", "aria"),
        "engine": state.get("tts_engine", "browser"),
        "api_available": state.get("tts_available", False)
    })


@app.route('/api/set_voice', methods=['POST'])
def set_voice():
    """Set the active TTS voice."""
    data = request.get_json(silent=True) or {}
    voice = data.get("voice", "aria")
    if nvidia_tts_module and voice in VOICES:
        state["tts_voice"] = voice
        return jsonify({"status": "ok", "voice": voice, "label": VOICES[voice]["label"]})
    return jsonify({"status": "invalid_voice"}), 400


@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Send current sentence as a chat message. TTS is handled client-side."""
    sentence = state.get("sentence", "").strip()
    if sentence:
        msg = {
            "text": sentence,
            "time": time.strftime("%I:%M %p"),
            "timestamp": time.time()
        }
        state["chat_messages"].append(msg)
        if len(state["chat_messages"]) > 200:
            state["chat_messages"] = state["chat_messages"][-200:]

        # Clear sentence after sending
        state["sentence"] = ""
        return jsonify({"status": "sent", "message": msg,
                        "auto_speak": state.get("auto_speak", True)})
    return jsonify({"status": "empty"})


@app.route('/api/clear', methods=['POST'])
def clear():
    """Clear the sentence."""
    state["sentence"] = ""
    state["history"] = []
    return jsonify({"status": "cleared"})


@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    """Clear all chat messages."""
    state["chat_messages"] = []
    return jsonify({"status": "chat_cleared"})


@app.route('/api/space', methods=['POST'])
def add_space():
    """Add a space to the sentence."""
    state["sentence"] += " "
    return jsonify({"status": "space_added", "sentence": state["sentence"]})


@app.route('/api/backspace', methods=['POST'])
def backspace():
    """Remove last character from sentence."""
    state["sentence"] = state["sentence"][:-1]
    return jsonify({"status": "deleted", "sentence": state["sentence"]})


@app.route('/api/toggle_autospeak', methods=['POST'])
def toggle_autospeak():
    """Toggle auto-speak mode."""
    state["auto_speak"] = not state.get("auto_speak", True)
    return jsonify({"auto_speak": state["auto_speak"]})


@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    """Toggle between word and letter detection mode."""
    current = state.get("mode", "word")
    state["mode"] = "letter" if current == "word" else "word"
    return jsonify({"mode": state["mode"]})


@app.route('/api/suggestions')
def suggestions():
    """Get word suggestions based on current sentence."""
    return jsonify({"suggestions": state.get("suggestions", [])})


@app.route('/api/use_suggestion', methods=['POST'])
def use_suggestion():
    """Replace current sentence with a suggestion."""
    data = request.get_json(silent=True) or {}
    suggestion = data.get("text", "").strip()
    if suggestion:
        # Replace the last word with the suggestion
        words = state["sentence"].split()
        if words:
            words[-1] = suggestion
            state["sentence"] = " ".join(words)
        else:
            state["sentence"] = suggestion
        state["suggestions"] = get_word_suggestions(state["sentence"])
        return jsonify({"status": "applied", "sentence": state["sentence"]})
    return jsonify({"status": "empty"})


@app.route('/api/quick_phrase', methods=['POST'])
def quick_phrase():
    """Send a prebuilt quick phrase. TTS is handled client-side."""
    data = request.get_json(silent=True) or {}
    phrase = data.get("phrase", "").strip()
    if phrase:
        msg = {
            "text": phrase,
            "time": time.strftime("%I:%M %p"),
            "timestamp": time.time()
        }
        state["chat_messages"].append(msg)
        return jsonify({"status": "sent", "message": msg,
                        "auto_speak": state.get("auto_speak", True)})
    return jsonify({"status": "empty"})


@app.route('/api/switch_camera', methods=['POST'])
def switch_camera():
    """Trigger the camera loop to switch to the next camera index."""
    state["switch_camera"] = True
    return jsonify({"status": "switching"})

# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  Sign Language Translator")
    print("=" * 60)
    print()

    # Initialize NVIDIA Magpie TTS
    print("  Initializing NVIDIA Magpie TTS (Cloud API)...")
    init_tts()
    print()

    print("  Starting camera with dual-hand detection...")

    # Start camera processing in background thread
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    print("  Launching web dashboard...")
    print()
    print("  [OK] Dashboard:  http://localhost:5000")
    print("  [OK] Network:    http://<your-ip>:5000")
    print()
    print("  Features:")
    print("    - Dual-hand tracking (both hands)")
    print("    - Real-time sign to text conversion")
    print("    - Chat-style message history")
    tts_mode = "NVIDIA Magpie TTS (Cloud)" if state.get("tts_available") else "Browser TTS (Fallback)"
    print(f"    - Text-to-speech: {tts_mode}")
    print("    - Quick phrases for common messages")
    print()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
