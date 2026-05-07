"""
=============================================================================
  SYNTHETIC DATA GENERATOR
  Sign Language to Text/Speech Converter
=============================================================================
  Generates synthetic MediaPipe-style hand landmark data for ASL gestures.
  This eliminates the need for manual webcam data collection.

  Each hand has 21 landmarks with (x, y, z) coordinates = 63 features.
  Landmarks are normalized relative to the wrist (landmark 0).

  HOW TO USE:
  1. Run: python generate_data.py
  2. Output: data/landmarks.csv
  3. Then run: python train_model.py
=============================================================================
"""

import os
import csv
import numpy as np
import random

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "landmarks.csv")

SAMPLES_PER_GESTURE = 500       # Number of synthetic samples per gesture
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3
TOTAL_FEATURES = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63

# ─── HAND SKELETON DEFINITION ──────────────────────────────────────────────
# MediaPipe landmark indices:
#  0: WRIST
#  1: THUMB_CMC,  2: THUMB_MCP,  3: THUMB_IP,   4: THUMB_TIP
#  5: INDEX_MCP,  6: INDEX_PIP,  7: INDEX_DIP,   8: INDEX_TIP
#  9: MIDDLE_MCP, 10: MIDDLE_PIP, 11: MIDDLE_DIP, 12: MIDDLE_TIP
# 13: RING_MCP,   14: RING_PIP,  15: RING_DIP,   16: RING_TIP
# 17: PINKY_MCP,  18: PINKY_PIP, 19: PINKY_DIP,  20: PINKY_TIP

# Base hand skeleton (relaxed open hand, palm facing camera)
# All coordinates are relative to wrist (0,0,0)
BASE_HAND = {
    0:  (0.0,    0.0,    0.0),     # WRIST (origin)
    # Thumb (extends to the left/right)
    1:  (-0.06, -0.04,  -0.01),    # THUMB_CMC
    2:  (-0.10, -0.08,  -0.02),    # THUMB_MCP
    3:  (-0.13, -0.12,  -0.02),    # THUMB_IP
    4:  (-0.15, -0.15,  -0.02),    # THUMB_TIP
    # Index finger
    5:  (-0.04, -0.12,   0.0),     # INDEX_MCP
    6:  (-0.04, -0.18,   0.0),     # INDEX_PIP
    7:  (-0.04, -0.22,   0.0),     # INDEX_DIP
    8:  (-0.04, -0.26,   0.0),     # INDEX_TIP
    # Middle finger
    9:  (0.0,   -0.13,   0.0),     # MIDDLE_MCP
    10: (0.0,   -0.19,   0.0),     # MIDDLE_PIP
    11: (0.0,   -0.24,   0.0),     # MIDDLE_DIP
    12: (0.0,   -0.28,   0.0),     # MIDDLE_TIP
    # Ring finger
    13: (0.04,  -0.12,   0.0),     # RING_MCP
    14: (0.04,  -0.18,   0.0),     # RING_PIP
    15: (0.04,  -0.22,   0.0),     # RING_DIP
    16: (0.04,  -0.25,   0.0),     # RING_TIP
    # Pinky finger
    17: (0.07,  -0.10,   0.0),     # PINKY_MCP
    18: (0.07,  -0.15,   0.0),     # PINKY_PIP
    19: (0.07,  -0.18,   0.0),     # PINKY_DIP
    20: (0.07,  -0.21,   0.0),     # PINKY_TIP
}


def curl_finger(hand, finger, curl_amount=0.8):
    """Curl a finger by moving tip/dip/pip towards palm."""
    finger_indices = {
        'thumb':  [1, 2, 3, 4],
        'index':  [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring':   [13, 14, 15, 16],
        'pinky':  [17, 18, 19, 20],
    }
    indices = finger_indices[finger]
    mcp = indices[0]
    mcp_pos = hand[mcp]

    for i, idx in enumerate(indices[1:], 1):
        orig = hand[idx]
        # Move towards MCP (curl inward)
        factor = curl_amount * (i / 3.0)
        new_x = orig[0] + (mcp_pos[0] - orig[0]) * factor
        new_y = orig[1] + (mcp_pos[1] - orig[1]) * factor * 0.6
        new_z = orig[2] - 0.02 * i * curl_amount  # Curl forward
        hand[idx] = (new_x, new_y, new_z)
    return hand


def extend_finger(hand, finger):
    """Ensure finger is fully extended (use base positions)."""
    finger_indices = {
        'thumb':  [1, 2, 3, 4],
        'index':  [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring':   [13, 14, 15, 16],
        'pinky':  [17, 18, 19, 20],
    }
    for idx in finger_indices[finger]:
        hand[idx] = BASE_HAND[idx]
    return hand


def touch_fingers(hand, idx1, idx2):
    """Move two fingertips to the same position (touching)."""
    mid_x = (hand[idx1][0] + hand[idx2][0]) / 2
    mid_y = (hand[idx1][1] + hand[idx2][1]) / 2
    mid_z = (hand[idx1][2] + hand[idx2][2]) / 2
    hand[idx1] = (mid_x, mid_y, mid_z)
    hand[idx2] = (mid_x, mid_y, mid_z)
    return hand


def make_fist(hand):
    """Curl all fingers into a fist."""
    for f in ['index', 'middle', 'ring', 'pinky']:
        hand = curl_finger(hand, f, 0.9)
    hand = curl_finger(hand, 'thumb', 0.5)
    return hand


# ═════════════════════════════════════════════════════════════════
#   ASL GESTURE DEFINITIONS
# ═════════════════════════════════════════════════════════════════

def generate_gesture(label):
    """Generate a hand pose for a given ASL gesture label."""
    hand = {k: v for k, v in BASE_HAND.items()}  # Deep copy

    if label == 'A':
        # Fist with thumb alongside
        hand = make_fist(hand)
        hand[4] = (-0.08, -0.10, -0.03)  # Thumb alongside fist

    elif label == 'B':
        # Flat hand, fingers together, thumb tucked
        hand = curl_finger(hand, 'thumb', 0.7)
        # Fingers close together
        for idx in [8, 12, 16, 20]:
            hand[idx] = (hand[idx][0] * 0.9, hand[idx][1], hand[idx][2])

    elif label == 'C':
        # Curved hand like holding a ball
        for f in ['index', 'middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.4)
        hand = curl_finger(hand, 'thumb', 0.3)
        hand[4] = (-0.10, -0.12, -0.03)

    elif label == 'D':
        # Index up, others curled touching thumb
        for f in ['middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand = touch_fingers(hand, 4, 12)

    elif label == 'E':
        # All fingers curled, tips near palm
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)

    elif label == 'F':
        # Thumb and index touching, others extended
        hand = touch_fingers(hand, 4, 8)

    elif label == 'G':
        # Index and thumb pointing sideways
        for f in ['middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand[8] = (-0.15, -0.12, 0.0)  # Index pointing right
        hand[7] = (-0.11, -0.12, 0.0)
        hand[4] = (-0.15, -0.10, -0.02)

    elif label == 'H':
        # Index and middle pointing sideways
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand = curl_finger(hand, 'thumb', 0.6)
        hand[8] = (-0.15, -0.12, 0.0)
        hand[7] = (-0.11, -0.12, 0.0)
        hand[12] = (-0.15, -0.13, 0.0)
        hand[11] = (-0.11, -0.13, 0.0)

    elif label == 'I':
        # Pinky up, all others curled
        hand = make_fist(hand)
        hand = extend_finger(hand, 'pinky')

    elif label == 'K':
        # Index and middle up spread, thumb between
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand[4] = (-0.05, -0.15, -0.02)  # Thumb between index/middle

    elif label == 'L':
        # L shape: thumb and index extended at 90 degrees
        for f in ['middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.9)
        hand[4] = (-0.15, -0.08, -0.02)  # Thumb out to side

    elif label == 'M':
        # Fist with thumb under three fingers
        hand = make_fist(hand)
        hand[4] = (0.02, -0.06, -0.04)

    elif label == 'N':
        # Fist with thumb under two fingers
        hand = make_fist(hand)
        hand[4] = (-0.01, -0.06, -0.04)

    elif label == 'O':
        # All fingertips touching thumb (circle)
        for f in ['index', 'middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.5)
        hand = touch_fingers(hand, 4, 8)
        hand[12] = (hand[4][0] + 0.01, hand[4][1] - 0.01, hand[4][2])

    elif label == 'P':
        # Like K but pointing down
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand[8] = (-0.04, -0.20, 0.03)  # Index pointing down-ish
        hand[12] = (0.0, -0.20, 0.03)
        hand[4] = (-0.05, -0.13, -0.02)

    elif label == 'Q':
        # Like G but pointing down
        for f in ['middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand[8] = (-0.06, -0.08, 0.04)  # Index pointing down
        hand[4] = (-0.08, -0.06, 0.02)

    elif label == 'R':
        # Index and middle crossed
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand = curl_finger(hand, 'thumb', 0.6)
        hand[8] = (-0.01, -0.26, 0.0)  # Index
        hand[12] = (-0.02, -0.27, 0.0)  # Middle crossed over

    elif label == 'S':
        # Tight fist, thumb over fingers
        hand = make_fist(hand)
        hand[4] = (-0.04, -0.09, -0.04)

    elif label == 'T':
        # Fist with thumb between index and middle
        hand = make_fist(hand)
        hand[4] = (-0.03, -0.10, -0.04)

    elif label == 'U':
        # Index and middle up, close together
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand = curl_finger(hand, 'thumb', 0.6)
        hand[8] = (-0.02, -0.26, 0.0)
        hand[12] = (0.0, -0.27, 0.0)

    elif label == 'V':
        # Peace sign - index and middle spread
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand = curl_finger(hand, 'thumb', 0.6)

    elif label == 'W':
        # Three fingers up (index, middle, ring)
        hand = curl_finger(hand, 'pinky', 0.85)
        hand = curl_finger(hand, 'thumb', 0.6)

    elif label == 'X':
        # Index finger hooked
        for f in ['middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.85)
        hand = curl_finger(hand, 'thumb', 0.6)
        hand = curl_finger(hand, 'index', 0.4)  # Partially curled

    elif label == 'Y':
        # Thumb and pinky out (shaka/hang loose)
        for f in ['index', 'middle', 'ring']:
            hand = curl_finger(hand, f, 0.9)
        hand[4] = (-0.14, -0.10, -0.02)  # Thumb extended out

    elif label == '0':
        # O shape
        return generate_gesture('O')

    elif label == '1':
        # Index up only
        for f in ['middle', 'ring', 'pinky']:
            hand = curl_finger(hand, f, 0.9)
        hand = curl_finger(hand, 'thumb', 0.6)

    elif label == '2':
        return generate_gesture('V')

    elif label == '3':
        # Thumb, index, middle up
        for f in ['ring', 'pinky']:
            hand = curl_finger(hand, f, 0.9)

    elif label == '4':
        # Four fingers up, thumb tucked
        hand = curl_finger(hand, 'thumb', 0.7)

    elif label == '5':
        # All fingers spread wide
        hand[8] = (-0.06, -0.27, 0.0)
        hand[20] = (0.10, -0.22, 0.0)
        hand[4] = (-0.16, -0.12, -0.02)

    return hand


def add_noise(hand, noise_level=0.008):
    """Add random noise to make data more realistic."""
    noisy = {}
    for idx, (x, y, z) in hand.items():
        nx = x + np.random.normal(0, noise_level)
        ny = y + np.random.normal(0, noise_level)
        nz = z + np.random.normal(0, noise_level * 0.5)
        noisy[idx] = (nx, ny, nz)
    return noisy


def apply_random_transform(hand):
    """Apply random rotation, scaling, and offset for augmentation."""
    # Random scale (hand size variation)
    scale = np.random.uniform(0.7, 1.3)

    # Random rotation in radians (-20 to +20 degrees)
    angle = np.random.uniform(-0.35, 0.35)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    transformed = {}
    for idx, (x, y, z) in hand.items():
        # Scale
        sx, sy, sz = x * scale, y * scale, z * scale
        # Rotate in xy plane
        rx = sx * cos_a - sy * sin_a
        ry = sx * sin_a + sy * cos_a
        transformed[idx] = (rx, ry, sz)
    return transformed


def hand_to_features(hand):
    """Convert hand dict to flat feature list (63 values), normalized to wrist."""
    features = []
    wrist = hand[0]
    for i in range(NUM_LANDMARKS):
        x, y, z = hand[i]
        features.extend([x - wrist[0], y - wrist[1], z - wrist[2]])
    return features


# ═════════════════════════════════════════════════════════════════
#   MAIN GENERATOR
# ═════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SYNTHETIC DATA GENERATOR")
    print("=" * 60)
    print()

    # Define gestures to generate
    gestures = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ['1', '2', '3', '4', '5']

    print(f"[INFO] Generating {SAMPLES_PER_GESTURE} samples for {len(gestures)} gestures")
    print(f"[INFO] Total samples: {SAMPLES_PER_GESTURE * len(gestures)}")
    print(f"[INFO] Features per sample: {TOTAL_FEATURES}")
    print()

    # Create CSV header
    header = []
    for i in range(NUM_LANDMARKS):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    all_data = []

    for gesture in gestures:
        print(f"  Generating '{gesture}'...", end=" ")
        count = 0
        for _ in range(SAMPLES_PER_GESTURE):
            # Generate base pose
            hand = generate_gesture(gesture)
            # Apply random augmentation
            hand = apply_random_transform(hand)
            # Add noise
            hand = add_noise(hand, noise_level=np.random.uniform(0.005, 0.012))
            # Convert to features
            features = hand_to_features(hand)
            all_data.append(features + [gesture])
            count += 1
        print(f"OK - {count} samples")

    # Shuffle
    random.shuffle(all_data)

    # Save CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)

    print()
    print("=" * 60)
    print("  GENERATION COMPLETE!")
    print("=" * 60)
    print(f"  Total samples: {len(all_data)}")
    print(f"  Gestures: {gestures}")
    print(f"  Output: {OUTPUT_CSV}")
    print("=" * 60)
    print()
    print("  [OK] Now run: python train_model.py")
    print()


if __name__ == "__main__":
    main()
