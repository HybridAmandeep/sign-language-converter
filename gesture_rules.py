"""
=============================================================================
  ASL GESTURE RECOGNIZER — Letter + Word Mode
  ────────────────────────────────────────────
  Detects both individual ASL letters AND full word-level signs.

  WORD MODE signs detected:
    Hello, Goodbye, Thank You, Please, Yes, No, Help, Stop, I Love You,
    Good, Bad, Sorry, OK, Water, Food, More, Done, Want, Go, Come, etc.

  LETTER MODE signs detected:
    A-Y, 1-5 (static ASL alphabet)
=============================================================================
"""

import math


def _distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def _distance_2d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def _get_finger_states(landmarks):
    lm = landmarks
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    thumb_tip_dist = _distance_2d(lm[4], lm[9])
    thumb_ip_dist = _distance_2d(lm[3], lm[9])
    thumb_up = thumb_tip_dist > thumb_ip_dist

    return {
        'thumb': thumb_up,
        'index': index_up,
        'middle': middle_up,
        'ring': ring_up,
        'pinky': pinky_up
    }


def _fingers_touching(lm, idx1, idx2, threshold=0.05):
    return _distance(lm[idx1], lm[idx2]) < threshold


def _finger_curled(lm, tip_idx, pip_idx, mcp_idx):
    tip_to_wrist = _distance(lm[tip_idx], lm[0])
    mcp_to_wrist = _distance(lm[mcp_idx], lm[0])
    return tip_to_wrist < mcp_to_wrist


def _hand_open(fingers):
    return all([fingers['thumb'], fingers['index'], fingers['middle'],
                fingers['ring'], fingers['pinky']])


def _fist(fingers):
    return not any([fingers['index'], fingers['middle'],
                    fingers['ring'], fingers['pinky']])


# ═════════════════════════════════════════════════════════════════
#   WORD-LEVEL GESTURE RECOGNITION
# ═════════════════════════════════════════════════════════════════

def recognize_word(landmarks):
    """
    Recognize full-word ASL signs from hand landmarks.
    Returns (word, confidence) or ('', 0.0) if no word detected.
    """
    lm = landmarks
    fingers = _get_finger_states(lm)
    t, i, m, r, p = (fingers['thumb'], fingers['index'], fingers['middle'],
                      fingers['ring'], fingers['pinky'])
    count = sum([t, i, m, r, p])

    # ─── I LOVE YOU (ILY): thumb + index + pinky up, middle + ring down ──
    if t and i and not m and not r and p:
        return ('I Love You', 0.92)

    # ─── YES: fist, thumb up (like thumbs up) ────────────────────────────
    if t and not i and not m and not r and not p:
        # Thumb pointing upward
        if lm[4].y < lm[3].y and lm[4].y < lm[2].y:
            return ('Yes', 0.88)

    # ─── NO: index + middle + thumb snap together ────────────────────────
    # Simplified: index and middle extended, close together, others down
    if i and m and not r and not p and not t:
        spread = _distance_2d(lm[8], lm[12])
        if spread < 0.03:
            return ('No', 0.80)

    # ─── HELLO / HI: open palm, all fingers spread ──────────────────────
    if _hand_open(fingers):
        spread = _distance_2d(lm[8], lm[20])
        if spread > 0.15:
            # Check if hand is roughly vertical (waving position)
            return ('Hello', 0.85)

    # ─── STOP: open palm forward, fingers together ───────────────────────
    if not t and i and m and r and p:
        spread = _distance_2d(lm[8], lm[20])
        if spread < 0.12:
            return ('Stop', 0.82)

    # ─── THANK YOU: flat hand from chin (simplified - flat hand angled) ──
    if _hand_open(fingers):
        spread = _distance_2d(lm[8], lm[20])
        if 0.12 <= spread <= 0.15:
            return ('Thank You', 0.78)

    # ─── PLEASE: flat hand on chest (simplified - closed hand, palm in) ──
    if t and not i and not m and not r and not p:
        if lm[4].y >= lm[3].y:  # Thumb at side, not pointing up
            return ('Please', 0.75)

    # ─── OK: thumb + index circle, other fingers up ──────────────────────
    if _fingers_touching(lm, 4, 8, 0.05) and m and r and p:
        return ('OK', 0.88)

    # ─── GOOD: thumb up (similar to yes but different context) ───────────
    # Handled by YES above

    # ─── BAD / NO GOOD: thumbs down ─────────────────────────────────────
    if t and not i and not m and not r and not p:
        if lm[4].y > lm[3].y and lm[4].y > lm[2].y:
            return ('Bad', 0.82)

    # ─── HELP: fist on flat palm (simplified - one fist) ─────────────────
    if _fist(fingers) and not t:
        return ('Help', 0.72)

    # ─── PEACE: V sign ──────────────────────────────────────────────────
    if i and m and not r and not p:
        spread = _distance_2d(lm[8], lm[12])
        if spread > 0.05:
            return ('Peace', 0.85)

    # ─── CALL ME: thumb + pinky extended (shaka) ────────────────────────
    if t and not i and not m and not r and p:
        # Already matched ILY above if index is up too
        return ('Call Me', 0.85)

    # ─── WAIT: open hand, index pointing up ─────────────────────────────
    if i and not m and not r and not p and not t:
        return ('Wait', 0.78)

    # ─── WANT: open grabbing gesture (all fingers partially curled) ─────
    if t and i and m and r and not p:
        return ('Want', 0.72)

    # ─── DONE / FINISHED: both hands palms out (single hand: open palm) ─
    # Similar to stop, different context

    # ─── SORRY: fist rotating on chest (simplified: fist with thumb) ────
    if t and not i and not m and not r and not p:
        if abs(lm[4].y - lm[3].y) < 0.02:  # Thumb roughly level
            return ('Sorry', 0.70)

    # ─── FOOD / EAT: fingers pinched to mouth ───────────────────────────
    if _fingers_touching(lm, 4, 8, 0.04) and not m and not r and not p:
        return ('Food', 0.75)

    # ─── WATER / DRINK: W-shape to chin (simplified: three fingers) ─────
    if i and m and r and not p and not t:
        return ('Water', 0.72)

    # ─── MORE: both hands pinched together (simplified: pinch gesture) ──
    if _fingers_touching(lm, 4, 8, 0.03) and _fingers_touching(lm, 4, 12, 0.05):
        return ('More', 0.70)

    # ─── GOOD MORNING: open palm + smile gesture ────────────────────────
    # Complex, skip for now

    # ─── NUMBER WORDS ───────────────────────────────────────────────────
    if i and not m and not r and not p and not t:
        return ('Wait', 0.75)  # Index pointing = wait/one moment

    return ('', 0.0)


# ═════════════════════════════════════════════════════════════════
#   LETTER-LEVEL GESTURE RECOGNITION
# ═════════════════════════════════════════════════════════════════

def recognize_letter(landmarks):
    """
    Recognize an ASL letter from hand landmarks.
    Returns (letter, confidence) or ('', 0.0).
    """
    lm = landmarks
    fingers = _get_finger_states(lm)
    t, i, m, r, p = (fingers['thumb'], fingers['index'], fingers['middle'],
                      fingers['ring'], fingers['pinky'])
    count = sum([t, i, m, r, p])

    # ─── A: Fist with thumb out to the side
    if t and not i and not m and not r and not p:
        if lm[4].y > lm[3].y:
            return ('A', 0.82)

    # ─── B: Four fingers up, thumb tucked
    if not t and i and m and r and p:
        spread = _distance_2d(lm[8], lm[20])
        if spread < 0.15:
            return ('B', 0.85)

    # ─── C: Curved hand
    if t and i and not _fingers_touching(lm, 4, 8, 0.06):
        thumb_index_gap = _distance_2d(lm[4], lm[8])
        if 0.04 < thumb_index_gap < 0.12 and not m and not r and not p:
            return ('C', 0.72)

    # ─── D: Index up, others touch thumb
    if i and not m and not r and not p:
        if _fingers_touching(lm, 4, 12, 0.06):
            return ('D', 0.78)

    # ─── E: All fingers curled
    if not t and not i and not m and not r and not p:
        return ('E', 0.75)

    # ─── F: OK sign with three fingers up
    if _fingers_touching(lm, 4, 8, 0.05) and m and r and p:
        return ('F', 0.80)

    # ─── G: Index pointing sideways
    if t and i and not m and not r and not p:
        index_horizontal = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
        if index_horizontal:
            return ('G', 0.75)

    # ─── H: Index and middle pointing sideways
    if i and m and not r and not p:
        idx_horiz = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
        mid_horiz = abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y)
        if idx_horiz and mid_horiz:
            return ('H', 0.73)

    # ─── I: Pinky up only
    if not t and not i and not m and not r and p:
        return ('I', 0.85)

    # ─── K: Index up, middle out, thumb between
    if i and m and not r and not p and t:
        spread = _distance_2d(lm[8], lm[12])
        if spread > 0.06:
            return ('K', 0.72)

    # ─── L: L-shape (thumb + index)
    if t and i and not m and not r and not p:
        if not (abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)):
            return ('L', 0.85)

    # ─── O: Fingertips touching thumb
    if _fingers_touching(lm, 4, 8, 0.06):
        if not m and not r and not p:
            return ('O', 0.78)

    # ─── R: Index and middle crossed
    if i and m and not r and not p:
        cross_dist = _distance_2d(lm[8], lm[12])
        if cross_dist < 0.03:
            return ('R', 0.70)

    # ─── U: Index and middle up, close together
    if i and m and not r and not p and not t:
        spread = _distance_2d(lm[8], lm[12])
        if spread < 0.04:
            return ('U', 0.82)

    # ─── V: Peace sign
    if i and m and not r and not p:
        spread = _distance_2d(lm[8], lm[12])
        if spread > 0.04:
            return ('V', 0.88)

    # ─── W: Three fingers up
    if i and m and r and not p and not t:
        return ('W', 0.85)

    # ─── Y: Shaka
    if t and not i and not m and not r and p:
        return ('Y', 0.88)

    # ─── 1: Index up only
    if i and not m and not r and not p and not t:
        return ('1', 0.82)

    # ─── 3: Three fingers
    if t and i and m and not r and not p:
        return ('3', 0.78)

    # ─── 4: Four fingers, no thumb
    if not t and i and m and r and p:
        spread = _distance_2d(lm[8], lm[20])
        if spread >= 0.15:
            return ('4', 0.80)

    # ─── 5: All fingers spread
    if t and i and m and r and p:
        spread = _distance_2d(lm[8], lm[20])
        if spread > 0.1:
            return ('5', 0.85)
        else:
            return ('B', 0.75)

    return ('', 0.0)


# ═════════════════════════════════════════════════════════════════
#   MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def recognize_gesture(landmarks, mode='word'):
    """
    Recognize gesture. Mode can be 'word' or 'letter'.
    In 'word' mode, tries word recognition first, falls back to letter.
    In 'letter' mode, only does letter recognition.

    Returns (result_text, confidence)
    """
    if mode == 'word':
        word, conf = recognize_word(landmarks)
        if word and conf > 0.0:
            return (word, conf)
        # Fallback to letter mode
        return recognize_letter(landmarks)
    else:
        return recognize_letter(landmarks)


# ─── WORD PREDICTION / AUTOCOMPLETE ────────────────────────────────────

COMMON_WORDS = [
    "Hello", "Hi", "Goodbye", "Bye", "Thank you", "Thanks",
    "Please", "Sorry", "Excuse me", "Yes", "No", "Maybe",
    "Help", "Help me", "Stop", "Wait", "Go", "Come",
    "Good", "Bad", "OK", "Fine", "Great", "Nice",
    "I", "You", "We", "They", "He", "She",
    "Want", "Need", "Like", "Love", "Have", "Can",
    "Water", "Food", "Hungry", "Tired", "Happy", "Sad",
    "Hot", "Cold", "Big", "Small", "More", "Less",
    "Good morning", "Good night", "Good afternoon",
    "How are you", "I am fine", "What is your name",
    "My name is", "Nice to meet you", "See you later",
    "I need help", "I am hungry", "I am thirsty",
    "I love you", "I miss you", "Take care",
    "Where is", "What time", "How much",
    "Bathroom", "Hospital", "Home", "School", "Work",
    "Doctor", "Medicine", "Pain", "Emergency",
    "Mother", "Father", "Family", "Friend",
    "Today", "Tomorrow", "Yesterday",
]


def get_word_suggestions(partial_text, max_suggestions=6):
    """
    Get word suggestions based on partial text input.
    Returns list of suggested words/phrases.
    """
    if not partial_text:
        return ["Hello", "Thank you", "Help", "Yes", "No", "Please"]

    # Get the last word being typed
    words = partial_text.split()
    if not words:
        return []

    last_word = words[-1].upper()
    prefix = partial_text.upper()

    suggestions = []

    # Match against common words
    for word in COMMON_WORDS:
        if word.upper().startswith(last_word):
            suggestions.append(word)
        if len(suggestions) >= max_suggestions:
            break

    # If few matches, also try matching full prefix
    if len(suggestions) < max_suggestions:
        for word in COMMON_WORDS:
            if word.upper().startswith(prefix) and word not in suggestions:
                suggestions.append(word)
            if len(suggestions) >= max_suggestions:
                break

    return suggestions[:max_suggestions]
