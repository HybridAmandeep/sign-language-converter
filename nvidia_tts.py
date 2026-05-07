"""
=============================================================================
  NVIDIA MAGPIE TTS — Cloud gRPC API Integration
  ──────────────────────────────────────────────
  Uses NVIDIA Riva gRPC cloud endpoint (grpc.nvcf.nvidia.com:443) for
  high-quality text-to-speech synthesis via the Magpie TTS Multilingual
  model. Falls back to pyttsx3 when offline.

  Supported voices:
    English : English-US.Female-1
    Spanish : English-US (with language_code override)

  Usage:
    from nvidia_tts import NvidiaTTS
    tts = NvidiaTTS()
    tts.speak("Hello, this is a test!")
    audio_bytes = tts.synthesize("Hello world")
=============================================================================
"""

import os
import io
import time
import wave
import threading
import tempfile

try:
    import grpc
    import riva.client
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "****"
)

NVIDIA_GRPC_URL = "grpc.nvcf.nvidia.com:443"

# Function ID for Magpie TTS Multilingual on NVCF
# Found on https://build.nvidia.com/nvidia/magpie-tts-multilingual
NVCF_FUNCTION_ID = "*****"

# Available voices
VOICES = {
    "aria": {
        "voice": "Magpie-Multilingual.EN-US.Aria",
        "lang": "en-US",
        "label": "Aria (English Female)",
        "sample_rate": 44100,
    },
    "diego": {
        "voice": "Magpie-Multilingual.ES-US.Diego",
        "lang": "es-US",
        "label": "Diego (Spanish Male)",
        "sample_rate": 44100,
    },
    "louise": {
        "voice": "Magpie-Multilingual.FR-FR.Louise",
        "lang": "fr-FR",
        "label": "Louise (French Female)",
        "sample_rate": 44100,
    },
}

DEFAULT_VOICE = "aria"


class NvidiaTTS:
    """
    NVIDIA Magpie TTS client using Riva gRPC cloud API.
    Thread-safe, with automatic fallback to pyttsx3.
    """

    def __init__(self, api_key=None, default_voice=None):
        self.api_key = api_key or NVIDIA_API_KEY
        self.default_voice = default_voice or DEFAULT_VOICE
        self._lock = threading.Lock()
        self._speaking = False
        self._pyttsx3_engine = None
        self._tts_service = None
        self._auth = None

        # Initialize Riva client
        self.api_available = self._init_riva()

        if not self.api_available:
            print("[NVIDIA TTS] Cloud gRPC API not reachable. Will use pyttsx3 fallback.")
            self._init_pyttsx3()
        else:
            print("[NVIDIA TTS] Connected to NVIDIA Riva Cloud (grpc.nvcf.nvidia.com)!")
            print(f"  -> Default voice: {VOICES[self.default_voice]['label']}")

    def _init_riva(self):
        """Initialize Riva gRPC client with cloud auth."""
        if not RIVA_AVAILABLE:
            print("[NVIDIA TTS] 'nvidia-riva-client' package not installed.")
            return False
        if not self.api_key:
            print("[NVIDIA TTS] No API key configured.")
            return False

        try:
            # Set up metadata for cloud API auth
            metadata = [
                ("function-id", NVCF_FUNCTION_ID),
                ("authorization", f"Bearer {self.api_key}")
            ]

            self._auth = riva.client.Auth(
                use_ssl=True,
                uri=NVIDIA_GRPC_URL,
                metadata_args=metadata,
            )

            self._tts_service = riva.client.SpeechSynthesisService(self._auth)

            # Test with a minimal synthesis
            voice_info = VOICES[self.default_voice]
            resp = self._tts_service.synthesize(
                "test",
                voice_name=voice_info["voice"],
                language_code=voice_info["lang"],
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=voice_info["sample_rate"],
            )

            if resp and hasattr(resp, 'audio') and len(resp.audio) > 0:
                return True
            else:
                print("[NVIDIA TTS] API responded but no audio data.")
                return False

        except Exception as e:
            print(f"[NVIDIA TTS] Riva gRPC init failed: {e}")
            return False

    def _init_pyttsx3(self):
        """Initialize pyttsx3 as fallback TTS engine."""
        if not PYTTSX3_AVAILABLE:
            print("[NVIDIA TTS] pyttsx3 not available either. TTS disabled.")
            return
        try:
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty('rate', 150)
            self._pyttsx3_engine.setProperty('volume', 1.0)
            voices = self._pyttsx3_engine.getProperty('voices')
            for voice in voices:
                if 'english' in voice.name.lower() or 'zira' in voice.name.lower():
                    self._pyttsx3_engine.setProperty('voice', voice.id)
                    break
            print("[NVIDIA TTS] pyttsx3 fallback initialized.")
        except Exception as e:
            print(f"[NVIDIA TTS] pyttsx3 init failed: {e}")

    @property
    def is_speaking(self):
        return self._speaking

    def get_voices(self):
        """Return dict of available voices."""
        return {k: v["label"] for k, v in VOICES.items()}

    def synthesize(self, text, voice_key=None):
        """
        Synthesize text to WAV audio bytes using NVIDIA Riva gRPC.

        Args:
            text: Text to synthesize
            voice_key: Voice key ('aria', 'diego') or None for default

        Returns:
            bytes: WAV audio data, or None on failure
        """
        if not text or not text.strip():
            return None

        voice_key = voice_key or self.default_voice
        voice_info = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])

        if not self.api_available or not self._tts_service:
            return None

        try:
            resp = self._tts_service.synthesize(
                text.strip(),
                voice_name=voice_info["voice"],
                language_code=voice_info["lang"],
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hz=voice_info["sample_rate"],
            )

            if resp and hasattr(resp, 'audio') and len(resp.audio) > 0:
                # Wrap raw PCM in WAV container for browser playback
                wav_bytes = self._pcm_to_wav(
                    resp.audio,
                    sample_rate=voice_info["sample_rate"],
                    channels=1,
                    sample_width=2
                )
                return wav_bytes
            else:
                print("[NVIDIA TTS] Synthesis returned no audio.")
                return None

        except Exception as e:
            print(f"[NVIDIA TTS] Synthesis error: {e}")
            return None

    def _pcm_to_wav(self, pcm_data, sample_rate=44100, channels=1, sample_width=2):
        """Wrap raw PCM bytes in a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()

    def speak(self, text, voice_key=None):
        """
        Synthesize and play text audio. Non-blocking (runs in thread).
        Falls back to pyttsx3 if cloud API is unavailable.
        """
        def _speak():
            self._speaking = True
            try:
                if self.api_available:
                    audio_data = self.synthesize(text, voice_key)
                    if audio_data:
                        self._play_wav_bytes(audio_data)
                        return

                # Fallback to pyttsx3
                if self._pyttsx3_engine:
                    with self._lock:
                        self._pyttsx3_engine.say(text)
                        self._pyttsx3_engine.runAndWait()
                else:
                    print(f"[NVIDIA TTS] No TTS engine available. Text: '{text}'")
            except Exception as e:
                print(f"[NVIDIA TTS] Speak error: {e}")
            finally:
                self._speaking = False

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def speak_blocking(self, text, voice_key=None):
        """Synthesize and play text audio. Blocking version."""
        self._speaking = True
        try:
            if self.api_available:
                audio_data = self.synthesize(text, voice_key)
                if audio_data:
                    self._play_wav_bytes(audio_data)
                    return

            if self._pyttsx3_engine:
                with self._lock:
                    self._pyttsx3_engine.say(text)
                    self._pyttsx3_engine.runAndWait()
        except Exception as e:
            print(f"[NVIDIA TTS] Speak error: {e}")
        finally:
            self._speaking = False

    def _play_wav_bytes(self, wav_bytes):
        """Play WAV audio bytes using the system's default audio player."""
        try:
            # Try pygame first (best cross-platform)
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=1)

            sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
            sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.05)
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[NVIDIA TTS] pygame playback failed: {e}")

        try:
            # Fallback: save to temp file and play with system command
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(wav_bytes)
                temp_path = f.name

            import platform
            system = platform.system()
            if system == 'Windows':
                import winsound
                winsound.PlaySound(temp_path, winsound.SND_FILENAME)
            elif system == 'Darwin':
                os.system(f'afplay "{temp_path}"')
            else:
                os.system(f'aplay "{temp_path}" 2>/dev/null || paplay "{temp_path}" 2>/dev/null')

            os.unlink(temp_path)
        except Exception as e:
            print(f"[NVIDIA TTS] Audio playback failed: {e}")

    def save_audio(self, text, output_path, voice_key=None):
        """Synthesize text and save audio to a WAV file."""
        audio_data = self.synthesize(text, voice_key)
        if audio_data:
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            print(f"[NVIDIA TTS] Audio saved to: {output_path}")
            return True
        return False


# ─── MODULE-LEVEL SINGLETON ─────────────────────────────────────────────────
_tts_instance = None


def get_tts():
    """Get or create the global TTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = NvidiaTTS()
    return _tts_instance


# ─── STANDALONE TEST ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NVIDIA MAGPIE TTS — Riva gRPC Cloud Test")
    print("=" * 60)
    print()

    tts = NvidiaTTS()
    print()

    if tts.api_available:
        print("[TEST 1] Synthesizing 'Hello, this is a speech synthesizer.'")
        audio = tts.synthesize("Hello, this is a speech synthesizer.")
        if audio:
            size_kb = len(audio) / 1024
            print(f"  -> Audio received: {size_kb:.1f} KB")

            # Save test audio
            test_path = os.path.join(os.path.dirname(__file__), "test_output.wav")
            tts.save_audio("Hello, this is a speech synthesizer.", test_path)
        else:
            print("  -> Synthesis failed!")

        print()
        print("[TEST 2] Speaking 'Welcome to the sign language converter.'")
        tts.speak_blocking("Welcome to the sign language converter.")
        print("  -> Done!")

        print()
        print("[TEST 3] Available voices:")
        for key, label in tts.get_voices().items():
            print(f"  -> {key}: {label}")
    else:
        print("[TEST] API not available. Testing pyttsx3 fallback...")
        tts.speak_blocking("This is a fallback test using local speech synthesis.")
        print("  -> Fallback test done!")

    print()
    print("=" * 60)
    print("  Tests complete!")
    print("=" * 60)
