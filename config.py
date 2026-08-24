import os

from dotenv import load_dotenv

load_dotenv()

TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-3.7-flash")
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.1-flash-tts-preview")
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "400"))
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "low")


def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()
