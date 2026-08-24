import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_config(key: str, default=None):
    """Get config from Streamlit secrets first, then environment variables."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key, default)


TEXT_MODEL = get_config("TEXT_MODEL", "gemini-3.7-flash")
TTS_MODEL = get_config("TTS_MODEL", "gemini-3.1-flash-tts-preview")
TTS_VOICE = get_config("TTS_VOICE", "Kore")

MAX_OUTPUT_TOKENS = int(get_config("MAX_OUTPUT_TOKENS", "400"))
THINKING_LEVEL = get_config("THINKING_LEVEL", "low")


def get_api_key() -> str:
    return get_config("GEMINI_API_KEY", "").strip()